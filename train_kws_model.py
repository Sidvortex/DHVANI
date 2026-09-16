"""
Keyword-spotting training pipeline: DS-CNN, int8 quantized, TFLite Micro export.

USAGE:
  1. Record real data with record_samples.py into data/keyword, data/background, data/unknown
  2. Run: python3 train_kws_model.py
  3. If no data/ folder is found yet, this generates synthetic placeholder audio so you
     can verify the whole pipeline runs and see real parameter counts / model size before
     your real recordings exist. Re-run once data/ exists to train the real model.

Outputs:
  - kws_model.tflite   (int8 quantized model, for reference/testing on a PC)
  - model_data.h       (C array to #include in the ESP-IDF firmware)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

SAMPLE_RATE = 16000
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
N_MELS = 40
FRAME_LENGTH = 400   # 25 ms
FRAME_STEP = 160     # 10 ms
LABELS = ["background", "unknown", "keyword"]   # index order = model output order
DATA_DIR = "data"
SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load real data if present, else generate synthetic placeholder audio
# ---------------------------------------------------------------------------

def fit_length(audio):
    if len(audio) > N_SAMPLES:
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start:start + N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        pad = N_SAMPLES - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    return audio

def load_real_data():
    clips, labels = [], []
    for label_idx, label in enumerate(LABELS):
        folder = os.path.join(DATA_DIR, label)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        for f in files:
            sr, audio = wavfile.read(f)
            audio = audio.astype(np.float32) / 32768.0
            clips.append(fit_length(audio))
            labels.append(label_idx)
    return clips, labels

def synth_clip(label_idx, rng):
    t = np.linspace(0, DURATION_S, N_SAMPLES, endpoint=False)
    if label_idx == 0:  # background: quiet noise
        audio = rng.normal(0, 0.01, N_SAMPLES)
    elif label_idx == 1:  # unknown: broadband noise burst, variable envelope
        env = np.clip(rng.normal(0.5, 0.2, N_SAMPLES), 0, 1)
        audio = rng.normal(0, 0.15, N_SAMPLES) * env
    else:  # "keyword" stand-in: overlaid tones with a word-like envelope, so
           # there's at least some learnable structure separating it from noise
        freqs = rng.uniform(200, 900, 3)
        audio = sum(np.sin(2 * np.pi * f * t) for f in freqs) / 3
        center = N_SAMPLES / 2
        env = np.exp(-((np.arange(N_SAMPLES) - center) ** 2) / (2 * (N_SAMPLES / 5) ** 2))
        audio = audio * env * 0.3 + rng.normal(0, 0.02, N_SAMPLES)
    return audio.astype(np.float32)

def load_synthetic_data(n_per_class=80):
    print("!! No data/ folder found -- generating SYNTHETIC placeholder audio.")
    print("!! This verifies the pipeline runs and reports real model size, but it")
    print("!! will NOT produce a working keyword detector. Record real audio with")
    print("!! record_samples.py, then re-run this script.\n")
    rng = np.random.default_rng(SEED)
    clips, labels = [], []
    for label_idx in range(len(LABELS)):
        for _ in range(n_per_class):
            clips.append(synth_clip(label_idx, rng))
            labels.append(label_idx)
    return clips, labels

# ---------------------------------------------------------------------------
# 2. Augmentation (applied on the fly to real data; skipped for synthetic demo)
# ---------------------------------------------------------------------------

def augment(audio, rng):
    audio = audio.copy()
    if rng.random() < 0.7:  # background noise mix at a random SNR
        snr_db = rng.uniform(5, 20)
        noise = rng.normal(0, 1, len(audio))
        sig_p = np.mean(audio ** 2) + 1e-9
        noise_p = np.mean(noise ** 2) + 1e-9
        scale = np.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10)))
        audio = audio + noise * scale
    if rng.random() < 0.5:  # time shift
        shift = int(rng.uniform(-0.1, 0.1) * SAMPLE_RATE)
        audio = np.roll(audio, shift)
    if rng.random() < 0.3:  # gain
        audio = audio * rng.uniform(0.7, 1.3)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# 3. Feature extraction: log-mel spectrogram (40 mels). The ESP32 firmware
#    must reproduce these exact params (frame length/step, mel count, range)
#    with esp-dsp at inference time, or accuracy will silently fall apart.
# ---------------------------------------------------------------------------

FFT_LENGTH = 512  # power of 2, required by esp-dsp's radix-2 FFT on-device;
                   # frame_length (400) is still the actual analysis window,
                   # zero-padded to FFT_LENGTH before transform.

_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS,
    num_spectrogram_bins=FFT_LENGTH // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=20.0,
    upper_edge_hertz=8000.0,
)

def extract_features(audio):
    stft = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                           fft_length=FFT_LENGTH)
    spec = tf.abs(stft)
    mel = tf.matmul(spec, _mel_matrix)
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel.numpy()  # (num_frames, N_MELS)

# ---------------------------------------------------------------------------
# 4. Build dataset
# ---------------------------------------------------------------------------

def build_dataset():
    have_real = os.path.isdir(DATA_DIR) and any(
        glob.glob(os.path.join(DATA_DIR, l, "*.wav")) for l in LABELS
    )
    if have_real:
        clips, labels = load_real_data()
        is_synthetic = False
    else:
        clips, labels = load_synthetic_data()
        is_synthetic = True

    rng = np.random.default_rng(SEED)
    X, y = [], []
    n_aug = 1 if is_synthetic else 4   # multiply real recordings harder
    for audio, label in zip(clips, labels):
        for i in range(n_aug):
            a = audio if i == 0 else augment(audio, rng)
            X.append(extract_features(a))
            y.append(label)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, is_synthetic

# ---------------------------------------------------------------------------
# 5. Model: compact DS-CNN (3-class head: background / unknown / keyword)
# ---------------------------------------------------------------------------

def build_model(input_shape, n_classes):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(4):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

# ---------------------------------------------------------------------------
# 6. Train + quantize + export
# ---------------------------------------------------------------------------

def export_c_array(tflite_bytes, out_path="model_data.h", var_name="g_model_data"):
    with open(out_path, "w") as f:
        f.write("// Auto-generated by train_kws_model.py. Do not edit by hand.\n")
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"alignas(16) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")

def export_mel_matrix(out_path="mel_filterbank.h"):
    mat = _mel_matrix.numpy().astype(np.float32)  # (FFT_LENGTH//2+1, N_MELS)
    with open(out_path, "w") as f:
        f.write("// Auto-generated by train_kws_model.py. Do not edit by hand.\n")
        f.write("// Row-major: [spectrogram_bin][mel_bin], multiply |FFT| (1..FFT_LENGTH/2+1 bins) by this.\n")
        f.write("#pragma once\n\n")
        f.write(f"#define MEL_SPEC_BINS {mat.shape[0]}\n")
        f.write(f"#define MEL_N_MELS {mat.shape[1]}\n\n")
        f.write("const float g_mel_filterbank[MEL_SPEC_BINS][MEL_N_MELS] = {\n")
        for row in mat:
            f.write("  {" + ", ".join(f"{v:.8f}f" for v in row) + "},\n")
        f.write("};\n")
    print(f"Wrote {out_path}: {mat.shape[0]}x{mat.shape[1]} mel filterbank")

def main():
    X, y, is_synthetic = build_dataset()
    X = X[..., np.newaxis]  # add channel dim -> (N, frames, mels, 1)
    print(f"Dataset: {X.shape[0]} clips, feature shape {X.shape[1:]}, synthetic={is_synthetic}")

    idx = np.random.default_rng(SEED).permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    model = build_model(X.shape[1:], len(LABELS))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    epochs = 5 if is_synthetic else 30
    model.fit(X[train_idx], y[train_idx], validation_data=(X[val_idx], y[val_idx]),
              epochs=epochs, batch_size=32, verbose=2)

    n_params = model.count_params()

    def representative_dataset():
        for i in range(min(100, len(X))):
            yield [X[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open("kws_model.tflite", "wb") as f:
        f.write(tflite_model)
    export_c_array(tflite_model)
    export_mel_matrix()

    print("\n" + "=" * 60)
    print(f"Parameters:        {n_params:,}")
    print(f"int8 model size:   {len(tflite_model):,} bytes ({len(tflite_model)/1024:.1f} KB)")
    print(f"Input shape:       {X.shape[1:]}  (frames x mel bins x channel)")
    print(f"Classes:           {LABELS}")
    print("=" * 60)
    if is_synthetic:
        print("\nReminder: trained on SYNTHETIC placeholder data. Record real")
        print("audio with record_samples.py, then re-run this script.")

if __name__ == "__main__":
    main()
