"""
Minimal receiver for the ESP32 firmware's trigger-audio POST.

Run:  python3 server.py
Test: python3 -c "import requests; requests.post('http://localhost:5000/audio',
        data=open('some.wav','rb').read()[44:], headers={'X-Sample-Rate':'16000'})"

Today's scope: plain HTTP POST of raw 16-bit PCM, no WSS/TLS, no persistent
connection. This exists to prove "trigger -> server receives audio" end to
end. Vosk transcription is optional -- if it's not installed/model not
downloaded, the server still saves the clip and returns 200 so the pipeline
isn't blocked on it.
"""

import os
import time
import wave
from flask import Flask, request, jsonify

app = Flask(__name__)
OUT_DIR = "received_audio"
os.makedirs(OUT_DIR, exist_ok=True)

_vosk_model = None
def get_vosk_model():
    global _vosk_model
    if _vosk_model is None:
        try:
            from vosk import Model
            # Download a small model yourself (e.g. vosk-model-small-en-us-0.15)
            # and point this at the extracted folder.
            _vosk_model = Model("model")
        except Exception as e:
            print(f"Vosk not available ({e}) -- will just save audio, no transcript.")
            _vosk_model = False
    return _vosk_model

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/audio", methods=["POST"])
def receive_audio():
    t0 = time.time()
    pcm_bytes = request.get_data()
    sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))

    fname = os.path.join(OUT_DIR, f"trigger_{int(t0*1000)}.wav")
    with wave.open(fname, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    n_samples = len(pcm_bytes) // 2
    duration_s = n_samples / sample_rate
    print(f"[{time.strftime('%H:%M:%S')}] received {len(pcm_bytes)} bytes "
          f"({duration_s:.2f}s @ {sample_rate}Hz) -> {fname}")

    transcript = None
    model = get_vosk_model()
    if model:
        import json
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(model, sample_rate)
        rec.AcceptWaveform(pcm_bytes)
        transcript = json.loads(rec.FinalResult()).get("text", "")
        print(f"  transcript: {transcript!r}")

    elapsed_ms = (time.time() - t0) * 1000
    return jsonify(status="received", bytes=len(pcm_bytes), duration_s=round(duration_s, 2),
                   transcript=transcript, server_processing_ms=round(elapsed_ms, 1))

if __name__ == "__main__":
    print(f"Listening on :5000 -- saving clips to {OUT_DIR}/")
    print("Point SERVER_URL in the firmware at http://<this-machine's-LAN-IP>:5000/audio")
    app.run(host="0.0.0.0", port=5000)
