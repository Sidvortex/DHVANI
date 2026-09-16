// main.cc -- KWS wake-word detector + trigger streaming, for ESP32-S3.
//
// Pipeline: I2S mic -> energy VAD gate -> (on speech) 1s window -> FFT + mel
// filterbank -> int8 quantize -> TFLite Micro inference -> posterior
// smoothing -> on confirmed "keyword": WiFi HTTP POST the raw audio.
//
// ONE-TIME SETUP in your ESP-IDF project:
//   idf.py add-dependency "espressif/esp-tflite-micro"
//   idf.py add-dependency "espressif/esp-dsp"
//   Copy model_data.h and mel_filterbank.h (from train_kws_model.py output)
//   into this main/ directory, next to this file.
//   Fill in WIFI_SSID / WIFI_PASS / SERVER_URL below.
//
// UNTESTED ON HARDWARE -- there's no ESP32 in the sandbox this was written
// in, so this has not been compiled or flashed. Written from well-established
// ESP-IDF/TFLM patterns, but budget time to debug. Likeliest trouble spots,
// in order:
//   1. SHIFT_AMOUNT below (I2S sample scaling -- if audio reads as silence
//      or garbage, this is the first thing to try adjusting)
//   2. kTensorArenaSize (see the comment at its definition)
//   3. compute_log_mel() -- the one function whose output I could not check
//      against the Python training script's output on real hardware
//   4. resolver.Add<Op>() calls -- if AllocateTensors() fails, the log names
//      the missing op

#include <cstring>
#include <cmath>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/i2s.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_client.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_dsp.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "mel_filterbank.h"

static const char* TAG = "kws";

// ---------------- Fill these in ----------------
#define WIFI_SSID   "YOUR_WIFI_SSID"
#define WIFI_PASS   "YOUR_WIFI_PASSWORD"
#define SERVER_URL  "http://YOUR_SERVER_IP:5000/audio"

// ---------------- Audio / I2S ----------------
#define I2S_BCK_GPIO   GPIO_NUM_4
#define I2S_WS_GPIO    GPIO_NUM_5
#define I2S_DIN_GPIO   GPIO_NUM_6
#define SAMPLE_RATE    16000
#define N_SAMPLES      16000   // 1 second @ 16kHz -- must match train_kws_model.py
#define SHIFT_AMOUNT   14      // INMP441's 24-bit sample in a 32-bit slot -> ~16-bit
                                // range. Empirical; 11-14 is the usual range. If your
                                // logged energy looks near-zero, lower this; if it
                                // saturates/clips, raise it.

// ---------------- Feature extraction -- must mirror train_kws_model.py ----------------
#define FRAME_LEN   400   // 25ms
#define FRAME_STEP  160   // 10ms
#define FFT_LEN     512
#define N_FRAMES    98    // (N_SAMPLES - FRAME_LEN) / FRAME_STEP + 1
#define N_MELS      40

// ---------------- VAD / trigger ----------------
#define VAD_RMS_THRESHOLD  400.0f   // tune against your own room's noise floor --
                                     // log `energy` for a few seconds of silence
                                     // and set this a bit above what you see
#define TRIGGER_THRESHOLD  0.85f
#define SMOOTH_WINDOW      3

// Must match LABELS order in train_kws_model.py.
enum { LABEL_BACKGROUND = 0, LABEL_UNKNOWN = 1, LABEL_KEYWORD = 2 };

// ---------------- TFLM ----------------
// Generous starting size -- deliberately not hand-tuned. The single largest
// activation tensor in this model (right after the first conv) is ~61KB, so
// TFLM's real arena need is likely well above a naive "model is tiny"
// guess. After first successful boot, read the "Tensor arena used" log line
// and shrink this #define to about that + 10% headroom.
constexpr int kTensorArenaSize = 200 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

// ================= WiFi =================
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static void wifi_event_handler(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();  // brute-force reconnect -- fine for a 1-day build
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void) {
    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi connected");
}

// Simplest-thing-that-works: one HTTP POST per trigger, raw PCM body, sample
// rate in a header. No WSS/TLS/pre-roll buffer yet -- add those only after
// this path is proven end to end. Matches server.py, which expects the same.
static void stream_audio(const int16_t* pcm, size_t n_samples) {
    esp_http_client_config_t config = {};
    config.url = SERVER_URL;
    config.method = HTTP_METHOD_POST;
    config.timeout_ms = 5000;
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_header(client, "Content-Type", "application/octet-stream");
    esp_http_client_set_header(client, "X-Sample-Rate", "16000");

    esp_err_t err = esp_http_client_open(client, n_samples * sizeof(int16_t));
    if (err == ESP_OK) {
        esp_http_client_write(client, (const char*)pcm, n_samples * sizeof(int16_t));
        esp_http_client_fetch_headers(client);
        ESP_LOGI(TAG, "POST status=%d", esp_http_client_get_status_code(client));
    } else {
        ESP_LOGE(TAG, "HTTP open failed: %s", esp_err_to_name(err));
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
}

// ================= I2S mic =================
static void i2s_mic_init(void) {
    i2s_config_t i2s_config = {};
    i2s_config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    i2s_config.sample_rate = SAMPLE_RATE;
    i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
    i2s_config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    i2s_config.dma_buf_count = 4;
    i2s_config.dma_buf_len = 256;
    i2s_config.use_apll = false;

    i2s_pin_config_t pin_config = {};
    pin_config.bck_io_num = I2S_BCK_GPIO;
    pin_config.ws_io_num = I2S_WS_GPIO;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num = I2S_DIN_GPIO;

    ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pin_config));
}

// NOTE ON RAM: this raw scratch buffer is 16000 * 4 bytes = 64KB, on top of
// the 32KB int16 pcm_window it feeds. That's ~96KB just for audio capture --
// more than a napkin estimate suggests. If you're tight on RAM once this is
// running, read+shift in smaller chunks (e.g. 512-sample blocks in a loop)
// instead of one full-second raw buffer. Left as the simple version for now.
static void i2s_read_window(int16_t* out) {
    static int32_t raw[N_SAMPLES];
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, raw, sizeof(raw), &bytes_read, portMAX_DELAY);
    size_t got = bytes_read / sizeof(int32_t);
    for (size_t i = 0; i < got && i < N_SAMPLES; i++) {
        out[i] = (int16_t)(raw[i] >> SHIFT_AMOUNT);
    }
}

static float rms_of(const int16_t* pcm, size_t n) {
    double sum_sq = 0;
    for (size_t i = 0; i < n; i++) sum_sq += (double)pcm[i] * pcm[i];
    return sqrtf((float)(sum_sq / n));
}

// ================= Feature extraction =================
static float hann[FRAME_LEN];
static void init_hann(void) {
    for (int i = 0; i < FRAME_LEN; i++) {
        hann[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (FRAME_LEN - 1));
    }
}

// Fills mel_out[N_FRAMES][N_MELS] (log-mel spectrogram) from a 1s int16
// window. This is the function most likely to have a subtle bug -- it was
// never run against real audio or cross-checked against the Python side.
// To sanity check: log mel_out for a loud, sustained tone and confirm energy
// shows up concentrated in a plausible mel bin, not spread uniformly or zero.
static void compute_log_mel(const int16_t* pcm, float mel_out[N_FRAMES][N_MELS]) {
    static float fft_buf[FFT_LEN * 2];  // interleaved re/im

    for (int f = 0; f < N_FRAMES; f++) {
        int start = f * FRAME_STEP;
        for (int i = 0; i < FFT_LEN; i++) {
            float sample = (i < FRAME_LEN && (start + i) < (int)N_SAMPLES)
                               ? (pcm[start + i] / 32768.0f) * hann[i]
                               : 0.0f;
            fft_buf[2 * i] = sample;
            fft_buf[2 * i + 1] = 0.0f;
        }
        dsps_fft2r_fc32(fft_buf, FFT_LEN);
        dsps_bit_rev_fc32(fft_buf, FFT_LEN);

        float mag[FFT_LEN / 2 + 1];
        for (int k = 0; k <= FFT_LEN / 2; k++) {
            float re = fft_buf[2 * k];
            float im = fft_buf[2 * k + 1];
            mag[k] = sqrtf(re * re + im * im);
        }
        for (int m = 0; m < N_MELS; m++) {
            float acc = 0.0f;
            for (int k = 0; k <= FFT_LEN / 2; k++) {
                acc += mag[k] * g_mel_filterbank[k][m];
            }
            mel_out[f][m] = logf(acc + 1e-6f);
        }
    }
}

// ================= TFLM setup =================
static void tflm_init(void) {
    dsps_fft2r_init_fc32(NULL, FFT_LEN);  // one-time twiddle-factor init

    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!");
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();           // GlobalAveragePooling2D lowers to Mean
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();
    // If AllocateTensors() below fails with "didn't find op", the error
    // names the missing op -- add resolver.Add<Op>() for it here.

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
    ESP_LOGI(TAG, "Tensor arena used: %d / %d bytes",
             (int)interpreter->arena_used_bytes(), kTensorArenaSize);
}

// Quantizes mel[] into the model's int8 input using its own scale/zero_point
// (read at runtime -- stays correct even if you retrain/requantize later).
static void quantize_into_input(float mel[N_FRAMES][N_MELS]) {
    float scale = model_input->params.scale;
    int zero_point = model_input->params.zero_point;
    int8_t* dst = model_input->data.int8;
    int idx = 0;
    for (int f = 0; f < N_FRAMES; f++) {
        for (int m = 0; m < N_MELS; m++) {
            int32_t q = (int32_t)roundf(mel[f][m] / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            dst[idx++] = (int8_t)q;
        }
    }
}

// ================= Main loop =================
extern "C" void app_main(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    init_hann();
    i2s_mic_init();
    tflm_init();
    wifi_init();

    static int16_t pcm_window[N_SAMPLES];
    static float mel[N_FRAMES][N_MELS];
    float recent_conf[SMOOTH_WINDOW] = {0};
    int smooth_idx = 0;

    ESP_LOGI(TAG, "Listening...");
    while (true) {
        i2s_read_window(pcm_window);
        float energy = rms_of(pcm_window, N_SAMPLES);

        if (energy < VAD_RMS_THRESHOLD) {
            vTaskDelay(pdMS_TO_TICKS(20));  // idle: cheap, keeps CPU low
            continue;
        }

        compute_log_mel(pcm_window, mel);
        quantize_into_input(mel);

        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed");
            continue;
        }

        float out_scale = model_output->params.scale;
        int out_zp = model_output->params.zero_point;
        float keyword_conf = (model_output->data.int8[LABEL_KEYWORD] - out_zp) * out_scale;

        recent_conf[smooth_idx] = keyword_conf;
        smooth_idx = (smooth_idx + 1) % SMOOTH_WINDOW;
        float avg = (recent_conf[0] + recent_conf[1] + recent_conf[2]) / SMOOTH_WINDOW;

        ESP_LOGI(TAG, "energy=%.0f keyword_conf=%.2f smoothed=%.2f", energy, keyword_conf, avg);

        if (avg > TRIGGER_THRESHOLD) {
            ESP_LOGI(TAG, "TRIGGER -- streaming audio");
            stream_audio(pcm_window, N_SAMPLES);
            for (int i = 0; i < SMOOTH_WINDOW; i++) recent_conf[i] = 0;  // avoid re-trigger spam
        }
    }
}
