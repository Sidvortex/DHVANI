"""
Quick sample recorder. Run this on YOUR LAPTOP (not the ESP32) -- it needs a
real microphone, which this sandbox doesn't have.

Install once:  pip install sounddevice numpy scipy

Usage:
  python3 record_samples.py keyword      # say your wake word each time
  python3 record_samples.py unknown      # say random OTHER words/phrases
  python3 record_samples.py background   # stay quiet / normal room noise

Recommended counts for a 1-day build: ~120 keyword, ~120 unknown, ~80 background.
Vary distance from the mic and tone of voice across takes -- that variation
matters more than raw count.

Press Enter to record 1 second, Ctrl+C to stop. Files land in data/<label>/.
"""

import sys
import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SAMPLE_RATE = 16000
DURATION_S = 1.0

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("keyword", "unknown", "background"):
        print("Usage: python3 record_samples.py [keyword|unknown|background]")
        sys.exit(1)

    label = sys.argv[1]
    out_dir = os.path.join("data", label)
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    count = len(existing)

    print(f"Recording '{label}' -> {out_dir}/  (starting at #{count})")
    print("Press Enter to record 1s, Ctrl+C when done.\n")

    try:
        while True:
            input(f"[{count}] Enter to record...")
            audio = sd.rec(int(DURATION_S * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                            channels=1, dtype="int16")
            sd.wait()
            path = os.path.join(out_dir, f"{label}_{count:04d}.wav")
            wavfile.write(path, SAMPLE_RATE, audio)
            peak = int(np.abs(audio).max())
            flag = "  <-- very quiet, check mic/distance" if peak < 500 else ""
            print(f"  saved {path}  (peak={peak}){flag}")
            count += 1
    except KeyboardInterrupt:
        print(f"\nStopped. {count} total clips in {out_dir}/")

if __name__ == "__main__":
    main()
