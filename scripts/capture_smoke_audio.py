from __future__ import annotations

import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "smoke"
SAMPLE_RATE = 16000
CHANNELS = 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Record short consented Day-2 smoke-test clips.")
    parser.add_argument("--speaker", required=True, help="Speaker code, e.g. speaker01")
    parser.add_argument("--count", required=True, type=int, help="Number of clips")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", default=None, help="Optional sounddevice input device index")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass

    print("Use only a consenting participant. Do not record private calls or personal financial information.")
    for i in range(1, args.count + 1):
        path = OUT / f"{args.speaker}_{i:02d}.wav"
        print(f"\nPreparing: {path.name}")
        input("Press Enter, then recording starts in 2 seconds...")
        time.sleep(2)
        print("RECORDING...")
        audio = sd.rec(
            int(round(args.seconds * SAMPLE_RATE)),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device,
        )
        sd.wait()
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"Saved: {path}")

    print("\nSmoke capture complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
