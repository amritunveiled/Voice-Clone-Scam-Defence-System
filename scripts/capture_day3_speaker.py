from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Day-3 enrollment + scripted utterances for one consented speaker.")
    parser.add_argument("--speaker", required=True, help="Example: spk01")
    parser.add_argument("--language", required=True, choices=["hindi", "english", "kannada"])
    parser.add_argument("--category", choices=["all", "benign", "urgent_legit", "scam"], default="all")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        device = int(args.device) if args.device is not None else None
    except ValueError:
        device = args.device

    root = Path(__file__).resolve().parents[1]
    speaker = root / "data" / "raw" / "participants" / args.speaker
    enrollment = speaker / "enrollment"
    consent = speaker / "consent"
    qc = speaker / "qc"
    for p in [enrollment, consent, qc]:
        p.mkdir(parents=True, exist_ok=True)

    if not (consent / "consent_received.txt").exists():
        raise SystemExit(
            f"STOP: {consent / 'consent_received.txt'} not found. "
            "Create it only after written consent has been obtained."
        )

    print(f"Speaker: {args.speaker} | Language: {args.language}")
    print("All recordings are mono, 16 kHz, PCM-16 WAV.")

    # Enrollment: five independent clips, 5-10 s.
    for idx in range(1, 6):
        path = enrollment / f"{args.speaker}_enroll_{idx:02d}.wav"
        input(f"Press Enter when ready for enrollment clip {idx}/5...")
        time.sleep(1)
        print("RECORDING")
        audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device)
        sd.wait()
        sf.write(path, audio, SAMPLE_RATE, subtype=SUBTYPE)
        print(f"Saved {path}")

    # A compact 20-utterance session per speaker: 7 benign, 7 urgent-legit, 6 scam.
    counts = {"benign": 7, "urgent_legit": 7, "scam": 6}
    categories = [args.category] if args.category != "all" else ["benign", "urgent_legit", "scam"]
    for category in categories:
        for idx in range(1, counts[category] + 1):
            path = speaker / "utterances" / category / f"{args.speaker}_{category}_{idx:02d}.wav"
            input(f"Press Enter when ready for {category} utterance {idx}/{counts[category]}...")
            time.sleep(0.75)
            print("RECORDING")
            audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device)
            sd.wait()
            sf.write(path, audio, SAMPLE_RATE, subtype=SUBTYPE)
            print(f"Saved {path}")

    metadata = speaker / "qc" / "recording_session.txt"
    metadata.write_text(
        f"speaker_id={args.speaker}\n"
        f"language={args.language}\n"
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"sample_rate_hz={SAMPLE_RATE}\n"
        f"channels={CHANNELS}\n"
        f"enrollment_clips=5\n"
        f"scripted_utterances=20\n",
        encoding="utf-8",
    )
    print("\nSpeaker recording session complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
