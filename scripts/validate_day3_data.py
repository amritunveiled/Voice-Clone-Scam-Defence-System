from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

SR = 16000


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Day-3 speaker folders, consent markers, audio format, and script-bank tags.")
    parser.add_argument("--min-speakers", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "participants"
    manifests = root / "data" / "manifests"

    speakers = sorted([p for p in raw.iterdir() if p.is_dir()]) if raw.exists() else []
    print(f"Speakers found: {len(speakers)}")

    errors: list[str] = []
    stats = Counter()

    if len(speakers) < args.min_speakers:
        errors.append(f"Need at least {args.min_speakers} speakers; found {len(speakers)}")

    for speaker in speakers:
        consent_marker = speaker / "consent" / "consent_received.txt"
        if not consent_marker.exists():
            errors.append(f"{speaker.name}: missing consent_received.txt")

        enrollment = sorted((speaker / "enrollment").glob("*.wav"))
        if len(enrollment) != 5:
            errors.append(f"{speaker.name}: expected 5 enrollment clips; found {len(enrollment)}")
        else:
            stats["enrollment_clips"] += len(enrollment)

        for category, expected in [("benign", 7), ("urgent_legit", 7), ("scam", 6)]:
            files = sorted((speaker / "utterances" / category).glob("*.wav"))
            if len(files) != expected:
                errors.append(f"{speaker.name}: {category} expected {expected}; found {len(files)}")
            stats[category] += len(files)

        # Validate every WAV for mono/16 kHz and report duration only as QA, not as a performance result.
        for wav in speaker.rglob("*.wav"):
            try:
                info = sf.info(wav)
                if info.samplerate != SR:
                    errors.append(f"{wav}: sample rate {info.samplerate}, expected {SR}")
                if info.channels != 1:
                    errors.append(f"{wav}: channels={info.channels}, expected mono")
            except Exception as exc:
                errors.append(f"{wav}: unreadable WAV: {exc}")

    script_file = manifests / "script_bank.csv"
    required_columns = {"script_id", "split", "category", "language", "text"}
    splits = Counter()
    seen = set()
    if not script_file.exists():
        errors.append(f"Missing {script_file}")
    else:
        with script_file.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            errors.append("script_bank.csv is empty")
        else:
            if not required_columns.issubset(rows[0].keys()):
                errors.append(f"script_bank.csv missing columns: {sorted(required_columns - set(rows[0].keys()))}")
            for row in rows:
                key=(row["script_id"],row["language"])
                if key in seen:
                    errors.append(f"Duplicate script/language row: {key}")
                seen.add(key)
                if row["split"] not in {"train","dev","test"}:
                    errors.append(f"Invalid split: {row['split']}")
                if row["category"] not in {"benign","urgent_legit","scam"}:
                    errors.append(f"Invalid category: {row['category']}")
                splits[row["split"]]+=1

    print("Audio counts:", dict(stats))
    print("Script rows by split:", dict(splits))
    print("Required script splits present:", all(x in splits for x in ("train","dev","test")))

    if errors:
        print("\nSTATUS: FAIL")
        for e in errors:
            print("-", e)
        return 1

    report = root / "results" / "tables" / "day3_data_validation.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"speakers={len(speakers)}\n"
        f"enrollment_clips={stats['enrollment_clips']}\n"
        f"benign_utterances={stats['benign']}\n"
        f"urgent_legit_utterances={stats['urgent_legit']}\n"
        f"scam_utterances={stats['scam']}\n"
        f"script_splits={dict(splits)}\n"
        "format=mono_16k_pcm_wav\n"
        "status=PASS\n",
        encoding="utf-8",
    )
    print(f"\nSaved validation report: {report}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
