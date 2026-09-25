from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path

SEED = 12345
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "speaker_splits_day4.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze speaker-disjoint train/dev/test folds once.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"STOP: {OUT} already exists. The speaker split is frozen. Do not regenerate it.")
    speakers_csv = ROOT / "data" / "manifests" / "speakers.csv"
    if not speakers_csv.exists():
        raise SystemExit(f"Missing {speakers_csv}. Run Day-3 dataset setup first.")
    rows = list(csv.DictReader(speakers_csv.open(encoding="utf-8", newline="")))
    if not 12 <= len(rows) <= 15:
        raise SystemExit(f"Expected 12-15 speakers, found {len(rows)}")
    missing = [r["speaker_id"] for r in rows if not (ROOT / "data" / "raw" / "participants" / r["speaker_id"] / "consent" / "consent_received.txt").exists()]
    if missing:
        raise SystemExit("Missing consent markers for: " + ", ".join(missing))
    ids = sorted(r["speaker_id"] for r in rows)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(2, round(n * 0.20))
    n_dev = max(2, round(n * 0.15))
    while n_test + n_dev >= n:
        n_dev -= 1
    split = {"train": sorted(ids[: n - n_dev - n_test]), "dev": sorted(ids[n - n_dev - n_test : n - n_test]), "test": sorted(ids[n - n_test:])}
    payload = {"seed": args.seed, "frozen": True, "speaker_disjoint": True, "counts": {k: len(v) for k, v in split.items()}, "splits": split}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved frozen split: {OUT}")
    print("STATUS: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
