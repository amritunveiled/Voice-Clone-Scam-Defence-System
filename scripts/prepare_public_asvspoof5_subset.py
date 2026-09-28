from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 12345


def read_protocol(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    # Official ASVspoof 5 protocol files are space-separated and have 10 fields.
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            rows.append(
                {
                    "speaker_id": parts[0],
                    "filename": parts[1],
                    "gender": parts[2],
                    # Track-1 uses '-' for uncompressed/clean audio. The
                    # project calls that required condition C00.
                    "codec": "C00" if parts[3] == "-" else parts[3],
                    "codec_q": parts[4],
                    "codec_seed": parts[5],
                    "attack_tag": parts[6],
                    "attack_label": parts[7],
                    "label": parts[8],
                    "line_no": str(line_no),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--codecs", nargs="+", default=["C00", "C09"])
    ap.add_argument("--out", type=Path, default=ROOT / "data/manifests/day6_public_asvspoof5.csv")
    args = ap.parse_args()

    protocol_rows = read_protocol(args.protocol)

    # Build a basename index once. This works for either an extracted flac_D tree
    # or a smaller directory containing only the selected files.
    file_index = {}
    print(f"Indexing FLAC files under {args.audio_root} ...")
    for p in args.audio_root.rglob("*.flac"):
        # Track-1 protocol IDs omit the .flac suffix.
        file_index[p.stem] = p

    if not file_index:
        raise RuntimeError("No .flac files found under --audio-root")

    rng = random.Random(SEED)
    selected = []

    for codec in args.codecs:
        for label in ("bonafide", "spoof"):
            candidates = [
                r for r in protocol_rows
                if r["codec"].upper() == codec.upper()
                and r["label"].lower() == label
                and r["filename"] in file_index
            ]
            if len(candidates) < args.per_cell:
                raise RuntimeError(
                    f"Not enough local files for codec={codec}, label={label}: "
                    f"need {args.per_cell}, found {len(candidates)}. "
                    "The ASVspoof5 Track-1 protocol only labels raw files as C00; "
                    "provide a separately transcoded C09 audio root or run with "
                    "--codecs C00."
                )
            rng.shuffle(candidates)
            chosen = candidates[:args.per_cell]
            selected.extend(chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "path", "dataset", "split", "speaker_id", "codec",
            "label", "attack_label", "codec_q", "source_protocol_line"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "path": str(file_index[r["filename"]].resolve()),
                "dataset": "ASVspoof5",
                "split": "dev_track_1",
                "speaker_id": r["speaker_id"],
                "codec": r["codec"],
                "label": r["label"].lower(),
                "attack_label": r["attack_label"],
                "codec_q": r["codec_q"],
                "source_protocol_line": r["line_no"],
            })

    print(f"Saved {len(selected)} public subset rows: {args.out}")
    print("Cells:")
    from collections import Counter
    print(dict(Counter((r["codec"], r["label"]) for r in selected)))
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
