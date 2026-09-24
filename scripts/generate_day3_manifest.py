from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "participants"
    manifests = root / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    rows=[]
    for wav in sorted(raw.rglob("*.wav")) if raw.exists() else []:
        rel=wav.relative_to(root).as_posix()
        parts=wav.relative_to(raw).parts
        speaker_id=parts[0] if parts else ""
        if "enrollment" in parts:
            kind="enrollment"; category="enrollment"; script_id=""
        elif "utterances" in parts:
            category=parts[2]
            kind="utterance"
            script_id=""
        else:
            continue
        rows.append({"path":rel,"speaker_id":speaker_id,"kind":kind,"category":category,"script_id":script_id})
    out=manifests/"audio_manifest_day3.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["path","speaker_id","kind","category","script_id"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
