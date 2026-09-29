from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "day4_evaluation_manifest.csv"
CODEC_MANIFEST = ROOT / "data" / "manifests" / "day4_codec_manifest.csv"
SCRIPT_BANK = ROOT / "data" / "manifests" / "script_bank.csv"
OUT_ROWS = ROOT / "results" / "tables" / "day7_whisper_cer_scores.csv"
OUT_SUM = ROOT / "results" / "tables" / "day7_whisper_cer_summary.csv"
OUT_JSON = ROOT / "results" / "tables" / "day7_whisper_cer_results.json"

LANGUAGE_CODES = {"english": "en", "hindi": "hi", "kannada": "kn", "en": "en", "hi": "hi", "kn": "kn"}


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm_language(x: str) -> str:
    return x.strip().lower().replace("_", "-")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cer(ref: str, hyp: str) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    # Character error rate over Unicode code points. Whitespace is normalized above,
    # but Indic characters and Latin characters are otherwise preserved.
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i]
        for j, hc in enumerate(hyp, start=1):
            cost = 0 if rc == hc else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return float(prev[-1] / len(ref))


def resolve(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def build_text_lookup():
    rows = read_csv(SCRIPT_BANK)
    lookup = {}
    for r in rows:
        sid = r.get("script_id", "").strip()
        lang = norm_language(r.get("language", ""))
        text = r.get("text") or r.get("script_text") or ""
        if sid and lang and text:
            lookup[(sid, lang)] = text
    return lookup


def codec_path_lookup():
    rows = read_csv(CODEC_MANIFEST)
    return {(r["source_path"], r["codec"]): r["decoded_path"] for r in rows}


def build_rows():
    own = read_csv(MANIFEST)
    split_rows = [r for r in own if r.get("evaluation_split") == "dev"]
    if not split_rows:
        raise RuntimeError("No development rows in day4_evaluation_manifest.csv")

    texts = build_text_lookup()
    codecs = codec_path_lookup()
    out = []
    for r in split_rows:
        if r.get("source_type") not in {"human", "clone"}:
            continue
        lang = norm_language(r.get("language", ""))
        key = (r.get("script_id", ""), lang)
        reference = texts.get(key)
        if not reference:
            raise RuntimeError(f"No reference text for script_id={key[0]!r}, language={lang!r}")
        out.append({
            "dataset": "own",
            "evaluation_split": "dev",
            "speaker_id": r.get("speaker_id", ""),
            "source_type": r.get("source_type", ""),
            "language": lang,
            "script_id": r.get("script_id", ""),
            "category": r.get("category", ""),
            "codec": "clean",
            "path": resolve(r["path"]),
            "reference_text": reference,
        })
        for codec_name in sorted({k[1] for k in codecs if k[0] == r["path"]}):
            decoded = codecs.get((r["path"], codec_name))
            if not decoded:
                continue
            out.append({
                **{k: r.get(k, "") for k in ["speaker_id", "source_type", "language", "script_id", "category"]},
                "dataset": "own",
                "evaluation_split": "dev",
                "codec": codec_name,
                "path": resolve(decoded),
                "reference_text": reference,
            })
    return out


def transcribe(model, row):
    lang_key = norm_language(row["language"])
    language = LANGUAGE_CODES.get(lang_key)
    if language is None:
        raise RuntimeError(f"Unsupported/unknown language in manifest: {row['language']}")
    segments, info = model.transcribe(
        str(row["path"]),
        language=language,
        beam_size=1,
        vad_filter=False,
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    return text, info


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 7 Whisper CER by language and codec on development data only.")
    ap.add_argument("--max-files", type=int, default=0, help="Optional smoke limit; 0 = all development files")
    ap.add_argument("--model", default="small")
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()

    rows = build_rows()
    if args.max_files > 0:
        rows = rows[:args.max_files]
    if not rows:
        raise RuntimeError("No ASR rows available.")

    from faster_whisper import WhisperModel

    print("=== DAY 7 WHISPER CER ===")
    print(f"Rows: {len(rows)}")
    print(f"Model: faster-whisper:{args.model}")
    print("Device: CPU | compute_type=int8 | beam_size=1")
    print("Final test: PROTECTED")

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, args.cpu_threads),
        num_workers=1,
    )

    output = []
    for i, row in enumerate(rows, 1):
        hyp, info = transcribe(model, row)
        score = cer(row["reference_text"], hyp)
        rec = {k: (str(v) if isinstance(v, Path) else v) for k, v in row.items()}
        rec.update({
            "hypothesis": hyp,
            "cer": score,
            "whisper_detected_language": getattr(info, "language", ""),
        })
        output.append(rec)
        if i % 10 == 0 or i == len(rows):
            print(f"  transcribed {i}/{len(rows)} | latest CER={score:.4f}")

    grouped = defaultdict(list)
    for r in output:
        grouped[(r["language"], r["codec"], r["source_type"])].append(float(r["cer"]))

    summary=[]
    for (language, codec, source_type), vals in sorted(grouped.items()):
        arr=np.asarray(vals,dtype=float)
        summary.append({
            "language": language,
            "codec": codec,
            "source_type": source_type,
            "n": int(arr.size),
            "cer_mean": float(arr.mean()),
            "cer_median": float(np.median(arr)),
            "cer_std": float(arr.std(ddof=0)),
            "cer_min": float(arr.min()),
            "cer_max": float(arr.max()),
        })

    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in output for k in r.keys()})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(output)
    with OUT_SUM.open("w", newline="", encoding="utf-8") as f:
        fields=sorted({k for r in summary for k in r.keys()})
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    OUT_JSON.write_text(json.dumps({
        "day":7,
        "model":f"faster-whisper:{args.model}",
        "split":"dev",
        "final_test_touched":False,
        "rows":len(output),
        "files":{"scores":str(OUT_ROWS.relative_to(ROOT)).replace('\\','/'),"summary":str(OUT_SUM.relative_to(ROOT)).replace('\\','/')},
        "normalization":"Unicode NFKC + whitespace collapse + trim",
        "note":"CER is computed against the scripted reference text. Results are development-only and must not be presented as final test performance."
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== CER SUMMARY ===")
    for r in summary:
        print(f"{r['language']:9s} | {r['codec']:16s} | {r['source_type']:6s} | n={r['n']:3d} | CER={r['cer_mean']:.4f}")
    print(f"Saved: {OUT_ROWS}")
    print(f"Saved: {OUT_SUM}")
    print(f"Saved: {OUT_JSON}")
    print("STATUS: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
