from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "manifests" / "speaker_splits_day4.json"
SCRIPT_FILE = ROOT / "data" / "manifests" / "script_bank.csv"
SPEAKER_FILE = ROOT / "data" / "manifests" / "speakers.csv"
OUT = ROOT / "data" / "manifests" / "day4_clone_jobs.csv"


def normalize_language(x: str) -> str:
    return x.strip().lower()


def language_warning(language: str, text: str) -> str:
    if language == "hindi":
        return "warning_non_devanagari_text" if not re.search(r"[\u0900-\u097F]", text) else ""
    if language == "kannada":
        return "warning_non_kannada_text" if not re.search(r"[\u0C80-\u0CFF]", text) else ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=1, help="Number of benign and scam clone targets per speaker. Default 1 keeps CPU runtime manageable.")
    args = parser.parse_args()
    if args.per_category < 1:
        raise SystemExit("--per-category must be >= 1")
    for p in (SPLIT_FILE, SCRIPT_FILE, SPEAKER_FILE):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
    splits = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["splits"]
    speaker_fold = {sid: fold for fold, ids in splits.items() for sid in ids}
    speakers = list(csv.DictReader(SPEAKER_FILE.open(encoding="utf-8", newline="")))
    scripts = list(csv.DictReader(SCRIPT_FILE.open(encoding="utf-8", newline="")))
    by_key = {(r["language"].lower(), r["category"], r["split"]): [] for r in scripts}
    for r in scripts:
        by_key.setdefault((r["language"].lower(), r["category"], r["split"]), []).append(r)
    jobs=[]
    for s in speakers:
        sid=s["speaker_id"]; lang=normalize_language(s["language"]); fold=speaker_fold[sid]
        speaker_dir=ROOT/"data"/"raw"/"participants"/sid
        ref_audio=speaker_dir/"utterances"/"benign"/f"{sid}_benign_01.wav"
        ref_candidates=by_key.get((lang,"benign","train"), [])
        if not ref_audio.exists():
            raise SystemExit(f"{sid}: missing reference audio {ref_audio}")
        if not ref_candidates:
            raise SystemExit(f"{sid}: no benign train script found for language={lang}")
        ref_row=ref_candidates[0]
        target_rows=[]
        for category in ("benign","scam"):
            candidates=by_key.get((lang,category,fold), [])
            if len(candidates) < args.per_category:
                raise SystemExit(f"{sid}: need {args.per_category} {category} scripts in split={fold}, language={lang}; found {len(candidates)}")
            target_rows.extend(candidates[:args.per_category])
        for row in target_rows:
            job_id=f"{sid}__clone__{row['script_id']}"
            out=ROOT/"data"/"clones"/"raw_24k"/sid/fold/row["category"]/f"{job_id}.wav"
            jobs.append({
                "job_id":job_id,"speaker_id":sid,"speaker_language":lang,"speaker_split":fold,
                "source_type":"clone","reference_audio":str(ref_audio.relative_to(ROOT)).replace('\\','/'),
                "reference_script_id":ref_row["script_id"],"reference_text":ref_row["text"],
                "target_script_id":row["script_id"],"target_category":row["category"],"target_script_split":row["split"],
                "target_text":row["text"],"language_text_warning":language_warning(lang,row["text"]),
                "output_path":str(out.relative_to(ROOT)).replace('\\','/'),"status":"PENDING"
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=jobs[0].keys()); w.writeheader(); w.writerows(jobs)
    print(f"Prepared {len(jobs)} clone jobs for {len(speakers)} speakers")
    warns=sum(1 for j in jobs if j["language_text_warning"])
    print(f"Language-text warnings: {warns}")
    if warns:
        print("WARNING: review the script bank before synthesis; the job is not blocked, but language/script mismatch may reduce TTS validity.")
    print(f"Saved: {OUT}")
    print("STATUS: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
