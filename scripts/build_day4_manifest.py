from __future__ import annotations
import csv, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def idx_to_script(category, idx):
    prefix={"benign":"B","urgent_legit":"U","scam":"S"}[category]
    return f"{prefix}{idx:03d}"

def parse_human(p):
    # <speaker>_<category>_<NN>.wav
    m=re.match(r"(?P<sid>.+?)_(?P<cat>benign|urgent_legit|scam)_(?P<idx>\d+)\.wav$",p.name)
    if not m: return None
    return m.group("sid"),m.group("cat"),int(m.group("idx"))

def main():
    split=json.loads((ROOT/"data/manifests/speaker_splits_day4.json").read_text(encoding="utf-8"))["splits"]
    speaker_fold={sid:fold for fold,ids in split.items() for sid in ids}
    speakers={r["speaker_id"]:r for r in csv.DictReader((ROOT/"data/manifests/speakers.csv").open(encoding="utf-8",newline=""))}
    scripts={(r["script_id"],r["language"].lower()):r for r in csv.DictReader((ROOT/"data/manifests/script_bank.csv").open(encoding="utf-8",newline=""))}
    rows=[]
    # human utterances
    for p in sorted((ROOT/"data/raw/participants").glob("*/utterances/*/*.wav")):
        parsed=parse_human(p)
        if not parsed: continue
        sid,cat,idx=parsed; lang=speakers[sid]["language"].lower(); script_id=idx_to_script(cat,idx); sr=scripts.get((script_id,lang),{})
        spkfold=speaker_fold[sid]; scriptsplit=sr.get("split", "UNVERIFIED")
        evalsplit="train" if spkfold=="train" else ("dev" if spkfold=="dev" and scriptsplit=="dev" else ("test" if spkfold=="test" and scriptsplit=="test" else "holdout"))
        rows.append({"path":str(p.relative_to(ROOT)).replace('\\','/'),"source_type":"human","speaker_id":sid,"language":lang,"category":cat,"script_id":script_id,"script_split":scriptsplit,"speaker_split":spkfold,"evaluation_split":evalsplit,"codec":"clean"})
    # clones from job CSV
    if (ROOT/"data/manifests/day4_clone_jobs.csv").exists():
        for j in csv.DictReader((ROOT/"data/manifests/day4_clone_jobs.csv").open(encoding="utf-8",newline="")):
            if j.get("status")!="GENERATED": continue
            spkfold=j["speaker_split"]; scriptsplit=j["target_script_split"]; evalsplit="train" if spkfold=="train" else ("dev" if spkfold=="dev" and scriptsplit=="dev" else ("test" if spkfold=="test" and scriptsplit=="test" else "holdout"))
            rows.append({"path":j["output_path"],"source_type":"clone","speaker_id":j["speaker_id"],"language":j["speaker_language"],"category":j["target_category"],"script_id":j["target_script_id"],"script_split":scriptsplit,"speaker_split":spkfold,"evaluation_split":evalsplit,"codec":"clean"})
    out=ROOT/"data/manifests/day4_evaluation_manifest.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} records: {out}")
    print("Evaluation split counts:")
    from collections import Counter
    print(dict(Counter(r["evaluation_split"] for r in rows)))
    print("STATUS: PASS")
if __name__=="__main__": main()
