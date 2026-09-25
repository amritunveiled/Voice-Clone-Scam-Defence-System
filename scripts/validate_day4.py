from __future__ import annotations
import csv, json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    errors=[]
    split=json.loads((ROOT/"data/manifests/speaker_splits_day4.json").read_text(encoding="utf-8"))
    groups=split["splits"]
    assigned=[sid for ids in groups.values() for sid in ids]
    if len(assigned)!=len(set(assigned)): errors.append("speaker leakage/duplicate speaker assignment in split file")
    for fold,ids in groups.items():
        if not ids: errors.append(f"empty {fold} speaker split")
    qcf=ROOT/"results/tables/day4_clone_qc.csv"
    if not qcf.exists(): errors.append("missing day4_clone_qc.csv")
    else:
        qc=list(csv.DictReader(qcf.open(encoding="utf-8",newline="")))
        if any(r["technical_status"]!="PASS" for r in qc): errors.append("technical clone QC has failures")
        if any(r["manual_listen"]!="PASS" for r in qc): errors.append("manual listening QC not PASS for every clone")
    jobs=ROOT/"data/manifests/day4_clone_jobs.csv"
    if not jobs.exists(): errors.append("missing day4_clone_jobs.csv")
    else:
        jrows=list(csv.DictReader(jobs.open(encoding="utf-8",newline="")))
        if any(r["status"]!="GENERATED" for r in jrows): errors.append("not every scheduled clone job is generated")
        if any(not (ROOT/r["output_path"]).exists() for r in jrows if r["status"]=="GENERATED"): errors.append("missing generated clone file")
    em=ROOT/"data/manifests/day4_evaluation_manifest.csv"
    if not em.exists(): errors.append("missing day4_evaluation_manifest.csv")
    else:
        rows=list(csv.DictReader(em.open(encoding="utf-8",newline="")))
        for r in rows:
            if r["evaluation_split"] in {"dev","test"} and r["speaker_split"] not in {"dev","test"}:
                errors.append(f"bad eval split mapping: {r}")
        dev_scripts={r["script_id"] for r in rows if r["evaluation_split"]=="dev"}
        test_scripts={r["script_id"] for r in rows if r["evaluation_split"]=="test"}
        if dev_scripts & test_scripts: errors.append("script leakage: dev/test script IDs overlap")
    codec=ROOT/"data/manifests/day4_codec_manifest.csv"
    if not codec.exists(): errors.append("missing day4_codec_manifest.csv")
    else:
        cr=list(csv.DictReader(codec.open(encoding="utf-8",newline="")))
        bad=[r for r in cr if r["status"]!="PASS" or r["decoded_sample_rate_hz"]!="16000" or r["decoded_channels"]!="1"]
        if bad: errors.append(f"codec failures or invalid decoded files: {len(bad)}")
    if errors:
        print("STATUS: FAIL")
        for e in errors: print("-",e)
        return 1
    print("Speaker split: PASS")
    print("Clone technical + manual QC: PASS")
    print("Evaluation manifest: PASS")
    print("Codec manifest: PASS")
    print("STATUS: PASS")
    return 0
if __name__=="__main__": raise SystemExit(main())
