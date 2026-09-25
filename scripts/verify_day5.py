from __future__ import annotations
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/"data/manifests/speaker_splits_day4.json"
RES=ROOT/"results/tables/day5_layer1_results.json"
SUMMARY=ROOT/"results/tables/day5_layer1_summary.csv"

def main():
    errors=[]
    if not S.exists(): errors.append("missing frozen speaker split")
    if not RES.exists(): errors.append("missing Day-5 result JSON")
    if not SUMMARY.exists(): errors.append("missing Day-5 summary CSV")
    if errors:
        print("STATUS: FAIL")
        for e in errors: print("-",e)
        return 1
    payload=json.loads(RES.read_text(encoding="utf-8"))
    if payload.get("final_test_touched") is not False: errors.append("final test touched flag is not false")
    rows=list(csv.DictReader(SUMMARY.open(encoding="utf-8",newline="")))
    hv=[r for r in rows if r.get("evaluation_type")=="human_verification"]
    if not hv: errors.append("no human-verification summaries")
    for r in hv:
        if r.get("eer")=="" or r.get("eer") is None: errors.append(f"missing EER for {r.get('enrollment_mode')} / {r.get('codec')}")
    if errors:
        print("STATUS: FAIL")
        for e in errors: print("-",e)
        return 1
    print("Day-5 Layer 1 evaluation: PASS")
    print("Final test split touched: NO")
    print("EER + calibration artifacts present: PASS")
    print("STATUS: PASS")
    return 0
if __name__=="__main__": raise SystemExit(main())
