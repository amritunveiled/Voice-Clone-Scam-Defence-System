from __future__ import annotations
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
summary=ROOT/"results/tables/day6_layer2_summary.csv"
scores=ROOT/"results/tables/day6_layer2_scores.csv"
report=ROOT/"results/tables/day6_layer2_results.json"
if not (summary.exists() and scores.exists() and report.exists()):
    raise SystemExit("Missing Day-6 result artifact(s). Run evaluate_layer2_day6.py first.")
payload=json.loads(report.read_text(encoding="utf-8"))
if payload.get("final_test_touched") is not False:
    raise SystemExit("FAIL: final test split was not protected.")
rows=list(csv.DictReader(summary.open(encoding="utf-8",newline="")))
models={r["model"] for r in rows}
required={"AASIST-L","XLSR-SLS"}
if not required.issubset(models):
    raise SystemExit(f"FAIL: missing models: {required-models}")
own={r["codec"] for r in rows if r["dataset"]=="own"}
if "clean" not in own:
    raise SystemExit("FAIL: own clean condition missing.")
print("Models evaluated: PASS")
print("Own clean condition: PASS")
print("Final test touched: NO")
print("RESULT STATUS: PASS")
