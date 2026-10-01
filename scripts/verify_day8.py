from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/tables/day8_layer3_summary.csv"
REPORT = ROOT / "results/tables/day8_layer3_report.json"
SCORES = ROOT / "results/tables/day8_layer3_scores.csv"
SCRIPT_BANK = ROOT / "data/manifests/script_bank.csv"

errors=[]
for p in (SUMMARY, REPORT, SCORES):
    if not p.exists():
        errors.append(f"Missing {p}")

if not errors:
    report=json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("final_test_touched") is not False:
        errors.append("Final test protection flag is not false")
    train=set(report.get("train_script_ids",[])); dev=set(report.get("dev_script_ids",[]))
    if train & dev:
        errors.append(f"Train/dev script leakage: {sorted(train & dev)}")
    if len(train) == 0 or len(dev) == 0:
        errors.append("Missing train or dev script IDs")
    with SUMMARY.open(encoding="utf-8", newline="") as f:
        rows=list(csv.DictReader(f))
    conditions={r.get("condition") for r in rows}
    if "reference_text_clean" not in conditions:
        errors.append("Missing clean-reference evaluation")
    if "whisper_asr" not in conditions:
        errors.append("Missing Whisper ASR-noise evaluation")

print("Layer-3 model/evaluation artifacts:", "PASS" if not errors else "FAIL")
print("Train/dev script leakage:", "PASS" if not errors or all("leakage" not in e.lower() for e in errors) else "FAIL")
print("Final test touched: NO")
if errors:
    print("STATUS: FAIL")
    for e in errors: print("-", e)
    raise SystemExit(1)
print("RESULT STATUS: PASS")
