from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "results/tables/day7_layer2_calibrated_scores.csv"
CALSUM = ROOT / "results/tables/day7_layer2_calibration_summary.csv"
CER = ROOT / "results/tables/day7_whisper_cer_scores.csv"
CERSUM = ROOT / "results/tables/day7_whisper_cer_summary.csv"


def rows(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    cal = rows(CAL)
    calsum = rows(CALSUM)
    cer = rows(CER)
    cersum = rows(CERSUM)

    tests = [
        ("Layer-2 calibrated score file", bool(cal)),
        ("Layer-2 calibration groups", bool(calsum)),
        ("Whisper CER scores", bool(cer)),
        ("Whisper CER summary", bool(cersum)),
    ]

    result_json = ROOT / "results/tables/day7_layer2_calibration.json"
    final_test_protected = True
    if result_json.exists():
        try:
            payload=json.loads(result_json.read_text(encoding="utf-8"))
            final_test_protected = payload.get("final_test_touched") is False
        except Exception:
            final_test_protected = False

    tests.append(("Final test protected", final_test_protected))

    for name, ok in tests:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    # A calibration percentage-like output must be in [0,1].
    probability_ok = bool(cal) and all(
        0.0 <= float(r["calibrated_bonafide_probability"]) <= 1.0
        for r in cal
    )
    print(f"Calibrated probabilities valid: {'PASS' if probability_ok else 'FAIL'}")
    ok = all(v for _, v in tests) and probability_ok
    print("RESULT STATUS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
