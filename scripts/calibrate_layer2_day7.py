from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results" / "tables" / "day6_layer2_scores.csv"
OUT_CAL = ROOT / "results" / "tables" / "day7_layer2_calibrated_scores.csv"
OUT_SUM = ROOT / "results" / "tables" / "day7_layer2_calibration_summary.csv"
OUT_JSON = ROOT / "results" / "tables" / "day7_layer2_calibration.json"


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_float(v: str) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite score: {v!r}")
    return x


def fit_platt(scores: np.ndarray, labels: np.ndarray):
    if len(np.unique(labels)) != 2:
        raise ValueError("Both bonafide and spoof labels are required for Platt calibration.")
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=12345)
    model.fit(scores.reshape(-1, 1), labels)
    p = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    return model, p


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 7 Layer-2 Platt calibration using Day-6 development scores only.")
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--split", choices=["dev"], default="dev")
    args = ap.parse_args()

    rows = read_csv(args.input)
    rows = [r for r in rows if r.get("split", "dev") == args.split]
    if not rows:
        raise RuntimeError("No development rows found in Day-6 score file.")

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["codec"])
        grouped.setdefault(key, []).append(r)

    calibrated = []
    summaries = []
    skipped = []

    for key in sorted(grouped):
        model_name, dataset, codec = key
        grp = grouped[key]
        scores = np.asarray([clean_float(r["bonafide_score"]) for r in grp], dtype=float)
        labels = np.asarray([int(r["label"]) for r in grp], dtype=int)

        if len(grp) < 4 or len(np.unique(labels)) != 2:
            skipped.append({"model": model_name, "dataset": dataset, "codec": codec, "n": len(grp), "reason": "too_few_rows_or_single_class"})
            continue

        calibrator, probs = fit_platt(scores, labels)
        brier = float(brier_score_loss(labels, probs))
        ll = float(log_loss(labels, probs, labels=[0, 1]))
        a = float(calibrator.coef_[0, 0])
        b = float(calibrator.intercept_[0])

        for r, p in zip(grp, probs):
            out = dict(r)
            out["calibrated_bonafide_probability"] = float(p)
            out["calibration_method"] = "platt_sigmoid"
            calibrated.append(out)

        summaries.append({
            "model": model_name,
            "dataset": dataset,
            "codec": codec,
            "n": len(grp),
            "bonafide": int(labels.sum()),
            "spoof": int((labels == 0).sum()),
            "platt_a": a,
            "platt_b": b,
            "brier_score_in_sample": brier,
            "log_loss_in_sample": ll,
        })

    OUT_CAL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CAL.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in calibrated for k in r})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(calibrated)

    with OUT_SUM.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in summaries for k in r}) or ["model", "dataset", "codec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)

    payload = {
        "day": 7,
        "final_test_touched": False,
        "source_scores": str(args.input),
        "split": args.split,
        "method": "platt_sigmoid",
        "groups_calibrated": len(summaries),
        "groups_skipped": skipped,
        "note": "Calibration is fit separately per model/dataset/codec using development rows. Brier/log loss here are in-sample calibration diagnostics, not unbiased generalization estimates.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== DAY 7 LAYER-2 CALIBRATION ===")
    print(f"Input rows: {len(rows)}")
    print(f"Groups calibrated: {len(summaries)}")
    print(f"Groups skipped: {len(skipped)}")
    for r in summaries:
        print(f"{r['model']:9s} | {r['dataset']:10s} | {r['codec']:16s} | n={r['n']:4d} | Brier={r['brier_score_in_sample']:.4f} | LogLoss={r['log_loss_in_sample']:.4f}")
    print(f"Saved: {OUT_CAL}")
    print(f"Saved: {OUT_SUM}")
    print(f"Saved: {OUT_JSON}")
    print("Final test: PROTECTED")
    print("STATUS: PASS" if summaries and not skipped else "STATUS: PASS_WITH_NOTES")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
