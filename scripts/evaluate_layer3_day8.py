from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BANK = ROOT / "data/manifests/script_bank.csv"
DAY4_MANIFEST = ROOT / "data/manifests/day4_evaluation_manifest.csv"
WHISPER_SCORES = ROOT / "results/tables/day7_whisper_cer_scores.csv"
OUT_ROWS = ROOT / "results/tables/day8_layer3_scores.csv"
OUT_SUMMARY = ROOT / "results/tables/day8_layer3_summary.csv"
OUT_REPORT = ROOT / "results/tables/day8_layer3_report.json"

sys.path.insert(0, str(ROOT))
from src.layer3_intent import LABELS, fit_intent_model  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm(x: str) -> str:
    return (x or "").strip().lower().replace("_", "-")


def build_script_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(SCRIPT_BANK)
    out = {}
    for r in rows:
        sid = r.get("script_id", "").strip()
        lang = norm(r.get("language", ""))
        text = r.get("text") or r.get("script_text") or ""
        if sid and lang and text:
            out[(sid, lang)] = r
    return out


def build_train_dev_texts() -> tuple[list[dict], list[dict]]:
    lookup = build_script_lookup()
    rows = []
    for key, r in lookup.items():
        if r.get("split") in {"train", "dev"}:
            rows.append({
                "script_id": key[0],
                "language": key[1],
                "split": r["split"],
                "category": r["category"],
                "reference_text": r.get("text") or r.get("script_text") or "",
            })
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    if not train or not dev:
        raise RuntimeError("Need both train and dev script rows in script_bank.csv")
    return train, dev


def verify_script_leakage(train: list[dict], dev: list[dict]) -> None:
    train_ids = {r["script_id"] for r in train}
    dev_ids = {r["script_id"] for r in dev}
    overlap = train_ids & dev_ids
    if overlap:
        raise RuntimeError(f"Script leakage: train/dev share script IDs {sorted(overlap)}")


def build_asr_rows(dev_script_rows: list[dict]) -> list[dict]:
    """Use Day-7 development ASR hypotheses as the ASR-noise condition."""
    if not WHISPER_SCORES.exists():
        raise FileNotFoundError(
            f"Missing {WHISPER_SCORES}. Run Day 7 Whisper CER first."
        )
    whisper = read_csv(WHISPER_SCORES)
    dev_ids = {r["script_id"] for r in dev_script_rows}
    rows = []
    for r in whisper:
        if r.get("evaluation_split") != "dev":
            continue
        if r.get("script_id") not in dev_ids:
            continue
        rows.append(r)
    if not rows:
        raise RuntimeError("No Day-7 ASR hypotheses matched Day-8 dev script IDs")
    return rows


def evaluate_predictions(
    model,
    rows: list[dict],
    text_field: str,
    condition: str,
    language_override: bool = False,
) -> tuple[list[dict], dict]:
    texts = [r[text_field] for r in rows]
    langs = [r["language"] for r in rows]
    y_true = [r["category"] for r in rows]
    pred = model.predict(texts, langs).tolist()
    probs = model.predict_proba(texts, langs)

    out = []
    for r, p, pr in zip(rows, pred, probs):
        out.append({
            "condition": condition,
            "language": r["language"],
            "script_id": r["script_id"],
            "category": r["category"],
            "text_source": text_field,
            "source_type": r.get("source_type", "script_bank"),
            "codec": r.get("codec", "reference"),
            "prediction": p,
            "prob_benign": float(pr[0]),
            "prob_urgent_legit": float(pr[1]),
            "prob_scam": float(pr[2]),
            "text": r[text_field],
        })

    accuracy = float(accuracy_score(y_true, pred))
    macro_f1 = float(f1_score(y_true, pred, labels=list(LABELS), average="macro", zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=list(LABELS)).tolist()
    report = classification_report(y_true, pred, labels=list(LABELS), output_dict=True, zero_division=0)
    return out, {
        "condition": condition,
        "n": len(rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def add_group_summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["language"], r["source_type"], r["codec"])].append(r)
    summary = []
    for key, vals in sorted(groups.items()):
        y = [r["category"] for r in vals]
        p = [r["prediction"] for r in vals]
        summary.append({
            "condition": key[0],
            "language": key[1],
            "source_type": key[2],
            "codec": key[3],
            "n": len(vals),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)),
        })
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 8 Layer-3 intent evaluation with ASR-noise ablation.")
    ap.add_argument("--include-human-audio", action="store_true", help="Also evaluate Day-4 development human/clone script rows when available.")
    args = ap.parse_args()

    train, dev_scripts = build_train_dev_texts()
    verify_script_leakage(train, dev_scripts)

    print("=== DAY 8 LAYER 3 ===")
    print(f"Train scripts: {len(train)}")
    print(f"Dev scripts  : {len(dev_scripts)}")
    print("Final test   : PROTECTED")

    x_train = [r["reference_text"] for r in train]
    y_train = [r["category"] for r in train]
    if len(set(y_train)) < 3:
        raise RuntimeError(f"Training scripts need all 3 classes; found {sorted(set(y_train))}")
    model = fit_intent_model(x_train, y_train)

    all_scores = []
    report = {
        "day": 8,
        "final_test_touched": False,
        "train_script_ids": sorted({r["script_id"] for r in train}),
        "dev_script_ids": sorted({r["script_id"] for r in dev_scripts}),
        "model": {
            "classifier": "logistic_regression",
            "text_features": "character TF-IDF 3-5 grams",
            "lexicon": "language-aware fuzzy indicators",
            "fuzzy_weight": model.fuzzy_weight,
            "classes": list(LABELS),
        },
        "evaluations": [],
    }

    clean_rows, clean_metrics = evaluate_predictions(
        model, dev_scripts, "reference_text", "reference_text_clean"
    )
    all_scores.extend(clean_rows)
    report["evaluations"].append(clean_metrics)

    asr_rows = build_asr_rows(dev_scripts)
    asr_scores, asr_metrics = evaluate_predictions(
        model, asr_rows, "hypothesis", "whisper_asr"
    )
    all_scores.extend(asr_scores)
    report["evaluations"].append(asr_metrics)

    # ASR-noise delta at matched rows: reference-text vs the corresponding ASR transcript.
    ref_by_key = {(r["script_id"], norm(r["language"])): r for r in clean_rows}
    asr_by_key = {(r["script_id"], norm(r["language"]), r.get("codec", "clean"), r.get("source_type", "")): r for r in asr_scores}
    deltas = []
    for key, asr in asr_by_key.items():
        base = ref_by_key.get((key[0], key[1]))
        if base:
            deltas.append({
                "script_id": key[0],
                "language": key[1],
                "source_type": key[3],
                "codec": key[2],
                "reference_prediction": base["prediction"],
                "asr_prediction": asr["prediction"],
                "prediction_changed": base["prediction"] != asr["prediction"],
                "reference_true_category": base["category"],
            })
    report["asr_noise_ablation"] = {
        "matched_comparisons": len(deltas),
        "prediction_flip_count": int(sum(int(x["prediction_changed"]) for x in deltas)),
    }

    summary = add_group_summaries(all_scores)

    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in all_scores for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_scores)

    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in summary for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)

    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DAY 8 SUMMARY ===")
    for item in report["evaluations"]:
        print(
            f"{item['condition']:22s} | n={item['n']:3d} | "
            f"accuracy={item['accuracy']:.4f} | macro-F1={item['macro_f1']:.4f}"
        )
    print(f"ASR matched comparisons : {report['asr_noise_ablation']['matched_comparisons']}")
    print(f"ASR prediction flips   : {report['asr_noise_ablation']['prediction_flip_count']}")
    print(f"Saved: {OUT_ROWS}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_REPORT}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
