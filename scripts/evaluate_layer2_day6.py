from __future__ import annotations

import librosa
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.codecs import CODEC_SPECS
from src.layer2_spoof import ONNXSpoofScorer

SEED = 12345
SAMPLE_RATE = 16000
OWN_MANIFEST = ROOT / "data/manifests/day4_evaluation_manifest.csv"
CODEC_MANIFEST = ROOT / "data/manifests/day4_codec_manifest.csv"
RESULTS = ROOT / "results/tables"
SCORES_OUT = RESULTS / "day6_layer2_scores.csv"
SUMMARY_OUT = RESULTS / "day6_layer2_summary.csv"
JSON_OUT = RESULTS / "day6_layer2_results.json"
PUBLIC_DEFAULT = ROOT / "data/manifests/day6_public_asvspoof5.csv"
SPLIT_FILE = ROOT / "data/manifests/speaker_splits_day4.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / p
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def find_model(model_dir: str) -> Path:
    root = ROOT / "models" / model_dir
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {model_dir}. Run .\\.venv\\Scripts\\python.exe "
            ".\\scripts\\download_spoof_models.py"
        )
    candidates = sorted(root.rglob("*.onnx"), key=lambda p: p.stat().st_size)
    if not candidates:
        raise FileNotFoundError(f"No ONNX file under {root}")
    return candidates[-1]


def eer_from_bonafide(scores: np.ndarray, labels_bonafide: np.ndarray) -> tuple[float, float]:
    # roc_curve assumes larger score => positive class (bonafide).
    fpr, tpr, thresholds = roc_curve(labels_bonafide, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def tpr_at_5_fpr(scores: np.ndarray, labels_bonafide: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels_bonafide, scores, pos_label=1)
    valid = fpr <= 0.05
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_eer(records: list[dict], n_boot: int = 1000) -> tuple[float, float, float]:
    speakers = sorted({r["speaker_id"] for r in records if r.get("speaker_id")})
    if len(speakers) < 2:
        return math.nan, math.nan, math.nan
    grouped = defaultdict(list)
    for r in records:
        grouped[r["speaker_id"]].append(r)
    rng = random.Random(SEED)
    vals = []
    for _ in range(n_boot):
        sampled = [rng.choice(speakers) for _ in speakers]
        s = [r for sid in sampled for r in grouped[sid]]
        try:
            scores = np.asarray([r["bonafide_score"] for r in s], dtype=float)
            labels = np.asarray([r["label"] for r in s], dtype=int)
            vals.append(eer_from_bonafide(scores, labels)[0])
        except Exception:
            continue
    if not vals:
        return math.nan, math.nan, math.nan
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def build_own_rows(split: str) -> list[dict]:
    own = read_csv(OWN_MANIFEST)
    codec = read_csv(CODEC_MANIFEST)
    codec_lookup = {(r["source_path"], r["codec"]): r["decoded_path"] for r in codec}

    if split == "test":
        raise ValueError("Day 6 must not evaluate the final test split. It is reserved for Day 14.")

    base = [r for r in own if r.get("evaluation_split") == split]
    if not base:
        raise RuntimeError(f"No own-data evaluation rows for split={split}")

    out = []
    for r in base:
        # Human speech = bona fide. Clone speech = spoof.
        if r.get("source_type") not in {"human", "clone"}:
            continue
        label = 1 if r["source_type"] == "human" else 0

        clean_path = resolve(r["path"])
        out.append({
            "dataset": "own",
            "split": split,
            "speaker_id": r.get("speaker_id", ""),
            "language": r.get("language", ""),
            "category": r.get("category", ""),
            "script_id": r.get("script_id", ""),
            "source_type": r.get("source_type", ""),
            "codec": "clean",
            "path": clean_path,
            "label": label,
        })

        for cname in CODEC_SPECS:
            decoded = codec_lookup.get((r["path"], cname))
            if not decoded:
                continue
            out.append({
                **{k: r.get(k, "") for k in ["speaker_id", "language", "category", "script_id", "source_type"]},
                "dataset": "own",
                "split": split,
                "codec": cname,
                "path": resolve(decoded),
                "label": label,
            })
    return out


def build_public_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    out = []
    for r in rows:
        label = 1 if r["label"].lower() == "bonafide" else 0
        out.append({
            "dataset": "ASVspoof5",
            "split": r.get("split", "dev_track_1"),
            "speaker_id": r.get("speaker_id", ""),
            "language": "English",
            "category": "public_antispoof",
            "script_id": "",
            "source_type": "public_bonafide" if label else "public_spoof",
            "codec": r.get("codec", "-"),
            "path": resolve(r["path"]),
            "label": label,
        })
    return out


def run_model(model_label: str, model_path: Path, rows: list[dict], bootstrap: int) -> tuple[list[dict], list[dict]]:
    print(f"\nLoading {model_label}: {model_path}")
    scorer = ONNXSpoofScorer(model_path)

    scored = []
    for i, r in enumerate(rows, 1):
        import soundfile as sf
        audio, sr = librosa.load(
            r["path"],
            sr=16000,
            mono=True,
        )

        audio = audio.astype("float32", copy=False)
        score = scorer.score_audio(audio, sr)
        rec = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
        rec["model"] = model_label
        rec["bonafide_score"] = score
        rec["spoof_score"] = -score
        scored.append(rec)
        if i % 100 == 0:
            print(f"  scored {i}/{len(rows)}")

    summaries = []
    for (dataset, codec), grp in sorted(_group(scored, ("dataset", "codec")).items()):
        scores = np.asarray([float(r["bonafide_score"]) for r in grp])
        labels = np.asarray([int(r["label"]) for r in grp])
        if len(np.unique(labels)) < 2:
            continue
        e, thr = eer_from_bonafide(scores, labels)
        auc = safe_auc(scores, labels)
        tpr5 = tpr_at_5_fpr(scores, labels)
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        b = [float(s) for s, y in zip(scores, labels) if y == 1]
        s = [float(s) for s, y in zip(scores, labels) if y == 0]
        _, lo, hi = bootstrap_eer(grp, bootstrap)
        summaries.append({
            "model": model_label,
            "dataset": dataset,
            "codec": codec,
            "n_trials": len(grp),
            "n_bonafide": len(b),
            "n_spoof": len(s),
            "bonafide_score_mean": float(np.mean(b)) if b else math.nan,
            "spoof_score_mean": float(np.mean(s)) if s else math.nan,
            "eer": e,
            "eer_threshold_bonafide_score": thr,
            "auc": auc,
            "tpr_at_5_fpr": tpr5,
            "fpr_at_eer_threshold": fpr,
            "fnr_at_eer_threshold": fnr,
            "eer_bootstrap_ci_low": lo,
            "eer_bootstrap_ci_high": hi,
        })
    return scored, summaries


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    d = defaultdict(list)
    for r in rows:
        d[tuple(r[k] for k in keys)].append(r)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 6 Layer-2 anti-spoof evaluation; final test protected.")
    ap.add_argument("--split", choices=["dev"], default="dev")
    ap.add_argument("--public-manifest", type=Path, default=PUBLIC_DEFAULT)
    ap.add_argument("--skip-public", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    if args.bootstrap < 100:
        raise SystemExit("Use at least 100 bootstrap resamples.")

    own_rows = build_own_rows(args.split)
    public_rows = [] if args.skip_public else build_public_rows(args.public_manifest)
    all_rows = own_rows + public_rows

    if not own_rows:
        raise RuntimeError("No own clone/human rows found in the dev split.")
    if not args.skip_public and not public_rows:
        raise RuntimeError("Public manifest is empty.")

    print(f"Own rows: {len(own_rows)}")
    print(f"Public rows: {len(public_rows)}")
    print("Final test split: PROTECTED")

    models = {
        "AASIST-L": find_model("AASIST-L"),
        "XLSR-SLS": find_model("XLSR-SLS"),
    }

    all_scores = []
    all_summaries = []
    for label, path in models.items():
        scores, summaries = run_model(label, path, all_rows, args.bootstrap)
        all_scores.extend(scores)
        all_summaries.extend(summaries)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with SCORES_OUT.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in all_scores for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_scores)

    with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in all_summaries for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_summaries)

    # Clean-vs-codec EER drop for own data only.
    clean = {(r["model"], r["dataset"]): r for r in []}
    drops = []
    for model in models:
        own_summ = [r for r in all_summaries if r["model"] == model and r["dataset"] == "own"]
        clean_rows = [r for r in own_summ if r["codec"] == "clean"]
        if not clean_rows:
            continue
        base = clean_rows[0]
        for r in own_summ:
            if r["codec"] == "clean":
                continue
            drops.append({
                "model": model,
                "codec": r["codec"],
                "clean_eer": base["eer"],
                "codec_eer": r["eer"],
                "eer_change": float(r["eer"] - base["eer"]),
                "clean_bonafide_score_mean": base["bonafide_score_mean"],
                "codec_bonafide_score_mean": r["bonafide_score_mean"],
                "bonafide_score_drop": float(base["bonafide_score_mean"] - r["bonafide_score_mean"]),
            })

    with (RESULTS / "day6_codec_drops.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(drops[0].keys()) if drops else ["model", "codec", "clean_eer", "codec_eer", "eer_change"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(drops)

    payload = {
        "day": 6,
        "evaluation_split": args.split,
        "final_test_touched": False,
        "seed": SEED,
        "bootstrap_resamples": args.bootstrap,
        "models": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in models.items()},
        "own_rows": len(own_rows),
        "public_rows": len(public_rows),
        "public_manifest": str(args.public_manifest) if not args.skip_public else None,
        "score_direction": "higher_is_bonafide",
        "files": {
            "scores": str(SCORES_OUT.relative_to(ROOT)).replace("\\", "/"),
            "summary": str(SUMMARY_OUT.relative_to(ROOT)).replace("\\", "/"),
            "codec_drops": "results/tables/day6_codec_drops.csv",
        },
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== DAY 6 SUMMARY ===")
    for r in all_summaries:
        print(
            f"{r['model']:9s} | {r['dataset']:10s} | {r['codec']:16s} | "
            f"n={r['n_trials']:4d} | EER={r['eer']:.4f} | AUC={r['auc']:.4f}"
        )
    print("\nSaved:")
    print(SCORES_OUT)
    print(SUMMARY_OUT)
    print(RESULTS / "day6_codec_drops.csv")
    print(JSON_OUT)
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
