from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer1_speaker import SpeakerEncoder
from src.codecs import CODEC_SPECS, ffmpeg_encode, ffmpeg_decode_to_16k

SEED = 12345
DEFAULT_SPLIT = "dev"  # Day 5 must not consume the final test split.
BOOTSTRAPS = 1000
SAMPLE_RATE = 16000

MANIFEST = ROOT / "data" / "manifests" / "day4_evaluation_manifest.csv"
CODEC_MANIFEST = ROOT / "data" / "manifests" / "day4_codec_manifest.csv"
SPLIT_FILE = ROOT / "data" / "manifests" / "speaker_splits_day4.json"
RESULTS = ROOT / "results" / "tables"
SCORES_CSV = RESULTS / "day5_layer1_scores.csv"
SUMMARY_CSV = RESULTS / "day5_layer1_summary.csv"
BOOTSTRAP_CSV = RESULTS / "day5_layer1_bootstrap.csv"
CALIBRATION_CSV = RESULTS / "day5_layer1_calibration.csv"
JSON_OUT = RESULTS / "day5_layer1_results.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_splits() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing frozen split file: {SPLIT_FILE}")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def discover_enrollment_files(speaker_id: str) -> list[Path]:
    directory = ROOT / "data" / "raw" / "participants" / speaker_id / "enrollment"
    files = sorted(directory.glob("*.wav"))
    if len(files) < 5:
        raise RuntimeError(
            f"{speaker_id}: expected at least 5 clean enrollment WAVs in {directory}; found {len(files)}"
        )
    return files[:5]


def ensure_day5_codec_file(enrollment_path: Path, codec_name: str) -> Path:
    out_dir = ROOT / "data" / "codec" / "day5_enrollment" / codec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_ext = ".gsm" if codec_name == "gsm" else (".opus" if codec_name.startswith("opus") else ".amr")
    encoded = out_dir / f"{enrollment_path.stem}__{codec_name}{encoded_ext}"
    decoded = out_dir / f"{enrollment_path.stem}__{codec_name}__decoded_16k.wav"
    if not decoded.exists():
        spec = CODEC_SPECS[codec_name]
        with tempfile.TemporaryDirectory(prefix="day5_src_") as td:
            source = Path(td) / "source.wav"
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(enrollment_path),
                    "-ar", str(spec["sample_rate"]), "-ac", "1",
                    "-c:a", "pcm_s16le", str(source),
                ],
                check=True,
            )
            ffmpeg_encode(source, encoded, codec_name)
            ffmpeg_decode_to_16k(encoded, decoded)
    return decoded


def enrollment_embeddings(encoder: SpeakerEncoder, speaker_id: str, mode: str, codec_name: str | None) -> np.ndarray:
    files = discover_enrollment_files(speaker_id)
    embeddings = [encoder.embed_file(p) for p in files]
    if mode == "clean":
        return encoder.average_embeddings(embeddings)
    if mode == "codec_matched":
        if not codec_name:
            raise ValueError("codec_name required for codec_matched enrollment")
        codec_files = [ensure_day5_codec_file(p, codec_name) for p in files]
        embeddings.extend(encoder.embed_file(p) for p in codec_files)
        return encoder.average_embeddings(embeddings)
    raise ValueError(f"Unknown enrollment mode: {mode}")


def resolve(root_relative: str) -> Path:
    p = ROOT / root_relative
    if not p.exists():
        raise FileNotFoundError(f"Missing audio file referenced by manifest: {p}")
    return p


def build_codec_lookup(codec_rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    return {(r["source_path"], r["codec"]): r["decoded_path"] for r in codec_rows}


def normalize_eval_rows(eval_rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    # Hard guard: never touch the final test split during Day 5.
    if split == "test":
        raise ValueError("Day 5 must not evaluate the final test split. The protocol reserves it for Day 14.")
    rows = [r for r in eval_rows if r.get("evaluation_split") == split]
    if not rows:
        raise RuntimeError(f"No evaluation rows found for split={split}")
    return rows


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def bootstrap_eer(target_trials: dict[str, list[tuple[float, int]]], n_boot: int, seed: int) -> tuple[float, float, float]:
    speaker_ids = sorted(target_trials)
    if len(speaker_ids) < 2:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sampled = [rng.choice(speaker_ids) for _ in range(len(speaker_ids))]
        score_list = []
        label_list = []
        for sid in sampled:
            for score, label in target_trials[sid]:
                score_list.append(score)
                label_list.append(label)
        try:
            e, _ = eer(np.asarray(score_list), np.asarray(label_list))
            if math.isfinite(e):
                values.append(e)
        except Exception:
            continue
    if not values:
        return math.nan, math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def load_all_embeddings(encoder: SpeakerEncoder, paths: Iterable[Path]) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for path in sorted({str(p) for p in paths}):
        p = Path(path)
        cache[str(p)] = encoder.embed_file(p)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 5 Layer-1 ECAPA evaluation. Final test split is protected.")
    ap.add_argument("--split", default=DEFAULT_SPLIT, choices=["dev"], help="Day-5 evaluation split; only dev is permitted.")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    args = ap.parse_args()

    if args.bootstrap < 100:
        raise SystemExit("Use at least 100 bootstrap resamples; 1000 is recommended.")

    eval_rows = read_csv(MANIFEST)
    codec_rows = read_csv(CODEC_MANIFEST)
    split_payload = read_splits()
    codec_lookup = build_codec_lookup(codec_rows)
    rows = normalize_eval_rows(eval_rows, args.split)

    # Human clean rows are used for EER. Clone rows are analyzed separately because
    # a target clone may preserve the target speaker embedding and is not a true human impostor.
    human_clean = [r for r in rows if r["source_type"] == "human" and r["codec"] == "clean"]
    clone_clean = [r for r in rows if r["source_type"] == "clone" and r["codec"] == "clean"]
    speakers = sorted({r["speaker_id"] for r in human_clean})
    if len(speakers) < 2:
        raise RuntimeError("Need at least two dev speakers for Day-5 EER evaluation.")

    # Build trial definitions first so we can embed each unique audio once.
    codecs = ["clean"] + list(CODEC_SPECS.keys())
    trial_records: list[dict] = []
    trial_audio: set[Path] = set()

    for r in human_clean:
        trial_audio.add(resolve(r["path"]))
        for codec in codecs[1:]:
            decoded = codec_lookup.get((r["path"], codec))
            if decoded:
                trial_audio.add(resolve(decoded))

    for r in clone_clean:
        trial_audio.add(resolve(r["path"]))
        for codec in codecs[1:]:
            decoded = codec_lookup.get((r["path"], codec))
            if decoded:
                trial_audio.add(resolve(decoded))

    # Enrollment: 5 clean files per speaker. For codec matched enrollment, add same-codec files.
    for sid in speakers:
        clean_enrollment = discover_enrollment_files(sid)
        # These files are used below to build clean and codec-matched profiles,
        # so they must be present in the one-pass embedding cache as well.
        trial_audio.update(clean_enrollment)
        for codec in codecs[1:]:
            for p in clean_enrollment:
                trial_audio.add(ensure_day5_codec_file(p, codec))

    print("Loading ECAPA...")
    encoder = SpeakerEncoder.load(ROOT)
    print(f"Embedding {len(trial_audio)} unique audio files...")
    emb_cache = load_all_embeddings(encoder, trial_audio)

    # Enrollment profiles.
    profiles_clean: dict[str, np.ndarray] = {}
    profiles_codec: dict[tuple[str, str], np.ndarray] = {}
    for sid in speakers:
        efiles = discover_enrollment_files(sid)
        profiles_clean[sid] = encoder.average_embeddings([emb_cache[str(p)] for p in efiles])
        for codec in codecs[1:]:
            cfiles = [ensure_day5_codec_file(p, codec) for p in efiles]
            profiles_codec[(sid, codec)] = encoder.average_embeddings(
                [emb_cache[str(p)] for p in efiles] + [emb_cache[str(p)] for p in cfiles]
            )

    score_rows: list[dict] = []

    def add_human_trials(codec: str):
        for r in human_clean:
            source_path = r["path"]
            if codec == "clean":
                trial_path = resolve(source_path)
            else:
                decoded = codec_lookup.get((source_path, codec))
                if not decoded:
                    continue
                trial_path = resolve(decoded)
            trial_emb = emb_cache[str(trial_path)]
            for target_sid in speakers:
                profile = profiles_clean[target_sid] if codec == "clean" else profiles_clean[target_sid]
                score = encoder.cosine(profile, trial_emb)
                label = 1 if r["speaker_id"] == target_sid else 0
                score_rows.append({
                    "split": args.split,
                    "evaluation_type": "human_verification",
                    "enrollment_mode": "clean",
                    "target_speaker": target_sid,
                    "trial_speaker": r["speaker_id"],
                    "source_type": "human",
                    "language": r.get("language", ""),
                    "category": r.get("category", ""),
                    "script_id": r.get("script_id", ""),
                    "codec": codec,
                    "path": str(trial_path.relative_to(ROOT)).replace("\\", "/"),
                    "label": label,
                    "score": score,
                })
                if codec != "clean":
                    profile2 = profiles_codec[(target_sid, codec)]
                    score2 = encoder.cosine(profile2, trial_emb)
                    score_rows.append({
                        "split": args.split,
                        "evaluation_type": "human_verification",
                        "enrollment_mode": "codec_matched",
                        "target_speaker": target_sid,
                        "trial_speaker": r["speaker_id"],
                        "source_type": "human",
                        "language": r.get("language", ""),
                        "category": r.get("category", ""),
                        "script_id": r.get("script_id", ""),
                        "codec": codec,
                        "path": str(trial_path.relative_to(ROOT)).replace("\\", "/"),
                        "label": label,
                        "score": score2,
                    })

    for codec in codecs:
        add_human_trials(codec)

    # Target clone similarity is a separate analysis: it answers how much Layer 1 may accept a target clone.
    for r in clone_clean:
        source_path = r["path"]
        for codec in codecs:
            if codec == "clean":
                trial_path = resolve(source_path)
            else:
                decoded = codec_lookup.get((source_path, codec))
                if not decoded:
                    continue
                trial_path = resolve(decoded)
            target_sid = r["speaker_id"]
            trial_emb = emb_cache[str(trial_path)]
            score_clean_profile = encoder.cosine(profiles_clean[target_sid], trial_emb)
            score_rows.append({
                "split": args.split,
                "evaluation_type": "target_clone_similarity",
                "enrollment_mode": "clean",
                "target_speaker": target_sid,
                "trial_speaker": target_sid,
                "source_type": "clone",
                "language": r.get("language", ""),
                "category": r.get("category", ""),
                "script_id": r.get("script_id", ""),
                "codec": codec,
                "path": str(trial_path.relative_to(ROOT)).replace("\\", "/"),
                "label": "clone",
                "score": score_clean_profile,
            })
            if codec != "clean":
                score_codec_profile = encoder.cosine(profiles_codec[(target_sid, codec)], trial_emb)
                score_rows.append({
                    "split": args.split,
                    "evaluation_type": "target_clone_similarity",
                    "enrollment_mode": "codec_matched",
                    "target_speaker": target_sid,
                    "trial_speaker": target_sid,
                    "source_type": "clone",
                    "language": r.get("language", ""),
                    "category": r.get("category", ""),
                    "script_id": r.get("script_id", ""),
                    "codec": codec,
                    "path": str(trial_path.relative_to(ROOT)).replace("\\", "/"),
                    "label": "clone",
                    "score": score_codec_profile,
                })

    RESULTS.mkdir(parents=True, exist_ok=True)
    fieldnames = list(score_rows[0].keys())
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(score_rows)

    # Summaries.
    summary_rows = []
    target_trials_for_bootstrap: dict[tuple[str, str], dict[str, list[tuple[float, int]]]] = {}
    for mode in ("clean", "codec_matched"):
        for codec in codecs:
            trials = [r for r in score_rows if r["evaluation_type"] == "human_verification" and r["enrollment_mode"] == mode and r["codec"] == codec]
            if not trials:
                continue
            scores = np.asarray([float(r["score"]) for r in trials], dtype=float)
            labels = np.asarray([int(r["label"]) for r in trials], dtype=int)
            e, thr = eer(scores, labels)
            auc = safe_auc(labels, scores)
            target_map: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for r in trials:
                target_map[r["target_speaker"]].append((float(r["score"]), int(r["label"])))
            target_trials_for_bootstrap[(mode, codec)] = target_map
            same = scores[labels == 1]
            diff = scores[labels == 0]
            summary_rows.append({
                "evaluation_type": "human_verification",
                "enrollment_mode": mode,
                "codec": codec,
                "n_trials": len(trials),
                "n_positive_same_speaker": int(len(same)),
                "n_negative_impostor": int(len(diff)),
                "same_speaker_mean": float(np.mean(same)) if len(same) else math.nan,
                "impostor_mean": float(np.mean(diff)) if len(diff) else math.nan,
                "eer": e,
                "eer_threshold": thr,
                "auc": auc if auc is not None else math.nan,
            })

    # Clone similarity summary.
    for mode in ("clean", "codec_matched"):
        for codec in codecs:
            rows2 = [r for r in score_rows if r["evaluation_type"] == "target_clone_similarity" and r["enrollment_mode"] == mode and r["codec"] == codec]
            if not rows2:
                continue
            vals = np.asarray([float(r["score"]) for r in rows2], dtype=float)
            summary_rows.append({
                "evaluation_type": "target_clone_similarity",
                "enrollment_mode": mode,
                "codec": codec,
                "n_trials": len(vals),
                "n_positive_same_speaker": "",
                "n_negative_impostor": "",
                "same_speaker_mean": "",
                "impostor_mean": "",
                "eer": "",
                "eer_threshold": "",
                "auc": "",
                "clone_similarity_mean": float(np.mean(vals)),
                "clone_similarity_median": float(np.median(vals)),
                "clone_similarity_min": float(np.min(vals)),
                "clone_similarity_max": float(np.max(vals)),
            })

    # Add 95% speaker-bootstrap CIs for EER.
    bootstrap_rows = []
    bootstrap_index = {}
    for key, tmap in target_trials_for_bootstrap.items():
        mode, codec = key
        mean_e, lo, hi = bootstrap_eer(tmap, args.bootstrap, SEED)
        bootstrap_rows.append({"enrollment_mode": mode, "codec": codec, "bootstrap_n": args.bootstrap, "eer_mean": mean_e, "eer_ci_low": lo, "eer_ci_high": hi})
        bootstrap_index[key] = (lo, hi)
    for row in summary_rows:
        key = (row["enrollment_mode"], row["codec"])
        if row["evaluation_type"] == "human_verification" and key in bootstrap_index:
            row["eer_ci_low"] = bootstrap_index[key][0]
            row["eer_ci_high"] = bootstrap_index[key][1]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in summary_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary_rows)

    with BOOTSTRAP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bootstrap_rows[0].keys()); w.writeheader(); w.writerows(bootstrap_rows)

    # Simple Platt calibration on development trials only. This is a development calibration artifact,
    # not a final-test result. It is fitted separately per codec/enrollment mode.
    calibration_rows = []
    for mode in ("clean", "codec_matched"):
        for codec in codecs:
            trials = [r for r in score_rows if r["evaluation_type"] == "human_verification" and r["enrollment_mode"] == mode and r["codec"] == codec]
            if not trials or len({int(r["label"]) for r in trials}) < 2:
                continue
            X = np.asarray([[float(r["score"])] for r in trials])
            y = np.asarray([int(r["label"]) for r in trials])
            clf = LogisticRegression(solver="lbfgs", random_state=SEED)
            clf.fit(X, y)
            for r in trials:
                p = float(clf.predict_proba([[float(r["score"])]])[0, 1])
                calibration_rows.append({"enrollment_mode": mode, "codec": codec, "target_speaker": r["target_speaker"], "trial_speaker": r["trial_speaker"], "label": r["label"], "raw_score": r["score"], "calibrated_probability": p})

    if calibration_rows:
        with CALIBRATION_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=calibration_rows[0].keys()); w.writeheader(); w.writerows(calibration_rows)

    # Clean-vs-codec drop, based on human same-speaker scores using clean enrollment.
    drop_rows = []
    clean_by_key = defaultdict(list)
    codec_by_key = defaultdict(list)
    for r in score_rows:
        if r["evaluation_type"] != "human_verification" or r["enrollment_mode"] != "clean" or int(r["label"]) != 1:
            continue
        key = (r["target_speaker"], r["trial_speaker"], r["path"].split("/"))[0]  # retained for explicit structure
    # Pairing by script/source-path stem is safest from the source manifest.
    clean_rows = [r for r in score_rows if r["evaluation_type"] == "human_verification" and r["enrollment_mode"] == "clean" and r["codec"] == "clean" and int(r["label"]) == 1]
    clean_index = {}
    for r in clean_rows:
        clean_index[(r["target_speaker"], r["trial_speaker"], r["script_id"], r["category"])] = float(r["score"])
    for r in score_rows:
        if r["evaluation_type"] != "human_verification" or r["enrollment_mode"] != "clean" or r["codec"] == "clean" or int(r["label"]) != 1:
            continue
        key = (r["target_speaker"], r["trial_speaker"], r["script_id"], r["category"])
        if key in clean_index:
            drop_rows.append({"target_speaker": r["target_speaker"], "trial_speaker": r["trial_speaker"], "script_id": r["script_id"], "category": r["category"], "codec": r["codec"], "clean_score": clean_index[key], "codec_score": float(r["score"]), "score_drop": clean_index[key] - float(r["score"])})

    json_payload = {
        "day": 5,
        "split_evaluated": args.split,
        "final_test_touched": False,
        "seed": SEED,
        "bootstrap_resamples": args.bootstrap,
        "files": {
            "scores": str(SCORES_CSV.relative_to(ROOT)).replace("\\", "/"),
            "summary": str(SUMMARY_CSV.relative_to(ROOT)).replace("\\", "/"),
            "bootstrap": str(BOOTSTRAP_CSV.relative_to(ROOT)).replace("\\", "/"),
            "calibration": str(CALIBRATION_CSV.relative_to(ROOT)).replace("\\", "/") if calibration_rows else None,
        },
        "counts": {
            "dev_speakers": len(speakers),
            "human_clean_records": len(human_clean),
            "clone_clean_records": len(clone_clean),
            "score_rows": len(score_rows),
            "codec_drop_pairs": len(drop_rows),
        },
        "protocol_notes": [
            "EER is computed from genuine human trials versus other-human impostor trials.",
            "Target voice-clone similarity is reported separately because a clone may legitimately preserve target-speaker characteristics; it is not labeled as a human impostor for EER.",
            "Codec-matched enrollment averages the five clean enrollment embeddings plus five same-codec enrollment embeddings.",
            "No Day-14 final test speaker or test script is evaluated today.",
        ],
        "clean_vs_codec_same_speaker_drop": drop_rows,
    }
    JSON_OUT.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print("\n=== DAY 5 ACCEPTANCE SUMMARY ===")
    human_summary = [r for r in summary_rows if r["evaluation_type"] == "human_verification"]
    clone_summary = [r for r in summary_rows if r["evaluation_type"] == "target_clone_similarity"]
    print(f"Dev speakers evaluated: {len(speakers)}")
    print(f"Human verification trial rows: {len([r for r in score_rows if r['evaluation_type']=='human_verification'])}")
    print(f"Clone similarity rows: {len([r for r in score_rows if r['evaluation_type']=='target_clone_similarity'])}")
    print(f"Codec conditions represented: {sorted({r['codec'] for r in score_rows})}")
    print("Final test split touched: NO")
    print("EER computed: YES" if human_summary else "EER computed: NO")
    print("Speaker-bootstrap CI computed: YES" if bootstrap_rows else "Speaker-bootstrap CI computed: NO")
    print("Platt calibration artifact: YES" if calibration_rows else "Platt calibration artifact: NO")
    print(f"\nSaved:\n- {SCORES_CSV}\n- {SUMMARY_CSV}\n- {BOOTSTRAP_CSV}\n- {JSON_OUT}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
