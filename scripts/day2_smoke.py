from __future__ import annotations

import json
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Make sure "src" can be imported even when this script is launched
# as: python .\scripts\day2_smoke.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMOKE_DIR = ROOT / "data" / "raw" / "smoke"
CODEC_DIR = ROOT / "data" / "codec" / "day2"
RESULT_DIR = ROOT / "results" / "tables"
RESULT_FILE = RESULT_DIR / "day2_smoke.json"

MODEL_DIR = ROOT / "models"

EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1

# XLSR-SLS / AASIST-L ONNX input length from the model interface
DETECTOR_SAMPLES = 64_600


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_speaker_id(path: Path) -> str:
    """
    speaker01_01.wav -> speaker01
    speaker02_03.wav -> speaker02
    """
    match = re.match(r"^(.+?)_\d+$", path.stem)

    if match:
        return match.group(1)

    # Fallback if filename does not follow expected convention
    return path.stem


def find_smoke_files() -> list[Path]:
    """
    Find the five Day-2 smoke recordings.
    """
    if not SMOKE_DIR.exists():
        raise FileNotFoundError(
            f"Smoke directory does not exist:\n{SMOKE_DIR}"
        )

    files = sorted(SMOKE_DIR.glob("*.wav"))

    if len(files) < 5:
        raise RuntimeError(
            f"Expected at least 5 WAV files in:\n{SMOKE_DIR}\n"
            f"Found: {len(files)}"
        )

    return files[:5]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """
    Load mono float32 WAV.
    """
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    return audio, sr


def validate_audio(path: Path) -> dict:
    """
    Verify basic properties of a smoke-test WAV.
    """
    audio, sr = load_audio(path)

    duration = len(audio) / sr

    return {
        "file": path.name,
        "speaker_id": get_speaker_id(path),
        "sample_rate_hz": sr,
        "channels": 1,
        "samples": len(audio),
        "duration_s": float(duration),
        "valid_sample_rate": sr == EXPECTED_SAMPLE_RATE,
    }


def prepare_detector_audio(audio: np.ndarray) -> np.ndarray:
    """
    XLSR-SLS/AASIST-L smoke input.

    Their ONNX input is expected to contain 64,600 samples.
    Crop if longer and zero-pad if shorter.
    """
    if len(audio) >= DETECTOR_SAMPLES:
        return audio[:DETECTOR_SAMPLES]

    pad_length = DETECTOR_SAMPLES - len(audio)

    return np.pad(
        audio,
        (0, pad_length),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)


def locate_onnx_model(model_name: str) -> Path:
    """
    Search common model-folder naming variants.
    """
    candidates = [
        MODEL_DIR / model_name,
        MODEL_DIR / model_name.upper(),
        MODEL_DIR / model_name.lower(),
        MODEL_DIR / model_name.replace("-", "_"),
        MODEL_DIR / model_name.replace("-", "").lower(),
    ]

    found: list[Path] = []

    for directory in candidates:
        if directory.exists():
            found.extend(directory.rglob("*.onnx"))

    # Remove duplicate paths
    found = sorted(set(found))

    if not found:
        raise FileNotFoundError(
            f"No ONNX model found for '{model_name}'.\n"
            f"Expected a model under:\n{MODEL_DIR}\n"
            f"Run:\n"
            f"  .\\.venv\\Scripts\\python.exe "
            f".\\scripts\\download_spoof_models.py"
        )

    # If multiple files exist, choose the largest ONNX file.
    return max(found, key=lambda p: p.stat().st_size)


# ============================================================
# AUDIO VALIDATION
# ============================================================

def run_audio_validation(files: list[Path]) -> list[dict]:
    print_header("AUDIO VALIDATION")

    results = []

    for path in files:
        info = validate_audio(path)
        results.append(info)

        print(
            f"{path.name:25s} | "
            f"speaker={info['speaker_id']:12s} | "
            f"sr={info['sample_rate_hz']:5d} Hz | "
            f"duration={info['duration_s']:.2f}s | "
            f"valid_sr={info['valid_sample_rate']}"
        )

    invalid = [
        x for x in results
        if not x["valid_sample_rate"]
    ]

    if invalid:
        raise RuntimeError(
            "One or more recordings are not 16 kHz. "
            "Convert them to mono 16 kHz WAV before continuing."
        )

    return results


# ============================================================
# LAYER 1 — ECAPA SPEAKER MATCH
# ============================================================

def run_ecapa(files: list[Path]) -> dict:
    print_header("LAYER 1 — ECAPA SPEAKER MATCH")

    from speechbrain.inference.speaker import EncoderClassifier

    model_dir = ROOT / "models" / "ecapa"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ECAPA model...")

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(model_dir),
        run_opts={"device": "cpu"},
    )

    embeddings: dict[str, torch.Tensor] = {}

    print("Generating embeddings...")

    for path in files:
        audio, sr = load_audio(path)

        if sr != EXPECTED_SAMPLE_RATE:
            raise RuntimeError(
                f"{path.name}: expected 16 kHz, got {sr} Hz"
            )

        waveform = torch.from_numpy(audio).unsqueeze(0)

        with torch.no_grad():
            embedding = model.encode_batch(waveform)

        embedding = embedding.reshape(-1)
        embedding = F.normalize(embedding, p=2, dim=0)

        embeddings[path.name] = embedding

        print(f"Embedding generated: {path.name}")

    same_pairs = []
    different_pairs = []

    for file_a, file_b in combinations(files, 2):

        score = float(
            torch.dot(
                embeddings[file_a.name],
                embeddings[file_b.name],
            ).item()
        )

        row = {
            "file_a": file_a.name,
            "file_b": file_b.name,
            "speaker_a": get_speaker_id(file_a),
            "speaker_b": get_speaker_id(file_b),
            "cosine_similarity": score,
        }

        if get_speaker_id(file_a) == get_speaker_id(file_b):
            same_pairs.append(row)
        else:
            different_pairs.append(row)

    if not same_pairs:
        raise RuntimeError(
            "No same-speaker pairs found. "
            "You need multiple recordings from the same speaker."
        )

    if not different_pairs:
        raise RuntimeError(
            "No different-speaker pairs found. "
            "You need recordings from at least two speakers."
        )

    same_mean = float(
        np.mean([
            x["cosine_similarity"]
            for x in same_pairs
        ])
    )

    different_mean = float(
        np.mean([
            x["cosine_similarity"]
            for x in different_pairs
        ])
    )

    smoke_check = same_mean > different_mean

    print("\nSame-speaker pairs:")

    for row in same_pairs:
        print(
            f"  {row['file_a']} vs {row['file_b']} : "
            f"{row['cosine_similarity']:.6f}"
        )

    print("\nDifferent-speaker pairs:")

    for row in different_pairs:
        print(
            f"  {row['file_a']} vs {row['file_b']} : "
            f"{row['cosine_similarity']:.6f}"
        )

    print("\nECAPA smoke-test summary:")
    print(f"same_speaker_mean      = {same_mean:.6f}")
    print(f"different_speaker_mean = {different_mean:.6f}")
    print(f"smoke_check             = {smoke_check}")

    return {
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "device": "cpu",
        "same_speaker_pairs": same_pairs,
        "different_speaker_pairs": different_pairs,
        "same_speaker_mean": same_mean,
        "different_speaker_mean": different_mean,
        "smoke_check": smoke_check,
    }


# ============================================================
# LAYER 2 — ONNX ANTI-SPOOF MODEL
# ============================================================

def run_detector(
    files: list[Path],
    model_label: str,
    folder_name: str,
) -> dict:

    print_header(f"LAYER 2 — {model_label}")

    import onnxruntime as ort

    model_path = locate_onnx_model(folder_name)

    print(f"Model file: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]

    print(f"Input name : {input_meta.name}")
    print(f"Input shape: {input_meta.shape}")

    file_results = []

    for path in files:

        audio, sr = load_audio(path)

        if sr != EXPECTED_SAMPLE_RATE:
            raise RuntimeError(
                f"{path.name}: expected 16 kHz, got {sr} Hz"
            )

        detector_audio = prepare_detector_audio(audio)

        model_input = detector_audio.reshape(
            1,
            -1,
        ).astype(np.float32)

        outputs = session.run(
            None,
            {
                input_meta.name: model_input
            },
        )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        logits_flat = [
            float(x)
            for x in logits.reshape(-1)
        ]

        row = {
            "file": path.name,
            "logits": logits_flat,
            "output_shapes": [
                list(np.asarray(x).shape)
                for x in outputs
            ],
        }

        file_results.append(row)

        print(
            f"{path.name:25s} -> "
            f"logits={logits_flat}"
        )

    return {
        "status": "PASS",
        "model": model_label,
        "model_path": str(model_path),
        "device": "CPUExecutionProvider",
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "files": file_results,
        "note": (
            "Day-2 smoke test records raw detector logits only. "
            "No class direction, threshold, or calibrated probability "
            "is assumed yet."
        ),
    }


# ============================================================
# LAYER 3 — WHISPER
# ============================================================

def run_whisper(files: list[Path]) -> dict:
    print_header("LAYER 3 — WHISPER")

    from faster_whisper import WhisperModel

    cpu_threads = max(
        4,
        min(
            12,
            os.cpu_count() or 4,
        ),
    )

    print(
        f"Loading faster-whisper small "
        f"(CPU INT8, {cpu_threads} threads)..."
    )

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
    )

    results = []

    for path in files:

        print(f"\nTranscribing: {path.name}")

        segments, info = model.transcribe(
            str(path),
            beam_size=1,
            vad_filter=False,
        )

        transcript_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()

        row = {
            "file": path.name,
            "detected_language": info.language,
            "language_probability": float(
                getattr(
                    info,
                    "language_probability",
                    0.0,
                )
            ),
            "transcript": transcript,
        }

        results.append(row)

        print(
            f"Language   : {row['detected_language']}"
        )

        print(
            f"Transcript : {row['transcript']}"
        )

    return {
        "status": "PASS",
        "model": "faster-whisper:small",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "files": results,
    }


# ============================================================
# CODEC PIPELINE
# ============================================================

def run_codec_pipeline(files: list[Path]) -> list[dict]:
    print_header("TELEPHONY CODEC PIPELINE")

    try:
        from src.codecs import CODEC_SPECS, ffmpeg_encode, ffmpeg_decode_to_16k
    except Exception as exc:
        raise RuntimeError(
            "Could not import src.codecs.\n"
            "Make sure src\\codecs.py exists in the project."
        ) from exc

    CODEC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for audio_file in files:

        for codec_name, spec in CODEC_SPECS.items():

            print(
                f"\n{audio_file.name} -> {codec_name}"
            )

            extension_map = {
                "amr_nb_4_75": ".amr",
                "amr_nb_7_4": ".amr",
                "amr_nb_12_2": ".amr",
                "amr_wb_12_65": ".amr",
                "gsm": ".gsm",
                "opus_8": ".opus",
                "opus_10": ".opus",
                "opus_12": ".opus",
            }

            extension = extension_map.get(
                codec_name,
                ".audio",
            )

            encoded_file = (
                CODEC_DIR
                / f"{audio_file.stem}__{codec_name}{extension}"
            )

            decoded_file = (
                CODEC_DIR
                / f"{audio_file.stem}__{codec_name}__decoded_16k.wav"
            )

            # Encode
            ffmpeg_encode(
                audio_file,
                encoded_file,
                codec_name,
            )

            # Decode back to 16 kHz
            ffmpeg_decode_to_16k(
                encoded_file,
                decoded_file,
            )

            # Verify decoded audio
            decoded_audio, decoded_sr = load_audio(
                decoded_file
            )

            if decoded_sr != EXPECTED_SAMPLE_RATE:
                raise RuntimeError(
                    f"Decoded file {decoded_file.name} "
                    f"is {decoded_sr} Hz instead of 16 kHz."
                )

            result = {
                "input": str(audio_file),
                "codec": codec_name,
                "encoder": spec["encoder"],
                "target_bitrate": spec["bitrate"],
                "source_codec_rate_hz": spec["sample_rate"],
                "encoded_file": str(encoded_file),
                "decoded_file": str(decoded_file),
                "decoded_sample_rate_hz": decoded_sr,
                "decoded_samples": int(len(decoded_audio)),
                "status": "PASS",
            }

            results.append(result)

            print(
                f"PASS: {codec_name}"
            )

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_header(
        "DAY 2 — COMPLETE MODEL + CODEC SMOKE TEST"
    )

    print(f"Project root: {ROOT}")
    print(f"Smoke files : {SMOKE_DIR}")
    print(f"Results     : {RESULT_FILE}")

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files = find_smoke_files()

    print("\nSmoke files:")

    for path in files:
        print(
            f"  {path.name} "
            f"(speaker={get_speaker_id(path)})"
        )

    # --------------------------------------------------------
    # Audio validation
    # --------------------------------------------------------

    audio_validation = run_audio_validation(
        files
    )

    # --------------------------------------------------------
    # Layer 1
    # --------------------------------------------------------

    ecapa_result = run_ecapa(
        files
    )

    # --------------------------------------------------------
    # Layer 2 — AASIST-L
    # --------------------------------------------------------

    aasist_result = run_detector(
        files,
        "SpeechAntiSpoofingBenchmarks/AASIST-L",
        "AASIST-L",
    )

    # --------------------------------------------------------
    # Layer 2 — XLSR-SLS
    # --------------------------------------------------------

    xlsr_result = run_detector(
        files,
        "SpeechAntiSpoofingBenchmarks/XLSR-SLS",
        "XLSR-SLS",
    )

    # --------------------------------------------------------
    # Layer 3
    # --------------------------------------------------------

    whisper_result = run_whisper(
        files
    )

    # --------------------------------------------------------
    # Codec pipeline
    # --------------------------------------------------------

    codec_result = run_codec_pipeline(
        files
    )

    # --------------------------------------------------------
    # Build final JSON
    # --------------------------------------------------------

    final_result = {
        "day": 2,
        "project_root": str(ROOT),
        "num_smoke_files": len(files),
        "files": [x.name for x in files],
        "audio_validation": audio_validation,
        "ecapa": ecapa_result,
        "aasist_l": aasist_result,
        "xlsr_sls": xlsr_result,
        "whisper": whisper_result,
        "codec_pipeline": codec_result,
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Acceptance summary
    # --------------------------------------------------------

    print_header(
        "DAY 2 ACCEPTANCE SUMMARY"
    )

    ecapa_pass = bool(
        ecapa_result["smoke_check"]
    )

    aasist_pass = (
        aasist_result["status"] == "PASS"
        and len(aasist_result["files"]) == len(files)
    )

    xlsr_pass = (
        xlsr_result["status"] == "PASS"
        and len(xlsr_result["files"]) == len(files)
    )

    whisper_pass = (
        whisper_result["status"] == "PASS"
        and len(whisper_result["files"]) == len(files)
    )

    codec_pass = (
        len(codec_result) > 0
        and all(
            x["status"] == "PASS"
            for x in codec_result
        )
    )

    print(
        f"[{'PASS' if len(files) >= 5 else 'FAIL'}] "
        f"5 smoke files available"
    )

    print(
        f"[{'PASS' if ecapa_pass else 'FAIL'}] "
        f"ECAPA same-speaker mean > different-speaker mean"
    )

    print(
        f"[{'PASS' if aasist_pass else 'FAIL'}] "
        f"AASIST-L produced scores"
    )

    print(
        f"[{'PASS' if xlsr_pass else 'FAIL'}] "
        f"XLSR-SLS produced scores"
    )

    print(
        f"[{'PASS' if whisper_pass else 'FAIL'}] "
        f"Whisper produced transcripts"
    )

    print(
        f"[{'PASS' if codec_pass else 'FAIL'}] "
        f"Codec pipeline completed"
    )

    all_pass = (
        len(files) >= 5
        and ecapa_pass
        and aasist_pass
        and xlsr_pass
        and whisper_pass
        and codec_pass
    )

    print("\nOverall Day-2 model/codec smoke status:")
    print(
        "STATUS: PASS"
        if all_pass
        else "STATUS: FAIL / INVESTIGATE"
    )

    print(
        f"\nResults saved to:\n{RESULT_FILE}"
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())