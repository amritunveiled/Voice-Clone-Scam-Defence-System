from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SEED = 12345
SAMPLE_RATE = 16000
AUDIO_SECONDS = 5.0
REPEATS = 3
BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)
AUDIO = RESULTS / "benchmark_5s_16k.wav"


def make_audio() -> None:
    if AUDIO.exists():
        return
    t = np.arange(int(SAMPLE_RATE * AUDIO_SECONDS), dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(SEED)
    signal = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.size).astype(np.float32)
    sf.write(AUDIO, signal, SAMPLE_RATE, subtype="PCM_16")


def bench(fn):
    fn()  # warm-up
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    mean = float(np.mean(times))
    return times, mean, mean / AUDIO_SECONDS


def find_onnx(folder: str) -> Path:
    candidates = sorted((BASE / "models" / folder).rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(
            f"No ONNX file found under models/{folder}. Run scripts/download_spoof_models.py first."
        )
    return max(candidates, key=lambda p: p.stat().st_size)


def run_onnx(path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_inputs()[0]
    length = meta.shape[-1]
    if not isinstance(length, int):
        length = 64600
    rng = np.random.default_rng(SEED)
    waveform = rng.standard_normal(length).astype(np.float32)[None, :]
    outputs = session.run(None, {meta.name: waveform})
    return [list(np.asarray(out).shape) for out in outputs]


def run_ecapa() -> None:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(BASE / "models" / "ecapa"),
        run_opts={"device": "cpu"},
    )
    waveform, sr = sf.read(AUDIO, dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Benchmark WAV must be 16 kHz, got {sr}")
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    with torch.no_grad():
        model.encode_batch(tensor)


def run_whisper() -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8, num_workers=1)
    segments, _ = model.transcribe(str(AUDIO), beam_size=1)
    list(segments)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["ecapa", "aasist_l", "xlsr_sls", "whisper", "all"],
        default="all",
    )
    args = parser.parse_args()
    make_audio()

    jobs = [args.only] if args.only != "all" else ["ecapa", "aasist_l", "xlsr_sls", "whisper"]
    output = {
        "date": time.strftime("%Y-%m-%d"),
        "benchmark_audio": str(AUDIO.relative_to(BASE)),
        "audio_duration_s": AUDIO_SECONDS,
        "seed": SEED,
        "results": {},
    }

    for job in jobs:
        print(f"\n=== {job} ===")
        holder = {}
        if job == "ecapa":
            fn = run_ecapa
            model = "speechbrain/spkrec-ecapa-voxceleb"
            device = "cpu"
        elif job == "aasist_l":
            path = find_onnx("AASIST-L")
            fn = lambda: holder.update(shapes=run_onnx(path))
            model = "SpeechAntiSpoofingBenchmarks/AASIST-L"
            device = "CPUExecutionProvider"
        elif job == "xlsr_sls":
            path = find_onnx("XLSR-SLS")
            fn = lambda: holder.update(shapes=run_onnx(path))
            model = "SpeechAntiSpoofingBenchmarks/XLSR-SLS"
            device = "CPUExecutionProvider"
        else:
            fn = run_whisper
            model = "faster-whisper:small"
            device = "cpu / int8"

        runs, mean, rtf = bench(fn)
        record = {
            "runs_s": runs,
            "mean_s": mean,
            "rtf": rtf,
            "audio_duration_s": AUDIO_SECONDS,
            "model": model,
            "device": device,
            "interpretation": "<1.0 is faster than real time; >=1.0 is slower than real time",
        }
        if job in {"aasist_l", "xlsr_sls"}:
            record["output_shapes"] = holder.get("shapes")
        if job == "whisper":
            record.update({"compute_type": "int8", "beam_size": 1})
        output["results"][job] = record
        print(json.dumps({job: record}, indent=2))

    out = RESULTS / "day1_rtf.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSaved:", out)
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
