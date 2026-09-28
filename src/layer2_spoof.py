from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

FIXED_SAMPLES = 64600
SAMPLE_RATE = 16000


def pad_fixed(audio: np.ndarray, max_len: int = FIXED_SAMPLES) -> np.ndarray:
    """Deterministic evaluation window used by the published wrappers.

    The first `max_len` samples are used; shorter signals are tile-repeated.
    This avoids zero-padding changing the acoustic distribution at the end of
    a short clip.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Audio is empty")
    if x.size >= max_len:
        return x[:max_len]
    reps = max_len // x.size + 1
    return np.tile(x, reps)[:max_len].astype(np.float32)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class ONNXSpoofScorer:
    """Thin wrapper for AASIST-L / XLSR-SLS ONNX files.

    Both current SpeechAntiSpoofingBenchmarks wrappers expose class 1 as
    bona-fide, so higher returned score means more bona-fide.
    """

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]

    def score_audio(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz input, got {sample_rate} Hz. "
                "Resample to 16 kHz before scoring."
            )

        x = pad_fixed(audio).reshape(1, -1).astype(np.float32)
        outputs = self.session.run(None, {self.input.name: x})
        arr = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        if arr.size < 2:
            raise RuntimeError(
                f"Expected a 2-class anti-spoof output, got shape {np.asarray(outputs[0]).shape}"
            )

        # Published wrappers return output[:, 1] as the bona-fide score.
        return float(arr[1])

    def score_files(self, paths: Iterable[Path]) -> list[float]:
        import soundfile as sf

        scores = []
        for path in paths:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            scores.append(self.score_audio(audio, sr))
        return scores
