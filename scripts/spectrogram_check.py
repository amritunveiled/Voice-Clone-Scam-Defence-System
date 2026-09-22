from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AMR-NB decoded spectrum for Day 2.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figures/day2_amr_nb_spectrogram.png"))
    args = parser.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frame = 1024
    hop = 256
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame-len(audio)))
    window = np.hanning(frame).astype(np.float32)
    columns = []
    for start in range(0, len(audio)-frame+1, hop):
        columns.append(np.abs(np.fft.rfft(audio[start:start+frame]*window)))
    spec = np.stack(columns, axis=1)
    freqs = np.fft.rfftfreq(frame, 1/sr)
    times = np.arange(spec.shape[1])*hop/sr
    spec_db = 20*np.log10(spec+1e-8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mesh=ax.pcolormesh(times, freqs, spec_db, shading="auto")
    ax.axhline(3400, linestyle="--", linewidth=1.5, label="3.4 kHz check line")
    ax.set_ylim(0, min(sr/2, 8000))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"AMR-NB decoded spectrogram: {args.wav.name}")
    ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print("Saved:", args.out)
    print("MANUAL ACCEPTANCE: material energy should not extend meaningfully above about 3.4 kHz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
