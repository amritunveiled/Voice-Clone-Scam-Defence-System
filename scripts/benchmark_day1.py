from __future__ import annotations
import argparse, json, time
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


def make_audio():
    if not AUDIO.exists():
        t = np.arange(int(SAMPLE_RATE*AUDIO_SECONDS), dtype=np.float32) / SAMPLE_RATE
        rng = np.random.default_rng(SEED)
        x = 0.05*np.sin(2*np.pi*220*t) + 0.01*rng.standard_normal(t.size).astype(np.float32)
        sf.write(AUDIO, x, SAMPLE_RATE, subtype="PCM_16")


def bench(fn):
    times=[]
    for _ in range(REPEATS):
        t0=time.perf_counter(); fn(); times.append(time.perf_counter()-t0)
    mean=float(np.mean(times))
    return times, mean, mean/AUDIO_SECONDS


def find_onnx(name: str) -> Path:
    cands=sorted((BASE/"models"/name).rglob("*.onnx"))
    if not cands:
        raise FileNotFoundError(f"No ONNX file found under models/{name}. Run scripts/download_spoof_models.py")
    return max(cands, key=lambda p: p.stat().st_size)


def run_onnx(path: Path):
    import onnxruntime as ort
    sess=ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp=sess.get_inputs()[0]
    shape=inp.shape
    n=shape[-1]
    if not isinstance(n, int):
        n=64600
    rng=np.random.default_rng(SEED)
    wav=rng.standard_normal(n).astype(np.float32)[None, :]
    out=sess.run(None, {inp.name:wav})
    return [list(o.shape) for o in out]


def run_ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    wav, sr = sf.read(AUDIO, dtype="float32")
    x=torch.from_numpy(wav).unsqueeze(0)
    if sr != 16000:
        raise RuntimeError(f"Benchmark WAV must be 16 kHz, got {sr}")
    model.encode_batch(x)


def run_whisper():
    from faster_whisper import WhisperModel
    model=WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(AUDIO), beam_size=1)
    list(segments)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ecapa","aasist_l","xlsr_sls","whisper","all"], default="all")
    args=ap.parse_args()
    make_audio()
    jobs=[] if args.only=="all" else [args.only]
    if args.only=="all": jobs=["ecapa","aasist_l","xlsr_sls","whisper"]
    output={"date": time.strftime("%Y-%m-%d"), "benchmark_audio": str(AUDIO.relative_to(BASE)), "audio_duration_s": AUDIO_SECONDS, "seed": SEED, "results": {}}
    for j in jobs:
        print(f"\n=== {j} ===")
        if j=="ecapa":
            fn=run_ecapa; model="speechbrain/spkrec-ecapa-voxceleb"; device="cpu"
        elif j=="aasist_l":
            path=find_onnx("AASIST-L"); holder={}
            fn=lambda: holder.update(shapes=run_onnx(path)); model="SpeechAntiSpoofingBenchmarks/AASIST-L"; device="CPUExecutionProvider"
        elif j=="xlsr_sls":
            path=find_onnx("XLSR-SLS"); holder={}
            fn=lambda: holder.update(shapes=run_onnx(path)); model="SpeechAntiSpoofingBenchmarks/XLSR-SLS"; device="CPUExecutionProvider"
        else:
            fn=run_whisper; model="faster-whisper:small"; device="cpu / int8"
        runs, mean, rtf=bench(fn)
        rec={"runs_s":runs,"mean_s":mean,"rtf":rtf,"audio_duration_s":AUDIO_SECONDS,"model":model,"device":device,"interpretation":"<1.0 is faster than real time; >=1.0 is slower than real time"}
        if j in {"aasist_l","xlsr_sls"}: rec["output_shapes"]=holder.get("shapes")
        if j=="whisper": rec.update({"compute_type":"int8","beam_size":1})
        output["results"][j]=rec
        print(json.dumps({j:rec}, indent=2))
    out=RESULTS/"day1_rtf.json"; out.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print("\nSaved:", out)
    print("STATUS: PASS")

if __name__=="__main__": main()
