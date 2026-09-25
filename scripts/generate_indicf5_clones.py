from __future__ import annotations
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "manifests" / "day4_clone_jobs.csv"
MODEL_ID = "ai4bharat/IndicF5"
META = ROOT / "results" / "tables" / "day4_clone_generation.json"


def normalize_audio(audio):
    if isinstance(audio, torch.Tensor):
        audio=audio.detach().cpu().numpy()
    audio=np.asarray(audio)
    if audio.dtype == np.int16:
        audio=audio.astype(np.float32)/32768.0
    else:
        audio=audio.astype(np.float32)
    if audio.ndim > 1:
        audio=np.squeeze(audio)
    peak=float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak>1.0:
        audio=audio/peak
    return audio


def check_reference(path: Path):
    info=sf.info(path)
    if info.channels != 1:
        raise RuntimeError(f"Reference must be mono: {path}")
    if info.duration >= 15.0:
        raise RuntimeError(f"Reference must be <15 seconds: {path} is {info.duration:.2f}s")


def main() -> int:
    parser=argparse.ArgumentParser(description="Generate consented research clones with AI4Bharat IndicF5.")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    args=parser.parse_args()
    if not JOBS.exists(): raise SystemExit(f"Missing {JOBS}; run prepare_day4_clone_jobs.py first.")
    rows=list(csv.DictReader(JOBS.open(encoding="utf-8", newline="")))
    if args.max_jobs: rows=rows[:args.max_jobs]
    if not rows: raise SystemExit("No clone jobs found.")
    for r in rows:
        consent=ROOT/"data"/"raw"/"participants"/r["speaker_id"]/"consent"/"voice_clone_consent.txt"
        if not consent.exists(): raise SystemExit(f"STOP: missing explicit voice-clone consent marker: {consent}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Loading {MODEL_ID} on CPU...")
    model=AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model=model.to("cpu").eval()
    try:
        revision=HfApi().model_info(MODEL_ID).sha
    except Exception:
        revision="UNVERIFIED"
    generated=[]
    for i,r in enumerate(rows,1):
        ref=ROOT/r["reference_audio"]
        check_reference(ref)
        out=ROOT/r["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(rows)}] {r['job_id']} | {r['speaker_language']} | {r['target_category']}")
        t0=time.perf_counter()
        audio=model(r["target_text"], ref_audio_path=str(ref), ref_text=r["reference_text"])
        audio=normalize_audio(audio)
        if audio.size == 0:
            raise RuntimeError(f"Model returned empty audio for {r['job_id']}")
        sf.write(out,audio,24000,subtype="PCM_16")
        elapsed=time.perf_counter()-t0
        r["status"]="GENERATED"; r["generation_seconds"]=f"{elapsed:.3f}"; r["output_sample_rate_hz"]="24000"; r["output_duration_s"]=f"{len(audio)/24000:.3f}"
        generated.append(r)
        print(f"  saved {out} ({len(audio)/24000:.2f}s, {elapsed:.2f}s)")
    # update job CSV statuses for generated rows
    all_rows=list(csv.DictReader(JOBS.open(encoding="utf-8", newline="")))
    gen_map={r["job_id"]:r for r in generated}
    for r in all_rows:
        if r["job_id"] in gen_map: r.update(gen_map[r["job_id"]])
    with JOBS.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps({"model_id":MODEL_ID,"model_revision":revision,"seed":args.seed,"num_generated":len(generated),"outputs":generated},indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved metadata: {META}")
    print("STATUS: PASS")
    return 0

if __name__=="__main__": raise SystemExit(main())
