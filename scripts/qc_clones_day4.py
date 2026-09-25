from __future__ import annotations
import csv, wave
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
JOBS=ROOT/"data"/"manifests"/"day4_clone_jobs.csv"
OUT=ROOT/"results"/"tables"/"day4_clone_qc.csv"


def main():
    rows=list(csv.DictReader(JOBS.open(encoding="utf-8",newline="")))
    qc=[]
    for r in rows:
        p=ROOT/r["output_path"]
        item={"job_id":r["job_id"],"speaker_id":r["speaker_id"],"language":r["speaker_language"],"category":r["target_category"],"path":r["output_path"],"technical_status":"FAIL","duration_s":"","peak":"","rms":"","clipping_fraction":"","manual_listen":"PENDING","notes":""}
        try:
            audio,sr=sf.read(p,dtype="float32")
            if audio.ndim==2: audio=audio.mean(axis=1)
            peak=float(np.max(np.abs(audio))) if len(audio) else 0.0
            rms=float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            clipping=float(np.mean(np.abs(audio)>=0.999)) if len(audio) else 1.0
            duration=len(audio)/sr if sr else 0.0
            technical=(sr==24000 and len(audio)>0 and duration>0.2 and clipping<0.01 and rms>1e-4)
            item.update({"technical_status":"PASS" if technical else "FAIL","duration_s":f"{duration:.3f}","peak":f"{peak:.5f}","rms":f"{rms:.5f}","clipping_fraction":f"{clipping:.6f}"})
        except Exception as exc: item["notes"]=str(exc)
        qc.append(item)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=qc[0].keys()); w.writeheader(); w.writerows(qc)
    print(f"Wrote {len(qc)} QC rows to {OUT}")
    print("TECHNICAL PASS:",sum(x["technical_status"]=="PASS" for x in qc),"/",len(qc))
    print("MANUAL LISTEN: mark every generated clone as PASS or FAIL in column 'manual_listen'.")
    print("STATUS: PASS" if all(x["technical_status"]=="PASS" for x in qc) else "STATUS: FAIL")
    return 0 if all(x["technical_status"]=="PASS" for x in qc) else 1
if __name__=="__main__": raise SystemExit(main())
