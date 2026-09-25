from __future__ import annotations
import csv, sys, time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
QCF=ROOT/"results"/"tables"/"day4_clone_qc.csv"

def main():
    if not QCF.exists(): raise SystemExit(f"Missing {QCF}; run qc_clones_day4.py first.")
    rows=list(csv.DictReader(QCF.open(encoding="utf-8",newline="")))
    for i,r in enumerate(rows,1):
        if r["manual_listen"] in {"PASS","FAIL"}: continue
        p=ROOT/r["path"]
        audio,sr=sf.read(p,dtype="float32")
        print(f"\n[{i}/{len(rows)}] {r['job_id']}")
        print(f"Language: {r['language']} | Category: {r['category']}")
        print("Playing audio...")
        sd.play(audio,sr); sd.wait()
        decision=input("Manual QC — P=pass, F=fail, S=skip: ").strip().upper()
        if decision=="P": r["manual_listen"]="PASS"; r["notes"]="" if not r["notes"] else r["notes"]
        elif decision=="F": r["manual_listen"]="FAIL"; note=input("Why did it fail? ").strip(); r["notes"]=note
        else: continue
        with QCF.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Manual QC session complete.")
if __name__=="__main__": main()
