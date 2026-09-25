from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
# Support direct execution (`python scripts/apply_day4_codecs.py`) by exposing
# the repository root, which contains the `src` package, to Python's importer.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.codecs import CODEC_SPECS, ffmpeg_encode, ffmpeg_decode_to_16k

def iter_inputs():
    # Evaluation utterances: all human scripted utterances, plus generated clone audio.
    for p in sorted((ROOT/"data"/"raw"/"participants").glob("*/utterances/*/*.wav")):
        yield "human", p
    for p in sorted((ROOT/"data"/"clones"/"raw_24k").glob("*/*/*/*.wav")):
        yield "clone", p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--overwrite",action="store_true"); args=ap.parse_args()
    outroot=ROOT/"data"/"codec"/"day4"; outroot.mkdir(parents=True,exist_ok=True)
    inputs=list(iter_inputs())
    if not inputs: raise SystemExit("No Day-4 human utterances or clone files found.")
    rows=[]
    for source_type,src in inputs:
        rel=src.relative_to(ROOT)
        for codec,spec in CODEC_SPECS.items():
            target_dir=outroot/source_type/src.parent.name/src.stem
            target_dir.mkdir(parents=True,exist_ok=True)
            # The extension selects FFmpeg's output container.  Use the codec
            # key here: `spec["encoder"]` is `libgsm`, not `gsm`.
            ext={"gsm":".gsm"}.get(codec, ".opus" if codec.startswith("opus_") else ".amr")
            encoded=target_dir/f"{src.stem}__{codec}{ext}"
            decoded=target_dir/f"{src.stem}__{codec}__decoded_16k.wav"
            if not args.overwrite and decoded.exists():
                pass
            else:
                # src.codecs handles input resampling/encoding. Its encoder selection is fixed in protocol.
                temp=target_dir/(src.stem+"__source_for_codec.wav")
                spec_rate=spec["sample_rate"]
                subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(src),"-ar",str(spec_rate),"-ac","1","-c:a","pcm_s16le",str(temp)],check=True)
                ffmpeg_encode(temp,encoded,codec)
                ffmpeg_decode_to_16k(encoded,decoded)
                temp.unlink(missing_ok=True)
            info=sf.info(decoded)
            rows.append({"source_type":source_type,"source_path":str(rel).replace('\\','/'),"codec":codec,"encoded_path":str(encoded.relative_to(ROOT)).replace('\\','/'),"decoded_path":str(decoded.relative_to(ROOT)).replace('\\','/'),"decoded_sample_rate_hz":info.samplerate,"decoded_channels":info.channels,"status":"PASS"})
    manifest=ROOT/"data"/"manifests"/"day4_codec_manifest.csv"; manifest.parent.mkdir(parents=True,exist_ok=True)
    import csv
    with manifest.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Generated {len(rows)} codec records from {len(inputs)} inputs")
    print(f"Saved: {manifest}")
    print("STATUS: PASS")
if __name__=="__main__": main()
