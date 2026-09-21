from __future__ import annotations
from pathlib import Path
import subprocess

CODEC_SPECS = {
    "amr_nb_4_75": {"encoder": "libopencore_amrnb", "sample_rate": 8000, "bitrate": "4.75k"},
    "amr_nb_7_4": {"encoder": "libopencore_amrnb", "sample_rate": 8000, "bitrate": "7.4k"},
    "amr_nb_12_2": {"encoder": "libopencore_amrnb", "sample_rate": 8000, "bitrate": "12.2k"},
    "amr_wb_12_65": {"encoder": "libvo_amrwbenc", "sample_rate": 16000, "bitrate": "12.65k"},
    "gsm": {"encoder": "libgsm", "sample_rate": 8000, "bitrate": None},
    "opus_8": {"encoder": "libopus", "sample_rate": 16000, "bitrate": "8k"},
    "opus_10": {"encoder": "libopus", "sample_rate": 16000, "bitrate": "10k"},
    "opus_12": {"encoder": "libopus", "sample_rate": 16000, "bitrate": "12k"},
}

def ffmpeg_encode(input_wav: str | Path, output_file: str | Path, codec_name: str):
    spec = CODEC_SPECS[codec_name]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_wav), "-ar", str(spec["sample_rate"]), "-ac", "1"]
    if spec["bitrate"]:
        cmd += ["-b:a", spec["bitrate"]]
    cmd += ["-c:a", spec["encoder"], str(output_file)]
    subprocess.run(cmd, check=True)

def ffmpeg_decode_to_16k(input_file: str | Path, output_wav: str | Path):
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(input_file),"-ar","16000","-ac","1","-c:a","pcm_s16le",str(output_wav)]
    subprocess.run(cmd,check=True)
