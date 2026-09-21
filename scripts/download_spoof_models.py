from __future__ import annotations
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = [
    "SpeechAntiSpoofingBenchmarks/AASIST-L",
    "SpeechAntiSpoofingBenchmarks/XLSR-SLS",
]
root = Path("models")
root.mkdir(exist_ok=True)
for repo in MODELS:
    target = root / repo.split("/")[-1]
    print(f"Downloading {repo} -> {target}")
    snapshot_download(repo_id=repo, local_dir=str(target))
    onnx = sorted(target.rglob("*.onnx"))
    print("ONNX files:")
    for f in onnx:
        print(" ", f)
