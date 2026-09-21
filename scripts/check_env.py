from __future__ import annotations
import importlib
import platform
import sys

EXPECTED = {
    "torch": "2.11.0",
    "torchaudio": "2.11.0",
    "speechbrain": "1.1.1",
    "faster_whisper": "1.2.1",
    "huggingface_hub": "1.32.0",
    "onnxruntime": "1.30.0",
}

print("=== DAY 1 ENVIRONMENT CHECK ===")
print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("Machine:", platform.machine())
fail = False
for module, expected in EXPECTED.items():
    try:
        m = importlib.import_module(module)
        got = getattr(m, "__version__", "unknown")
        ok = got == expected or (module in {"torch", "torchaudio"} and got.startswith(expected + "+"))
        print(("OK   " if ok else "FAIL ") + f"{module:18s} {got} (expected {expected})")
        fail |= not ok
    except Exception as exc:
        print(f"FAIL {module:18s} import error: {exc}")
        fail = True
try:
    import torch
    print("Torch CUDA available:", torch.cuda.is_available())
    print("Torch thread count:", torch.get_num_threads())
except Exception:
    pass
try:
    import onnxruntime as ort
    print("ONNX Runtime providers:", ort.get_available_providers())
except Exception:
    pass
print("STATUS:", "FAIL" if fail else "PASS")
sys.exit(1 if fail else 0)
