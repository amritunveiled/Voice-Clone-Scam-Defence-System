$ErrorActionPreference = "Stop"
Write-Host "=== IndicF5 isolated environment setup ==="
Write-Host "The official IndicF5 README currently recommends Python 3.10 and installation from the AI4Bharat GitHub repo."
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' not found." }
py -3.10 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 is required for the isolated IndicF5 environment. Install Python 3.10 x64, then rerun." }
if (-not (Test-Path ".venv_indicf5")) { py -3.10 -m venv .venv_indicf5 }
& ".\.venv_indicf5\Scripts\python.exe" -m pip install --upgrade pip
# IndicF5's unpinned dependencies can otherwise resolve to incompatible future
# torch/torchaudio releases.  Keep this known-compatible CPU pair together.
& ".\.venv_indicf5\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" "torchaudio==2.5.1+cpu"
& ".\.venv_indicf5\Scripts\python.exe" -m pip install git+https://github.com/AI4Bharat/IndicF5.git
# IndicF5 declares transformers<4.50.  Do not override it with the latest
# major release after installing the model package.
& ".\.venv_indicf5\Scripts\python.exe" -m pip install "transformers==4.49.0" huggingface-hub soundfile numpy
& ".\.venv_indicf5\Scripts\python.exe" -c "from transformers import AutoModel; print('IndicF5 loader import: PASS')"
Write-Host "STATUS: PASS"
