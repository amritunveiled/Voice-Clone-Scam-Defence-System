$ErrorActionPreference = "Stop"
Write-Host "=== Voice Clone Scam Defense: clean Day-1 environment setup ==="

if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' not found. Install Python 3.11 x64 first." }
py -3.11 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required." }

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu" "torchaudio==2.11.0+cpu"
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt

Write-Host "Environment installation complete."
Write-Host "Next: .\.venv\Scripts\python.exe .\scripts\check_env.py"
