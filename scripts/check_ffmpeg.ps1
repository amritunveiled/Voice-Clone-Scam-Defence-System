$ErrorActionPreference = "Stop"
Write-Host "=== FFMPEG ENCODER CHECK ==="
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "STATUS: FAIL - ffmpeg not found on PATH"
    exit 1
}
ffmpeg -version | Select-Object -First 1
$matches = ffmpeg -hide_banner -encoders 2>$null | Select-String -Pattern 'amr|gsm|opus'
$matches | ForEach-Object { $_.Line }
$projectRoot = Split-Path -Parent $PSScriptRoot
$out = Join-Path (Join-Path $projectRoot "results\tables") "ffmpeg_encoders.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
ffmpeg -hide_banner -encoders 2>$null | Out-File -Encoding utf8 $out
if ($matches) { Write-Host "STATUS: PASS - inspect $out" } else { Write-Host "STATUS: FAIL - no requested codec encoders found"; exit 1 }
