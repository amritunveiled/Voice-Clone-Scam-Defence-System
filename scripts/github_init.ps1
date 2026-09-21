$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Git is not installed or not on PATH." }
if (-not (Test-Path ".git")) { git init }
git branch -M main
git add .
git commit -m "chore: initialize 15-day voice clone scam defense project"
if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh auth status
    if ($LASTEXITCODE -eq 0) {
        gh repo create voice_clone_scam_defense --private --source=. --remote=origin --push
        Write-Host "GitHub repository created and main pushed."
        exit 0
    }
}
Write-Host "Local Git repository initialized. GitHub CLI is not authenticated/available."
Write-Host "Follow docs/GITHUB_SETUP.md to create or connect the GitHub remote."
