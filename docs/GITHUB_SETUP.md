# GitHub Setup

The repository is designed to live as one GitHub project for the full 15-day build. The remote can be created after GitHub authentication is available.

## Recommended: GitHub CLI
GitHub CLI supports creating a remote from an existing local repository with `--source`, setting visibility with `--private`/`--public`, and pushing with `--push`.

```powershell
git --version
gh --version
git init -b main
git add .
git commit -m "chore: initialize 15-day voice clone scam defense project"
gh auth login
gh repo create voice_clone_scam_defense --private --source=. --remote=origin --push
```

The command above is documented by GitHub CLI and GitHub Docs. Do not initialize a new remote with another README/license/gitignore when importing this existing repository.

## If you create the empty repository in the browser
Create an **empty** repository named `voice_clone_scam_defense`, then run:

```powershell
git init -b main
git add .
git commit -m "chore: initialize 15-day voice clone scam defense project"
git remote add origin https://github.com/<YOUR_USERNAME>/voice_clone_scam_defense.git
git push -u origin main
```

Never commit passwords, API keys, access tokens, private voice data, or other secrets.

## Daily save
At the end of each completed day:

```powershell
git add .
git commit -m "Day N: <actual work completed>"
git push
```

The project should have one continuous Git history from Day 1 through Day 15.
