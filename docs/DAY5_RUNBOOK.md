# Day 5 Runbook

Run from the repository root.

## 1. Check the frozen split
```powershell
Get-Content .\data\manifests\speaker_splits_day4.json
```
Confirm it exists and is already frozen.

## 2. Run Layer 1 evaluation
```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_layer1_day5.py
```

## 3. Plot clean-vs-codec score distributions
```powershell
.\.venv\Scripts\python.exe .\scripts\plot_layer1_day5.py
```

## 4. Verify Day 5
```powershell
.\.venv\Scripts\python.exe .\scripts\verify_day5.py
```

## Important
Never run the evaluator with a test split. The script accepts only `dev` by design.

## Outputs
- `results/tables/day5_layer1_scores.csv`
- `results/tables/day5_layer1_summary.csv`
- `results/tables/day5_layer1_bootstrap.csv`
- `results/tables/day5_layer1_calibration.csv`
- `results/tables/day5_layer1_results.json`
- `results/figures/day5_layer1_score_distributions.png`
