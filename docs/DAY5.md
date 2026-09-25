# Day 5 — Layer 1 Speaker Verification

## Goal
Evaluate ECAPA speaker matching using the frozen Day-4 speaker splits without touching the final Day-14 test split.

## What is measured
- 5-clip clean enrollment per speaker
- genuine human trials vs other-human impostor trials
- clean vs every codec condition
- codec-matched enrollment: 5 clean enrollment embeddings + 5 same-codec enrollment embeddings
- target-clone similarity reported separately, not folded into human-impostor EER
- EER and threshold
- AUC (supporting metric)
- speaker-bootstrap 95% CI for EER
- Platt calibration artifact on development trials
- clean-vs-codec score drop for same-speaker human trials

## Why clones are separate
A target voice clone can preserve the target speaker embedding. Treating it as a human impostor would mix two different questions. We therefore report clone similarity directly today; later Layer 2 and fusion handle synthetic/anti-spoof evidence.

## Protected data rule
Day 5 uses `evaluation_split=dev` only. The final test split is reserved for Day 14 exactly once. The evaluator contains a hard guard against `--split test`.

## Outputs
- `results/tables/day5_layer1_scores.csv`
- `results/tables/day5_layer1_summary.csv`
- `results/tables/day5_layer1_bootstrap.csv`
- `results/tables/day5_layer1_calibration.csv`
- `results/tables/day5_layer1_results.json`
- `results/figures/day5_layer1_score_distributions.png`
