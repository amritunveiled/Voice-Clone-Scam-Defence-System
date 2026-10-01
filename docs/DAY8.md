# Day 8 — Layer 3 Scam Intent

## Goal
Build the frozen Layer-3 design: fuzzy lexicon + character n-gram TF-IDF logistic regression.

## Training/evaluation rule
- Train only on script IDs tagged `train` in `data/manifests/script_bank.csv`.
- Evaluate only on script IDs tagged `dev`.
- Never use `test` scripts on Day 8.
- Compare clean reference text against Whisper ASR hypotheses from Day 7 to measure ASR-noise impact.

## Languages
The model is language-aware for English, Hindi, and Kannada. Language is taken from the manifest; it is never inferred from speaker identity.

## Outputs
- `results/tables/day8_layer3_scores.csv`
- `results/tables/day8_layer3_summary.csv`
- `results/tables/day8_layer3_report.json`

## Interpretation
Report accuracy and macro-F1 by language/source/codec. Treat results as development evidence only. Do not tune a production threshold today; fusion/calibration happens on Day 9.
