# Day 7 Runbook

1. Confirm Day-6 score file exists: `results/tables/day6_layer2_scores.csv`.
2. Run Layer-2 calibration. Inspect the number of groups calibrated and any skipped groups.
3. Run Whisper CER on the full development set. Use `--max-files 10` first if you want to verify the ASR path before starting the full CPU job.
4. Inspect `day7_whisper_cer_summary.csv` by language and codec.
5. Run `verify_day7.py`.
6. Only after all checks pass, commit the code/docs and generated result tables intentionally selected for the research record.

Do not evaluate `test` and do not tune final decision thresholds today.
