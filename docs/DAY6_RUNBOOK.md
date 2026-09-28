# Day 6 Runbook

1. Confirm Day-4 manifests exist.
2. Confirm AASIST-L and XLSR-SLS ONNX files exist.
3. Prepare public ASVspoof5 dev C00/C09 subset if not already present.
4. Run `evaluate_layer2_day6.py` on dev only.
5. Inspect `day6_layer2_summary.csv` and `day6_codec_drops.csv`.
6. Run `verify_day6.py`.
7. Commit code/results that are safe to commit. Keep raw audio and model weights out of Git.

The evaluation reports raw score metrics today. Calibration and final threshold work belong to Day 7/9.
