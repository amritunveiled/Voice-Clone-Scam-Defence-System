# Day 7 — Layer 2 calibration + Whisper CER

## Scope
- Calibrate raw AASIST-L and XLSR-SLS scores separately by model/dataset/codec using development data only.
- Use Platt sigmoid calibration; do not use isotonic with these small own-data cells.
- Measure Whisper character error rate (CER) by language, codec, and human/clone source type on development data only.
- Use the known manifest language (`en`, `hi`, `kn`) for ASR instead of language auto-detection.
- Do not touch the final test split.

## Commands

### Calibration
```powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_layer2_day7.py
```

### CER
```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_whisper_cer_day7.py
```

For a quick smoke run before the full CPU run:
```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_whisper_cer_day7.py --max-files 10
```

### Verify
```powershell
.\.venv\Scripts\python.exe .\scripts\verify_day7.py
```

## Outputs
- `results/tables/day7_layer2_calibrated_scores.csv`
- `results/tables/day7_layer2_calibration_summary.csv`
- `results/tables/day7_whisper_cer_scores.csv`
- `results/tables/day7_whisper_cer_summary.csv`

The calibration diagnostics are explicitly labelled in-sample. They are not final generalization estimates.
