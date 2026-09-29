# Day 7 result fields

## Layer 2 calibration
- `model`: AASIST-L or XLSR-SLS
- `dataset`: own or ASVspoof5
- `codec`: clean, codec name, C00/C09
- `calibrated_bonafide_probability`: Platt sigmoid output for class 1 (bona fide)
- `platt_a`, `platt_b`: sigmoid parameters for that group
- `brier_score_in_sample`, `log_loss_in_sample`: calibration diagnostics on the same development observations used to fit the calibrator

## Whisper CER
- `language`: expected recording language from the manifest
- `codec`: channel condition
- `source_type`: human or clone
- `reference_text`: scripted target text
- `hypothesis`: Whisper transcription
- `cer`: character error rate after Unicode NFKC and whitespace normalization
