# Frozen Study Protocol

## Research claim boundary
This is an application and evaluation study, not a claim of a new fusion algorithm.

## Factorial design
Cross speaker type with script type:
- Genuine enrolled speaker × benign script
- Genuine enrolled speaker × urgent-but-legitimate script
- Human impostor × benign script
- Human impostor × scam script
- Voice clone of target × benign script
- Voice clone of target × scam script

## Audio conditions
Apply every condition to every class so codec type cannot become a shortcut label:
- clean
- AMR-NB: 4.75, 7.4, 12.2 kbps
- AMR-WB: 12.65 kbps
- GSM
- Opus: 8–12 kbps

The project simulates telephony and compressed VoIP conditions; real mobile-network behaviour varies by route and device.

## Splits
- speaker-disjoint
- fixed random seed
- scripts disjoint between development and test
- test split is used once, on Day 14

## Systems
- A: detector only
- A2: detector + speaker match (SASV-style baseline)
- B: all three signals
- all seven non-empty subsets of {Voice, Deepfake, Intent}

## Metrics
EER, AUC, TPR at 5% FPR, FNR, precision, recall, F1, confusion matrices, per-cell flag-rate heatmap, clean-vs-codec drop per layer, and bootstrap 95% confidence intervals resampled by speaker.

Calibrate raw scores with Platt scaling on development data before displaying percentage-like scores.

## Live metrics
Time to first verdict, per-update latency, accuracy versus seconds of audio heard, false-alarm rate on genuine and urgent-but-legitimate calls, and comparison of Route A, Route B, and offline processing.

## Freeze rule
After Day 1, do not alter the scientific protocol without an explicit documented deviation. Engineering changes that do not alter the experimental question are allowed and should be logged in `PROJECT_STATE.md`.
