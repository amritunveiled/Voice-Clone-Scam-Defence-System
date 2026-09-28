# Day 6 — Layer 2 anti-spoofing evaluation

## Goal
Compare the lightweight AASIST-L baseline with the main XLSR-SLS detector on the development split, under clean and all project codec conditions, plus a small public ASVspoof 5 codec subset and the project's own generated clones.

## Safety / research rules
- Do not evaluate `test`; Day 14 owns the final test split.
- Do not tune thresholds today.
- Do not calibrate today; Day 7 handles calibration.
- Own data labels: human = bona fide, clone = spoof.
- Public ASVspoof 5 labels are taken from its official dev Track-1 protocol.
- Report raw detector score direction as higher = more bona fide.

## Models
Current SpeechAntiSpoofingBenchmarks wrappers define class 1 as bona fide for both AASIST-L and XLSR-SLS. Both use a deterministic first 64,600-sample evaluation window; shorter audio is tile-repeated.

## Public subset
Use ASVspoof5 **development Track 1**, not the final evaluation set. For the required codec stress check, select C00 (none) and C09 (AMR at 8 kHz). C09 covers 4.75–12.20 kbps in the official ASVspoof 5 codec table.

Do not download the full 142+ GB database just for this day. Use only enough local dev audio to construct a small balanced C00/C09 subset.
