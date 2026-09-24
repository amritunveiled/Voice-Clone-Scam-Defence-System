# Day 3 Acceptance

Day 3 passes when:

1. 12–15 speaker folders exist.
2. Every speaker has a consent marker.
3. Every speaker has exactly 5 enrollment WAVs.
4. Every speaker has 20 scripted utterances: 7 benign, 7 urgent-legit, 6 scam-test.
5. All audio is mono 16 kHz PCM-16 WAV.
6. `script_bank.csv` contains train/dev/test tags.
7. `validate_day3_data.py` reports `STATUS: PASS`.

This is a recording/data-readiness gate. Do not generate voice clones on Day 3.
