# 15-Day Master Plan

## Day 1 — Scope freeze, repo, environment
Acceptance: environment runs; RTF table exists; FFmpeg encoders confirmed or fallback chosen; `PROTOCOL.md` frozen.

## Day 2 — Smoke-test models + codec pipeline + capture prototype
Acceptance: five files processed; AMR-NB spectrogram has no material energy above ~3.4 kHz; same-speaker ECAPA similarity exceeds different-speaker similarity; detector outputs scores; Whisper transcribes.

## Day 3 — Recording sessions + script bank
Acceptance: per-speaker folders; consent on file; scripts tagged train/dev/test; Hindi/English core and small Kannada exploratory set.

## Day 4 — Clone generation + manifest + codec application
Acceptance: manifest has no missing files or speaker leakage; QC log complete; speaker-disjoint splits frozen.

## Day 5 — Layer 1 evaluation
Enrollment, scoring, clean-vs-codec distributions, codec-augmented enrollment, EER and calibration.

## Day 6 — Layer 2 evaluation
AASIST-L baseline versus XLS-R detector on clean, codecs, public codec subset, and own consented synthetic clones.

## Day 7 — Layer 2 calibration + ASR CER
Calibration per condition; Whisper CER by language/condition; optional fine-tuning only if schedule gate allows.

## Day 8 — Layer 3 intent
Fuzzy lexicon + character n-gram TF-IDF logistic regression; held-out scripts; ASR-noise ablation.

## Day 9 — Fusion + ablations
All seven non-empty layer subsets, calibration, leave-one-speaker-out fusion; freeze models and thresholds.

## Day 10 — Real-channel test set
Speakerphone-to-mic and WhatsApp Desktop loopback conditions.

## Day 11 — Streaming engine
VAD, rolling 4 s windows with 1–2 s hop, evidence accumulation, hysteresis, rolling transcription. No model changes.

## Day 12 — Live UI
Family-member selection, Start Listening, meters, verdict states, recommended actions. Route B first, Route A second.

## Day 13 — End-to-end live tests
Injected clone audio; time-to-verdict, latency, accuracy versus audio duration, false alarms.

## Day 14 — Final test run
Run test split once; produce final tables/figures and error analysis.

## Day 15 — Demo + reproducibility
README, one-command reproduction, rehearsal, final cleanup.

## Cut order if behind
1. Kannada
2. Tamil (not currently in selected recording languages)
3. Detector fine-tuning
4. Route A
5. Streaming polish

Never cut the factorial evaluation.
