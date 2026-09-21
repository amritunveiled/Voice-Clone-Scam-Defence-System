# Voice Clone Scam Defense

A 15-day research prototype for a personalized, telephony-robust scam-call guardian targeting voice-cloning fraud in Indian languages.

## Project rule
This is **one evolving repository for all 15 days**. The same files are modified throughout the project. Do not create separate day folders or patch projects.

## Restart policy
This repository is a clean restart. Previous experimental numbers, datasets, and benchmark results are **not part of this run**. New results must be generated and logged from this repository.

## Core pipeline
1. Voice Match — ECAPA-TDNN speaker embeddings, enrollment from 3–5 consented clips, cosine similarity.
2. Deepfake Risk — XLS-R-based anti-spoofing detector as main detector; AASIST-L as lightweight baseline.
3. Scam Intent — Whisper transcription followed by fuzzy lexicon + character n-gram TF-IDF logistic regression.
4. Fusion — calibrated logistic regression plus a hand-set rule baseline.

## Safety
Use only consented voice recordings. Keep voice data local. Never use real phone numbers, UPI IDs, OTPs, or personal identifiers in scripts. Scam examples are short, generic detection-test content.

## Day 1 starting point
- Freeze the study protocol.
- Create/rebuild the Python environment.
- Benchmark ECAPA, XLS-R anti-spoofing, and Whisper on this machine.
- Verify FFmpeg codec encoders.
- Record consent/recruitment process.

See `MASTER_15_DAY_PLAN.md`, `PROTOCOL.md`, and `docs/GITHUB_SETUP.md`.
