# Day 3 Recording Guide

## Goal
Create the first real research dataset structure: 12–15 consented speakers, five enrollment clips per speaker, and about 20 scripted utterances per speaker.

## Required audio
- mono
- 16 kHz
- PCM-16 WAV
- quiet room
- enrollment: 5 clips, roughly 5–10 seconds each
- scripted utterances: 20 per speaker (7 benign, 7 urgent-but-legitimate, 6 scam-test)

## Languages for this team
- Core: Hindi, English
- Exploratory: Kannada
- Tamil is not required because the current team has not selected it for recording.

## Session order
1. Confirm consent.
2. Create the participant's `consent/consent_received.txt` marker.
3. Record 5 enrollment clips.
4. Record 7 benign utterances.
5. Record 7 urgent-but-legitimate utterances.
6. Record 6 generic scam-test utterances.
7. Do a quick listen/QC check.
8. Do not edit the audio except for removing a failed take and re-recording it.

## Data separation
Never commit raw voice files to GitHub. Keep them under the `.gitignore` rules.

## Consent marker
The `consent_received.txt` file should be created only after the written consent is actually obtained. A reasonable content format is:

participant_code=spk01
written_consent_received=yes
voice_clone_consent=separate_yes_or_no
recording_date=YYYY-MM-DD
researcher=team_member

Keep signed consent forms separate from the audio repository; the marker is only an internal status flag.
