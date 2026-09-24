from __future__ import annotations

import argparse
import csv
from pathlib import Path

CATEGORIES = ("benign", "urgent_legit", "scam")
SPLITS = ("train", "dev", "test")
LANGUAGES = ("hindi", "english", "kannada")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the Day-3 speaker/script dataset scaffold.")
    parser.add_argument("--speakers", type=int, default=12)
    args = parser.parse_args()

    if args.speakers < 12 or args.speakers > 15:
        raise SystemExit("--speakers must be between 12 and 15")

    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw" / "participants"
    manifests = root / "data" / "manifests"
    raw.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    # Equal-ish language allocation. Hindi/English are core; Kannada is exploratory.
    language_plan = []
    for i in range(1, args.speakers + 1):
        if i <= max(4, args.speakers // 3):
            language_plan.append("kannada")
        elif i % 2 == 0:
            language_plan.append("hindi")
        else:
            language_plan.append("english")

    speaker_rows = []
    for i in range(1, args.speakers + 1):
        sid = f"spk{i:02d}"
        lang = language_plan[i - 1]
        speaker_dir = raw / sid
        (speaker_dir / "enrollment").mkdir(parents=True, exist_ok=True)
        (speaker_dir / "utterances" / "benign").mkdir(parents=True, exist_ok=True)
        (speaker_dir / "utterances" / "urgent_legit").mkdir(parents=True, exist_ok=True)
        (speaker_dir / "utterances" / "scam").mkdir(parents=True, exist_ok=True)
        (speaker_dir / "qc").mkdir(parents=True, exist_ok=True)
        (speaker_dir / "consent").mkdir(parents=True, exist_ok=True)
        speaker_rows.append({
            "speaker_id": sid,
            "language": lang,
            "consent_status": "pending",
            "recording_status": "pending",
        })

    with (manifests / "speakers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=speaker_rows[0].keys())
        writer.writeheader()
        writer.writerows(speaker_rows)

    # Script IDs are intentionally generic and non-operational. They are used to test
    # detection categories, not to optimize persuasion.
    script_bank = [
        ("B001", "benign", "Please remind me to bring the documents for tomorrow's meeting."),
        ("B002", "benign", "I will call you back after I finish my work."),
        ("B003", "benign", "The train is arriving later than expected today."),
        ("B004", "benign", "Please send me the grocery list when you have time."),
        ("B005", "benign", "I am reaching home in about twenty minutes."),
        ("B006", "benign", "The meeting has been moved to the afternoon."),
        ("B007", "benign", "Can you remind me about the appointment tomorrow?"),
        ("U001", "urgent_legit", "I have an urgent family matter, but do not send any money."),
        ("U002", "urgent_legit", "Please call me back quickly because I need your help at home."),
        ("U003", "urgent_legit", "There is a genuine emergency, so please verify this with the family first."),
        ("U004", "urgent_legit", "I need an immediate reply, but please use our normal verification step."),
        ("U005", "urgent_legit", "Please contact the hospital directly before taking any action."),
        ("U006", "urgent_legit", "This is time sensitive, but do not share any password or OTP."),
        ("U007", "urgent_legit", "Please come as soon as you can and call the known family number."),
        ("S001", "scam", "This is an example detection-test message asking for an immediate transfer."),
        ("S002", "scam", "This is a generic scam-test message asking for an OTP during a call."),
        ("S003", "scam", "This detection-test example claims to be a relative and requests money."),
        ("S004", "scam", "This generic example creates urgency and asks the listener to transfer funds."),
        ("S005", "scam", "This test sentence asks for private account information on an unexpected call."),
        ("S006", "scam", "This generic scam-test example says the payment must be completed immediately."),
    ]

    # Train/dev/test are script-level tags. The actual assignment is fixed by script ID.
    rows = []
    for idx, (script_id, category, text) in enumerate(script_bank):
        if script_id.startswith("B"):
            split = "train" if idx < 4 else ("dev" if idx < 6 else "test")
        elif script_id.startswith("U"):
            split = "train" if idx < 12 else ("dev" if idx < 14 else "test")
        else:
            split = "train" if idx < 18 else ("dev" if idx < 19 else "test")
        for lang in LANGUAGES:
            rows.append({
                "script_id": script_id,
                "split": split,
                "category": category,
                "language": lang,
                "text": text,
            })

    with (manifests / "script_bank.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {args.speakers} speaker folders under {raw}")
    print(f"Created speaker manifest: {manifests / 'speakers.csv'}")
    print(f"Created script bank: {manifests / 'script_bank.csv'}")
    print("IMPORTANT: category text is generic research-test content, not guidance for real-world deception.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
