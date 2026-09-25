from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOBS_FILE = ROOT / "data" / "manifests" / "day4_clone_jobs.csv"


def main() -> int:
    if not JOBS_FILE.exists():
        raise SystemExit(f"Missing: {JOBS_FILE}")

    with JOBS_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("day4_clone_jobs.csv is empty.")

    generated = 0
    still_pending = []

    for row in rows:
        output_path = row.get("output_path", "").strip()

        if not output_path:
            still_pending.append(row["job_id"])
            continue

        output_file = ROOT / output_path

        if output_file.exists() and output_file.stat().st_size > 44:
            # The actual audio file exists, so the job is generated.
            row["status"] = "GENERATED"
            generated += 1
        else:
            still_pending.append(row["job_id"])

    with JOBS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("DAY 4 — REPAIR CLONE JOB STATUS")
    print("=" * 70)

    print(f"Total jobs       : {len(rows)}")
    print(f"Marked GENERATED : {generated}")
    print(f"Still pending    : {len(still_pending)}")

    if still_pending:
        print("\nJobs still missing audio:")
        for job_id in still_pending:
            print(f"  - {job_id}")

    else:
        print("\nEvery scheduled job has an existing output file.")

    print(f"\nUpdated: {JOBS_FILE}")

    if still_pending:
        print("STATUS: INCOMPLETE")
        return 1

    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())