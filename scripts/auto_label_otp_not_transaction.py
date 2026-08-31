#!/usr/bin/env python3
"""Auto-label unambiguous OTP / verification security messages as not_transaction in the local workbench."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lfm25.annotation_service import AnnotationService  # noqa: E402
from lfm25.annotation_sources import load_blinded_workspace  # noqa: E402
from lfm25.annotation_store import WorkbenchStore  # noqa: E402
from lfm25.annotation_workbench import DEFAULT_BLINDED_DB, SOURCE_PREFILL_OFF  # noqa: E402

OTP_REGEX = re.compile(
    r"\b("
    r"otp|one[- ]time[- ]password|verification[- ]code|verify[- ]code|"
    r"auth[- ]code|authentication[- ]code|security[- ]code|login[- ]code|"
    r"login[- ]otp|passcode|secret[- ]code"
    r")\b",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-label unambiguous OTP messages as not_transaction in the blinded test workbench."
    )
    parser.add_argument(
        "--reviewer",
        default="toji",
        help="Reviewer ID to record for the automated OTP annotations (default: toji)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many OTP rows would be labeled without modifying the database.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_BLINDED_DB,
        help="Path to the SQLite database (default: PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3)",
    )
    arguments = parser.parse_args()

    repo_root = SCRIPT_ROOT.resolve()
    db_path = arguments.db if arguments.db.is_absolute() else repo_root / arguments.db

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}", file=sys.stderr)
        return 1

    definition = load_blinded_workspace(
        repo_root,
        include_initial_annotations=False,
        source_prefill=SOURCE_PREFILL_OFF,
    )
    store = WorkbenchStore(db_path, workspace_binding=definition.binding)
    service = AnnotationService(
        repo_root=repo_root,
        definition=definition,
        store=store,
        reviewer=arguments.reviewer,
        batch_size=50,
    )

    all_rows = service._all_rows(include_private=True)
    pending_rows = [r for r in all_rows if r["status"] == "pending"]

    otp_rows = []
    for r in pending_rows:
        sms = str(r["reviewer_fields"]["sms"])
        if OTP_REGEX.search(sms):
            otp_rows.append(r)

    print(f"Total pending rows in database: {len(pending_rows)}")
    print(f"Pending rows matching OTP patterns: {len(otp_rows)}")

    if arguments.dry_run:
        print("Dry run mode: No changes written to database.")
        return 0

    if not otp_rows:
        print("No pending OTP rows to label.")
        return 0

    labeled_count = 0
    for r in otp_rows:
        row_id = r["row_id"]
        expected_revision = r["revision"]
        annotation_payload = {
            "decision": "not_transaction",
            "amount_decimal": None,
            "amount_span": None,
            "type": None,
            "account_span": None,
            "counterparty_span": None,
            "counterparty_absent": False,
            "notes": "auto-labeled: unambiguous OTP/verification message",
            "uncertain": False,
        }
        service.save(
            row_id=row_id,
            expected_revision=expected_revision,
            annotation=annotation_payload,
            submit=True,
        )
        labeled_count += 1

    progress = service.progress()
    print(f"Successfully auto-labeled {labeled_count} OTP messages as not_transaction.")
    print("Updated Progress:")
    print(f"  Completed: {progress['completed_rows']} / {progress['total_rows']}")
    print(f"  Pending:   {progress['pending_rows']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
