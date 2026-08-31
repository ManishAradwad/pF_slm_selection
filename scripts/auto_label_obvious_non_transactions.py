#!/usr/bin/env python3
"""Conservatively auto-label 100% obvious non-financial / non-transaction messages in the blinded test workbench."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lfm25.annotation_service import AnnotationService
from lfm25.annotation_sources import load_blinded_workspace
from lfm25.annotation_store import WorkbenchStore
from lfm25.annotation_workbench import DEFAULT_BLINDED_DB, SOURCE_PREFILL_OFF

# Broad pattern for ANY financial, monetary, banking, or transactional indicator
ANY_FINANCIAL_OR_TRANSACTION_PATTERN = re.compile(
    r"(?:"
    r"₹|\brs\.?|\binr\b|\busd\b|\beur\b|\bgbp\b|\b\d+\s*/-\b|"
    r"\bdebited\b|\bcredited\b|\bspent\b|\bsent\b|\btransferred\b|\bwithdrawn\b|"
    r"\brefund\b|\brefunded\b|\bpaid\b|\breceived\b|\bpayment\b|\bpay\b|"
    r"\bbill\b|\bdue\b|\bbalance\b|\bbal\b|\bavl\s*bal\b|\bavailable\s*limit\b|"
    r"\ba/c\b|\bacct\b|\baccount\b|\bcredit\s*card\b|\bdebit\s*card\b|\bcard\b|"
    r"\bupi\b|\bimps\b|\bneft\b|\brtgs\b|\bwallet\b|\bbnpl\b|"
    r"\bcashback\b|\breward\s*points\b|\bvpa\b|\btxn\b|\btransaction\b"
    r")",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively auto-label obvious non-financial messages as not_transaction in the blinded test workbench."
    )
    parser.add_argument(
        "--reviewer",
        default="toji",
        help="Reviewer ID to record for the automated annotations (default: toji)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many obvious non-transaction rows would be labeled without modifying the database.",
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

    obvious_non_txn_rows = []
    skipped_rows = []

    for r in pending_rows:
        sms = str(r["reviewer_fields"]["sms"])
        if not ANY_FINANCIAL_OR_TRANSACTION_PATTERN.search(sms):
            obvious_non_txn_rows.append(r)
        else:
            skipped_rows.append(r)

    print(f"Total pending rows in database: {len(pending_rows)}")
    print(f"Obvious non-financial rows identified for auto-labeling: {len(obvious_non_txn_rows)}")
    print(f"Rows reserved for manual human review (contains financial/banking terms): {len(skipped_rows)}")

    if arguments.dry_run:
        print("Dry run mode: No changes written to database.")
        return 0

    if not obvious_non_txn_rows:
        print("No obvious non-transaction rows to label.")
        return 0

    labeled_count = 0
    for r in obvious_non_txn_rows:
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
            "notes": "auto-labeled: unambiguous non-financial / non-transaction message",
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
    print(f"Successfully auto-labeled {labeled_count} messages as not_transaction.")
    print("Updated Workbench Progress:")
    print(f"  Total:     {progress['total_rows']}")
    print(f"  Completed: {progress['completed_rows']}")
    print(f"  Pending:   {progress['pending_rows']}")
    print(f"  Uncertain: {progress['uncertain_rows']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
