"""Synthetic-only workbench workflow, safety, recovery, and export tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.corpus.grouping import build_grouping
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.types import CandidateKind
from pocketfinancer_sms.workbench.service import WorkbenchService, WorkbenchValidationError
from pocketfinancer_sms.workbench.store import WorkbenchConflict, WorkbenchStore


def _manifest_record(*, pool: str = "protected_test") -> dict:
    source_id = "src_" + "a" * 32
    source = "INR 42.50 was credited to account **7788 from FRIEND."
    analysis = DeterministicSmsAnalyzer(
        CurrencyContext("INR", ("core-en", "india"))
    ).analyze(source, operation_id=source_id, is_outgoing=False)
    triage = evaluate_triage(analysis)
    grouping = build_grouping(b"synthetic-key" * 3, source, "SYNTH-BANK", "2024-01-01T00:00:00Z")
    return {
        "contract": "pocketfinancer.corpus-record/1",
        "source_id": source_id,
        "source": {"body": source, "sender": "SYNTH-BANK"},
        "source_metadata": {
            "source_record_id": "synthetic-1",
            "source_row_index": 0,
            "timestamp": "2024-01-01T00:00:00Z",
            "service": "SMS",
            "is_outgoing": False,
        },
        "analysis": analysis.to_dict(),
        "weak_facets": {
            "disposition": triage.disposition.value,
            "selector_action": triage.selector_action.value,
            "operational_class": "posted_candidate",
            "event_state": "posted",
            "financial_family": "bank_transfer",
            "payment_rail": "bank_internal",
            "confidence": "medium",
            "reason_codes": list(triage.reason_codes),
        },
        "grouping": asdict(grouping),
        "pool": pool,
        "review_state": "unreviewed",
        "provenance": {"corpus_run_id": "synthetic-run"},
    }


def _store(tmp_path: Path, *, pool: str = "protected_test") -> tuple[WorkbenchStore, dict]:
    record = _manifest_record(pool=pool)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store = WorkbenchStore(tmp_path / "private" / "workbench.sqlite3")
    assert store.import_manifest(manifest, corpus_run_id="synthetic-run") == 1
    return store, record


def _posted_payload(record: dict) -> dict:
    candidates = record["analysis"]["candidates"]

    def present(kind: str) -> dict:
        return next(
            item
            for item in candidates
            if item["kind"] == kind and not item["explicit_absence"]
        )

    amount = present(CandidateKind.AMOUNT.value)
    direction = present(CandidateKind.DIRECTION.value)
    account = present(CandidateKind.ACCOUNT.value)
    counterparty = present(CandidateKind.COUNTERPARTY.value)

    def span(item: dict) -> dict:
        return {
            "start_char": item["evidence"]["start_char"],
            "end_char": item["evidence"]["end_char"],
        }

    return {
        "decision": "posted",
        "operational_class": "posted_candidate",
        "event_state": "posted",
        "financial_family": "bank_transfer",
        "payment_rail": "bank_internal",
        "events": [
            {
                "amount_span": span(amount),
                "currency": "INR",
                "currency_provenance": "explicit_code",
                "direction": "credit",
                "direction_span": span(direction),
                "account_state": "present",
                "account_span": span(account),
                "counterparty_state": "present",
                "counterparty_span": span(counterparty),
                "financial_family": "bank_transfer",
                "payment_rail": "bank_internal",
            }
        ],
        "uncertain": False,
        "notes": "",
    }


def test_protected_review_is_blind_until_submission_and_explicit_reveal(tmp_path: Path) -> None:
    store, record = _store(tmp_path)
    service = WorkbenchService(store)
    source_id = record["source_id"]

    blind = service.view_row(source_id, "reviewer-one")
    assert blind["blind_locked"] is True
    assert "analysis" not in blind
    listed = service.list_rows(reviewer_id="reviewer-one", filters={})
    assert listed["rows"][0]["operational_class"] is None

    with pytest.raises(WorkbenchConflict, match="blind review must be submitted"):
        service.reveal(source_id, "reviewer-one")

    draft = service.save_draft(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload={"decision": "posted"},
    )
    assert draft["revision"] == 1
    submitted = service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=1,
        payload=_posted_payload(record),
    )
    assert submitted["revision"] == 2
    assert service.view_row(source_id, "reviewer-one")["blind_locked"] is True

    revealed = service.reveal(source_id, "reviewer-one")
    assert revealed["blind_locked"] is False
    assert revealed["analysis"]["analysis_id"] == record["analysis"]["analysis_id"]


def test_revision_conflicts_and_invalid_submissions_never_downgrade_to_negative(
    tmp_path: Path,
) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    service.save_draft(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload={},
    )
    with pytest.raises(WorkbenchConflict, match="revision changed"):
        service.save_draft(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=0,
            payload={},
        )
    with pytest.raises(WorkbenchValidationError, match="label_event_count_inconsistent"):
        service.submit(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=1,
            payload={
                "decision": "posted",
                "operational_class": "posted_candidate",
                "event_state": "posted",
                "financial_family": None,
                "payment_rail": None,
                "events": [],
                "uncertain": False,
                "notes": "",
            },
        )
    assert store.current_revision(source_id, "reviewer-one") == 1
    assert store.latest_annotation(source_id, "reviewer-one")["status"] == "draft"


def test_target_preview_reports_candidate_oracle_without_silent_fallback(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    service.submit(
        source_id=record["source_id"],
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )

    preview = service.target_preview(record["source_id"], "reviewer-one")
    assert preview["convertible"] is True
    assert preview["target"]["decision"] == "posted"
    assert set(preview["target"]) == {
        "decision",
        "amount",
        "direction",
        "account",
        "counterparty",
    }


def test_weak_corrections_are_separate_from_raw_source_and_human_truth(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    before = store.get_record(record["source_id"])["source"]
    correction = service.correct_weak_facets(
        source_id=record["source_id"],
        reviewer_id="reviewer-one",
        expected_revision=0,
        facets={
            "operational_class": "ambiguous",
            "event_state": "unknown",
            "financial_family": "unknown",
            "payment_rail": "unknown",
            "reason": "synthetic correction",
        },
    )

    assert correction["revision"] == 1
    assert store.get_record(record["source_id"])["source"] == before
    assert store.latest_annotation(record["source_id"], "reviewer-one") is None


def test_backup_recovery_integrity_and_reproducible_hash_bound_export(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    service.submit(
        source_id=record["source_id"],
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    assert store.integrity_check() is True

    backup = store.create_backup(tmp_path / "backups")
    backup_path = tmp_path / "backups" / backup["backup"]
    restored_path = tmp_path / "restored" / "workbench.sqlite3"
    WorkbenchStore.restore_backup(backup_path, restored_path, backup["sha256"])
    restored = WorkbenchStore(restored_path)
    assert restored.integrity_check() is True
    assert restored.get_record(record["source_id"])["source_id"] == record["source_id"]

    first = store.export_labels(tmp_path / "exports")
    second = store.export_labels(tmp_path / "exports")
    assert first == second
    assert first["label_count"] == 1
    export_dir = tmp_path / "exports" / first["export_id"]
    assert (export_dir / "canonical_labels.jsonl").is_file()
    assert (export_dir / "manifest.json").is_file()
