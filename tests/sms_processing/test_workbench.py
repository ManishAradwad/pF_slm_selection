"""Synthetic-only workbench workflow, safety, recovery, and export tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pocketfinancer_sms.analyzer import DeterministicSmsAnalyzer
from pocketfinancer_sms.corpus.grouping import build_grouping
from pocketfinancer_sms.currency import CurrencyContext
from pocketfinancer_sms.extractor import SourceSpan
from pocketfinancer_sms.provenance import PrivateArtifactError
from pocketfinancer_sms.triage import evaluate_triage
from pocketfinancer_sms.types import CandidateKind
from pocketfinancer_sms.workbench.service import WorkbenchService, WorkbenchValidationError
from pocketfinancer_sms.workbench.store import WorkbenchConflict, WorkbenchStore


def _manifest_record(*, pool: str = "protected_test") -> dict:
    source_id = "src_" + "a" * 32
    source = "INR 42.50 was credited to account **7788 from FRIEND."
    analysis = DeterministicSmsAnalyzer(CurrencyContext("INR", ("core-en", "india"))).analyze(
        source, operation_id=source_id, is_outgoing=False
    )
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
            item for item in candidates if item["kind"] == kind and not item["explicit_absence"]
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
    with pytest.raises(WorkbenchValidationError, match="disagreement details remain hidden"):
        service.disagreements(source_id, "reviewer-one")

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
    with pytest.raises(WorkbenchValidationError, match="preview remains hidden"):
        service.target_preview(source_id, "reviewer-one")
    with pytest.raises(WorkbenchValidationError, match="cannot be corrected"):
        service.correct_weak_facets(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=0,
            facets={
                "operational_class": "ambiguous",
                "event_state": "unknown",
                "financial_family": "unknown",
                "payment_rail": "unknown",
                "reason": "synthetic blind correction",
            },
        )

    revealed = service.reveal(source_id, "reviewer-one")
    assert revealed["blind_locked"] is False
    assert revealed["analysis"]["analysis_id"] == record["analysis"]["analysis_id"]
    assert service.target_preview(source_id, "reviewer-one")["convertible"] is True


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
    assert store.progress(reviewer_id="reviewer-one")["validation_failures"] == 1
    assert store.progress(reviewer_id="reviewer-one")["my_remaining"] == 1
    assert store.latest_annotation(source_id, "reviewer-one")["status"] == "draft"
    row = service.view_row(source_id, "reviewer-one")
    assert [item["revision"] for item in row["annotation_history"]] == [1]
    assert row["annotation_history"][0]["status"] == "draft"


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


def test_global_review_progress_never_downgrades_when_another_reviewer_drafts(
    tmp_path: Path,
) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    service.save_draft(
        source_id=source_id,
        reviewer_id="reviewer-two",
        expected_revision=0,
        payload={"decision": "ambiguous"},
    )

    assert store.get_record(source_id)["review_state"] == "submitted"
    assert store.progress()["review_states"] == {"submitted": 1}


def test_group_time_and_category_navigation_filters_are_available_off_protected_pools(
    tmp_path: Path,
) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    grouping = record["grouping"]
    result = service.list_rows(
        reviewer_id="reviewer-one",
        filters={
            "normalized_template_group": grouping["normalized_template_hash"],
            "sender_family_group": grouping["sender_family_hash"],
            "sender_template_group": grouping["sender_template_group_hash"],
            "time_group": grouping["time_group"],
            "time_from": "2024-01-01",
            "time_to": "2024-01-31",
            "financial_family": "bank_transfer",
            "selector_action": "run_normal",
        },
    )

    assert result["total"] == 1
    assert result["rows"][0]["normalized_template_group"] == grouping["normalized_template_hash"]


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
    assert backup["corpus_run_id"] == "synthetic-run"
    backup_path = tmp_path / "backups" / backup["backup"]
    assert not list((tmp_path / "backups").glob(".*.tmp*"))
    assert not list((tmp_path / "backups").glob("*-wal"))
    assert not list((tmp_path / "backups").glob("*-shm"))
    restored_path = tmp_path / "restored" / "workbench.sqlite3"
    WorkbenchStore.restore_backup(
        backup_path,
        restored_path,
        backup["sha256"],
        backup["corpus_run_id"],
    )
    assert not Path(f"{restored_path}-wal").exists()
    assert not Path(f"{restored_path}-shm").exists()
    restored = WorkbenchStore(restored_path)
    assert restored.integrity_check() is True
    assert restored.get_record(record["source_id"])["source_id"] == record["source_id"]

    with pytest.raises(PrivateArtifactError, match="different corpus run"):
        WorkbenchStore.restore_backup(
            backup_path,
            tmp_path / "wrong-run" / "workbench.sqlite3",
            backup["sha256"],
            "different-synthetic-run",
        )

    revision = store.latest_annotation(record["source_id"], "reviewer-one")
    selection = [{
        "source_id": record["source_id"], "reviewer_id": "reviewer-one",
        "revision": revision["revision"], "revision_hash": revision["revision_hash"],
    }]
    with pytest.raises(PrivateArtifactError, match="explicit consent"):
        store.export_labels(tmp_path / "exports", selected_revisions=selection)
    with pytest.raises(PrivateArtifactError, match="nonempty revision selection"):
        store.export_labels(tmp_path / "exports", explicit_consent=True)
    first = store.export_labels(
        tmp_path / "exports", explicit_consent=True, selected_revisions=selection,
    )
    second = store.export_labels(
        tmp_path / "exports", explicit_consent=True, selected_revisions=selection,
    )
    assert first == second
    assert first["label_count"] == 1
    export_dir = tmp_path / "exports" / first["export_id"]
    assert (export_dir / "canonical_labels.jsonl").is_file()
    assert (export_dir / "manifest.json").is_file()


def test_export_and_backup_fail_closed_if_revision_history_is_tampered(
    tmp_path: Path,
) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    WorkbenchService(store).submit(
        source_id=record["source_id"],
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE annotation_revisions SET payload_json = ?",
            ('{"decision":"tampered"}',),
        )
        connection.commit()

    with pytest.raises(PrivateArtifactError, match="revision chain is invalid"):
        store.export_labels(
            tmp_path / "exports", explicit_consent=True,
            selected_revisions=[{
                "source_id": record["source_id"], "reviewer_id": "reviewer-one",
                "revision": 1, "revision_hash": "0" * 64,
            }],
        )
    with pytest.raises(PrivateArtifactError, match="revision chain is invalid"):
        store.create_backup(tmp_path / "backups")


def test_adjudication_requires_disagreement_and_binds_resolved_revision_hashes(
    tmp_path: Path,
) -> None:
    store, record = _store(tmp_path, pool="annotation_development")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    service.submit(
        source_id=source_id,
        reviewer_id="reviewer-two",
        expected_revision=0,
        payload={
            "decision": "non_financial",
            "operational_class": "non_financial",
            "event_state": "no_event",
            "financial_family": None,
            "payment_rail": None,
            "events": [],
            "uncertain": False,
            "notes": "synthetic disagreement",
        },
    )
    disagreement = service.disagreements(source_id, "adjudicator")
    assert disagreement["has_disagreement"] is True
    assert service.list_rows(
        reviewer_id="adjudicator", filters={"disagreement": "yes"},
    )["total"] == 1
    result = service.submit(
        source_id=source_id,
        reviewer_id="adjudicator",
        expected_revision=0,
        payload=_posted_payload(record),
        adjudicated=True,
    )

    assert result["status"] == "adjudicated"
    assert store.progress(reviewer_id="adjudicator")["disagreements"] == 1
    assert store.progress(reviewer_id="adjudicator")["my_remaining"] == 0
    assert store.progress()["submitted_last_day"] >= 3
    latest = store.latest_annotation(source_id, "adjudicator")
    assert len(latest["payload"]["adjudication_of"]) == 2
    assert set(latest["payload"]["adjudication_of"]) == {
        item["revision_hash"] for item in disagreement["annotations"]
    }


def _posted_v2_payload(record: dict) -> dict:
    source = record["source"]["body"]

    def span(text: str) -> dict:
        start = source.index(text)
        return SourceSpan.from_source(source, start, start + len(text)).to_dict()

    return {
        "contract": "pocketfinancer.canonical-label/2",
        "decision": "posted",
        "operational_class": "posted_candidate",
        "event_state": "posted",
        "financial_family": "bank_transfer",
        "payment_rail": "bank_internal",
        "event": {
            "amount_value": "42.50",
            "currency": "INR",
            "amount_span": span("INR 42.50"),
            "direction": "credit",
            "direction_span": span("credited"),
            "account_reference": "**7788",
            "account_span": span("**7788"),
            "existing_account_id": None,
            "counterparty": "FRIEND",
            "counterparty_span": span("FRIEND"),
        },
        "uncertain": False,
        "notes": "",
    }


def test_v2_submission_projects_direct_extractor_without_analyzer_candidates(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    payload = _posted_v2_payload(record)
    saved = service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=payload,
    )
    assert saved["revision"] == 1
    assert store.latest_annotation(source_id, "reviewer-one")["canonical_label"]["contract"] == "pocketfinancer.canonical-label/2"
    preview = service.target_preview(source_id, "reviewer-one")
    assert preview["convertible"] is True
    assert preview["target_contract"] == "pocketfinancer.sms-extractor/1"
    assert preview["target"]["amount"]["value"] == "42.50"
    assert preview["target"]["account"]["evidence"]["text"] == "**7788"
    assert "candidate_id" not in json.dumps(preview)


def test_v2_drafts_and_explicit_transition_preserve_legacy_revisions(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    legacy = service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    assert legacy["revision"] == 1
    assert store.latest_annotation(source_id, "reviewer-one")["canonical_label"]["contract"] == "pocketfinancer.canonical-label/1"
    draft = service.save_draft(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=1,
        payload={"contract": "pocketfinancer.canonical-label/2", "decision": "posted"},
    )
    assert draft["revision"] == 2
    latest_draft = store.latest_annotation(source_id, "reviewer-one")
    assert latest_draft["canonical_label"] is None
    assert latest_draft["payload"]["contract"] == "pocketfinancer.canonical-label/2"
    with pytest.raises(WorkbenchValidationError, match="legacy v1"):
        service.save_draft(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=2,
            payload={"decision": "posted"},
        )
    posted = service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=2,
        payload=_posted_v2_payload(record),
    )
    assert posted["revision"] == 3
    history = service.view_row(source_id, "reviewer-one")["annotation_history"]
    assert [item["revision"] for item in history] == [1, 2, 3]
    assert history[0]["canonical_label"]["contract"] == "pocketfinancer.canonical-label/1"
    assert service.target_preview(source_id, "reviewer-one")["target_contract"] == "pocketfinancer.sms-extractor/1"


def test_v2_invalid_grounding_and_decision_do_not_append_revision(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    payload = _posted_v2_payload(record)
    payload["event"]["account_span"]["text"] = "not in source"
    with pytest.raises(WorkbenchValidationError, match="annotation source span is invalid"):
        service.submit(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=0,
            payload=payload,
        )
    assert store.current_revision(source_id, "reviewer-one") == 0
    payload = _posted_v2_payload(record)
    payload["decision"] = "unknown"
    payload["event"] = None
    with pytest.raises(WorkbenchValidationError, match="label_decision_invalid"):
        service.submit(
            source_id=source_id,
            reviewer_id="reviewer-one",
            expected_revision=0,
            payload=payload,
        )
    assert store.current_revision(source_id, "reviewer-one") == 0


def test_v2_protected_review_keeps_direct_preview_blind_until_reveal(tmp_path: Path) -> None:
    store, record = _store(tmp_path)
    service = WorkbenchService(store)
    source_id = record["source_id"]
    service.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_v2_payload(record),
    )
    with pytest.raises(WorkbenchValidationError, match="preview remains hidden"):
        service.target_preview(source_id, "reviewer-one")
    service.reveal(source_id, "reviewer-one")
    assert service.target_preview(source_id, "reviewer-one")["convertible"] is True


def test_progress_does_not_expose_protected_weak_facets(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    progress = store.progress()
    assert progress["total"] == 1
    assert progress["pools"]["protected_test"] == 1
    assert progress["classes"] == {}
    assert progress["families"] == {}
    assert progress["rails"] == {}
    assert progress["class_coverage"] == {}


def test_durable_resume_and_reviewer_unfinished_queue(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    saved = service.save_resume(
        reviewer_id="reviewer-one",
        source_id=source_id,
        offset=0,
        filters={"pool": "annotation_training", "reviewer_state": "unfinished"},
        search="",
        sort="timestamp",
        descending=False,
    )
    reopened = WorkbenchService(WorkbenchStore(store.database_path))
    assert reopened.load_resume("reviewer-one") == saved
    assert reopened.load_resume("reviewer-two") is None
    assert reopened.list_rows(
        reviewer_id="reviewer-one",
        filters={"reviewer_state": "unfinished"},
    )["total"] == 1

    reopened.submit(
        source_id=source_id,
        reviewer_id="reviewer-one",
        expected_revision=0,
        payload=_posted_payload(record),
    )
    assert reopened.list_rows(
        reviewer_id="reviewer-one",
        filters={"reviewer_state": "unfinished"},
    )["total"] == 0
    assert reopened.list_rows(
        reviewer_id="reviewer-two",
        filters={"reviewer_state": "unfinished"},
    )["total"] == 1
    assert reopened.list_rows(
        reviewer_id="reviewer-one",
        filters={"reviewer_state": "completed"},
    )["total"] == 1

    with pytest.raises(WorkbenchValidationError, match="resume filters"):
        reopened.save_resume(
            reviewer_id="reviewer-one",
            source_id=source_id,
            offset=0,
            filters={"unsafe": "value"},
        )
    assert reopened.load_resume("reviewer-one") == saved


def test_export_rejects_stale_or_duplicate_selection(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    result = WorkbenchService(store).submit(
        source_id=record["source_id"], reviewer_id="reviewer-one",
        expected_revision=0, payload=_posted_payload(record),
    )
    selection = {
        "source_id": record["source_id"], "reviewer_id": "reviewer-one",
        "revision": result["revision"], "revision_hash": result["revision_hash"],
    }
    with pytest.raises(PrivateArtifactError, match="missing or changed"):
        store.export_labels(
            tmp_path / "exports", explicit_consent=True,
            selected_revisions=[{**selection, "revision_hash": "0" * 64}],
        )
    with pytest.raises(PrivateArtifactError, match="duplicates"):
        store.export_labels(
            tmp_path / "exports", explicit_consent=True,
            selected_revisions=[selection, selection],
        )
    assert not (tmp_path / "exports" / "canonical_labels.jsonl").exists()


def test_candidate_coverage_filter_preserves_protected_blindness(tmp_path: Path) -> None:
    store, _record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    complete = service.list_rows(
        reviewer_id="reviewer-one", filters={"candidate_coverage": "core_complete"},
    )["total"]
    missing = service.list_rows(
        reviewer_id="reviewer-one", filters={"candidate_coverage": "core_missing"},
    )["total"]
    assert complete + missing == 1

    protected_root = tmp_path / "protected"
    protected_root.mkdir()
    protected_store, _ = _store(protected_root, pool="protected_test")
    protected = WorkbenchService(protected_store)
    assert protected.list_rows(
        reviewer_id="reviewer-one", filters={"candidate_coverage": "core_complete"},
    )["total"] == 0
    with pytest.raises(WorkbenchValidationError, match="blind review"):
        protected.list_rows(
            reviewer_id="reviewer-one",
            filters={"pool": "protected_test", "candidate_coverage": "core_complete"},
        )


def test_label_correction_appends_new_draft_and_preserves_submission(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_training")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    submitted = service.submit(
        source_id=source_id, reviewer_id="reviewer-one",
        expected_revision=0, payload=_posted_v2_payload(record),
    )
    draft = service.save_draft(
        source_id=source_id, reviewer_id="reviewer-one",
        expected_revision=submitted["revision"],
        payload=_posted_v2_payload(record),
    )
    assert draft["revision"] == submitted["revision"] + 1
    assert [item["status"] for item in store.annotation_history(
        source_id, "reviewer-one"
    )] == ["submitted", "draft"]
    assert store.progress(reviewer_id="reviewer-one")["my_remaining"] == 1
    store.verify_revision_chains()


def test_field_level_disagreement_enters_adjudication_queue(tmp_path: Path) -> None:
    store, record = _store(tmp_path, pool="annotation_development")
    service = WorkbenchService(store)
    source_id = record["source_id"]
    first = _posted_v2_payload(record)
    second = _posted_v2_payload(record)
    second["event"]["counterparty"] = None
    second["event"]["counterparty_span"] = None
    service.submit(
        source_id=source_id, reviewer_id="reviewer-one",
        expected_revision=0, payload=first,
    )
    service.submit(
        source_id=source_id, reviewer_id="reviewer-two",
        expected_revision=0, payload=second,
    )
    assert service.disagreements(source_id, "adjudicator")["has_disagreement"] is True
    assert service.list_rows(
        reviewer_id="adjudicator", filters={"disagreement": "yes"},
    )["total"] == 1
    assert store.progress()["disagreements"] == 1
