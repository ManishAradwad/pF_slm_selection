from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest

from lfm25.annotation_service import AnnotationService, validate_blinded_import_gate
from lfm25.annotation_sources import load_blinded_workspace
from lfm25.annotation_store import (
    WorkbenchStore,
    WorkbenchStoreError,
    recover_backup,
)
from lfm25.blinded_review import run_export
from lfm25.annotation_workbench import (
    BLINDED_MODE,
    SOURCE_PREFILL_OFF,
    SOURCE_PREFILL_UNAMBIGUOUS,
    TRAINING_MODE,
    WorkspaceDefinition,
    WorkbenchError,
    WorkbenchSourceRow,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
)


TRANSACTION_SMS = (
    "Synthetic alert: INR 1,234.50 debited from A/c XX0042 at CAFÉ NILA."
)
REMINDER_SMS = "Synthetic reminder: statement is ready for A/c XX0099."
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _span(sms: str, text: str) -> dict[str, object]:
    character_start = sms.index(text)
    character_end = character_start + len(text)
    return {
        "text": text,
        "start": len(sms[:character_start].encode("utf-8")),
        "end": len(sms[:character_end].encode("utf-8")),
    }


def _transaction_annotation(
    *,
    transaction_type: str = "debit",
    notes: str = "Synthetic blind-first note.",
) -> dict[str, object]:
    value = empty_annotation()
    value.update(
        {
            "decision": "transaction",
            "amount_decimal": "1234.50",
            "amount_span": _span(TRANSACTION_SMS, "INR 1,234.50"),
            "type": transaction_type,
            "account_span": _span(TRANSACTION_SMS, "A/c XX0042"),
            "counterparty_span": _span(TRANSACTION_SMS, "CAFÉ NILA"),
            "counterparty_absent": False,
            "notes": notes,
        }
    )
    return value


def _null_annotation() -> dict[str, object]:
    value = empty_annotation()
    value["decision"] = "not_transaction"
    return value


def _definition(*, include_reminder: bool = False) -> WorkspaceDefinition:
    private_transaction = {
        "record_hash": "synthetic-transaction-1",
        "split": "train",
        "sender": "SYNTH-TXN",
        "sms": TRANSACTION_SMS,
        "silver_label": {
            "amount": "1234.50",
            "counterparty": "CAFÉ NILA",
            "type": "debit",
            "account": "A/c XX0042",
        },
        "confidence": Decimal("0.80"),
        "heuristic_reason_codes": ["synthetic-fixture"],
        "local_model_proposals": [
            {
                "model_id": "synthetic-local-model",
                "model_family": "synthetic-family",
                "label": {"type": "credit"},
                "confidence": Decimal("0.70"),
                "confidence_basis": "synthetic-only",
                "schema_valid": True,
            }
        ],
    }
    rows = [
        WorkbenchSourceRow(
            row_id="synthetic-transaction-1",
            position=0,
            sender="SYNTH-TXN",
            sms=TRANSACTION_SMS,
            source_json=exact_json_dumps(private_transaction),
            split="train",
            queue_tags=("synthetic-queue",),
        )
    ]
    if include_reminder:
        rows.append(
            WorkbenchSourceRow(
                row_id="synthetic-reminder-1",
                position=1,
                sender="SYNTH-REMINDER",
                sms=REMINDER_SMS,
                source_json=exact_json_dumps(
                    {
                        "record_hash": "synthetic-reminder-1",
                        "split": "train",
                        "sender": "SYNTH-REMINDER",
                        "sms": REMINDER_SMS,
                        "silver_label": None,
                        "confidence": Decimal("0.60"),
                        "heuristic_reason_codes": ["synthetic-reminder"],
                        "local_model_proposals": [],
                    }
                ),
                split="train",
                queue_tags=("synthetic-reminder",),
            )
        )
    row_ids = "\n".join(sorted(row.row_id for row in rows)) + "\n"
    return WorkspaceDefinition(
        mode=TRAINING_MODE,
        rows=tuple(rows),
        binding={
            "contract": "synthetic-annotation-service-v1",
            "mode": TRAINING_MODE,
            "row_count": len(rows),
            "record_id_set_sha256": hashlib.sha256(row_ids.encode("utf-8")).hexdigest(),
        },
        metadata={
            "schema_version": 1,
            "fixture": "invented-synthetic-only",
        },
    )


def _temporary_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo = tmp_path / "synthetic-repo"
    private_root = repo / "PRIVATE_DATA" / "lfm25"
    private_root.mkdir(parents=True)
    private_root.chmod(0o700)
    monkeypatch.setattr(
        "lfm25.annotation_service.require_private_ignore", lambda *_args: None
    )
    monkeypatch.setattr(
        "lfm25.annotation_sources.require_private_ignore", lambda *_args: None
    )
    return repo


def _db_path(repo: Path) -> Path:
    return repo / "PRIVATE_DATA" / "lfm25" / "annotation_workbench" / "synthetic.sqlite3"


def _active_learning_definition(*, mode: str = TRAINING_MODE) -> WorkspaceDefinition:
    specs = (
        ("rare", ("rare_sender_or_template",)),
        ("coverage-a", ("candidate_coverage_miss",)),
        (
            "disagreement",
            ("model_disagreement", "rare_sender_or_template"),
        ),
        ("ordinary", ()),
        (
            "coverage-b",
            ("candidate_coverage_miss", "refund_or_reversal"),
        ),
    )
    rows = []
    for position, (suffix, queue_tags) in enumerate(specs):
        row_id = f"invented-active-{suffix}"
        sms = f"Invented active-learning message {suffix}."
        split = "train" if mode == TRAINING_MODE else "test"
        rows.append(
            WorkbenchSourceRow(
                row_id=row_id,
                position=position,
                sender=f"INVENTED-{position}",
                sms=sms,
                source_json=exact_json_dumps(
                    {
                        "record_hash": row_id,
                        "split": split,
                        "sender": f"INVENTED-{position}",
                        "sms": sms,
                    }
                ),
                split=split,
                queue_tags=queue_tags,
            )
        )
    row_ids = "\n".join(sorted(row.row_id for row in rows)) + "\n"
    return WorkspaceDefinition(
        mode=mode,
        rows=tuple(rows),
        binding={
            "contract": "invented-active-learning-service-v1",
            "mode": mode,
            "row_count": len(rows),
            "record_id_set_sha256": hashlib.sha256(
                row_ids.encode("utf-8")
            ).hexdigest(),
        },
        metadata={"fixture": "invented-active-learning-only"},
    )


def test_active_learning_queue_priority_blindness_and_cycling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _active_learning_definition()

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="invented-reviewer",
        )

        first = service.navigate(
            position=-1,
            direction="first",
            filter_name="active_learning",
        )["row"]
        assert first["review_id"] == "invented-active-disagreement"
        assert first["queue_tags"] == []
        assert service.get_row("invented-active-coverage-a")["queue_tags"] == []

        current = service.navigate(
            position=1,
            direction="current",
            filter_name="active_learning",
        )["row"]
        assert current["review_id"] == "invented-active-coverage-a"
        missing_current = service.navigate(
            position=3,
            direction="current",
            filter_name="active_learning",
        )["row"]
        assert missing_current["review_id"] == "invented-active-disagreement"

        expected_cycle = (
            (2, "next", "invented-active-coverage-a"),
            (1, "next", "invented-active-coverage-b"),
            (4, "next", "invented-active-rare"),
            (0, "next", "invented-active-disagreement"),
            (2, "previous", "invented-active-rare"),
        )
        for position, direction, expected_id in expected_cycle:
            selected = service.navigate(
                position=position,
                direction=direction,
                filter_name="active_learning",
            )["row"]
            assert selected["review_id"] == expected_id

        draft = service.save(
            row_id="invented-active-coverage-a",
            expected_revision=0,
            annotation=empty_annotation(),
            submit=False,
        )["row"]
        assert draft["status"] == "draft"
        assert draft["queue_tags"] == []
        assert service.navigate(
            position=1,
            direction="current",
            filter_name="active_learning",
        )["row"]["review_id"] == "invented-active-coverage-a"

        completed = service.save(
            row_id="invented-active-disagreement",
            expected_revision=0,
            annotation=_null_annotation(),
            submit=True,
        )["row"]
        assert completed["queue_tags"] == [
            "model_disagreement",
            "rare_sender_or_template",
        ]
        after_drop = service.navigate(
            position=2,
            direction="next",
            filter_name="active_learning",
        )["row"]
        assert after_drop["review_id"] == "invented-active-coverage-a"


def test_active_learning_queue_rejected_in_qc_and_blinded_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_repo = _temporary_repo(tmp_path / "training", monkeypatch)
    training_definition = _active_learning_definition()
    with WorkbenchStore(
        _db_path(training_repo),
        workspace_binding=training_definition.binding,
    ) as store:
        training_service = AnnotationService(
            repo_root=training_repo,
            definition=training_definition,
            store=store,
            reviewer="invented-reviewer",
        )
        for row in training_definition.rows:
            training_service.save(
                row_id=row.row_id,
                expected_revision=0,
                annotation=_null_annotation(),
                submit=True,
            )
        training_service.start_qc()
        with pytest.raises(WorkbenchError, match="active-learning queue is unavailable"):
            training_service.navigate(
                position=0,
                direction="first",
                filter_name="active_learning",
            )

    blinded_repo = _temporary_repo(tmp_path / "blinded", monkeypatch)
    blinded_definition = _active_learning_definition(mode=BLINDED_MODE)
    monkeypatch.setattr(
        AnnotationService,
        "_reconcile_projection_on_open",
        lambda _self: None,
    )
    with WorkbenchStore(
        _db_path(blinded_repo),
        workspace_binding=blinded_definition.binding,
    ) as store:
        blinded_service = AnnotationService(
            repo_root=blinded_repo,
            definition=blinded_definition,
            store=store,
            reviewer="invented-reviewer",
        )
        with pytest.raises(WorkbenchError, match="active-learning queue is unavailable"):
            blinded_service.navigate(
                position=0,
                direction="first",
                filter_name="active_learning",
            )


def test_partial_transaction_draft_persists_across_service_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition()
    draft = empty_annotation()
    draft.update(
        {
            "amount_decimal": "1234.50",
            "amount_span": _span(TRANSACTION_SMS, "INR 1,234.50"),
            "type": "debit",
            "notes": "Synthetic partial transaction draft.",
        }
    )

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        result = service.save(
            row_id="synthetic-transaction-1",
            expected_revision=0,
            annotation=draft,
            submit=False,
        )
        assert result["row"]["status"] == "draft"
        assert result["row"]["revision"] == 1
        assert result["row"]["annotation"]["decision"] is None
        assert result["row"]["annotation"]["amount_decimal"] == "1234.50"

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as restarted:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=restarted,
            reviewer="synthetic-reviewer",
        )
        row = service.get_row("synthetic-transaction-1")
        assert row["status"] == "draft"
        assert row["revision"] == 1
        assert row["annotation"]["notes"] == "Synthetic partial transaction draft."
        assert service.history("synthetic-transaction-1")[0]["phase"] == "draft"
        assert service.progress()["pending_rows"] == 1


def test_service_rejects_workspace_above_supported_row_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(include_reminder=True)
    monkeypatch.setattr("lfm25.annotation_service.MAX_PAGE_ROWS", 1)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        with pytest.raises(WorkbenchError, match="supported row limit"):
            AnnotationService(
                repo_root=repo,
                definition=definition,
                store=store,
                reviewer="synthetic-reviewer",
            )
        assert store.get_progress()["total_rows"] == 0


def test_post_proposal_revision_keeps_blind_first_and_requires_new_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition()

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        service.save(
            row_id="synthetic-transaction-1",
            expected_revision=0,
            annotation=_transaction_annotation(),
            submit=True,
        )
        blind_first = store.get_row("synthetic-transaction-1")
        proposals = service.reveal_proposals("synthetic-transaction-1")
        assert proposals["model_proposals"][0]["model_id"] == "synthetic-local-model"
        assert proposals["model_proposals"][0]["confidence"] == Decimal("0.70")
        assert isinstance(proposals["model_proposals"][0]["confidence"], Decimal)
        assert proposals["heuristic_proposal"]["confidence"] == Decimal("0.80")
        assert isinstance(proposals["heuristic_proposal"]["confidence"], Decimal)

        service.save(
            row_id="synthetic-transaction-1",
            expected_revision=1,
            annotation=_transaction_annotation(),
            submit=True,
        )
        unchanged = store.get_row("synthetic-transaction-1")
        assert unchanged["revision"] == 2
        assert unchanged["phase"] == "proposal_revision"
        assert unchanged["initial_event_id"] == blind_first["initial_event_id"]
        assert unchanged["initial_event_hash"] == blind_first["initial_event_hash"]

        unchanged_note = _transaction_annotation(transaction_type="credit")
        with pytest.raises(WorkbenchError, match="requires a new note"):
            service.save(
                row_id="synthetic-transaction-1",
                expected_revision=2,
                annotation=unchanged_note,
                submit=True,
            )
        assert store.get_row("synthetic-transaction-1")["revision"] == 2

        revised = _transaction_annotation(
            transaction_type="credit",
            notes="Synthetic change after reviewing the local proposal.",
        )
        service.save(
            row_id="synthetic-transaction-1",
            expected_revision=2,
            annotation=revised,
            submit=True,
        )
        current = store.get_row("synthetic-transaction-1")
        assert current["revision"] == 3
        assert current["phase"] == "proposal_revision"
        assert current["initial_event_id"] == blind_first["initial_event_id"]
        assert current["initial_event_hash"] == blind_first["initial_event_hash"]
        assert [event["phase"] for event in store.get_history(current["row_id"])] == [
            "initial",
            "proposal_revision",
            "proposal_revision",
        ]
        output = repo / "PRIVATE_DATA" / "lfm25" / "synthetic-export.jsonl"
        report = repo / "PRIVATE_DATA" / "lfm25" / "synthetic-export-report.json"
        export = service.export_training(
            output_file=output,
            report_file=report,
        )
        assert export["completed_rows"] == 1
        raw_export = output.read_text(encoding="utf-8")
        exported = exact_json_loads(raw_export)
        assert exported["confidence"] == Decimal("0.80")
        assert isinstance(exported["confidence"], Decimal)
        proposal_confidence = exported["local_model_proposals"][0]["confidence"]
        assert proposal_confidence == Decimal("0.70")
        assert isinstance(proposal_confidence, Decimal)
        assert '"confidence":0.80' in raw_export
        assert '"confidence":"0.80"' not in raw_export


def test_qc_queue_survives_restart_and_hides_pending_labels_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(include_reminder=True)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        service.save(
            row_id="synthetic-transaction-1",
            expected_revision=0,
            annotation=_transaction_annotation(),
            submit=True,
        )
        service.save(
            row_id="synthetic-reminder-1",
            expected_revision=0,
            annotation=_null_annotation(),
            submit=True,
        )
        bulk_call_sizes: list[int] = []
        set_qc_requirements = store.set_qc_requirements

        def tracked_bulk(requirements, **kwargs):
            bulk_call_sizes.append(len(requirements))
            return set_qc_requirements(requirements, **kwargs)

        monkeypatch.setattr(store, "set_qc_requirements", tracked_bulk)
        service.start_qc()
        assert bulk_call_sizes == [2]
        queue_before = store.get_qc_requirements()
        service.start_qc()
        assert bulk_call_sizes == [2]
        assert store.get_qc_requirements() == queue_before
        required_ids = {item["row_id"] for item in queue_before if item["required"]}
        assert "synthetic-transaction-1" in required_ids
        assert service.history("synthetic-transaction-1") == []
        hidden_row = service.get_row("synthetic-transaction-1")
        assert hidden_row["history_available"] is False
        assert hidden_row["annotation"] == empty_annotation()
        assert hidden_row["queue_tags"] == []
        assert hidden_row["proposal_reveal_available"] is False
        for filter_name in ("noted", "transaction", "null"):
            with pytest.raises(WorkbenchError, match="hidden"):
                service.navigate(position=0, direction="first", filter_name=filter_name)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as restarted:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=restarted,
            reviewer="synthetic-reviewer",
        )
        assert service.session_phase == "qc"
        assert restarted.get_qc_requirements() == queue_before
        assert service.history("synthetic-transaction-1") == []
        selected = service.navigate(position=0, direction="first", filter_name="qc")
        assert selected["row"]["review_id"] in required_ids
        assert selected["row"]["annotation"] == empty_annotation()
        assert selected["row"]["history_available"] is False
        assert selected["row"]["proposal_reveal_available"] is False


def test_qc_save_rolls_back_event_and_status_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition()
    row_id = "synthetic-transaction-1"

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        service.save(
            row_id=row_id,
            expected_revision=0,
            annotation=_transaction_annotation(),
            submit=True,
        )
        service.start_qc()
        row_before = store.get_row(row_id)
        qc_before = store.get_qc_status(row_id)
        history_before = store.get_history(row_id)
        connection = store._require_connection()
        connection.execute(
            "CREATE TEMP TRIGGER synthetic_abort_qc_status "
            "BEFORE UPDATE OF status ON qc_requirements "
            "WHEN OLD.row_id = 'synthetic-transaction-1' "
            "BEGIN SELECT RAISE(ABORT, 'synthetic interruption'); END"
        )

        with pytest.raises(WorkbenchStoreError, match="write failed"):
            service.save_qc(
                row_id=row_id,
                expected_revision=int(row_before["revision"]),
                annotation=_null_annotation(),
            )

        assert connection.in_transaction is False
        connection.execute("DROP TRIGGER synthetic_abort_qc_status")
        assert store.get_row(row_id) == row_before
        assert store.get_qc_status(row_id) == qc_before
        assert store.get_history(row_id) == history_before

        failed = service.save_qc(
            row_id=row_id,
            expected_revision=int(row_before["revision"]),
            annotation=_null_annotation(),
        )
        assert failed["next_available"] is False
        assert store.get_qc_status(row_id)["status"] == "failed"
        disagreement = store.get_row(row_id)
        assert disagreement["status"] == "needs_adjudication"
        assert disagreement["projection_dirty"] is False

        resolved = service.save_qc(
            row_id=row_id,
            expected_revision=int(disagreement["revision"]),
            annotation=_transaction_annotation(),
        )
        assert resolved["next_available"] is True
        assert store.get_qc_status(row_id)["status"] == "passed"
        assert [
            event["phase"] for event in store.get_history(row_id)
        ] == ["initial", "qc", "adjudication"]


def test_partial_qc_queue_is_rejected_without_extending_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_repo(tmp_path, monkeypatch)
    definition = _definition(include_reminder=True)

    with WorkbenchStore(_db_path(repo), workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        service.save(
            row_id="synthetic-transaction-1",
            expected_revision=0,
            annotation=_transaction_annotation(),
            submit=True,
        )
        service.save(
            row_id="synthetic-reminder-1",
            expected_revision=0,
            annotation=_null_annotation(),
            submit=True,
        )
        transaction = store.get_row("synthetic-transaction-1")
        reference_hash = store.get_history("synthetic-transaction-1")[-1]["event_hash"]
        store.set_qc_requirement(
            "synthetic-transaction-1",
            required=True,
            requirement={
                "reasons": ["noted_second_pass", "transaction_second_pass"],
                "initial_event_hash": transaction["initial_event_hash"],
                "reference_event_hash": reference_hash,
            },
            expected_revision=0,
            session_id=service.session_id,
            reviewer="synthetic-reviewer",
        )
        before = store.get_qc_requirements()

        with pytest.raises(
            WorkbenchError, match="saved QC queue does not match current policy"
        ) as captured:
            service.start_qc()
        assert store.get_qc_requirements() == before
        by_id = {item["row_id"]: item for item in before}
        assert by_id["synthetic-reminder-1"]["revision"] == 0
        assert by_id["synthetic-reminder-1"]["required"] is False
        assert TRANSACTION_SMS not in str(captured.value)


def _prepare_completed_blinded_qc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    adjudicate_transaction: bool = False,
    source_prefill: str = SOURCE_PREFILL_OFF,
) -> tuple[Path, Path, str, str]:
    repo = _temporary_repo(tmp_path, monkeypatch)
    if source_prefill == SOURCE_PREFILL_UNAMBIGUOUS:
        contracts = repo / "configs" / "contracts"
        contracts.mkdir(parents=True)
        for name in (
            "pocketfinancer-android-current.json",
            "pocketfinancer-candidate-v1.json",
        ):
            contracts.joinpath(name).write_bytes(
                PROJECT_ROOT.joinpath("configs", "contracts", name).read_bytes()
            )
    handbook = repo / "docs" / "guides" / "ANNOTATION_HANDBOOK_V1.md"
    handbook.parent.mkdir(parents=True)
    handbook.write_text(
        "# Synthetic Annotation Handbook V1\n\nInvented test guidance only.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "lfm25.blinded_review.require_private_ignore", lambda *_args: None
    )
    null_messages = (
        "Synthetic null notice alpha: monthly summary is ready.",
        "Synthetic null notice beta: monthly summary is ready.",
    )
    manifest_rows = [
        {
            "record_hash": "synthetic-blinded-transaction",
            "split": "test",
            "sender": "SYNTH-BLIND-TXN",
            "sms": TRANSACTION_SMS,
        },
        {
            "record_hash": "synthetic-blinded-null-alpha",
            "split": "test",
            "sender": "SYNTH-BLIND-NULL-A",
            "sms": null_messages[0],
        },
        {
            "record_hash": "synthetic-blinded-null-beta",
            "split": "test",
            "sender": "SYNTH-BLIND-NULL-B",
            "sms": null_messages[1],
        },
    ]
    manifest = repo / "PRIVATE_DATA" / "lfm25" / "split_manifest.jsonl"
    manifest.write_text(
        "".join(exact_json_dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    run_export(repo, source_prefill=source_prefill)
    definition = load_blinded_workspace(
        repo,
        include_initial_annotations=False,
        source_prefill=source_prefill,
    )
    db_path = _db_path(repo)
    annotations: dict[str, dict[str, object]] = {}
    transaction_id = next(
        row.row_id for row in definition.rows if row.sms == TRANSACTION_SMS
    )
    null_ids = {
        row.row_id for row in definition.rows if row.sms in set(null_messages)
    }

    with WorkbenchStore(db_path, workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        for row in definition.rows:
            annotation = (
                _transaction_annotation()
                if row.row_id == transaction_id
                else _null_annotation()
            )
            annotations[row.row_id] = annotation
            service.save(
                row_id=row.row_id,
                expected_revision=0,
                annotation=annotation,
                submit=True,
            )

        service.start_qc()
        qc_by_id = {
            item["row_id"]: item for item in store.get_qc_requirements()
        }
        required_ids = {
            row_id for row_id, state in qc_by_id.items() if state["required"]
        }
        assert transaction_id in required_ids
        selected_nulls = null_ids & required_ids
        unselected_nulls = null_ids - required_ids
        assert len(selected_nulls) == 1
        assert len(unselected_nulls) == 1

        for row_id in required_ids:
            current = store.get_row(row_id)
            qc_annotation = (
                _null_annotation()
                if adjudicate_transaction and row_id == transaction_id
                else annotations[row_id]
            )
            result = service.save_qc(
                row_id=row_id,
                expected_revision=int(current["revision"]),
                annotation=qc_annotation,
            )
            if adjudicate_transaction and row_id == transaction_id:
                assert result["next_available"] is False
                assert store.get_qc_status(row_id)["status"] == "failed"
                current = store.get_row(row_id)
                adjudicated = service.save_qc(
                    row_id=row_id,
                    expected_revision=int(current["revision"]),
                    annotation=qc_annotation,
                )
                assert adjudicated["next_available"] is True
                assert [
                    event["phase"] for event in store.get_history(row_id)
                ] == ["initial", "qc", "adjudication"]
            else:
                assert result["next_available"] is True
            assert store.get_qc_status(row_id)["status"] == "passed"
        assert all(
            item["status"] in {"passed", "not_required"}
            for item in store.get_qc_requirements()
        )

    report = validate_blinded_import_gate(
        repo, db_path=db_path, source_prefill=source_prefill
    )
    assert report["valid"] is True
    return repo, db_path, transaction_id, unselected_nulls.pop()


def test_blinded_import_gate_accepts_adjudicated_decision_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, db_path, transaction_id, _unselected_null = _prepare_completed_blinded_qc(
        tmp_path,
        monkeypatch,
        adjudicate_transaction=True,
    )
    definition = load_blinded_workspace(repo, include_initial_annotations=False)

    with WorkbenchStore(db_path, workspace_binding=definition.binding) as store:
        row = store.get_row(transaction_id)
        qc = store.get_qc_status(transaction_id)
        assert row["phase"] == "adjudication"
        assert row["annotation"]["decision"] == "not_transaction"
        assert qc["status"] == "passed"
        assert qc["requirement"]["reasons"] == [
            "noted_second_pass",
            "transaction_second_pass",
        ]
        assert qc["requirement"]["initial_event_hash"] == row["initial_event_hash"]
        assert qc["requirement"]["reference_event_hash"] == row["initial_event_hash"]
        assert [event["phase"] for event in store.get_history(transaction_id)] == [
            "initial",
            "qc",
            "adjudication",
        ]


def test_blinded_import_gate_resolves_assisted_review_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, db_path, _transaction_id, _unselected_null = (
        _prepare_completed_blinded_qc(
            tmp_path,
            monkeypatch,
            source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
        )
    )

    report = validate_blinded_import_gate(
        repo,
        db_path=db_path,
        source_prefill=SOURCE_PREFILL_UNAMBIGUOUS,
    )

    assert report["valid"] is True
    assert report["source_prefill"] == SOURCE_PREFILL_UNAMBIGUOUS
    assert (
        repo
        / "PRIVATE_DATA"
        / "lfm25"
        / "blinded_test_candidate_assisted_review.jsonl"
    ).is_file()
    assert not (
        repo / "PRIVATE_DATA" / "lfm25" / "blinded_test_review.jsonl"
    ).exists()


def test_blinded_relaunch_rebuilds_projection_after_older_backup_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, db_path, transaction_id, _unselected_null = _prepare_completed_blinded_qc(
        tmp_path, monkeypatch
    )
    definition = load_blinded_workspace(repo, include_initial_annotations=False)
    review_file = repo / "PRIVATE_DATA" / "lfm25" / "blinded_test_review.jsonl"

    with WorkbenchStore(db_path, workspace_binding=definition.binding) as store:
        service = AnnotationService(
            repo_root=repo,
            definition=definition,
            store=store,
            reviewer="synthetic-reviewer",
        )
        restored_revision = int(store.get_row(transaction_id)["revision"])
        older_backup = store.create_backup()
        newer = store.save_annotation(
            transaction_id,
            expected_revision=restored_revision,
            phase="adjudication",
            payload=_transaction_annotation(
                transaction_type="credit",
                notes="Synthetic newer projection state.",
            ),
            session_id=service.session_id,
            reviewer="synthetic-reviewer",
            status="completed",
            projection_required=True,
        )
        service._write_blinded_projection()
        service._mark_projection_clean(transaction_id, int(newer["revision"]))
        newer_projection = review_file.read_bytes()

    displaced = recover_backup(
        db_path,
        older_backup,
        workspace_binding=definition.binding,
    )
    assert displaced is not None
    restored_definition = load_blinded_workspace(
        repo, include_initial_annotations=False
    )
    with WorkbenchStore(
        db_path, workspace_binding=restored_definition.binding
    ) as restored_store:
        AnnotationService(
            repo_root=repo,
            definition=restored_definition,
            store=restored_store,
            reviewer="synthetic-reviewer",
        )
        restored = restored_store.get_row(transaction_id)
        assert restored["revision"] == restored_revision
        assert restored["annotation"]["type"] == "debit"
        assert restored["projection_dirty"] is False

    repaired_projection = review_file.read_bytes()
    assert repaired_projection != newer_projection
    projected_rows = [
        exact_json_loads(line)
        for line in repaired_projection.decode("utf-8").splitlines()
        if line
    ]
    transaction_projection = next(
        row for row in projected_rows if row["review_id"] == transaction_id
    )
    assert transaction_projection["type"] == "debit"


@pytest.mark.parametrize(
    "hash_field",
    ("initial_event_hash", "reference_event_hash"),
)
def test_blinded_import_gate_rejects_tampered_qc_requirement_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hash_field: str,
) -> None:
    repo, db_path, transaction_id, _unselected_null = _prepare_completed_blinded_qc(
        tmp_path, monkeypatch
    )
    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT requirement_json FROM qc_requirements WHERE row_id = ?",
            (transaction_id,),
        ).fetchone()
        assert stored is not None
        requirement = json.loads(stored[0])
        original = requirement[hash_field]
        requirement[hash_field] = (
            "0" * 64 if original != "0" * 64 else "1" * 64
        )
        connection.execute(
            "UPDATE qc_requirements SET requirement_json = ? WHERE row_id = ?",
            (exact_json_dumps(requirement), transaction_id),
        )

    with pytest.raises(
        WorkbenchError, match="QC requirements do not match policy"
    ) as captured:
        validate_blinded_import_gate(repo, db_path=db_path)
    message = str(captured.value)
    assert TRANSACTION_SMS not in message
    assert transaction_id not in message


def test_blinded_import_gate_rejects_unexpected_qc_for_nonselected_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, db_path, _transaction_id, unselected_null = _prepare_completed_blinded_qc(
        tmp_path, monkeypatch
    )
    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            "SELECT initial_event_hash FROM current_state WHERE row_id = ?",
            (unselected_null,),
        ).fetchone()
        assert stored is not None
        initial_hash = stored[0]
        stale_requirement = {
            "reasons": ["deterministic_null_sample_10pct"],
            "initial_event_hash": initial_hash,
            "reference_event_hash": initial_hash,
        }
        connection.execute(
            "UPDATE qc_requirements "
            "SET required = 1, status = 'pending', revision = revision + 1, "
            "requirement_json = ? WHERE row_id = ?",
            (exact_json_dumps(stale_requirement), unselected_null),
        )

    with pytest.raises(
        WorkbenchError, match="unexpected QC state outside policy"
    ) as captured:
        validate_blinded_import_gate(repo, db_path=db_path)
    message = str(captured.value)
    assert "Synthetic null notice" not in message
    assert unselected_null not in message
