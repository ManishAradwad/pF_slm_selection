from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
from decimal import Decimal
from pathlib import Path

import pytest

from lfm25.annotation_store import (
    MAX_BACKUPS,
    StaleRevisionError,
    WorkbenchBindingError,
    WorkbenchCorruptionError,
    WorkbenchLockError,
    WorkbenchStore,
    WorkbenchStoreError,
    recover_backup,
    store_paths,
    validate_backup,
)


SYNTHETIC_TEXT = "₹50 paid to café"
BINDING = {
    "contract_sha256": "a" * 64,
    "source_sha256": "b" * 64,
    "proposal_temperature": 0.25,
}
STAMP = "2026-08-08T10:00:00+05:30"
ZONE = "Asia/Calcutta"


def _source_rows() -> list[dict]:
    return [
        {
            "row_id": "synthetic-0001",
            "position": 0,
            "reviewer_fields": {"sms": SYNTHETIC_TEXT, "sender": "SYNTH-A"},
            "source_json": json.dumps(
                {
                    "sms": SYNTHETIC_TEXT,
                    "sender": "SYNTH-A",
                    "private_confidence": 0.875,
                    "private_proposals": [{"amount": 50.0}],
                },
                ensure_ascii=False,
            ),
        },
        {
            "row_id": "synthetic-0002",
            "position": 1,
            "reviewer_fields": {"sms": "Synthetic reminder only", "sender": "SYNTH-B"},
            "source": {
                "sms": "Synthetic reminder only",
                "sender": "SYNTH-B",
                "private_confidence": 0.125,
            },
        },
    ]


def _valid_payload(amount: Decimal = Decimal("50.00")) -> dict:
    encoded = SYNTHETIC_TEXT.encode("utf-8")
    start = encoded.index(b"50")
    return {
        "decision": "transaction",
        "amount": amount,
        "spans": {
            "amount": {
                "start_byte": start,
                "end_byte": start + 2,
                "text": "50",
            }
        },
    }


def _bootstrap(store: WorkbenchStore) -> str:
    report = store.bootstrap(
        _source_rows(),
        workspace_binding=BINDING,
        workspace_metadata={"fixture": "synthetic-only"},
        created_at=STAMP,
    )
    assert report["row_count"] == 2
    session = store.start_session(
        "reviewer-fixture",
        session_id="session-fixture",
        started_at=STAMP,
        timezone_name=ZONE,
        metadata={"purpose": "synthetic-test"},
    )
    return session["session_id"]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_restart_hash_chain_qc_invalidation_and_parsed_reads(tmp_path: Path) -> None:
    db_path = tmp_path / "private-workspace" / "annotations.sqlite3"
    exact_payload = json.dumps(_valid_payload(), ensure_ascii=False, default=str, indent=2)

    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        requirement = store.set_qc_requirement(
            "synthetic-0001",
            required=True,
            requirement={"rule": "double-review"},
            expected_revision=0,
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        assert requirement["status"] == "pending"
        initial = store.save_annotation(
            "synthetic-0001",
            expected_revision=0,
            phase="initial",
            payload=exact_payload,
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        qc = store.save_annotation(
            "synthetic-0001",
            expected_revision=1,
            phase="qc",
            payload={"decision": "transaction", "amount": Decimal("50.00")},
            session_id=session_id,
            recorded_at="2026-08-08T10:05:00+05:30",
            timezone_name=ZONE,
        )
        assert qc["qc_for_initial_hash"] == initial["initial_event_hash"]
        store.update_qc_status(
            "synthetic-0001",
            "passed",
            expected_revision=2,
            session_id=session_id,
            recorded_at="2026-08-08T10:06:00+05:30",
            timezone_name=ZONE,
        )
        replacement = store.save_annotation(
            "synthetic-0001",
            expected_revision=2,
            phase="initial",
            payload=_valid_payload(Decimal("50.01")),
            session_id=session_id,
            status="needs_adjudication",
            recorded_at="2026-08-08T10:07:00+05:30",
            timezone_name=ZONE,
        )
        assert replacement["qc_event_id"] is None
        assert replacement["qc_event_hash"] is None
        assert store.get_qc_status("synthetic-0001")["status"] == "pending"
        history = store.get_history("synthetic-0001")
        assert [event["revision"] for event in history] == [1, 2, 3]
        assert history[0]["previous_hash"] is None
        assert history[1]["previous_hash"] == history[0]["event_hash"]
        assert history[2]["previous_hash"] == history[1]["event_hash"]
        assert history[0]["payload"]["amount"] == "50.00"
        assert all("payload_json" not in event for event in history)
        public_row = store.get_row("synthetic-0001")
        assert "source" not in public_row
        private_row = store.get_row("synthetic-0001", include_private=True)
        assert private_row["source"]["private_confidence"] == 0.875
        assert isinstance(private_row["source"]["private_confidence"], Decimal)
        assert store.verify_integrity()["event_count"] == 3

    with sqlite3.connect(db_path) as connection:
        stored_payload = connection.execute(
            "SELECT payload_json FROM annotation_events WHERE revision = 1"
        ).fetchone()[0]
    assert stored_payload == exact_payload

    with WorkbenchStore(db_path, workspace_binding=BINDING) as restarted:
        assert restarted.get_progress()["event_count"] == 3
        assert restarted.get_row("synthetic-0001")["status"] == "needs_adjudication"
        assert restarted.workspace_metadata()["binding"] == BINDING

    with pytest.raises(WorkbenchBindingError) as error:
        WorkbenchStore(db_path, workspace_binding={"source_sha256": "wrong"})
    assert SYNTHETIC_TEXT not in str(error.value)
    assert str(db_path) not in str(error.value)


def test_projection_dirty_is_durable_blocks_saves_and_reconciles(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        saved = store.save_annotation(
            "synthetic-0001",
            expected_revision=0,
            phase="initial",
            payload=_valid_payload(),
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
            projection_required=True,
        )
        assert saved["projection_dirty"] is True
        assert [row["row_id"] for row in store.get_dirty_rows()] == ["synthetic-0001"]
        assert store.get_progress()["dirty_rows"] == 1
        with pytest.raises(WorkbenchStoreError, match="reconciled"):
            store.save_annotation(
                "synthetic-0001",
                expected_revision=1,
                phase="draft",
                payload={"decision": "transaction"},
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )

    with WorkbenchStore(db_path, workspace_binding=BINDING) as restarted:
        assert restarted.get_row("synthetic-0001")["projection_dirty"] is True
        with pytest.raises(StaleRevisionError):
            restarted.mark_projection_clean("synthetic-0001", expected_revision=0)
        clean = restarted.mark_projection_clean("synthetic-0001", expected_revision=1)
        assert clean["projection_dirty"] is False
        assert restarted.get_dirty_rows() == []
        assert len(restarted.get_rows(projection_dirty=False)) == 2
        restarted.mark_projection_clean("synthetic-0001", expected_revision=1)
        restarted.save_annotation(
            "synthetic-0001",
            expected_revision=1,
            phase="draft",
            payload={"decision": "transaction"},
            session_id="session-fixture",
            recorded_at=STAMP,
            timezone_name=ZONE,
        )


def test_stale_revision_decimal_and_utf8_span_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        store.save_annotation(
            "synthetic-0001",
            expected_revision=0,
            phase="initial",
            payload=_valid_payload(),
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        with pytest.raises(StaleRevisionError):
            store.save_annotation(
                "synthetic-0001",
                expected_revision=0,
                phase="draft",
                payload={"decision": "transaction"},
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )
        assert len(store.get_history("synthetic-0001")) == 1
        with pytest.raises(WorkbenchStoreError, match="event is invalid"):
            store.save_annotation(
                "synthetic-0002",
                expected_revision=0,
                phase="initial",
                payload='{"amount": 12.5}',
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )
        with pytest.raises(WorkbenchStoreError, match="source span"):
            store.save_annotation(
                "synthetic-0001",
                expected_revision=1,
                phase="draft",
                payload={
                    "decision": "not_transaction",
                    "span": {"start_byte": 1, "end_byte": 2, "source_field": "sms"},
                },
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )


def test_qc_and_proposal_reveal_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        pending = store.set_qc_requirement(
            "synthetic-0002",
            required=True,
            requirement={"sample": "boundary"},
            expected_revision=0,
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        passed = store.update_qc_status(
            "synthetic-0002",
            "passed",
            expected_revision=pending["revision"],
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        assert passed["revision"] == 2
        reveal = store.record_proposal_reveal(
            "synthetic-0002",
            "proposal-local-1",
            session_id=session_id,
            revealed_at=STAMP,
            timezone_name=ZONE,
            details={"confidence": Decimal("0.75")},
        )
        assert reveal["details"] == {"confidence": "0.75"}
        assert store.get_proposal_reveals("synthetic-0002") == [reveal]
        assert [row["row_id"] for row in store.get_qc_requirements(status="passed")] == [
            "synthetic-0002"
        ]


def test_transaction_rolls_back_unexpected_interruption(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        _bootstrap(store)
        with pytest.raises(KeyboardInterrupt):
            with store._transaction() as connection:
                connection.execute(
                    "UPDATE qc_requirements SET revision = 1 "
                    "WHERE row_id = 'synthetic-0001'"
                )
                raise KeyboardInterrupt
        assert store._require_connection().in_transaction is False
        assert store.get_qc_status("synthetic-0001")["revision"] == 0


def test_bulk_qc_requirements_are_atomic_and_reject_partial_queue(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        store.set_qc_requirement(
            "synthetic-0002",
            required=False,
            requirement=None,
            expected_revision=0,
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        before = store.get_qc_requirements()
        stale_batch = [
            {
                "row_id": "synthetic-0001",
                "required": True,
                "requirement": {"reasons": ["synthetic-second-pass"]},
                "expected_revision": 0,
            },
            {
                "row_id": "synthetic-0002",
                "required": False,
                "requirement": None,
                "expected_revision": 0,
            },
        ]

        with pytest.raises(StaleRevisionError, match="revision is stale"):
            store.set_qc_requirements(
                stale_batch,
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )
        assert store.get_qc_requirements() == before

        with pytest.raises(WorkbenchStoreError, match="cover the workspace"):
            store.set_qc_requirements(
                stale_batch[:1],
                session_id=session_id,
                recorded_at=STAMP,
                timezone_name=ZONE,
            )
        assert store.get_qc_requirements() == before

        stale_batch[1]["expected_revision"] = 1
        saved = store.set_qc_requirements(
            stale_batch,
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        by_id = {item["row_id"]: item for item in saved}
        assert by_id["synthetic-0001"]["status"] == "pending"
        assert by_id["synthetic-0001"]["revision"] == 1
        assert by_id["synthetic-0002"]["status"] == "not_required"
        assert by_id["synthetic-0002"]["revision"] == 2
        assert all(item["session_id"] == session_id for item in saved)


def test_nonblocking_lock_and_secure_permissions_and_retention(tmp_path: Path) -> None:
    db_path = tmp_path / "secure-parent" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        _bootstrap(store)
        assert _mode(db_path.parent) == 0o700
        assert _mode(store.paths.database) == 0o600
        assert _mode(store.paths.lock) == 0o600
        assert _mode(store.paths.backups) == 0o700
        assert store.last_backup_path is not None
        assert _mode(store.last_backup_path) == 0o600
        descriptor = os.open(store.paths.lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        with pytest.raises(WorkbenchLockError, match="already open"):
            WorkbenchStore(db_path, workspace_binding=BINDING)

    for _ in range(MAX_BACKUPS + 3):
        with WorkbenchStore(db_path, workspace_binding=BINDING):
            pass
    paths = store_paths(db_path)
    backups = list(paths.backups.glob("*.bak"))
    assert len(backups) == MAX_BACKUPS
    assert all(_mode(path) == 0o600 for path in backups)


def test_corruption_backup_validation_and_recovery(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        store.save_annotation(
            "synthetic-0001",
            expected_revision=0,
            phase="initial",
            payload=_valid_payload(),
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        good_backup = store.last_backup_path
        assert good_backup is not None
        with pytest.raises(WorkbenchLockError):
            validate_backup(db_path, good_backup, workspace_binding=BINDING)
    validated = validate_backup(db_path, good_backup, workspace_binding=BINDING)
    assert validated.row_count == 2
    assert validated.event_count == 1
    with pytest.raises(WorkbenchBindingError):
        validate_backup(db_path, good_backup, workspace_binding={"wrong": "binding"})

    db_path.write_bytes(b"synthetic corrupt database bytes")
    db_path.chmod(0o600)
    with pytest.raises(WorkbenchStoreError) as error:
        WorkbenchStore(db_path, workspace_binding=BINDING)
    assert "synthetic" not in str(error.value)
    displaced = recover_backup(db_path, good_backup, workspace_binding=BINDING)
    assert displaced is not None
    assert displaced.read_bytes() == b"synthetic corrupt database bytes"
    assert _mode(displaced) == 0o600
    with WorkbenchStore(db_path, workspace_binding=BINDING) as recovered:
        assert recovered.get_row("synthetic-0001")["revision"] == 1
        bad_backup = recovered.create_backup()
    bad_backup.write_bytes(b"synthetic invalid backup")
    bad_backup.chmod(0o600)
    with pytest.raises(WorkbenchCorruptionError):
        validate_backup(db_path, bad_backup, workspace_binding=BINDING)


def test_schema_pragmas_append_only_and_schema_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        _bootstrap(store)
        connection = store._require_connection()
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA synchronous").fetchone()[0] >= 2
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        with pytest.raises(sqlite3.IntegrityError, match="append only"):
            connection.execute(
                "UPDATE sessions SET reviewer = 'changed' WHERE session_id = 'session-fixture'"
            )
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(WorkbenchStoreError, match="schema"):
        WorkbenchStore(db_path, workspace_binding=BINDING)


def test_proposal_revision_preserves_blind_initial_and_qc_pointers(tmp_path: Path) -> None:
    db_path = tmp_path / "workspace" / "annotations.sqlite3"
    with WorkbenchStore(db_path, workspace_binding=BINDING) as store:
        session_id = _bootstrap(store)
        initial = store.save_annotation(
            "synthetic-0001",
            expected_revision=0,
            phase="initial",
            payload=_valid_payload(),
            session_id=session_id,
            recorded_at=STAMP,
            timezone_name=ZONE,
        )
        qc = store.save_annotation(
            "synthetic-0001",
            expected_revision=1,
            phase="qc",
            payload=_valid_payload(),
            session_id=session_id,
            recorded_at="2026-08-08T10:05:00+05:30",
            timezone_name=ZONE,
        )
        store.record_proposal_reveal(
            "synthetic-0001",
            "proposal-bundle",
            session_id=session_id,
            revealed_at="2026-08-08T10:06:00+05:30",
            timezone_name=ZONE,
        )
        revised = store.save_annotation(
            "synthetic-0001",
            expected_revision=2,
            phase="proposal_revision",
            payload=_valid_payload(Decimal("50.01")),
            session_id=session_id,
            recorded_at="2026-08-08T10:07:00+05:30",
            timezone_name=ZONE,
        )

        current = store.get_row("synthetic-0001")
        assert current["revision"] == 3
        assert current["phase"] == "proposal_revision"
        assert current["status"] == "completed"
        assert current["annotation"]["amount"] == "50.01"
        assert current["initial_event_id"] == initial["event_id"]
        assert current["initial_event_hash"] == initial["event_hash"]
        assert current["qc_event_id"] == qc["event_id"]
        assert current["qc_event_hash"] == qc["event_hash"]
        assert current["qc_for_initial_hash"] == initial["event_hash"]
        assert revised["initial_event_id"] == initial["event_id"]
        assert revised["qc_event_id"] == qc["event_id"]
        assert [event["phase"] for event in store.get_history("synthetic-0001")] == [
            "initial",
            "qc",
            "proposal_revision",
        ]
