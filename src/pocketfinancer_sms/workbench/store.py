"""Crash-safe SQLite storage for local corpus review."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..provenance import (
    PrivateArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    ensure_private_directory,
    file_sha256,
    object_sha256,
)


SCHEMA_VERSION = 1


class WorkbenchConflict(RuntimeError):
    """Optimistic revision conflict with no private values in its message."""


class WorkbenchStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.resolve()
        ensure_private_directory(self.database_path.parent)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def import_manifest(self, manifest_path: Path, *, corpus_run_id: str) -> int:
        rows = _read_jsonl(manifest_path)
        with self.transaction() as connection:
            existing_run = connection.execute(
                "SELECT value FROM meta WHERE key = 'corpus_run_id'"
            ).fetchone()
            if existing_run and existing_run[0] != corpus_run_id:
                raise PrivateArtifactError("workbench database is bound to a different corpus run")
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('corpus_run_id', ?)",
                (corpus_run_id,),
            )
            for record in rows:
                _validate_manifest_record(record)
                grouping = record["grouping"]
                weak = record["weak_facets"]
                metadata = record["source_metadata"]
                connection.execute(
                    """
                    INSERT INTO corpus_rows(
                        source_id, record_json, pool, operational_class, event_state,
                        financial_family, payment_rail, sender_template_group, time_group,
                        disposition, selector_action, timestamp, review_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed')
                    ON CONFLICT(source_id) DO UPDATE SET
                        record_json = excluded.record_json,
                        pool = excluded.pool,
                        operational_class = excluded.operational_class,
                        event_state = excluded.event_state,
                        financial_family = excluded.financial_family,
                        payment_rail = excluded.payment_rail,
                        sender_template_group = excluded.sender_template_group,
                        time_group = excluded.time_group,
                        disposition = excluded.disposition,
                        selector_action = excluded.selector_action,
                        timestamp = excluded.timestamp
                    """,
                    (
                        record["source_id"],
                        _json(record),
                        record["pool"],
                        weak["operational_class"],
                        weak["event_state"],
                        weak.get("financial_family"),
                        weak.get("payment_rail"),
                        grouping["sender_template_group_hash"],
                        grouping["time_group"],
                        weak["disposition"],
                        weak["selector_action"],
                        metadata.get("timestamp"),
                    ),
                )
            count = connection.execute("SELECT COUNT(*) FROM corpus_rows").fetchone()[0]
            if count != len(rows):
                raise PrivateArtifactError("workbench import did not preserve manifest row count")
        os.chmod(self.database_path, 0o600)
        return len(rows)

    def get_record(self, source_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT record_json, review_state FROM corpus_rows WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        if row is None:
            return None
        record = json.loads(row["record_json"])
        record["review_state"] = row["review_state"]
        return record

    def list_rows(
        self,
        *,
        filters: dict[str, str | None],
        search: str | None,
        sort: str,
        descending: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        allowed_filters = {
            "pool",
            "operational_class",
            "event_state",
            "financial_family",
            "payment_rail",
            "sender_template_group",
            "time_group",
            "disposition",
            "selector_action",
            "review_state",
        }
        clauses: list[str] = []
        values: list[Any] = []
        for name, value in filters.items():
            if name not in allowed_filters:
                raise ValueError("workbench filter is unsupported")
            if value:
                clauses.append(f"{name} = ?")
                values.append(value)
        if search:
            clauses.append(
                "(json_extract(record_json, '$.source.body') LIKE ? ESCAPE '\\' "
                "OR json_extract(record_json, '$.source.sender') LIKE ? ESCAPE '\\')"
            )
            escaped = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            values.extend((escaped, escaped))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sort_columns = {
            "timestamp": "timestamp",
            "source_id": "source_id",
            "pool": "pool",
            "review_state": "review_state",
        }
        if sort not in sort_columns:
            raise ValueError("workbench sort is unsupported")
        direction = "DESC" if descending else "ASC"
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self.connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM corpus_rows{where}", values  # noqa: S608
            ).fetchone()[0]
            rows = connection.execute(
                f"""SELECT source_id, pool, operational_class, event_state,
                           financial_family, payment_rail, time_group, disposition,
                           selector_action, timestamp, review_state,
                           json_extract(record_json, '$.source.body') AS body,
                           json_extract(record_json, '$.source.sender') AS sender
                    FROM corpus_rows{where}
                    ORDER BY {sort_columns[sort]} {direction}, source_id ASC
                    LIMIT ? OFFSET ?""",  # noqa: S608
                (*values, limit, offset),
            ).fetchall()
        return {"total": total, "rows": [dict(row) for row in rows]}

    def current_revision(self, source_id: str, reviewer_id: str) -> int:
        with self.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM annotation_revisions "
                    "WHERE source_id = ? AND reviewer_id = ?",
                    (source_id, reviewer_id),
                ).fetchone()[0]
            )

    def append_annotation_revision(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        status: str,
        payload: dict[str, Any],
        canonical_label: dict[str, Any] | None,
        created_at_epoch_ms: int,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            current = int(
                connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM annotation_revisions "
                    "WHERE source_id = ? AND reviewer_id = ?",
                    (source_id, reviewer_id),
                ).fetchone()[0]
            )
            if current != expected_revision:
                raise WorkbenchConflict("annotation revision changed; reload before saving")
            previous = connection.execute(
                "SELECT revision_hash FROM annotation_revisions "
                "WHERE source_id = ? AND reviewer_id = ? ORDER BY revision DESC LIMIT 1",
                (source_id, reviewer_id),
            ).fetchone()
            previous_hash = previous[0] if previous else None
            revision = current + 1
            revision_body = {
                "source_id": source_id,
                "reviewer_id": reviewer_id,
                "revision": revision,
                "status": status,
                "payload": payload,
                "canonical_label": canonical_label,
                "created_at_epoch_ms": created_at_epoch_ms,
                "previous_hash": previous_hash,
            }
            revision_hash = object_sha256(revision_body)
            connection.execute(
                """
                INSERT INTO annotation_revisions(
                    source_id, reviewer_id, revision, status, payload_json,
                    canonical_label_json, created_at_epoch_ms, previous_hash, revision_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    reviewer_id,
                    revision,
                    status,
                    _json(payload),
                    _json(canonical_label) if canonical_label is not None else None,
                    created_at_epoch_ms,
                    previous_hash,
                    revision_hash,
                ),
            )
            next_state = "draft" if status == "draft" else status
            connection.execute(
                "UPDATE corpus_rows SET review_state = ? WHERE source_id = ?",
                (next_state, source_id),
            )
        return {"revision": revision, "revision_hash": revision_hash, "status": status}

    def latest_annotation(self, source_id: str, reviewer_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM annotation_revisions WHERE source_id = ? AND reviewer_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (source_id, reviewer_id),
            ).fetchone()
        return _annotation_row(row) if row else None

    def submitted_annotations(self, source_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT a.* FROM annotation_revisions a
                JOIN (
                    SELECT source_id, reviewer_id, MAX(revision) AS revision
                    FROM annotation_revisions
                    WHERE source_id = ? AND status IN ('submitted', 'adjudicated')
                    GROUP BY source_id, reviewer_id
                ) latest
                  ON a.source_id = latest.source_id
                 AND a.reviewer_id = latest.reviewer_id
                 AND a.revision = latest.revision
                ORDER BY a.reviewer_id
                """,
                (source_id,),
            ).fetchall()
        return [_annotation_row(row) for row in rows]

    def has_initial_submission(self, source_id: str, reviewer_id: str) -> bool:
        with self.connect() as connection:
            value = connection.execute(
                "SELECT 1 FROM annotation_revisions WHERE source_id = ? AND reviewer_id = ? "
                "AND status IN ('submitted', 'adjudicated') LIMIT 1",
                (source_id, reviewer_id),
            ).fetchone()
        return value is not None

    def reveal(self, source_id: str, reviewer_id: str, created_at_epoch_ms: int) -> None:
        if not self.has_initial_submission(source_id, reviewer_id):
            raise WorkbenchConflict("blind review must be submitted before reveal")
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO reveal_events(source_id, reviewer_id, created_at_epoch_ms) "
                "VALUES (?, ?, ?)",
                (source_id, reviewer_id, created_at_epoch_ms),
            )

    def is_revealed(self, source_id: str, reviewer_id: str) -> bool:
        with self.connect() as connection:
            value = connection.execute(
                "SELECT 1 FROM reveal_events WHERE source_id = ? AND reviewer_id = ?",
                (source_id, reviewer_id),
            ).fetchone()
        return value is not None

    def append_weak_correction(
        self,
        *,
        source_id: str,
        reviewer_id: str,
        expected_revision: int,
        facets: dict[str, Any],
        created_at_epoch_ms: int,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            current = int(
                connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM segregation_corrections "
                    "WHERE source_id = ?",
                    (source_id,),
                ).fetchone()[0]
            )
            if current != expected_revision:
                raise WorkbenchConflict("segregation revision changed; reload before saving")
            revision = current + 1
            previous = connection.execute(
                "SELECT correction_hash FROM segregation_corrections WHERE source_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (source_id,),
            ).fetchone()
            previous_hash = previous[0] if previous else None
            body = {
                "source_id": source_id,
                "revision": revision,
                "reviewer_id": reviewer_id,
                "facets": facets,
                "created_at_epoch_ms": created_at_epoch_ms,
                "previous_hash": previous_hash,
            }
            correction_hash = object_sha256(body)
            connection.execute(
                "INSERT INTO segregation_corrections VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    revision,
                    reviewer_id,
                    _json(facets),
                    created_at_epoch_ms,
                    previous_hash,
                    correction_hash,
                ),
            )
        return {"revision": revision, "correction_hash": correction_hash}

    def progress(self) -> dict[str, Any]:
        with self.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM corpus_rows").fetchone()[0]
            states = dict(
                connection.execute(
                    "SELECT review_state, COUNT(*) FROM corpus_rows GROUP BY review_state"
                ).fetchall()
            )
            pools = dict(
                connection.execute("SELECT pool, COUNT(*) FROM corpus_rows GROUP BY pool").fetchall()
            )
            classes = dict(
                connection.execute(
                    "SELECT operational_class, COUNT(*) FROM corpus_rows GROUP BY operational_class"
                ).fetchall()
            )
        return {"total": total, "review_states": states, "pools": pools, "classes": classes}

    def create_backup(self, backup_root: Path) -> dict[str, Any]:
        ensure_private_directory(backup_root)
        stamp = int(time.time() * 1000)
        temporary = backup_root / f".workbench-{stamp}.sqlite3.tmp"
        destination = backup_root / f"workbench-{stamp}.sqlite3"
        with self.connect() as source, sqlite3.connect(temporary) as target:
            source.backup(target)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        digest = file_sha256(destination)
        manifest = {"backup": destination.name, "sha256": digest, "schema_version": SCHEMA_VERSION}
        atomic_write_json(destination.with_suffix(".manifest.json"), manifest)
        return manifest

    def integrity_check(self) -> bool:
        with self.connect() as connection:
            return connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    @staticmethod
    def restore_backup(backup: Path, destination: Path, expected_sha256: str) -> None:
        if file_sha256(backup) != expected_sha256:
            raise PrivateArtifactError("workbench backup hash does not match its manifest")
        ensure_private_directory(destination.parent)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".restore-", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(backup, temporary)
            os.chmod(temporary, 0o600)
            with sqlite3.connect(temporary) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise PrivateArtifactError("workbench backup failed integrity validation")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def export_labels(self, output_root: Path) -> dict[str, Any]:
        ensure_private_directory(output_root)
        with self.connect() as connection:
            run_id = connection.execute(
                "SELECT value FROM meta WHERE key = 'corpus_run_id'"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT source_id, reviewer_id, revision, status, canonical_label_json,
                       revision_hash
                FROM annotation_revisions
                WHERE status IN ('submitted', 'adjudicated') AND canonical_label_json IS NOT NULL
                ORDER BY source_id, reviewer_id, revision
                """
            ).fetchall()
        values = [
            {
                "source_id": row["source_id"],
                "reviewer_id": row["reviewer_id"],
                "revision": row["revision"],
                "status": row["status"],
                "canonical_label": json.loads(row["canonical_label_json"]),
                "revision_hash": row["revision_hash"],
            }
            for row in rows
        ]
        binding = {"corpus_run_id": run_id, "labels": values, "export_contract": "pocketfinancer.workbench-export/1"}
        export_id = object_sha256(binding)[:20]
        export_dir = output_root / export_id
        ensure_private_directory(export_dir)
        labels_path = export_dir / "canonical_labels.jsonl"
        atomic_write_jsonl(labels_path, values)
        manifest = {
            "contract": "pocketfinancer.workbench-export/1",
            "export_id": export_id,
            "corpus_run_id": run_id,
            "label_count": len(values),
            "labels_sha256": file_sha256(labels_path),
        }
        atomic_write_json(export_dir / "manifest.json", manifest)
        return manifest

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS corpus_rows(
                    source_id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    pool TEXT NOT NULL,
                    operational_class TEXT NOT NULL,
                    event_state TEXT NOT NULL,
                    financial_family TEXT,
                    payment_rail TEXT,
                    sender_template_group TEXT NOT NULL,
                    time_group TEXT NOT NULL,
                    disposition TEXT NOT NULL,
                    selector_action TEXT NOT NULL,
                    timestamp TEXT,
                    review_state TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS corpus_filter_idx
                    ON corpus_rows(pool, operational_class, event_state, review_state);
                CREATE INDEX IF NOT EXISTS corpus_group_idx
                    ON corpus_rows(sender_template_group, time_group);
                CREATE TABLE IF NOT EXISTS annotation_revisions(
                    source_id TEXT NOT NULL REFERENCES corpus_rows(source_id),
                    reviewer_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    canonical_label_json TEXT,
                    created_at_epoch_ms INTEGER NOT NULL,
                    previous_hash TEXT,
                    revision_hash TEXT NOT NULL,
                    PRIMARY KEY(source_id, reviewer_id, revision)
                );
                CREATE TABLE IF NOT EXISTS reveal_events(
                    source_id TEXT NOT NULL REFERENCES corpus_rows(source_id),
                    reviewer_id TEXT NOT NULL,
                    created_at_epoch_ms INTEGER NOT NULL,
                    PRIMARY KEY(source_id, reviewer_id)
                );
                CREATE TABLE IF NOT EXISTS segregation_corrections(
                    source_id TEXT NOT NULL REFERENCES corpus_rows(source_id),
                    revision INTEGER NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    facets_json TEXT NOT NULL,
                    created_at_epoch_ms INTEGER NOT NULL,
                    previous_hash TEXT,
                    correction_hash TEXT NOT NULL,
                    PRIMARY KEY(source_id, revision)
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        os.chmod(self.database_path, 0o600)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrivateArtifactError("canonical manifest could not be read for workbench import") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise PrivateArtifactError("canonical manifest contains a non-object row")
    return rows


def _validate_manifest_record(record: dict[str, Any]) -> None:
    required = {"source_id", "source", "source_metadata", "analysis", "weak_facets", "grouping", "pool"}
    if not required <= set(record):
        raise PrivateArtifactError("canonical manifest record is incomplete")


def _annotation_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "source_id": row["source_id"],
        "reviewer_id": row["reviewer_id"],
        "revision": row["revision"],
        "status": row["status"],
        "payload": json.loads(row["payload_json"]),
        "canonical_label": (
            json.loads(row["canonical_label_json"]) if row["canonical_label_json"] else None
        ),
        "created_at_epoch_ms": row["created_at_epoch_ms"],
        "previous_hash": row["previous_hash"],
        "revision_hash": row["revision_hash"],
    }
