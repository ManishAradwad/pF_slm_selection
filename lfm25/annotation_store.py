"""Private, local-only SQLite persistence for the annotation workbench.

The caller owns path-policy validation.  This module deliberately has no default
database location and never logs row content.  It adds filesystem permissions,
single-writer locking, integrity validation, audit history, and crash-safe backups
around the explicitly supplied path.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Final


SCHEMA_VERSION: Final = 1
KNOWN_PHASES: Final = frozenset(
    {"initial", "qc", "adjudication", "draft", "uncertain", "proposal_revision"}
)
KNOWN_STATUSES: Final = frozenset(
    {"pending", "draft", "completed", "uncertain", "needs_adjudication"}
)
QC_STATUSES: Final = frozenset(
    {"not_required", "pending", "passed", "failed", "waived"}
)
MAX_BACKUPS: Final = 5
_EVENT_DOMAIN: Final = b"lfm25-annotation-event-v1\0"
_REQUIRED_TABLES: Final = frozenset(
    {
        "schema_version",
        "workspace_metadata",
        "sessions",
        "source_rows",
        "annotation_events",
        "current_state",
        "qc_requirements",
        "proposal_reveals",
    }
)
_REQUIRED_TRIGGERS: Final = frozenset(
    {
        "workspace_metadata_no_update",
        "workspace_metadata_no_delete",
        "sessions_no_update",
        "sessions_no_delete",
        "source_rows_no_update",
        "source_rows_no_delete",
        "annotation_events_no_update",
        "annotation_events_no_delete",
        "proposal_reveals_no_update",
        "proposal_reveals_no_delete",
    }
)


class WorkbenchStoreError(RuntimeError):
    """A persistence failure whose message is safe to show to a reviewer."""


class WorkbenchLockError(WorkbenchStoreError):
    """The workspace lock could not be acquired."""


class WorkbenchCorruptionError(WorkbenchStoreError):
    """The SQLite file or its audit projections failed validation."""


class WorkbenchSchemaError(WorkbenchStoreError):
    """The database schema cannot be used by this implementation."""


class WorkbenchBindingError(WorkbenchStoreError):
    """The frozen workspace binding differs from the caller's binding."""


class StaleRevisionError(WorkbenchStoreError):
    """An optimistic annotation or QC revision no longer matches."""


@dataclass(frozen=True)
class SourceRow:
    """An explicitly separated reviewer view and private source record."""

    row_id: str
    reviewer_fields: Mapping[str, Any]
    source_json: str | Mapping[str, Any]
    position: int | None = None


@dataclass(frozen=True)
class StorePaths:
    """Filesystem artifacts derived from the caller-supplied database path."""

    database: Path
    lock: Path
    backups: Path


@dataclass(frozen=True)
class BackupValidation:
    """Content-free result returned after validating a backup."""

    path: Path
    schema_version: int
    row_count: int
    event_count: int
    binding_sha256: str | None


_PROCESS_GUARD = threading.RLock()
_ACTIVE_LOCKS: set[str] = set()


def _absolute_path(value: os.PathLike[str] | str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise WorkbenchStoreError("an explicit annotation database path is required") from exc
    if not raw or "\x00" in raw:
        raise WorkbenchStoreError("an explicit annotation database path is required")
    return Path(os.path.abspath(raw))


def store_paths(db_path: os.PathLike[str] | str) -> StorePaths:
    """Return deterministic adjacent paths without applying repository policy."""

    database = _absolute_path(db_path)
    return StorePaths(
        database=database,
        lock=database.with_name(f"{database.name}.lock"),
        backups=database.with_name(f"{database.name}.backups"),
    )


def _ensure_secure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise WorkbenchStoreError("annotation workspace directory is unsafe")
        path.chmod(0o700)
    except WorkbenchStoreError:
        raise
    except OSError as exc:
        raise WorkbenchStoreError("annotation workspace directory is unavailable") from exc


def _require_regular_file(path: Path, message: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise WorkbenchStoreError(message) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WorkbenchStoreError(message)


def _secure_database_file(path: Path) -> None:
    _ensure_secure_directory(path.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        _require_regular_file(path, "annotation database path is unsafe")
    except OSError as exc:
        raise WorkbenchStoreError("annotation database could not be created") from exc
    else:
        os.close(descriptor)
    try:
        path.chmod(0o600)
    except OSError as exc:
        raise WorkbenchStoreError("annotation database permissions could not be secured") from exc


class _ExclusiveFileLock:
    """A nonblocking flock augmented with an in-process path registry."""

    def __init__(self, paths: StorePaths) -> None:
        self._paths = paths
        self._descriptor: int | None = None
        self._key = os.path.normcase(os.fspath(paths.lock))

    def acquire(self) -> None:
        _ensure_secure_directory(self._paths.database.parent)
        with _PROCESS_GUARD:
            if self._key in _ACTIVE_LOCKS:
                raise WorkbenchLockError("annotation workspace is already open")
            _ACTIVE_LOCKS.add(self._key)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._paths.lock, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise WorkbenchLockError("annotation workspace lock is unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            with _PROCESS_GUARD:
                _ACTIVE_LOCKS.discard(self._key)
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise WorkbenchLockError("annotation workspace is already open") from exc
            raise WorkbenchLockError("annotation workspace lock could not be acquired") from exc
        except WorkbenchStoreError:
            with _PROCESS_GUARD:
                _ACTIVE_LOCKS.discard(self._key)
            try:
                os.close(descriptor)
            except (OSError, UnboundLocalError):
                pass
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        with _PROCESS_GUARD:
            _ACTIVE_LOCKS.discard(self._key)

    def __enter__(self) -> _ExclusiveFileLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _reject_json_float(_value: str) -> None:
    raise ValueError


def _load_json(
    text: str,
    *,
    require_mapping: bool = True,
    reject_floats: bool = False,
) -> Any:
    if not isinstance(text, str):
        raise ValueError
    if reject_floats:
        value = json.loads(
            text,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    else:
        value = json.loads(text, parse_constant=_reject_json_constant)
    if require_mapping and not isinstance(value, dict):
        raise ValueError
    return value


def _load_source_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError
    value = json.loads(
        text,
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError
    return value


def _json_compatible(value: Any, *, allow_float: bool = True) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not allow_float or not math.isfinite(value):
            raise ValueError
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError
        return {
            key: _json_compatible(item, allow_float=allow_float)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item, allow_float=allow_float) for item in value]
    raise ValueError


def _canonical_json(value: Mapping[str, Any], *, allow_float: bool = True) -> str:
    compatible = _json_compatible(value, allow_float=allow_float)
    return json.dumps(
        compatible,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_or_canonical_json(value: str | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        parsed = _load_json(value)
        return value, parsed
    if not isinstance(value, Mapping):
        raise ValueError
    text = _canonical_json(value)
    return text, _load_json(text)


def _exact_or_canonical_source_json(
    value: str | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, _load_source_json(value)
    if not isinstance(value, Mapping):
        raise ValueError
    text = _canonical_json(value)
    return text, _load_source_json(text)


def _exact_or_canonical_payload(
    value: str | Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value, _load_json(value, reject_floats=True)
    if not isinstance(value, Mapping):
        raise ValueError
    text = _canonical_json(value, allow_float=False)
    return text, _load_json(text, reject_floats=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 512:
        raise ValueError
    return value


def _aware_timestamp(value: str | datetime | None) -> tuple[str, datetime]:
    if value is None:
        parsed = datetime.now().astimezone()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(candidate)
    else:
        raise ValueError
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError
    return parsed.isoformat(timespec="microseconds"), parsed


def _timestamp_and_zone(
    value: str | datetime | None,
    zone: str | None,
) -> tuple[str, str]:
    timestamp, parsed = _aware_timestamp(value)
    if zone is None:
        zone = parsed.tzname() or parsed.strftime("%z")
    return timestamp, _token(zone)


def _event_hash(
    *,
    chain_index: int,
    row_id: str,
    revision: int,
    phase: str,
    status: str,
    reviewer: str,
    session_id: str,
    recorded_at: str,
    timezone_name: str,
    payload_json: str,
    payload_sha256: str,
    previous_hash: str | None,
) -> str:
    metadata = _canonical_json(
        {
            "chain_index": chain_index,
            "payload_sha256": payload_sha256,
            "phase": phase,
            "previous_hash": previous_hash,
            "recorded_at": recorded_at,
            "reviewer": reviewer,
            "revision": revision,
            "row_id": row_id,
            "session_id": session_id,
            "status": status,
            "timezone": timezone_name,
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_EVENT_DOMAIN)
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    encoded_payload = payload_json.encode("utf-8")
    digest.update(len(encoded_payload).to_bytes(8, "big"))
    digest.update(encoded_payload)
    return digest.hexdigest()


def _validate_utf8_span(text: str, start: Any, end: Any) -> None:
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError
    encoded = text.encode("utf-8")
    if end > len(encoded):
        raise ValueError
    boundaries = {0, len(encoded)}
    cursor = 0
    for character in text:
        cursor += len(character.encode("utf-8"))
        boundaries.add(cursor)
    if start not in boundaries or end not in boundaries:
        raise ValueError


def _span_source_text(source: Mapping[str, Any], span: Mapping[str, Any]) -> str:
    field = span.get("source_field", span.get("field", "sms"))
    if not isinstance(field, str):
        raise ValueError
    text = source.get(field)
    if not isinstance(text, str):
        if field == "sms" and isinstance(source.get("text"), str):
            text = source["text"]
        else:
            raise ValueError
    return text


def _validate_span_object(span: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    if "start_byte" in span or "end_byte" in span:
        if "start_byte" not in span or "end_byte" not in span:
            raise ValueError
        start, end = span["start_byte"], span["end_byte"]
    elif "start" in span or "end" in span:
        if "start" not in span or "end" not in span:
            raise ValueError
        start, end = span["start"], span["end"]
    else:
        return False
    text = _span_source_text(source, span)
    _validate_utf8_span(text, start, end)
    expected = span.get("text")
    if expected is not None:
        if not isinstance(expected, str):
            raise ValueError
        raw = text.encode("utf-8")[start:end]
        if raw.decode("utf-8") != expected:
            raise ValueError
    return True


def _validate_payload_spans(value: Any, source: Mapping[str, Any], *, in_span: bool = False) -> None:
    if isinstance(value, Mapping):
        is_span = _validate_span_object(value, source) if in_span else False
        for key, child in value.items():
            child_in_span = (
                key in {"span", "source_span", "spans"}
                or key.endswith("_span")
                or key.endswith("_spans")
            )
            if key == "spans" and isinstance(child, Mapping):
                for nested in child.values():
                    _validate_payload_spans(nested, source, in_span=True)
            else:
                _validate_payload_spans(child, source, in_span=child_in_span)
        if in_span and not is_span and {"start", "end"}.intersection(value):
            raise ValueError
    elif isinstance(value, list):
        if in_span and len(value) == 2 and all(isinstance(item, int) for item in value):
            text = _span_source_text(source, {})
            _validate_utf8_span(text, value[0], value[1])
            return
        for child in value:
            _validate_payload_spans(child, source, in_span=in_span)


def _default_status(phase: str) -> str:
    if phase == "draft":
        return "draft"
    if phase == "uncertain":
        return "uncertain"
    return "completed"


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA busy_timeout = 0")
    connection.execute("PRAGMA foreign_keys = ON")
    journal = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
    if journal is None or str(journal[0]).casefold() != "delete":
        raise WorkbenchStoreError("annotation database journaling could not be secured")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA temp_store = MEMORY")
    foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
    synchronous = connection.execute("PRAGMA synchronous").fetchone()
    if foreign_keys != (1,) or synchronous is None or int(synchronous[0]) < 2:
        raise WorkbenchStoreError("annotation database safety settings were not applied")


def _quick_check(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute("PRAGMA quick_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise WorkbenchCorruptionError("annotation workspace is corrupt") from exc
    if rows != [("ok",)]:
        raise WorkbenchCorruptionError("annotation workspace is corrupt")


_MIGRATION_V1 = """
BEGIN IMMEDIATE;
CREATE TABLE schema_version (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL,
    migrated_at TEXT NOT NULL
);
INSERT INTO schema_version(singleton, version, migrated_at)
VALUES (1, 1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

CREATE TABLE workspace_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    binding_json TEXT NOT NULL,
    binding_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    frozen INTEGER NOT NULL CHECK (frozen = 1)
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    reviewer TEXT NOT NULL,
    started_at TEXT NOT NULL,
    timezone TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE source_rows (
    row_id TEXT PRIMARY KEY,
    position INTEGER NOT NULL UNIQUE CHECK (position >= 0),
    reviewer_json TEXT NOT NULL,
    source_json TEXT NOT NULL,
    source_sha256 TEXT NOT NULL
);

CREATE TABLE annotation_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_index INTEGER NOT NULL UNIQUE CHECK (chain_index > 0),
    row_id TEXT NOT NULL REFERENCES source_rows(row_id) ON DELETE RESTRICT,
    revision INTEGER NOT NULL CHECK (revision > 0),
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    recorded_at TEXT NOT NULL,
    timezone TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    previous_hash TEXT,
    event_hash TEXT NOT NULL UNIQUE,
    UNIQUE(row_id, revision)
);

CREATE TABLE current_state (
    row_id TEXT PRIMARY KEY REFERENCES source_rows(row_id) ON DELETE RESTRICT,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    projection_dirty INTEGER NOT NULL DEFAULT 0 CHECK (projection_dirty IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'pending',
    latest_event_id INTEGER REFERENCES annotation_events(event_id) ON DELETE RESTRICT,
    phase TEXT,
    payload_json TEXT,
    reviewer TEXT,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE RESTRICT,
    recorded_at TEXT,
    timezone TEXT,
    initial_event_id INTEGER REFERENCES annotation_events(event_id) ON DELETE RESTRICT,
    initial_event_hash TEXT,
    qc_event_id INTEGER REFERENCES annotation_events(event_id) ON DELETE RESTRICT,
    qc_event_hash TEXT,
    qc_for_initial_hash TEXT
);

CREATE TABLE qc_requirements (
    row_id TEXT PRIMARY KEY REFERENCES source_rows(row_id) ON DELETE RESTRICT,
    required INTEGER NOT NULL DEFAULT 0 CHECK (required IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'not_required',
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    requirement_json TEXT,
    updated_at TEXT,
    timezone TEXT,
    reviewer TEXT,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE RESTRICT
);

CREATE TABLE proposal_reveals (
    reveal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    row_id TEXT NOT NULL REFERENCES source_rows(row_id) ON DELETE RESTRICT,
    proposal_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    revealed_at TEXT NOT NULL,
    timezone TEXT NOT NULL,
    details_json TEXT
);

CREATE INDEX annotation_events_row_index
ON annotation_events(row_id, revision);
CREATE INDEX qc_requirements_status_index
ON qc_requirements(required, status);
CREATE INDEX proposal_reveals_row_index
ON proposal_reveals(row_id, reveal_id);

CREATE TRIGGER workspace_metadata_no_update BEFORE UPDATE ON workspace_metadata
BEGIN SELECT RAISE(ABORT, 'workspace metadata is frozen'); END;
CREATE TRIGGER workspace_metadata_no_delete BEFORE DELETE ON workspace_metadata
BEGIN SELECT RAISE(ABORT, 'workspace metadata is frozen'); END;
CREATE TRIGGER sessions_no_update BEFORE UPDATE ON sessions
BEGIN SELECT RAISE(ABORT, 'sessions are append only'); END;
CREATE TRIGGER sessions_no_delete BEFORE DELETE ON sessions
BEGIN SELECT RAISE(ABORT, 'sessions are append only'); END;
CREATE TRIGGER source_rows_no_update BEFORE UPDATE ON source_rows
BEGIN SELECT RAISE(ABORT, 'source rows are frozen'); END;
CREATE TRIGGER source_rows_no_delete BEFORE DELETE ON source_rows
BEGIN SELECT RAISE(ABORT, 'source rows are frozen'); END;
CREATE TRIGGER annotation_events_no_update BEFORE UPDATE ON annotation_events
BEGIN SELECT RAISE(ABORT, 'annotation events are append only'); END;
CREATE TRIGGER annotation_events_no_delete BEFORE DELETE ON annotation_events
BEGIN SELECT RAISE(ABORT, 'annotation events are append only'); END;
CREATE TRIGGER proposal_reveals_no_update BEFORE UPDATE ON proposal_reveals
BEGIN SELECT RAISE(ABORT, 'proposal reveals are append only'); END;
CREATE TRIGGER proposal_reveals_no_delete BEFORE DELETE ON proposal_reveals
BEGIN SELECT RAISE(ABORT, 'proposal reveals are append only'); END;

PRAGMA user_version = 1;
COMMIT;
"""


def _migrate(connection: sqlite3.Connection) -> None:
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if user_version == SCHEMA_VERSION:
        return
    if user_version != 0:
        raise WorkbenchSchemaError("annotation workspace schema is unsupported")
    objects = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger', 'index') "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if objects:
        raise WorkbenchSchemaError("annotation workspace schema is unsupported")
    try:
        connection.executescript(_MIGRATION_V1)
    except sqlite3.DatabaseError as exc:
        if connection.in_transaction:
            connection.rollback()
        raise WorkbenchSchemaError("annotation workspace schema migration failed") from exc


def _schema_names(connection: sqlite3.Connection, object_type: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?",
            (object_type,),
        )
    }


def _validate_schema(connection: sqlite3.Connection) -> None:
    try:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        version_rows = connection.execute(
            "SELECT version FROM schema_version WHERE singleton = 1"
        ).fetchall()
        tables = _schema_names(connection, "table")
        triggers = _schema_names(connection, "trigger")
    except sqlite3.DatabaseError as exc:
        raise WorkbenchSchemaError("annotation workspace schema is invalid") from exc
    if user_version != SCHEMA_VERSION or version_rows != [(SCHEMA_VERSION,)]:
        raise WorkbenchSchemaError("annotation workspace schema is unsupported")
    if not _REQUIRED_TABLES.issubset(tables) or not _REQUIRED_TRIGGERS.issubset(triggers):
        raise WorkbenchSchemaError("annotation workspace schema is invalid")


def _parsed_mapping(text: str) -> dict[str, Any]:
    try:
        return _load_json(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkbenchCorruptionError("annotation workspace contains invalid JSON") from exc


def _parsed_source_mapping(text: str) -> dict[str, Any]:
    try:
        return _load_source_json(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkbenchCorruptionError("annotation workspace contains invalid JSON") from exc


def _parsed_payload_mapping(text: str) -> dict[str, Any]:
    try:
        return _load_json(text, reject_floats=True)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkbenchCorruptionError("annotation workspace contains invalid JSON") from exc


def _validate_workspace_binding(
    connection: sqlite3.Connection,
    expected_binding_json: str | None,
) -> tuple[str | None, int]:
    rows = connection.execute(
        "SELECT binding_json, binding_sha256, metadata_json, frozen FROM workspace_metadata"
    ).fetchall()
    if len(rows) > 1:
        raise WorkbenchCorruptionError("annotation workspace metadata is invalid")
    if not rows:
        return None, 0
    binding_json, binding_hash, metadata_json, frozen = rows[0]
    binding = _parsed_mapping(binding_json)
    metadata = _parsed_mapping(metadata_json)
    if (
        frozen != 1
        or _canonical_json(binding) != binding_json
        or _canonical_json(metadata) != metadata_json
        or _sha256_text(binding_json) != binding_hash
    ):
        raise WorkbenchCorruptionError("annotation workspace metadata is invalid")
    if expected_binding_json is not None and binding_json != expected_binding_json:
        raise WorkbenchBindingError("annotation workspace binding does not match")
    return str(binding_hash), 1


def _validate_contents(
    connection: sqlite3.Connection,
    expected_binding_json: str | None = None,
) -> BackupValidation:
    _quick_check(connection)
    _validate_schema(connection)
    binding_hash, workspace_count = _validate_workspace_binding(
        connection, expected_binding_json
    )
    try:
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise WorkbenchCorruptionError("annotation workspace references are invalid")
        source_rows = connection.execute(
            "SELECT row_id, position, reviewer_json, source_json, source_sha256 "
            "FROM source_rows ORDER BY position"
        ).fetchall()
        state_rows = connection.execute(
            "SELECT row_id, revision, projection_dirty, status, latest_event_id, phase, "
            "payload_json, reviewer, session_id, recorded_at, timezone, initial_event_id, "
            "initial_event_hash, qc_event_id, qc_event_hash, qc_for_initial_hash "
            "FROM current_state"
        ).fetchall()
        qc_rows = connection.execute(
            "SELECT row_id, required, status, revision, requirement_json, updated_at, "
            "timezone, reviewer, session_id FROM qc_requirements"
        ).fetchall()
        session_rows = connection.execute(
            "SELECT session_id, reviewer, started_at, timezone, metadata_json FROM sessions"
        ).fetchall()
        events = connection.execute(
            "SELECT event_id, chain_index, row_id, revision, phase, status, reviewer, "
            "session_id, recorded_at, timezone, payload_json, payload_sha256, previous_hash, "
            "event_hash FROM annotation_events ORDER BY chain_index"
        ).fetchall()
        reveals = connection.execute(
            "SELECT proposal_id, reviewer, session_id, revealed_at, timezone, details_json "
            "FROM proposal_reveals"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise WorkbenchCorruptionError("annotation workspace contents are invalid") from exc

    if bool(source_rows) != bool(workspace_count):
        raise WorkbenchCorruptionError("annotation workspace bootstrap is incomplete")
    if len(state_rows) != len(source_rows) or len(qc_rows) != len(source_rows):
        raise WorkbenchCorruptionError("annotation workspace projections are incomplete")
    if [row[1] for row in source_rows] != list(range(len(source_rows))):
        raise WorkbenchCorruptionError("annotation workspace row ordering is invalid")

    source_ids: set[str] = set()
    for row_id, _position, reviewer_json, source_json, source_hash in source_rows:
        reviewer_fields = _parsed_mapping(reviewer_json)
        source = _parsed_source_mapping(source_json)
        if (
            not isinstance(row_id, str)
            or not row_id
            or _canonical_json(reviewer_fields) != reviewer_json
            or _sha256_text(source_json) != source_hash
        ):
            raise WorkbenchCorruptionError("annotation workspace source rows are invalid")
        source_ids.add(row_id)
        del source

    sessions: dict[str, str] = {}
    for session_id, reviewer, started_at, timezone_name, metadata_json in session_rows:
        try:
            _token(session_id)
            _token(reviewer)
            _token(timezone_name)
            _aware_timestamp(started_at)
            metadata = _parsed_mapping(metadata_json)
        except (TypeError, ValueError) as exc:
            raise WorkbenchCorruptionError("annotation workspace sessions are invalid") from exc
        if _canonical_json(metadata) != metadata_json:
            raise WorkbenchCorruptionError("annotation workspace sessions are invalid")
        sessions[session_id] = reviewer

    event_by_id: dict[int, tuple[Any, ...]] = {}
    latest_by_row: dict[str, tuple[Any, ...]] = {}
    row_revisions: dict[str, int] = {}
    previous_hash: str | None = None
    for expected_chain, event in enumerate(events, start=1):
        (
            event_id,
            chain_index,
            row_id,
            revision,
            phase,
            status,
            reviewer,
            session_id,
            recorded_at,
            timezone_name,
            payload_json,
            payload_hash,
            stored_previous,
            stored_hash,
        ) = event
        try:
            payload = _parsed_payload_mapping(payload_json)
            _aware_timestamp(recorded_at)
            _token(timezone_name)
        except (TypeError, ValueError) as exc:
            raise WorkbenchCorruptionError("annotation workspace history is invalid") from exc
        expected_revision = row_revisions.get(row_id, 0) + 1
        calculated = _event_hash(
            chain_index=chain_index,
            row_id=row_id,
            revision=revision,
            phase=phase,
            status=status,
            reviewer=reviewer,
            session_id=session_id,
            recorded_at=recorded_at,
            timezone_name=timezone_name,
            payload_json=payload_json,
            payload_sha256=payload_hash,
            previous_hash=stored_previous,
        )
        if (
            chain_index != expected_chain
            or row_id not in source_ids
            or revision != expected_revision
            or phase not in KNOWN_PHASES
            or status not in KNOWN_STATUSES
            or sessions.get(session_id) != reviewer
            or _sha256_text(payload_json) != payload_hash
            or stored_previous != previous_hash
            or calculated != stored_hash
        ):
            raise WorkbenchCorruptionError("annotation workspace history is invalid")
        del payload
        previous_hash = stored_hash
        row_revisions[row_id] = revision
        event_by_id[event_id] = event
        latest_by_row[row_id] = event

    states = {row[0]: row for row in state_rows}
    for row_id in source_ids:
        state = states.get(row_id)
        if state is None or state[2] not in {0, 1} or state[3] not in KNOWN_STATUSES:
            raise WorkbenchCorruptionError("annotation workspace projections are invalid")
        latest = latest_by_row.get(row_id)
        if latest is None:
            if state[1] != 0 or state[3] != "pending" or any(state[index] is not None for index in range(4, 16)):
                raise WorkbenchCorruptionError("annotation workspace projections are invalid")
            continue
        if (
            state[1] != latest[3]
            or state[3] != latest[5]
            or state[4] != latest[0]
            or state[5] != latest[4]
            or state[6] != latest[10]
            or state[7] != latest[6]
            or state[8] != latest[7]
            or state[9] != latest[8]
            or state[10] != latest[9]
        ):
            raise WorkbenchCorruptionError("annotation workspace projections are invalid")
        initial_id, initial_hash, qc_id, qc_hash, qc_for_initial = state[11:16]
        if initial_id is None:
            if any(value is not None for value in (initial_hash, qc_id, qc_hash, qc_for_initial)):
                raise WorkbenchCorruptionError("annotation workspace QC projection is invalid")
        else:
            initial = event_by_id.get(initial_id)
            if initial is None or initial[2] != row_id or initial[4] != "initial" or initial[13] != initial_hash:
                raise WorkbenchCorruptionError("annotation workspace QC projection is invalid")
            if qc_id is None:
                if qc_hash is not None or qc_for_initial is not None:
                    raise WorkbenchCorruptionError("annotation workspace QC projection is invalid")
            else:
                qc = event_by_id.get(qc_id)
                if (
                    qc is None
                    or qc[2] != row_id
                    or qc[4] != "qc"
                    or qc[13] != qc_hash
                    or qc_for_initial != initial_hash
                ):
                    raise WorkbenchCorruptionError("annotation workspace QC projection is invalid")

    if {row[0] for row in qc_rows} != source_ids:
        raise WorkbenchCorruptionError("annotation workspace QC requirements are invalid")
    for _row_id, required, status_value, revision, requirement_json, *audit in qc_rows:
        if (
            required not in {0, 1}
            or status_value not in QC_STATUSES
            or not isinstance(revision, int)
            or revision < 0
            or (required == 0 and status_value != "not_required")
            or (required == 1 and status_value == "not_required")
        ):
            raise WorkbenchCorruptionError("annotation workspace QC requirements are invalid")
        if requirement_json is not None:
            _parsed_mapping(requirement_json)
        updated_at, timezone_name, reviewer, session_id = audit
        if updated_at is None:
            if any(value is not None for value in (timezone_name, reviewer, session_id)):
                raise WorkbenchCorruptionError("annotation workspace QC requirements are invalid")
        else:
            try:
                _aware_timestamp(updated_at)
                _token(timezone_name)
            except (TypeError, ValueError) as exc:
                raise WorkbenchCorruptionError("annotation workspace QC requirements are invalid") from exc
            if session_id is not None and sessions.get(session_id) != reviewer:
                raise WorkbenchCorruptionError("annotation workspace QC requirements are invalid")

    for proposal_id, reviewer, session_id, revealed_at, timezone_name, details_json in reveals:
        try:
            _token(proposal_id)
            _aware_timestamp(revealed_at)
            _token(timezone_name)
        except (TypeError, ValueError) as exc:
            raise WorkbenchCorruptionError("annotation workspace reveal audit is invalid") from exc
        if sessions.get(session_id) != reviewer:
            raise WorkbenchCorruptionError("annotation workspace reveal audit is invalid")
        if details_json is not None:
            _parsed_mapping(details_json)

    return BackupValidation(
        path=Path(),
        schema_version=SCHEMA_VERSION,
        row_count=len(source_rows),
        event_count=len(events),
        binding_sha256=binding_hash,
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backup_name(db_path: Path, marker: str = "snapshot") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{db_path.name}.{stamp}.{uuid.uuid4().hex}.{marker}.bak"


def _backup_files(paths: StorePaths) -> list[Path]:
    if not paths.backups.is_dir():
        return []
    return sorted(
        (
            candidate
            for candidate in paths.backups.iterdir()
            if candidate.is_file()
            and not candidate.is_symlink()
            and candidate.name.startswith(f"{paths.database.name}.")
            and candidate.name.endswith(".bak")
        ),
        key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
    )


def _prune_backups(paths: StorePaths) -> None:
    files = _backup_files(paths)
    removed = False
    for candidate in files[:-MAX_BACKUPS]:
        candidate.unlink()
        removed = True
    if removed:
        _fsync_directory(paths.backups)


def _sqlite_backup(connection: sqlite3.Connection, paths: StorePaths) -> Path:
    _ensure_secure_directory(paths.backups)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{paths.database.name}.", suffix=".tmp", dir=paths.backups
    )
    temporary = Path(temporary_name)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    target = paths.backups / _backup_name(paths.database)
    destination: sqlite3.Connection | None = None
    try:
        destination = sqlite3.connect(temporary, timeout=0, isolation_level=None)
        connection.backup(destination)
        destination.close()
        destination = None
        temporary.chmod(0o600)
        _fsync_file(temporary)
        os.replace(temporary, target)
        target.chmod(0o600)
        _fsync_directory(paths.backups)
        _prune_backups(paths)
        return target
    except (OSError, sqlite3.DatabaseError) as exc:
        if destination is not None:
            destination.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise WorkbenchStoreError("annotation workspace backup could not be created") from exc


def _atomic_copy(source: Path, target: Path, directory: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    os.fchmod(descriptor, 0o600)
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_descriptor = os.open(source, source_flags)
        try:
            with os.fdopen(source_descriptor, "rb", closefd=True) as source_file:
                with os.fdopen(descriptor, "wb", closefd=True) as target_file:
                    descriptor = -1
                    shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
                    target_file.flush()
                    os.fsync(target_file.fileno())
        finally:
            source_descriptor = -1
        os.replace(temporary, target)
        target.chmod(0o600)
        _fsync_directory(directory)
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validated_backup_path(paths: StorePaths, backup_path: os.PathLike[str] | str) -> Path:
    candidate = _absolute_path(backup_path)
    try:
        candidate.relative_to(paths.backups)
    except ValueError as exc:
        raise WorkbenchStoreError("annotation backup path is outside the backup directory") from exc
    if candidate.parent != paths.backups:
        raise WorkbenchStoreError("annotation backup path is outside the backup directory")
    _require_regular_file(candidate, "annotation backup is unavailable")
    return candidate


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _validate_database_file(
    path: Path,
    expected_binding_json: str | None,
) -> BackupValidation:
    connection: sqlite3.Connection | None = None
    try:
        connection = _read_only_connection(path)
        result = _validate_contents(connection, expected_binding_json)
        return BackupValidation(
            path=path,
            schema_version=result.schema_version,
            row_count=result.row_count,
            event_count=result.event_count,
            binding_sha256=result.binding_sha256,
        )
    except (WorkbenchStoreError, sqlite3.DatabaseError) as exc:
        if isinstance(exc, WorkbenchBindingError):
            raise
        raise WorkbenchCorruptionError("annotation backup is invalid") from exc
    finally:
        if connection is not None:
            connection.close()


def validate_backup(
    db_path: os.PathLike[str] | str,
    backup_path: os.PathLike[str] | str,
    *,
    workspace_binding: Mapping[str, Any] | None = None,
) -> BackupValidation:
    """Validate a backup while holding the same exclusive workspace lock."""

    paths = store_paths(db_path)
    expected = _canonical_json(workspace_binding) if workspace_binding is not None else None
    with _ExclusiveFileLock(paths):
        candidate = _validated_backup_path(paths, backup_path)
        return _validate_database_file(candidate, expected)


def recover_backup(
    db_path: os.PathLike[str] | str,
    backup_path: os.PathLike[str] | str,
    *,
    workspace_binding: Mapping[str, Any] | None = None,
) -> Path | None:
    """Atomically recover a validated backup and preserve the displaced database."""

    paths = store_paths(db_path)
    expected = _canonical_json(workspace_binding) if workspace_binding is not None else None
    with _ExclusiveFileLock(paths):
        _ensure_secure_directory(paths.backups)
        candidate = _validated_backup_path(paths, backup_path)
        _validate_database_file(candidate, expected)
        displaced: Path | None = None
        try:
            if paths.database.exists():
                _require_regular_file(paths.database, "annotation database path is unsafe")
                displaced = paths.backups / _backup_name(paths.database, "displaced-current")
                _atomic_copy(paths.database, displaced, paths.backups)
            _atomic_copy(candidate, paths.database, paths.database.parent)
            result = _validate_database_file(paths.database, expected)
            if result.schema_version != SCHEMA_VERSION:
                raise WorkbenchCorruptionError("recovered annotation workspace is invalid")
            _prune_backups(paths)
            return displaced
        except WorkbenchStoreError:
            raise
        except OSError as exc:
            raise WorkbenchStoreError("annotation workspace recovery failed") from exc


class WorkbenchStore:
    """Single-writer, restart-safe annotation workspace persistence."""

    def __init__(
        self,
        db_path: os.PathLike[str] | str,
        *,
        workspace_binding: Mapping[str, Any] | None = None,
    ) -> None:
        self.paths = store_paths(db_path)
        self._expected_binding_json = (
            _canonical_json(workspace_binding) if workspace_binding is not None else None
        )
        self._mutex = threading.RLock()
        self._transaction_depth = 0
        self._lock = _ExclusiveFileLock(self.paths)
        self._connection: sqlite3.Connection | None = None
        self.last_backup_path: Path | None = None
        self._lock.acquire()
        try:
            _secure_database_file(self.paths.database)
            connection = sqlite3.connect(
                self.paths.database,
                timeout=0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection = connection
            _configure_connection(connection)
            _quick_check(connection)
            _migrate(connection)
            _validate_contents(connection, self._expected_binding_json)
            self.last_backup_path = _sqlite_backup(connection, self.paths)
            self.paths.database.chmod(0o600)
        except WorkbenchStoreError:
            self.close()
            raise
        except (OSError, sqlite3.DatabaseError) as exc:
            self.close()
            raise WorkbenchStoreError("annotation workspace could not be opened safely") from exc

    def __enter__(self) -> WorkbenchStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            connection, self._connection = self._connection, None
            if connection is not None:
                try:
                    connection.close()
                finally:
                    self._lock.release()
            else:
                self._lock.release()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise WorkbenchStoreError("annotation workspace is closed")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            connection = self._require_connection()
            nested = self._transaction_depth > 0
            savepoint = f"workbench_{uuid.uuid4().hex}" if nested else None

            def rollback() -> None:
                if not connection.in_transaction:
                    return
                if savepoint is None:
                    connection.rollback()
                    return
                try:
                    connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                except sqlite3.DatabaseError:
                    if connection.in_transaction:
                        connection.rollback()

            started = False
            try:
                if nested:
                    if not connection.in_transaction:
                        raise WorkbenchCorruptionError(
                            "annotation workspace transaction state is invalid"
                        )
                    connection.execute(f"SAVEPOINT {savepoint}")
                else:
                    if connection.in_transaction:
                        raise WorkbenchCorruptionError(
                            "annotation workspace transaction state is invalid"
                        )
                    connection.execute("BEGIN IMMEDIATE")
                self._transaction_depth += 1
                started = True
                yield connection
                if savepoint is None:
                    connection.commit()
                else:
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except WorkbenchStoreError:
                rollback()
                raise
            except sqlite3.DatabaseError as exc:
                rollback()
                raise WorkbenchStoreError("annotation workspace write failed") from exc
            except BaseException:
                rollback()
                raise
            finally:
                if started:
                    self._transaction_depth -= 1

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            connection = self._require_connection()
            try:
                yield connection
            except WorkbenchStoreError:
                raise
            except sqlite3.DatabaseError as exc:
                raise WorkbenchStoreError("annotation workspace read failed") from exc

    def create_backup(self) -> Path:
        """Create an additional crash-safe snapshot while this store holds the lock."""

        with self._mutex:
            return _sqlite_backup(self._require_connection(), self.paths)

    def verify_integrity(self) -> dict[str, Any]:
        """Re-run physical, logical, foreign-key, and audit-chain validation."""

        with self._read() as connection:
            result = _validate_contents(connection, self._expected_binding_json)
        return {
            "schema_version": result.schema_version,
            "row_count": result.row_count,
            "event_count": result.event_count,
            "binding_sha256": result.binding_sha256,
        }

    @staticmethod
    def _normalize_source_rows(
        rows: Sequence[SourceRow | Mapping[str, Any]],
    ) -> list[tuple[str, int, str, str, str]]:
        normalized: list[tuple[str, int, str, str, str]] = []
        seen_ids: set[str] = set()
        positions: set[int] = set()
        for fallback_position, item in enumerate(rows):
            try:
                if isinstance(item, SourceRow):
                    row_id = _token(item.row_id)
                    position = fallback_position if item.position is None else item.position
                    reviewer_fields = item.reviewer_fields
                    source_value = item.source_json
                elif isinstance(item, Mapping):
                    row_id = _token(item["row_id"])
                    position = item.get("position", fallback_position)
                    reviewer_fields = item["reviewer_fields"]
                    if "source_json" in item:
                        source_value = item["source_json"]
                    else:
                        source_value = item["source"]
                else:
                    raise ValueError
                if (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position < 0
                    or not isinstance(reviewer_fields, Mapping)
                ):
                    raise ValueError
                reviewer_json = _canonical_json(reviewer_fields)
                source_json, _source = _exact_or_canonical_source_json(source_value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise WorkbenchStoreError("annotation source row is invalid") from exc
            if row_id in seen_ids or position in positions:
                raise WorkbenchStoreError("annotation source rows contain duplicate identifiers")
            seen_ids.add(row_id)
            positions.add(position)
            normalized.append(
                (row_id, position, reviewer_json, source_json, _sha256_text(source_json))
            )
        normalized.sort(key=lambda row: row[1])
        if [row[1] for row in normalized] != list(range(len(normalized))):
            raise WorkbenchStoreError("annotation source row positions must be contiguous")
        return normalized

    def bootstrap(
        self,
        source_rows: Sequence[SourceRow | Mapping[str, Any]],
        *,
        workspace_binding: Mapping[str, Any],
        workspace_metadata: Mapping[str, Any] | None = None,
        created_at: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Freeze a workspace binding and stable ordered source rows transactionally."""

        try:
            binding_json = _canonical_json(workspace_binding)
            metadata_json = _canonical_json(workspace_metadata or {})
            timestamp, _ = _aware_timestamp(created_at)
        except (TypeError, ValueError) as exc:
            raise WorkbenchStoreError("annotation workspace metadata is invalid") from exc
        if self._expected_binding_json is not None and binding_json != self._expected_binding_json:
            raise WorkbenchBindingError("annotation workspace binding does not match")
        normalized = self._normalize_source_rows(source_rows)
        if not normalized:
            raise WorkbenchStoreError("annotation workspace requires source rows")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT binding_json, metadata_json FROM workspace_metadata WHERE singleton = 1"
            ).fetchone()
            if existing is not None:
                if existing != (binding_json, metadata_json):
                    raise WorkbenchBindingError("annotation workspace binding does not match")
                stored = connection.execute(
                    "SELECT row_id, position, reviewer_json, source_json, source_sha256 "
                    "FROM source_rows ORDER BY position"
                ).fetchall()
                if stored != normalized:
                    raise WorkbenchBindingError("annotation workspace source binding does not match")
                return {
                    "created": False,
                    "row_count": len(stored),
                    "binding_sha256": _sha256_text(binding_json),
                }
            if connection.execute("SELECT 1 FROM source_rows LIMIT 1").fetchone() is not None:
                raise WorkbenchCorruptionError("annotation workspace bootstrap is incomplete")
            connection.execute(
                "INSERT INTO workspace_metadata(singleton, binding_json, binding_sha256, "
                "metadata_json, created_at, frozen) VALUES (1, ?, ?, ?, ?, 1)",
                (binding_json, _sha256_text(binding_json), metadata_json, timestamp),
            )
            connection.executemany(
                "INSERT INTO source_rows(row_id, position, reviewer_json, source_json, "
                "source_sha256) VALUES (?, ?, ?, ?, ?)",
                normalized,
            )
            connection.executemany(
                "INSERT INTO current_state(row_id) VALUES (?)",
                ((row[0],) for row in normalized),
            )
            connection.executemany(
                "INSERT INTO qc_requirements(row_id) VALUES (?)",
                ((row[0],) for row in normalized),
            )
        self._expected_binding_json = binding_json
        return {
            "created": True,
            "row_count": len(normalized),
            "binding_sha256": _sha256_text(binding_json),
        }

    def workspace_metadata(self) -> dict[str, Any] | None:
        """Return parsed frozen metadata and binding, or ``None`` before bootstrap."""

        with self._read() as connection:
            row = connection.execute(
                "SELECT binding_json, binding_sha256, metadata_json, created_at, frozen "
                "FROM workspace_metadata WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "binding": _parsed_mapping(row[0]),
            "binding_sha256": row[1],
            "metadata": _parsed_mapping(row[2]),
            "created_at": row[3],
            "frozen": bool(row[4]),
        }

    def start_session(
        self,
        reviewer: str,
        *,
        session_id: str | None = None,
        started_at: str | datetime | None = None,
        timezone_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an immutable reviewer session."""

        try:
            reviewer = _token(reviewer)
            session_id = _token(session_id or f"session-{uuid.uuid4().hex}")
            timestamp, timezone_name = _timestamp_and_zone(started_at, timezone_name)
            metadata_json = _canonical_json(metadata or {})
        except (TypeError, ValueError) as exc:
            raise WorkbenchStoreError("annotation session metadata is invalid") from exc
        with self._transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO sessions(session_id, reviewer, started_at, timezone, "
                    "metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (session_id, reviewer, timestamp, timezone_name, metadata_json),
                )
            except sqlite3.IntegrityError as exc:
                raise WorkbenchStoreError("annotation session could not be started") from exc
        return {
            "session_id": session_id,
            "reviewer": reviewer,
            "started_at": timestamp,
            "timezone": timezone_name,
            "metadata": _parsed_mapping(metadata_json),
        }

    @staticmethod
    def _session_reviewer(connection: sqlite3.Connection, session_id: str) -> str:
        row = connection.execute(
            "SELECT reviewer FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise WorkbenchStoreError("annotation session is invalid")
        return str(row[0])

    @staticmethod
    def _row_result(row: tuple[Any, ...], include_private: bool) -> dict[str, Any]:
        result = {
            "row_id": row[0],
            "position": row[1],
            "reviewer_fields": _parsed_mapping(row[2]),
            "revision": row[4],
            "projection_dirty": bool(row[5]),
            "status": row[6],
            "phase": row[7],
            "annotation": None if row[8] is None else _parsed_mapping(row[8]),
            "reviewer": row[9],
            "session_id": row[10],
            "recorded_at": row[11],
            "timezone": row[12],
            "initial_event_id": row[13],
            "initial_event_hash": row[14],
            "qc_event_id": row[15],
            "qc_event_hash": row[16],
            "qc_for_initial_hash": row[17],
            "qc_required": bool(row[18]),
            "qc_status": row[19],
            "qc_revision": row[20],
        }
        if include_private:
            result["source"] = _parsed_source_mapping(row[3])
        return result

    _ROW_QUERY: Final = (
        "SELECT s.row_id, s.position, s.reviewer_json, s.source_json, c.revision, "
        "c.projection_dirty, c.status, c.phase, c.payload_json, c.reviewer, c.session_id, "
        "c.recorded_at, c.timezone, c.initial_event_id, c.initial_event_hash, "
        "c.qc_event_id, c.qc_event_hash, c.qc_for_initial_hash, q.required, q.status, "
        "q.revision FROM source_rows AS s JOIN current_state AS c ON c.row_id = s.row_id "
        "JOIN qc_requirements AS q ON q.row_id = s.row_id"
    )

    def get_rows(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        status: str | None = None,
        qc_only: bool = False,
        projection_dirty: bool | None = None,
        include_private: bool = False,
    ) -> list[dict[str, Any]]:
        """Return rows in frozen position order; raw source requires an explicit opt-in."""

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise WorkbenchStoreError("annotation row page is invalid")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise WorkbenchStoreError("annotation row page is invalid")
        if status is not None and status not in KNOWN_STATUSES:
            raise WorkbenchStoreError("annotation status filter is invalid")
        if projection_dirty is not None and not isinstance(projection_dirty, bool):
            raise WorkbenchStoreError("annotation projection filter is invalid")
        clauses: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            clauses.append("c.status = ?")
            parameters.append(status)
        if qc_only:
            clauses.append("q.required = 1")
        if projection_dirty is not None:
            clauses.append("c.projection_dirty = ?")
            parameters.append(int(projection_dirty))
        query = self._ROW_QUERY
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY s.position LIMIT ? OFFSET ?"
        parameters.extend((-1 if limit is None else limit, offset))
        with self._read() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_result(row, include_private) for row in rows]

    def get_row(self, row_id: str, *, include_private: bool = False) -> dict[str, Any]:
        """Return one parsed row without ever embedding its content in errors."""

        try:
            row_id = _token(row_id)
        except ValueError as exc:
            raise WorkbenchStoreError("annotation row was not found") from exc
        with self._read() as connection:
            row = connection.execute(
                self._ROW_QUERY + " WHERE s.row_id = ?", (row_id,)
            ).fetchone()
        if row is None:
            raise WorkbenchStoreError("annotation row was not found")
        return self._row_result(row, include_private)

    def get_progress(self) -> dict[str, Any]:
        """Return aggregate progress only, with no reviewer or source content."""

        with self._read() as connection:
            total = int(connection.execute("SELECT count(*) FROM source_rows").fetchone()[0])
            event_count = int(
                connection.execute("SELECT count(*) FROM annotation_events").fetchone()[0]
            )
            dirty_rows = int(
                connection.execute(
                    "SELECT count(*) FROM current_state WHERE projection_dirty = 1"
                ).fetchone()[0]
            )
            by_status = {
                str(status): int(count)
                for status, count in connection.execute(
                    "SELECT status, count(*) FROM current_state GROUP BY status"
                )
            }
            by_phase = {
                str(phase): int(count)
                for phase, count in connection.execute(
                    "SELECT coalesce(phase, 'none'), count(*) FROM current_state GROUP BY phase"
                )
            }
            qc = {
                str(status): int(count)
                for status, count in connection.execute(
                    "SELECT status, count(*) FROM qc_requirements GROUP BY status"
                )
            }
        return {
            "total_rows": total,
            "event_count": event_count,
            "dirty_rows": dirty_rows,
            "by_status": by_status,
            "by_phase": by_phase,
            "qc": qc,
        }

    def save_annotation(
        self,
        row_id: str,
        *,
        expected_revision: int,
        phase: str,
        payload: str | Mapping[str, Any],
        session_id: str,
        reviewer: str | None = None,
        status: str | None = None,
        recorded_at: str | datetime | None = None,
        timezone_name: str | None = None,
        projection_required: bool = False,
    ) -> dict[str, Any]:
        """Append an event and update its projection with optimistic concurrency."""

        try:
            row_id = _token(row_id)
            session_id = _token(session_id)
            phase = _token(phase)
            if phase not in KNOWN_PHASES:
                raise ValueError
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
                raise ValueError
            if not isinstance(projection_required, bool):
                raise ValueError
            status = _default_status(phase) if status is None else _token(status)
            if status not in KNOWN_STATUSES:
                raise ValueError
            payload_json, parsed_payload = _exact_or_canonical_payload(payload)
            timestamp, timezone_name = _timestamp_and_zone(recorded_at, timezone_name)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkbenchStoreError("annotation event is invalid") from exc

        with self._transaction() as connection:
            source_row = connection.execute(
                "SELECT source_json FROM source_rows WHERE row_id = ?", (row_id,)
            ).fetchone()
            if source_row is None:
                raise WorkbenchStoreError("annotation row was not found")
            try:
                _validate_payload_spans(parsed_payload, _parsed_source_mapping(source_row[0]))
            except (TypeError, ValueError) as exc:
                raise WorkbenchStoreError("annotation source span is invalid") from exc
            session_reviewer = self._session_reviewer(connection, session_id)
            if reviewer is not None:
                try:
                    reviewer = _token(reviewer)
                except ValueError as exc:
                    raise WorkbenchStoreError("annotation reviewer is invalid") from exc
                if reviewer != session_reviewer:
                    raise WorkbenchStoreError("annotation reviewer does not match the session")
            reviewer = session_reviewer
            state = connection.execute(
                "SELECT revision, projection_dirty, initial_event_id, initial_event_hash, "
                "qc_event_id, qc_event_hash, qc_for_initial_hash FROM current_state "
                "WHERE row_id = ?",
                (row_id,),
            ).fetchone()
            if state is None:
                raise WorkbenchStoreError("annotation row was not found")
            if state[0] != expected_revision:
                raise StaleRevisionError("annotation revision is stale")
            if state[1] != 0:
                raise WorkbenchStoreError(
                    "annotation projection must be reconciled before another save"
                )
            initial_id, initial_hash, qc_id, qc_hash, qc_for_initial = state[2:7]
            if phase == "qc" and initial_id is None:
                raise WorkbenchStoreError("QC requires an initial annotation")
            latest_chain = connection.execute(
                "SELECT chain_index, event_hash FROM annotation_events "
                "ORDER BY chain_index DESC LIMIT 1"
            ).fetchone()
            chain_index = 1 if latest_chain is None else int(latest_chain[0]) + 1
            previous_hash = None if latest_chain is None else str(latest_chain[1])
            revision = expected_revision + 1
            payload_hash = _sha256_text(payload_json)
            digest = _event_hash(
                chain_index=chain_index,
                row_id=row_id,
                revision=revision,
                phase=phase,
                status=status,
                reviewer=reviewer,
                session_id=session_id,
                recorded_at=timestamp,
                timezone_name=timezone_name,
                payload_json=payload_json,
                payload_sha256=payload_hash,
                previous_hash=previous_hash,
            )
            changed = connection.execute(
                "UPDATE current_state SET projection_dirty = 1 "
                "WHERE row_id = ? AND revision = ? AND projection_dirty = 0",
                (row_id, expected_revision),
            ).rowcount
            if changed != 1:
                raise StaleRevisionError("annotation revision is stale")
            cursor = connection.execute(
                "INSERT INTO annotation_events(chain_index, row_id, revision, phase, status, "
                "reviewer, session_id, recorded_at, timezone, payload_json, payload_sha256, "
                "previous_hash, event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chain_index,
                    row_id,
                    revision,
                    phase,
                    status,
                    reviewer,
                    session_id,
                    timestamp,
                    timezone_name,
                    payload_json,
                    payload_hash,
                    previous_hash,
                    digest,
                ),
            )
            event_id = int(cursor.lastrowid)
            if phase == "initial":
                initial_id, initial_hash = event_id, digest
                qc_id = qc_hash = qc_for_initial = None
                connection.execute(
                    "UPDATE qc_requirements SET status = 'pending', revision = revision + 1, "
                    "updated_at = ?, timezone = ?, reviewer = ?, session_id = ? "
                    "WHERE row_id = ? AND required = 1",
                    (timestamp, timezone_name, reviewer, session_id, row_id),
                )
            elif phase == "qc":
                qc_id, qc_hash, qc_for_initial = event_id, digest, initial_hash
            updated = connection.execute(
                "UPDATE current_state SET revision = ?, projection_dirty = ?, status = ?, "
                "latest_event_id = ?, phase = ?, payload_json = ?, reviewer = ?, "
                "session_id = ?, recorded_at = ?, timezone = ?, initial_event_id = ?, "
                "initial_event_hash = ?, qc_event_id = ?, qc_event_hash = ?, "
                "qc_for_initial_hash = ? WHERE row_id = ? AND revision = ? "
                "AND projection_dirty = 1",
                (
                    revision,
                    int(projection_required),
                    status,
                    event_id,
                    phase,
                    payload_json,
                    reviewer,
                    session_id,
                    timestamp,
                    timezone_name,
                    initial_id,
                    initial_hash,
                    qc_id,
                    qc_hash,
                    qc_for_initial,
                    row_id,
                    expected_revision,
                ),
            ).rowcount
            if updated != 1:
                raise WorkbenchCorruptionError("annotation projection update failed")
        return {
            "event_id": event_id,
            "chain_index": chain_index,
            "revision": revision,
            "phase": phase,
            "status": status,
            "reviewer": reviewer,
            "session_id": session_id,
            "recorded_at": timestamp,
            "timezone": timezone_name,
            "payload": parsed_payload,
            "payload_sha256": payload_hash,
            "previous_hash": previous_hash,
            "event_hash": digest,
            "initial_event_id": initial_id,
            "initial_event_hash": initial_hash,
            "qc_event_id": qc_id,
            "qc_event_hash": qc_hash,
            "qc_for_initial_hash": qc_for_initial,
            "projection_dirty": projection_required,
        }

    def get_dirty_rows(self, *, include_private: bool = False) -> list[dict[str, Any]]:
        """Return durable events that still require cross-artifact projection."""

        return self.get_rows(
            projection_dirty=True,
            include_private=include_private,
        )

    def mark_projection_clean(
        self,
        row_id: str,
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Optimistically mark one externally reconciled projection clean."""

        try:
            row_id = _token(row_id)
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
                raise ValueError
        except ValueError as exc:
            raise WorkbenchStoreError(
                "annotation projection reconciliation is invalid"
            ) from exc
        with self._transaction() as connection:
            state = connection.execute(
                "SELECT revision, projection_dirty FROM current_state WHERE row_id = ?",
                (row_id,),
            ).fetchone()
            if state is None:
                raise WorkbenchStoreError("annotation row was not found")
            if state[0] != expected_revision:
                raise StaleRevisionError("annotation revision is stale")
            if state[1] == 1:
                changed = connection.execute(
                    "UPDATE current_state SET projection_dirty = 0 "
                    "WHERE row_id = ? AND revision = ? AND projection_dirty = 1",
                    (row_id, expected_revision),
                ).rowcount
                if changed != 1:
                    raise StaleRevisionError("annotation revision is stale")
        return self.get_row(row_id)

    def get_history(self, row_id: str) -> list[dict[str, Any]]:
        """Return parsed append-only history in row revision order."""

        try:
            row_id = _token(row_id)
        except ValueError as exc:
            raise WorkbenchStoreError("annotation row was not found") from exc
        with self._read() as connection:
            if connection.execute(
                "SELECT 1 FROM source_rows WHERE row_id = ?", (row_id,)
            ).fetchone() is None:
                raise WorkbenchStoreError("annotation row was not found")
            rows = connection.execute(
                "SELECT event_id, chain_index, revision, phase, status, reviewer, session_id, "
                "recorded_at, timezone, payload_json, payload_sha256, previous_hash, event_hash "
                "FROM annotation_events WHERE row_id = ? ORDER BY revision",
                (row_id,),
            ).fetchall()
        return [
            {
                "event_id": row[0],
                "chain_index": row[1],
                "revision": row[2],
                "phase": row[3],
                "status": row[4],
                "reviewer": row[5],
                "session_id": row[6],
                "recorded_at": row[7],
                "timezone": row[8],
                "payload": _parsed_mapping(row[9]),
                "payload_sha256": row[10],
                "previous_hash": row[11],
                "event_hash": row[12],
            }
            for row in rows
        ]

    def set_qc_requirement(
        self,
        row_id: str,
        *,
        required: bool,
        requirement: str | Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        session_id: str | None = None,
        reviewer: str | None = None,
        recorded_at: str | datetime | None = None,
        timezone_name: str | None = None,
    ) -> dict[str, Any]:
        """Set or clear one QC requirement transactionally."""

        try:
            row_id = _token(row_id)
            if not isinstance(required, bool):
                raise ValueError
            if expected_revision is not None and (
                isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
            ):
                raise ValueError
            requirement_json = (
                None if requirement is None else _exact_or_canonical_json(requirement)[0]
            )
            timestamp, timezone_name = _timestamp_and_zone(recorded_at, timezone_name)
            if session_id is not None:
                session_id = _token(session_id)
            if reviewer is not None:
                reviewer = _token(reviewer)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkbenchStoreError("QC requirement is invalid") from exc
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT revision FROM qc_requirements WHERE row_id = ?", (row_id,)
            ).fetchone()
            if current is None:
                raise WorkbenchStoreError("annotation row was not found")
            if expected_revision is not None and current[0] != expected_revision:
                raise StaleRevisionError("QC requirement revision is stale")
            if session_id is not None:
                session_reviewer = self._session_reviewer(connection, session_id)
                if reviewer is not None and reviewer != session_reviewer:
                    raise WorkbenchStoreError("QC reviewer does not match the session")
                reviewer = session_reviewer
            elif reviewer is not None:
                reviewer = _token(reviewer)
            revision = int(current[0]) + 1
            status_value = "pending" if required else "not_required"
            connection.execute(
                "UPDATE qc_requirements SET required = ?, status = ?, revision = ?, "
                "requirement_json = ?, updated_at = ?, timezone = ?, reviewer = ?, "
                "session_id = ? WHERE row_id = ?",
                (
                    int(required),
                    status_value,
                    revision,
                    requirement_json,
                    timestamp,
                    timezone_name,
                    reviewer,
                    session_id,
                    row_id,
                ),
            )
        return self.get_qc_status(row_id)

    def set_qc_requirements(
        self,
        requirements: Sequence[Mapping[str, Any]],
        *,
        session_id: str,
        reviewer: str | None = None,
        recorded_at: str | datetime | None = None,
        timezone_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Persist one complete QC policy as a single transaction."""

        expected_fields = {
            "row_id",
            "required",
            "requirement",
            "expected_revision",
        }
        normalized: list[tuple[str, bool, str | None, int]] = []
        seen_ids: set[str] = set()
        try:
            session_id = _token(session_id)
            if reviewer is not None:
                reviewer = _token(reviewer)
            timestamp, timezone_name = _timestamp_and_zone(
                recorded_at, timezone_name
            )
            for item in requirements:
                if not isinstance(item, Mapping) or set(item) != expected_fields:
                    raise ValueError
                row_id = _token(item["row_id"])
                required = item["required"]
                expected_revision = item["expected_revision"]
                if (
                    not isinstance(required, bool)
                    or isinstance(expected_revision, bool)
                    or not isinstance(expected_revision, int)
                    or expected_revision < 0
                    or row_id in seen_ids
                ):
                    raise ValueError
                raw_requirement = item["requirement"]
                requirement_json: str | None = None
                if raw_requirement is not None:
                    requirement_json, parsed = _exact_or_canonical_json(
                        raw_requirement
                    )
                    if not isinstance(parsed, Mapping):
                        raise ValueError
                if (required and requirement_json is None) or (
                    not required and requirement_json is not None
                ):
                    raise ValueError
                seen_ids.add(row_id)
                normalized.append(
                    (row_id, required, requirement_json, expected_revision)
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkbenchStoreError("QC requirement batch is invalid") from exc

        with self._transaction() as connection:
            workspace_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT row_id FROM qc_requirements"
                ).fetchall()
            }
            if seen_ids != workspace_ids:
                raise WorkbenchStoreError(
                    "QC requirement batch does not cover the workspace"
                )
            session_reviewer = self._session_reviewer(connection, session_id)
            if reviewer is not None and reviewer != session_reviewer:
                raise WorkbenchStoreError("QC reviewer does not match the session")
            reviewer = session_reviewer
            for row_id, required, requirement_json, expected_revision in normalized:
                current = connection.execute(
                    "SELECT revision, required, status, requirement_json "
                    "FROM qc_requirements WHERE row_id = ?",
                    (row_id,),
                ).fetchone()
                if current is None:
                    raise WorkbenchStoreError(
                        "QC requirement batch does not cover the workspace"
                    )
                if (
                    current[1] != 0
                    or current[2] != "not_required"
                    or current[3] is not None
                ):
                    raise WorkbenchStoreError(
                        "QC requirement batch cannot replace an active queue"
                    )
                if current[0] != expected_revision:
                    raise StaleRevisionError("QC requirement revision is stale")
                connection.execute(
                    "UPDATE qc_requirements SET required = ?, status = ?, "
                    "revision = ?, requirement_json = ?, updated_at = ?, "
                    "timezone = ?, reviewer = ?, session_id = ? WHERE row_id = ?",
                    (
                        int(required),
                        "pending" if required else "not_required",
                        expected_revision + 1,
                        requirement_json,
                        timestamp,
                        timezone_name,
                        reviewer,
                        session_id,
                        row_id,
                    ),
                )
        return self.get_qc_requirements()

    def save_qc_result(
        self,
        row_id: str,
        *,
        expected_revision: int,
        expected_qc_revision: int,
        expected_qc_status: str,
        expected_initial_event_hash: str,
        expected_reference_event_hash: str,
        phase: str,
        payload: str | Mapping[str, Any],
        session_id: str,
        reviewer: str | None = None,
        status: str,
        qc_status: str,
        recorded_at: str | datetime | None = None,
        timezone_name: str | None = None,
        projection_required: bool = False,
    ) -> dict[str, Any]:
        """Atomically persist one QC/adjudication event and its queue transition."""

        legal_transitions = {
            ("pending", "qc", "completed", "passed"),
            ("pending", "qc", "needs_adjudication", "failed"),
            ("failed", "adjudication", "completed", "passed"),
        }
        try:
            row_id = _token(row_id)
            session_id = _token(session_id)
            phase = _token(phase)
            status = _token(status)
            qc_status = _token(qc_status)
            expected_qc_status = _token(expected_qc_status)
            expected_initial_event_hash = _token(expected_initial_event_hash)
            expected_reference_event_hash = _token(expected_reference_event_hash)
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or isinstance(expected_qc_revision, bool)
                or not isinstance(expected_qc_revision, int)
                or expected_revision < 0
                or expected_qc_revision < 0
                or not isinstance(projection_required, bool)
                or (
                    expected_qc_status,
                    phase,
                    status,
                    qc_status,
                )
                not in legal_transitions
                or (qc_status == "failed" and projection_required)
            ):
                raise ValueError
            for digest in (
                expected_initial_event_hash,
                expected_reference_event_hash,
            ):
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError
            timestamp, timezone_name = _timestamp_and_zone(
                recorded_at, timezone_name
            )
        except (TypeError, ValueError) as exc:
            raise WorkbenchStoreError("QC result is invalid") from exc

        with self._transaction() as connection:
            current = connection.execute(
                "SELECT q.required, q.status, q.revision, q.requirement_json, "
                "c.revision, c.projection_dirty, c.initial_event_hash, "
                "c.latest_event_id, c.phase, c.status, c.qc_event_id, "
                "c.qc_event_hash, c.qc_for_initial_hash "
                "FROM qc_requirements AS q JOIN current_state AS c "
                "ON c.row_id = q.row_id WHERE q.row_id = ?",
                (row_id,),
            ).fetchone()
            if current is None:
                raise WorkbenchStoreError("annotation row was not found")
            if current[4] != expected_revision:
                raise StaleRevisionError("annotation revision is stale")
            if current[2] != expected_qc_revision:
                raise StaleRevisionError("QC requirement revision is stale")
            if current[5] != 0:
                raise WorkbenchStoreError(
                    "annotation projection must be reconciled before another save"
                )
            if current[0] != 1:
                raise WorkbenchStoreError(
                    "QC is not required for this annotation row"
                )
            if current[1] != expected_qc_status:
                raise WorkbenchStoreError("QC requirement status is stale")

            requirement = (
                None
                if current[3] is None
                else _parsed_mapping(str(current[3]))
            )
            if (
                requirement is None
                or set(requirement)
                != {
                    "reasons",
                    "initial_event_hash",
                    "reference_event_hash",
                }
                or not isinstance(requirement["reasons"], list)
                or not requirement["reasons"]
                or any(
                    not isinstance(reason, str) or not reason
                    for reason in requirement["reasons"]
                )
                or requirement["initial_event_hash"]
                != expected_initial_event_hash
                or current[6] != expected_initial_event_hash
                or requirement["reference_event_hash"]
                != expected_reference_event_hash
            ):
                raise WorkbenchCorruptionError(
                    "QC requirement binding is invalid"
                )

            reference = connection.execute(
                "SELECT phase, status FROM annotation_events "
                "WHERE row_id = ? AND event_hash = ?",
                (row_id, expected_reference_event_hash),
            ).fetchone()
            if (
                reference is None
                or reference[0] not in {"initial", "proposal_revision"}
                or reference[1] != "completed"
            ):
                raise WorkbenchCorruptionError(
                    "QC requirement binding is invalid"
                )
            latest = connection.execute(
                "SELECT event_hash, phase, status FROM annotation_events "
                "WHERE event_id = ? AND row_id = ?",
                (current[7], row_id),
            ).fetchone()
            if latest is None:
                raise WorkbenchCorruptionError("QC event state is invalid")

            if expected_qc_status == "pending":
                if (
                    current[8] not in {"initial", "proposal_revision"}
                    or current[9] != "completed"
                    or latest[0] != expected_reference_event_hash
                    or latest[1] != current[8]
                    or latest[2] != "completed"
                    or any(value is not None for value in current[10:13])
                ):
                    raise WorkbenchCorruptionError("QC event state is invalid")
            elif (
                current[8] != "qc"
                or current[9] != "needs_adjudication"
                or current[10] != current[7]
                or current[11] != latest[0]
                or current[12] != expected_initial_event_hash
                or latest[1] != "qc"
                or latest[2] != "needs_adjudication"
            ):
                raise WorkbenchCorruptionError("QC event state is invalid")

            event = self.save_annotation(
                row_id,
                expected_revision=expected_revision,
                phase=phase,
                payload=payload,
                session_id=session_id,
                reviewer=reviewer,
                status=status,
                recorded_at=timestamp,
                timezone_name=timezone_name,
                projection_required=projection_required,
            )
            qc = self.update_qc_status(
                row_id,
                qc_status,
                expected_revision=expected_qc_revision,
                session_id=session_id,
                reviewer=reviewer,
                recorded_at=timestamp,
                timezone_name=timezone_name,
            )
        return {"event": event, "qc": qc}

    def update_qc_status(
        self,
        row_id: str,
        status: str,
        *,
        expected_revision: int,
        session_id: str,
        reviewer: str | None = None,
        recorded_at: str | datetime | None = None,
        timezone_name: str | None = None,
    ) -> dict[str, Any]:
        """Optimistically update the status of an active QC requirement."""

        try:
            row_id = _token(row_id)
            status = _token(status)
            session_id = _token(session_id)
            if status not in QC_STATUSES - {"not_required"}:
                raise ValueError
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
                raise ValueError
            timestamp, timezone_name = _timestamp_and_zone(recorded_at, timezone_name)
        except (TypeError, ValueError) as exc:
            raise WorkbenchStoreError("QC status is invalid") from exc
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT required, revision FROM qc_requirements WHERE row_id = ?", (row_id,)
            ).fetchone()
            if current is None:
                raise WorkbenchStoreError("annotation row was not found")
            if current[1] != expected_revision:
                raise StaleRevisionError("QC requirement revision is stale")
            if current[0] != 1:
                raise WorkbenchStoreError("QC is not required for this annotation row")
            session_reviewer = self._session_reviewer(connection, session_id)
            if reviewer is not None and reviewer != session_reviewer:
                raise WorkbenchStoreError("QC reviewer does not match the session")
            connection.execute(
                "UPDATE qc_requirements SET status = ?, revision = revision + 1, "
                "updated_at = ?, timezone = ?, reviewer = ?, session_id = ? "
                "WHERE row_id = ? AND revision = ?",
                (
                    status,
                    timestamp,
                    timezone_name,
                    session_reviewer,
                    session_id,
                    row_id,
                    expected_revision,
                ),
            )
        return self.get_qc_status(row_id)

    def get_qc_status(self, row_id: str) -> dict[str, Any]:
        """Return parsed current QC requirement state."""

        try:
            row_id = _token(row_id)
        except ValueError as exc:
            raise WorkbenchStoreError("annotation row was not found") from exc
        with self._read() as connection:
            row = connection.execute(
                "SELECT required, status, revision, requirement_json, updated_at, timezone, "
                "reviewer, session_id FROM qc_requirements WHERE row_id = ?", (row_id,)
            ).fetchone()
        if row is None:
            raise WorkbenchStoreError("annotation row was not found")
        return {
            "row_id": row_id,
            "required": bool(row[0]),
            "status": row[1],
            "revision": row[2],
            "requirement": None if row[3] is None else _parsed_mapping(row[3]),
            "updated_at": row[4],
            "timezone": row[5],
            "reviewer": row[6],
            "session_id": row[7],
        }

    def get_qc_requirements(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Return QC requirement states in frozen source order."""

        if status is not None and status not in QC_STATUSES:
            raise WorkbenchStoreError("QC status filter is invalid")
        query = (
            "SELECT q.row_id, q.required, q.status, q.revision, q.requirement_json, "
            "q.updated_at, q.timezone, q.reviewer, q.session_id FROM qc_requirements AS q "
            "JOIN source_rows AS s ON s.row_id = q.row_id"
        )
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE q.status = ?"
            parameters = (status,)
        query += " ORDER BY s.position"
        with self._read() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "row_id": row[0],
                "required": bool(row[1]),
                "status": row[2],
                "revision": row[3],
                "requirement": None if row[4] is None else _parsed_mapping(row[4]),
                "updated_at": row[5],
                "timezone": row[6],
                "reviewer": row[7],
                "session_id": row[8],
            }
            for row in rows
        ]

    def record_proposal_reveal(
        self,
        row_id: str,
        proposal_id: str,
        *,
        session_id: str,
        reviewer: str | None = None,
        revealed_at: str | datetime | None = None,
        timezone_name: str | None = None,
        details: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a proposal-reveal audit record."""

        try:
            row_id = _token(row_id)
            proposal_id = _token(proposal_id)
            session_id = _token(session_id)
            timestamp, timezone_name = _timestamp_and_zone(revealed_at, timezone_name)
            details_json = None if details is None else _exact_or_canonical_json(details)[0]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkbenchStoreError("proposal reveal audit is invalid") from exc
        with self._transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM source_rows WHERE row_id = ?", (row_id,)
            ).fetchone() is None:
                raise WorkbenchStoreError("annotation row was not found")
            session_reviewer = self._session_reviewer(connection, session_id)
            if reviewer is not None and reviewer != session_reviewer:
                raise WorkbenchStoreError("proposal reviewer does not match the session")
            cursor = connection.execute(
                "INSERT INTO proposal_reveals(row_id, proposal_id, reviewer, session_id, "
                "revealed_at, timezone, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row_id,
                    proposal_id,
                    session_reviewer,
                    session_id,
                    timestamp,
                    timezone_name,
                    details_json,
                ),
            )
            reveal_id = int(cursor.lastrowid)
        return {
            "reveal_id": reveal_id,
            "row_id": row_id,
            "proposal_id": proposal_id,
            "reviewer": session_reviewer,
            "session_id": session_id,
            "revealed_at": timestamp,
            "timezone": timezone_name,
            "details": None if details_json is None else _parsed_mapping(details_json),
        }

    def get_proposal_reveals(self, row_id: str | None = None) -> list[dict[str, Any]]:
        """Return parsed reveal audit entries without source or proposal payload leakage."""

        parameters: tuple[Any, ...] = ()
        query = (
            "SELECT reveal_id, row_id, proposal_id, reviewer, session_id, revealed_at, "
            "timezone, details_json FROM proposal_reveals"
        )
        if row_id is not None:
            try:
                row_id = _token(row_id)
            except ValueError as exc:
                raise WorkbenchStoreError("annotation row was not found") from exc
            query += " WHERE row_id = ?"
            parameters = (row_id,)
        query += " ORDER BY reveal_id"
        with self._read() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "reveal_id": row[0],
                "row_id": row[1],
                "proposal_id": row[2],
                "reviewer": row[3],
                "session_id": row[4],
                "revealed_at": row[5],
                "timezone": row[6],
                "details": None if row[7] is None else _parsed_mapping(row[7]),
            }
            for row in rows
        ]


__all__ = [
    "BackupValidation",
    "KNOWN_PHASES",
    "KNOWN_STATUSES",
    "MAX_BACKUPS",
    "QC_STATUSES",
    "SCHEMA_VERSION",
    "SourceRow",
    "StaleRevisionError",
    "StorePaths",
    "WorkbenchBindingError",
    "WorkbenchCorruptionError",
    "WorkbenchLockError",
    "WorkbenchSchemaError",
    "WorkbenchStore",
    "WorkbenchStoreError",
    "recover_backup",
    "store_paths",
    "validate_backup",
]
