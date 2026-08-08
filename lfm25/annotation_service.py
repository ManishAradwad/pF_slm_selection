"""Workflow service for the shared local annotation workbench.

This layer is the only bridge between workflow policy, persistent state, and
the legacy reviewer JSONL projection.  It returns explicit browser DTOs and
never emits or logs source values.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .annotation_sources import (
    load_blinded_workspace,
    public_row,
    qc_requirements,
    training_proposals,
)
from .annotation_store import (
    SourceRow,
    StaleRevisionError,
    WorkbenchStore,
    WorkbenchStoreError,
)
from .annotation_workbench import (
    ACTIVE_LEARNING_QUEUE_TAGS,
    BLINDED_MODE,
    DEFAULT_TRAINING_EXPORT,
    DEFAULT_TRAINING_REPORT,
    TRAINING_MODE,
    WORKBENCH_CONTRACT,
    AnnotationValidationError,
    WorkspaceDefinition,
    WorkbenchError,
    annotation_to_legacy_fields,
    annotations_equal,
    empty_annotation,
    exact_json_dumps,
    exact_json_loads,
    validate_annotation,
)
from .blinded_review import REVIEW_FIELDS, resolve_review_paths, run_validate
from .private_data import ensure_within, file_sha256, require_private_ignore


MAX_PAGE_ROWS = 10_000
MAX_REVIEW_BACKUPS = 5


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _review_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise WorkbenchError("an annotation event timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WorkbenchError("an annotation event timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkbenchError("an annotation event timestamp is invalid")
    return parsed


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(mode)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(exact_json_dumps(row) + "\n" for row in rows).encode("utf-8")


def _aggregate_report_bytes(report: Mapping[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _valid_reviewer(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise WorkbenchError("the reviewer identity is invalid")
    return value.strip()


def _store_sources(definition: WorkspaceDefinition) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for source in definition.rows:
        reviewer_fields: dict[str, Any] = {
            "review_id": source.row_id,
            "sender": source.sender,
            "sms": source.sms,
            "queue_tags": list(source.queue_tags),
        }
        if definition.mode == TRAINING_MODE:
            reviewer_fields["split"] = source.split
        if source.source_json is None:
            source_value: str | Mapping[str, Any] = {"sms": source.sms}
        else:
            try:
                parsed_source = exact_json_loads(source.source_json)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise WorkbenchError(
                    "a workbench source row has an invalid private record"
                ) from exc
            if not isinstance(parsed_source, Mapping):
                raise WorkbenchError(
                    "a workbench source row has an invalid private record"
                )
            source_value = source.source_json
        rows.append(
            SourceRow(
                row_id=source.row_id,
                reviewer_fields=reviewer_fields,
                source_json=source_value,
                position=source.position,
            )
        )
    return rows


class AnnotationService:
    """One locked reviewer session over either explicit workflow policy."""

    def __init__(
        self,
        *,
        repo_root: Path,
        definition: WorkspaceDefinition,
        store: WorkbenchStore,
        reviewer: str,
        batch_size: int = 50,
    ) -> None:
        if batch_size < 1 or batch_size > 10_000:
            raise WorkbenchError("the annotation batch size is invalid")
        if definition.row_count > MAX_PAGE_ROWS:
            raise WorkbenchError("the annotation workspace exceeds the supported row limit")
        self.repo_root = repo_root.resolve()
        self.definition = definition
        self.store = store
        self.reviewer = _valid_reviewer(reviewer)
        self.batch_size = batch_size
        metadata = {
            **definition.metadata,
            "contract": WORKBENCH_CONTRACT,
            "mode": definition.mode,
            "primary_reviewer": self.reviewer,
        }
        metadata.pop("completed_rows_at_bootstrap", None)
        bootstrapped = self.store.bootstrap(
            _store_sources(definition),
            workspace_binding=definition.binding,
            workspace_metadata=metadata,
        )
        if bootstrapped["row_count"] != definition.row_count:
            raise WorkbenchError("the annotation workspace row count changed")
        if bootstrapped["created"]:
            self._import_existing_annotations()
        qc_states = self.store.get_qc_requirements()
        qc_started = any(
            item["required"]
            or item["status"] != "not_required"
            or item["requirement"] is not None
            for item in qc_states
        )
        initial_phase = "qc" if qc_started else "initial"
        self.session = self.store.start_session(
            self.reviewer,
            metadata={"phase": initial_phase, "batch_size": batch_size},
        )
        self.session_phase = initial_phase
        self._review_backup_created = False
        self._reconcile_projection_on_open()

    @property
    def mode(self) -> str:
        return self.definition.mode

    @property
    def session_id(self) -> str:
        return str(self.session["session_id"])

    def _import_existing_annotations(self) -> None:
        existing = [row for row in self.definition.rows if row.initial_annotation is not None]
        if not existing:
            return
        bootstrap = self.store.start_session(
            self.reviewer,
            metadata={"phase": "legacy_projection_bootstrap"},
        )
        for source in existing:
            if source.initial_reviewer != self.reviewer:
                raise WorkbenchError("existing annotations use a different reviewer identity")
            self.store.save_annotation(
                source.row_id,
                expected_revision=0,
                phase="initial",
                payload=source.initial_annotation or empty_annotation(),
                session_id=str(bootstrap["session_id"]),
                reviewer=self.reviewer,
                status="completed",
                recorded_at=source.initial_reviewed_at,
            )

    def _all_rows(self, *, include_private: bool = False) -> list[dict[str, Any]]:
        return self.store.get_rows(limit=MAX_PAGE_ROWS, include_private=include_private)

    @staticmethod
    def _event_for_projection(
        row: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        if row.get("status") == "completed" and isinstance(row.get("annotation"), Mapping):
            return {
                "payload": row["annotation"],
                "reviewer": row.get("reviewer"),
                "recorded_at": row.get("recorded_at"),
                "status": "completed",
            }
        candidates = [
            event
            for event in history
            if event.get("status") == "completed"
            and event.get("phase") in {"initial", "qc", "adjudication"}
        ]
        return candidates[-1] if candidates else None

    def _projection_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stored in self._all_rows():
            fields = stored["reviewer_fields"]
            row = {
                "review_id": fields["review_id"],
                "sender": fields["sender"],
                "sms": fields["sms"],
                "decision": None,
                "amount": None,
                "counterparty": None,
                "type": None,
                "account": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
            }
            event = self._event_for_projection(stored, self.store.get_history(stored["row_id"]))
            if event is not None:
                annotation = validate_annotation(
                    event["payload"], fields["sms"], require_complete=True
                )
                projected = annotation_to_legacy_fields(annotation)
                row.update(projected)
                row["reviewer"] = event["reviewer"]
                row["reviewed_at"] = event["recorded_at"]
            if tuple(row) != REVIEW_FIELDS:
                raise WorkbenchError("the blinded projection schema is invalid")
            rows.append(row)
        return rows

    def _review_backup(self, review_file: Path) -> None:
        if self._review_backup_created:
            return
        backup_dir = review_file.parent / "annotation_workbench/review_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.chmod(0o700)
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        destination = backup_dir / f"{review_file.name}.{stamp}.bak"
        _atomic_bytes(destination, review_file.read_bytes())
        backups = sorted(backup_dir.glob(f"{review_file.name}.*.bak"))
        for old in backups[:-MAX_REVIEW_BACKUPS]:
            old.unlink()
        self._review_backup_created = True

    def _write_blinded_projection(self) -> None:
        if self.mode != BLINDED_MODE:
            return
        current = load_blinded_workspace(
            self.repo_root,
            include_initial_annotations=False,
        )
        if current.binding != self.definition.binding:
            raise WorkbenchError("the frozen blinded package changed after workbench launch")
        paths = resolve_review_paths(self.repo_root)
        source_hash = file_sha256(paths.source_manifest)
        if source_hash != self.definition.binding["source_manifest_sha256"]:
            raise WorkbenchError("the frozen source manifest changed after workbench launch")
        previous = paths.review_file.read_bytes()
        self._review_backup(paths.review_file)
        try:
            _atomic_bytes(paths.review_file, _jsonl_bytes(self._projection_rows()))
            report = run_validate(self.repo_root)
            if report["test_rows"] != self.definition.row_count:
                raise WorkbenchError("the blinded projection row count changed")
            if file_sha256(paths.source_manifest) != source_hash:
                raise WorkbenchError("the frozen source manifest changed during projection")
        except BaseException:
            _atomic_bytes(paths.review_file, previous)
            raise

    def _dirty_rows(self) -> list[dict[str, Any]]:
        method = getattr(self.store, "get_dirty_rows", None)
        return [] if method is None else list(method())

    def _mark_projection_clean(self, row_id: str, revision: int) -> None:
        method = getattr(self.store, "mark_projection_clean", None)
        if method is None:
            return
        method(row_id, expected_revision=revision)

    def _reconcile_projection_on_open(self) -> None:
        if self.mode != BLINDED_MODE:
            return
        dirty = self._dirty_rows()
        if dirty:
            self._write_blinded_projection()
            for row in dirty:
                self._mark_projection_clean(str(row["row_id"]), int(row["revision"]))
            return
        paths = resolve_review_paths(self.repo_root)
        expected = _jsonl_bytes(self._projection_rows())
        if hashlib.sha256(paths.review_file.read_bytes()).digest() != hashlib.sha256(expected).digest():
            self._write_blinded_projection()

    def progress(self) -> dict[str, Any]:
        rows = self._all_rows()
        total = len(rows)
        completed = sum(row["status"] == "completed" for row in rows)
        uncertain = sum(row["status"] in {"uncertain", "needs_adjudication"} for row in rows)
        pending = total - completed - uncertain
        batch_completed = sum(
            row.get("session_id") == self.session_id and row["status"] == "completed"
            for row in rows
        )
        qc_rows = [item for item in self.store.get_qc_requirements() if item["required"]]
        return {
            "total_rows": total,
            "completed_rows": completed,
            "pending_rows": pending,
            "uncertain_rows": uncertain,
            "batch_size": self.batch_size,
            "batch_completed": min(batch_completed, self.batch_size),
            "qc_required": len(qc_rows),
            "qc_passed": sum(item["status"] == "passed" for item in qc_rows),
            "ready_for_qc": completed == total and uncertain == 0,
        }

    @staticmethod
    def _matches_filter(row: Mapping[str, Any], filter_name: str) -> bool:
        annotation = row.get("annotation") or {}
        if filter_name == "pending":
            return row.get("status") in {"pending", "draft"}
        if filter_name == "active_learning":
            fields = row.get("reviewer_fields")
            return (
                row.get("status") in {"pending", "draft"}
                and isinstance(fields, Mapping)
                and bool(fields.get("queue_tags"))
            )
        if filter_name == "completed":
            return row.get("status") == "completed"
        if filter_name == "uncertain":
            return row.get("status") in {"uncertain", "needs_adjudication"}
        if filter_name == "noted":
            return bool(annotation.get("notes"))
        if filter_name == "transaction":
            return annotation.get("decision") == "transaction"
        if filter_name == "null":
            return annotation.get("decision") == "not_transaction"
        if filter_name == "qc":
            return bool(row.get("qc_required")) and row.get("qc_status") != "passed"
        raise WorkbenchError("the annotation filter is invalid")

    @staticmethod
    def _active_learning_sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
        fields = row.get("reviewer_fields")
        tags = fields.get("queue_tags", ()) if isinstance(fields, Mapping) else ()
        priorities = {
            tag: priority for priority, tag in enumerate(ACTIVE_LEARNING_QUEUE_TAGS)
        }
        queue_priority = min(
            (priorities.get(tag, len(ACTIVE_LEARNING_QUEUE_TAGS)) for tag in tags),
            default=len(ACTIVE_LEARNING_QUEUE_TAGS),
        )
        return queue_priority, int(row["position"])

    def _dto(self, row: Mapping[str, Any]) -> dict[str, Any]:
        fields = row["reviewer_fields"]
        source = {
            "row_id": row["row_id"],
            "position": row["position"],
            "sender": fields["sender"],
            "sms": fields["sms"],
            "queue_tags": (
                []
                if self.session_phase == "qc" or row.get("status") != "completed"
                else fields.get("queue_tags", [])
            ),
        }
        state = dict(row)
        qc_pending = (
            self.session_phase == "qc"
            and row.get("qc_required")
            and row.get("qc_status") == "pending"
        )
        if qc_pending:
            state["annotation"] = empty_annotation()
        value = public_row(source, state, mode=self.mode, total_rows=self.definition.row_count)
        value["history_available"] = not qc_pending
        value["proposal_reveal_available"] = (
            self.mode == TRAINING_MODE
            and row.get("status") == "completed"
            and self.session_phase != "qc"
        )
        value["adjudication_required"] = row.get("qc_status") == "failed"
        return value

    def navigate(self, *, position: int, direction: str, filter_name: str) -> dict[str, Any]:
        if direction not in {"first", "current", "next", "previous"}:
            raise WorkbenchError("the annotation navigation direction is invalid")
        if filter_name == "active_learning" and (
            self.mode != TRAINING_MODE or self.session_phase == "qc"
        ):
            raise WorkbenchError(
                "the active-learning queue is unavailable in this workflow"
            )
        if self.session_phase == "qc" and filter_name in {"noted", "transaction", "null"}:
            raise WorkbenchError("label-derived filters are hidden during delayed QC")

        rows = [row for row in self._all_rows() if self._matches_filter(row, filter_name)]
        if not rows:
            return {"row": None, "progress": self.progress()}
        if filter_name == "active_learning":
            rows.sort(key=self._active_learning_sort_key)
            current_index = next(
                (
                    index
                    for index, item in enumerate(rows)
                    if item["position"] == position
                ),
                None,
            )
            if direction in {"first", "current"}:
                selected = rows[0] if current_index is None else rows[current_index]
            elif direction == "next":
                selected = (
                    rows[0]
                    if current_index is None
                    else rows[(current_index + 1) % len(rows)]
                )
            else:
                selected = (
                    rows[-1]
                    if current_index is None
                    else rows[(current_index - 1) % len(rows)]
                )
        elif direction == "first":
            selected = rows[0]
        elif direction == "current":
            selected = min(rows, key=lambda item: (abs(item["position"] - position), item["position"]))
        elif direction == "next":
            selected = next((item for item in rows if item["position"] > position), rows[0])
        else:
            selected = next(
                (item for item in reversed(rows) if item["position"] < position), rows[-1]
            )
        return {"row": self._dto(selected), "progress": self.progress()}

    def get_row(self, row_id: str) -> dict[str, Any]:
        return self._dto(self.store.get_row(row_id))

    def history(self, row_id: str) -> list[dict[str, Any]]:
        row = self.store.get_row(row_id)
        if (
            self.session_phase == "qc"
            and row.get("qc_required")
            and row.get("qc_status") == "pending"
        ):
            return []
        events = self.store.get_history(row_id)
        return [
            {
                "phase": event["phase"],
                "status": event["status"],
                "reviewer": event["reviewer"],
                "recorded_at": event["recorded_at"],
                "annotation": event["payload"],
            }
            for event in events
        ]

    def save(
        self,
        *,
        row_id: str,
        expected_revision: int,
        annotation: Mapping[str, Any],
        submit: bool,
    ) -> dict[str, Any]:
        if self.session_phase != "initial":
            raise WorkbenchError("initial annotations are locked after QC begins")
        row = self.store.get_row(row_id)
        sms = str(row["reviewer_fields"]["sms"])
        uncertain = annotation.get("uncertain") is True
        validated = validate_annotation(
            annotation,
            sms,
            require_complete=submit and not uncertain,
        )
        proposal_reveals = (
            self.store.get_proposal_reveals(row_id)
            if self.mode == TRAINING_MODE
            else []
        )
        if submit and not uncertain and proposal_reveals:
            history = self.store.get_history(row_id)
            initial = next(
                (
                    event
                    for event in history
                    if event["event_hash"] == row["initial_event_hash"]
                ),
                None,
            )
            if initial is None:
                raise WorkbenchError("the blind-first annotation could not be verified")
            changed = not annotations_equal(validated, initial["payload"])
            change_reason = validated.get("notes")
            if changed and (
                not isinstance(change_reason, str)
                or not change_reason.strip()
                or change_reason == initial["payload"].get("notes")
            ):
                raise WorkbenchError("a post-proposal label change requires a new note")
        if submit and uncertain:
            phase, status = "uncertain", "uncertain"
        elif submit:
            phase = "proposal_revision" if proposal_reveals else "initial"
            status = "completed"
        else:
            phase, status = ("uncertain", "uncertain") if uncertain else ("draft", "draft")
        projection_required = self.mode == BLINDED_MODE and phase == "initial"
        save_arguments = {
            "expected_revision": expected_revision,
            "phase": phase,
            "payload": validated,
            "session_id": self.session_id,
            "reviewer": self.reviewer,
            "status": status,
        }
        event = self.store.save_annotation(
            row_id, projection_required=projection_required, **save_arguments
        )
        if projection_required:
            self._write_blinded_projection()
            self._mark_projection_clean(row_id, int(event["revision"]))
        return {
            "row": self.get_row(row_id),
            "progress": self.progress(),
            "next_available": submit,
        }

    def start_qc(self) -> dict[str, Any]:
        progress = self.progress()
        if not progress["ready_for_qc"]:
            raise WorkbenchError("QC cannot start until every annotation is complete and resolved")
        rows = self._all_rows()
        qc_input = []
        reference_hashes: dict[str, str] = {}
        for row in rows:
            row_id = str(row["row_id"])
            history = self.store.get_history(row_id)
            reference = next(
                (
                    event
                    for event in reversed(history)
                    if event["phase"] in {"initial", "proposal_revision"}
                    and event["status"] == "completed"
                ),
                None,
            )
            initial_hash = row["initial_event_hash"]
            if reference is None or not isinstance(initial_hash, str):
                raise WorkbenchError("the QC reference annotation could not be verified")
            try:
                reference_annotation = validate_annotation(
                    reference["payload"],
                    row["reviewer_fields"]["sms"],
                    require_complete=True,
                )
            except AnnotationValidationError as exc:
                raise WorkbenchError(
                    "the QC reference annotation could not be verified"
                ) from exc
            reference_hashes[row_id] = str(reference["event_hash"])
            qc_input.append(
                {
                    "row_id": row_id,
                    "annotation": reference_annotation,
                    "ever_uncertain": any(
                        event["phase"] == "uncertain" for event in history
                    ),
                }
            )
        seed = str(
            self.definition.binding.get("source_manifest_sha256")
            or self.definition.binding.get("record_id_set_sha256")
        )
        selected = qc_requirements(qc_input, deterministic_seed=seed)
        qc_states = {
            str(item["row_id"]): item for item in self.store.get_qc_requirements()
        }
        row_ids = {str(row["row_id"]) for row in rows}
        if set(qc_states) != row_ids:
            raise WorkbenchError("the saved QC queue does not match current policy")
        expected_queue: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = str(row["row_id"])
            reasons = selected.get(row_id)
            expected_queue[row_id] = {
                "required": reasons is not None,
                "requirement": (
                    None
                    if reasons is None
                    else {
                        "reasons": list(reasons),
                        "initial_event_hash": row["initial_event_hash"],
                        "reference_event_hash": reference_hashes[row_id],
                    }
                ),
            }
        qc_started = any(
            item["revision"] != 0
            or item["required"]
            or item["status"] != "not_required"
            or item["requirement"] is not None
            for item in qc_states.values()
        )
        if qc_started:
            for row_id, expected in expected_queue.items():
                current = qc_states[row_id]
                if (
                    current["required"] is not expected["required"]
                    or current["requirement"] != expected["requirement"]
                    or (
                        expected["required"] is False
                        and current["status"] != "not_required"
                    )
                ):
                    raise WorkbenchError(
                        "the saved QC queue does not match current policy"
                    )
        qc_session = self.store.start_session(
            self.reviewer,
            metadata={"phase": "qc", "batch_size": self.batch_size},
        )
        if not qc_started:
            self.store.set_qc_requirements(
                [
                    {
                        "row_id": row_id,
                        "required": expected["required"],
                        "requirement": expected["requirement"],
                        "expected_revision": int(qc_states[row_id]["revision"]),
                    }
                    for row_id, expected in expected_queue.items()
                ],
                session_id=str(qc_session["session_id"]),
                reviewer=self.reviewer,
            )
        self.session = qc_session
        self.session_phase = "qc"
        return {"progress": self.progress()}

    @staticmethod
    def _reference_event(
        row: Mapping[str, Any],
        qc: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        requirement = qc.get("requirement")
        if not isinstance(requirement, Mapping):
            raise WorkbenchError("the QC reference annotation could not be verified")
        initial_hash = requirement.get("initial_event_hash")
        reference_hash = requirement.get("reference_event_hash")
        if (
            not isinstance(initial_hash, str)
            or initial_hash != row.get("initial_event_hash")
            or not isinstance(reference_hash, str)
        ):
            raise WorkbenchError("the QC reference annotation could not be verified")
        for event in reversed(history):
            if event.get("event_hash") == reference_hash:
                return event
        raise WorkbenchError("the QC reference annotation could not be verified")

    def save_qc(
        self,
        *,
        row_id: str,
        expected_revision: int,
        annotation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.session_phase != "qc":
            raise WorkbenchError("a delayed QC session has not been started")
        row = self.store.get_row(row_id)
        qc = self.store.get_qc_status(row_id)
        if not qc["required"] or qc["status"] not in {"pending", "failed"}:
            raise WorkbenchError("this row is not pending QC")
        validated = validate_annotation(
            annotation,
            str(row["reviewer_fields"]["sms"]),
            require_complete=True,
        )
        history = self.store.get_history(row_id)
        reference = self._reference_event(row, qc, history)
        requirement = qc["requirement"]
        if qc["status"] == "failed":
            phase, status, qc_status = "adjudication", "completed", "passed"
            projection_required = self.mode == BLINDED_MODE
        elif annotations_equal(validated, reference["payload"]):
            phase, status, qc_status = "qc", "completed", "passed"
            projection_required = self.mode == BLINDED_MODE
        else:
            phase, status, qc_status = "qc", "needs_adjudication", "failed"
            projection_required = False
        result = self.store.save_qc_result(
            row_id,
            expected_revision=expected_revision,
            expected_qc_revision=int(qc["revision"]),
            expected_qc_status=str(qc["status"]),
            expected_initial_event_hash=str(requirement["initial_event_hash"]),
            expected_reference_event_hash=str(requirement["reference_event_hash"]),
            phase=phase,
            payload=validated,
            session_id=self.session_id,
            reviewer=self.reviewer,
            status=status,
            qc_status=qc_status,
            projection_required=projection_required,
        )
        event = result["event"]
        if projection_required:
            self._write_blinded_projection()
            self._mark_projection_clean(row_id, int(event["revision"]))
        return {
            "row": self.get_row(row_id),
            "progress": self.progress(),
            "next_available": qc_status == "passed",
        }

    def reveal_proposals(self, row_id: str) -> dict[str, Any]:
        if self.mode != TRAINING_MODE or self.session_phase == "qc":
            raise WorkbenchError("proposal reveal is unavailable in this workflow")
        row = self.store.get_row(row_id, include_private=True)
        if row["status"] != "completed":
            raise WorkbenchError("complete a blind-first label before revealing proposals")
        proposals = training_proposals(exact_json_dumps(row["source"]), mode=self.mode)
        self.store.record_proposal_reveal(
            row_id,
            "proposal-bundle",
            session_id=self.session_id,
            reviewer=self.reviewer,
            details={"proposal_count": len(proposals["model_proposals"])},
        )
        return proposals

    def export_training(
        self,
        *,
        output_file: Path = DEFAULT_TRAINING_EXPORT,
        report_file: Path = DEFAULT_TRAINING_REPORT,
        force: bool = False,
    ) -> dict[str, Any]:
        if self.mode != TRAINING_MODE:
            raise WorkbenchError("training export is unavailable in blinded-test mode")
        private_root = (self.repo_root / "PRIVATE_DATA/lfm25").resolve()
        require_private_ignore(self.repo_root, private_root)
        output = ensure_within(
            output_file if output_file.is_absolute() else self.repo_root / output_file,
            private_root,
        )
        report_path = ensure_within(
            report_file if report_file.is_absolute() else self.repo_root / report_file,
            private_root,
        )
        if output == report_path:
            raise WorkbenchError("training export paths must be distinct")
        if not force and any(path.exists() and path.stat().st_size for path in (output, report_path)):
            raise WorkbenchError("training export outputs are nonempty")
        rows: list[dict[str, Any]] = []
        counts = {"completed": 0, "pending": 0, "uncertain": 0}
        for stored in self._all_rows(include_private=True):
            source = dict(stored["source"])
            if source.get("split") not in {"train", "dev"}:
                raise WorkbenchError("training export encountered an ineligible split")
            if stored["status"] == "completed":
                annotation = validate_annotation(
                    stored["annotation"], str(source["sms"]), require_complete=True
                )
                projected = annotation_to_legacy_fields(annotation)
                source["review_status"] = "human_approved"
                source["human_label"] = (
                    None
                    if projected["decision"] == "not_transaction"
                    else {
                        "amount": projected["amount"],
                        "counterparty": projected["counterparty"],
                        "type": projected["type"],
                        "account": projected["account"],
                    }
                )
                source["human_reviewer"] = stored["reviewer"]
                source["human_reviewed_at"] = stored["recorded_at"]
                source["human_review_notes"] = annotation.get("notes")
                history = self.store.get_history(stored["row_id"])
                reveals = self.store.get_proposal_reveals(stored["row_id"])
                source["human_annotation_workbench"] = {
                    "contract": WORKBENCH_CONTRACT,
                    "annotation": annotation,
                    "blind_first_event_hash": stored["initial_event_hash"],
                    "final_event_hash": history[-1]["event_hash"],
                    "final_phase": stored["phase"],
                    "proposal_reveal_count": len(reveals),
                    "proposal_ids": sorted({item["proposal_id"] for item in reveals}),
                }
                counts["completed"] += 1
            elif stored["status"] in {"uncertain", "needs_adjudication"}:
                counts["uncertain"] += 1
            else:
                counts["pending"] += 1
            rows.append(source)
        payload = _jsonl_bytes(rows)
        report = {
            "contract": WORKBENCH_CONTRACT,
            "operation": "training_export",
            "valid": True,
            "rows": len(rows),
            "completed_rows": counts["completed"],
            "pending_rows": counts["pending"],
            "uncertain_rows": counts["uncertain"],
            "sealed_test_rows_exported": 0,
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_values_emitted_to_console": False,
        }
        _atomic_bytes(output, payload)
        _atomic_bytes(report_path, _aggregate_report_bytes(report))
        return report


def validate_blinded_import_gate(
    repo_root: Path,
    *,
    db_path: Path,
    source_manifest: Path | None = None,
    review_file: Path | None = None,
    mapping_file: Path | None = None,
    metadata_file: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the complete frozen package and every required QC pass."""

    definition = load_blinded_workspace(
        repo_root,
        source_manifest=source_manifest,
        review_file=review_file,
        mapping_file=mapping_file,
        metadata_file=metadata_file,
        include_initial_annotations=False,
    )
    with WorkbenchStore(db_path, workspace_binding=definition.binding) as store:
        rows = store.get_rows(limit=MAX_PAGE_ROWS)
        if len(rows) != definition.row_count:
            raise WorkbenchError("final import row count does not match the frozen package")
        if any(row["projection_dirty"] for row in rows):
            raise WorkbenchError("final import is blocked by an unrecovered projection")
        if any(row["status"] != "completed" for row in rows):
            raise WorkbenchError("final import requires every annotation to be completed")
        projection_rows: list[dict[str, Any]] = []
        for stored in rows:
            fields = stored["reviewer_fields"]
            projected = {
                "review_id": fields["review_id"],
                "sender": fields["sender"],
                "sms": fields["sms"],
                "decision": None,
                "amount": None,
                "counterparty": None,
                "type": None,
                "account": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
            }
            event = AnnotationService._event_for_projection(
                stored, store.get_history(stored["row_id"])
            )
            if event is not None:
                annotation = validate_annotation(
                    event["payload"], fields["sms"], require_complete=True
                )
                projected.update(annotation_to_legacy_fields(annotation))
                projected["reviewer"] = event["reviewer"]
                projected["reviewed_at"] = event["recorded_at"]
            projection_rows.append(projected)
        overrides = {
            name: value
            for name, value in {
                "source_manifest": source_manifest,
                "review_file": review_file,
                "mapping_file": mapping_file,
                "metadata_file": metadata_file,
            }.items()
            if value is not None
        }
        review_path = resolve_review_paths(repo_root, **overrides).review_file
        if review_path.read_bytes() != _jsonl_bytes(projection_rows):
            raise WorkbenchError(
                "final import projection does not match durable annotation history"
            )
        qc_input: list[dict[str, Any]] = []
        histories: dict[str, list[dict[str, Any]]] = {}
        initial_events: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = str(row["row_id"])
            history = store.get_history(row_id)
            initial_hash = row["initial_event_hash"]
            initial = next(
                (event for event in history if event["event_hash"] == initial_hash),
                None,
            )
            if (
                not isinstance(initial_hash, str)
                or initial is None
                or initial["event_id"] != row["initial_event_id"]
                or initial["phase"] != "initial"
                or initial["status"] != "completed"
            ):
                raise WorkbenchError(
                    "final import QC initial-event binding is invalid"
                )
            try:
                initial_annotation = validate_annotation(
                    initial["payload"],
                    row["reviewer_fields"]["sms"],
                    require_complete=True,
                )
            except AnnotationValidationError as exc:
                raise WorkbenchError(
                    "final import QC initial-event annotation is invalid"
                ) from exc
            histories[row_id] = history
            initial_events[row_id] = initial
            qc_input.append(
                {
                    "row_id": row_id,
                    "annotation": initial_annotation,
                    "ever_uncertain": any(event["phase"] == "uncertain" for event in history),
                }
            )
        seed = str(definition.binding["source_manifest_sha256"])
        required = qc_requirements(qc_input, deterministic_seed=seed)
        qc_states = {str(item["row_id"]): item for item in store.get_qc_requirements()}
        row_ids = {str(row["row_id"]) for row in rows}
        if set(qc_states) != row_ids:
            raise WorkbenchError("final import QC state does not match the frozen package")
        for row in rows:
            row_id = str(row["row_id"])
            reasons = required.get(row_id)
            qc = qc_states[row_id]
            history = histories[row_id]
            if reasons is None:
                has_qc_history = any(
                    event["phase"] in {"qc", "adjudication"} for event in history
                )
                if (
                    qc["required"] is not False
                    or qc["status"] != "not_required"
                    or qc["requirement"] is not None
                    or row["qc_event_id"] is not None
                    or row["qc_event_hash"] is not None
                    or row["qc_for_initial_hash"] is not None
                    or has_qc_history
                ):
                    raise WorkbenchError(
                        "final import contains unexpected QC state outside policy"
                    )
                continue

            initial = initial_events[row_id]
            initial_hash = initial["event_hash"]
            expected_requirement = {
                "reasons": list(reasons),
                "initial_event_hash": initial_hash,
                "reference_event_hash": initial_hash,
            }
            if qc["required"] is not True or qc["requirement"] != expected_requirement:
                raise WorkbenchError("final import QC requirements do not match policy")
            if qc["status"] != "passed" or row["qc_event_id"] is None:
                raise WorkbenchError("final import requires every selected QC review to pass")
            qc_event = next(
                (event for event in history if event["event_hash"] == row["qc_event_hash"]),
                None,
            )
            if (
                qc_event is None
                or qc_event["event_id"] != row["qc_event_id"]
                or qc_event["phase"] != "qc"
                or initial["session_id"] == qc_event["session_id"]
                or _review_timestamp(qc_event["recorded_at"])
                <= _review_timestamp(initial["recorded_at"])
                or row["qc_for_initial_hash"] != initial_hash
            ):
                raise WorkbenchError("final import QC is not a distinct delayed second pass")
            if not annotations_equal(qc_event["payload"], initial["payload"]):
                adjudications = [
                    event
                    for event in history
                    if event["phase"] == "adjudication"
                    and event["chain_index"] > qc_event["chain_index"]
                    and event["status"] == "completed"
                ]
                if (
                    not adjudications
                    or not annotations_equal(
                        adjudications[-1]["payload"], row["annotation"]
                    )
                ):
                    raise WorkbenchError(
                        "final import requires adjudication after a QC disagreement"
                    )
        return {
            "valid": True,
            "completed_rows": len(rows),
            "pending_rows": 0,
            "unresolved_uncertain_rows": 0,
            "qc_required_rows": len(required),
            "qc_passed_rows": len(required),
            "ready_for_import": True,
        }


def safe_service_error(error: BaseException) -> str:
    if isinstance(
        error,
        (WorkbenchError, WorkbenchStoreError, StaleRevisionError, AnnotationValidationError),
    ):
        return str(error)
    return "the local annotation workbench encountered an unexpected failure"
