"""Immutable processing traces for truthful local observability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID


TRACE_STAGES = frozenset(
    {
        "analysis",
        "triage",
        "selector_decode",
        "selector_validation",
        "reconstruction",
        "persistence_gate",
        "review",
    }
)
TRACE_STATUSES = frozenset({"started", "completed", "skipped", "failed", "retained"})
TRACE_V2_CONTRACT = "pocketfinancer.processing-trace/2"
TRACE_V2_STAGES = frozenset(
    {
        "admission",
        "configuration",
        "claim",
        "analysis",
        "triage",
        "selector_execution",
        "selector_validation",
        "reconstruction",
        "account_resolution",
        "persistence_gate",
        "settlement",
        "review",
        "recovery",
        "erase",
    }
)
TRACE_V2_STATUSES = frozenset(
    {"pending", "running", "completed", "skipped", "failed", "interrupted", "retained"}
)


@dataclass(frozen=True, slots=True)
class TraceStage:
    sequence: int
    stage: str
    status: str
    reason_codes: tuple[str, ...] = ()
    detail: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProcessingTrace:
    contract: str
    operation_id_hash: str
    config_hash: str
    stages: tuple[TraceStage, ...]
    previous_trace_hash: str | None = None

    @classmethod
    def create(
        cls,
        operation_id: str,
        config_hash: str,
        stages: tuple[TraceStage, ...],
        *,
        previous_trace_hash: str | None = None,
    ) -> ProcessingTrace:
        if not operation_id:
            raise ValueError("trace operation ID is required")
        if not _is_sha256(config_hash):
            raise ValueError("trace configuration hash is invalid")
        if previous_trace_hash is not None and not _is_sha256(previous_trace_hash):
            raise ValueError("previous trace hash is invalid")
        for expected_sequence, stage in enumerate(stages):
            if stage.sequence != expected_sequence:
                raise ValueError("trace stages must have contiguous sequence numbers")
            if stage.stage not in TRACE_STAGES or stage.status not in TRACE_STATUSES:
                raise ValueError("trace stage or status is unsupported")
        return cls(
            contract="pocketfinancer.processing-trace/1",
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            config_hash=config_hash,
            stages=stages,
            previous_trace_hash=previous_trace_hash,
        )

    @property
    def trace_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TraceEventV2:
    sequence: int
    event_id: str
    occurred_at_epoch_ms: int
    stage: str
    status: str
    reason_codes: tuple[str, ...] = ()
    detail: dict[str, Any] | None = None
    previous_event_hash: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("trace event sequence is invalid")
        _require_uuid(self.event_id, "trace event")
        if (
            isinstance(self.occurred_at_epoch_ms, bool)
            or not isinstance(self.occurred_at_epoch_ms, int)
            or self.occurred_at_epoch_ms < 0
        ):
            raise ValueError("trace event timestamp is invalid")
        if self.stage not in TRACE_V2_STAGES or self.status not in TRACE_V2_STATUSES:
            raise ValueError("trace event stage or status is unsupported")
        if len(set(self.reason_codes)) != len(self.reason_codes) or not all(
            isinstance(reason, str) and reason for reason in self.reason_codes
        ):
            raise ValueError("trace event reason codes are invalid")
        if self.detail is not None and not isinstance(self.detail, dict):
            raise ValueError("trace event detail must be an object")
        if self.sequence == 0 and self.previous_event_hash is not None:
            raise ValueError("first trace event cannot reference a predecessor")
        if self.sequence > 0 and not _is_sha256(self.previous_event_hash or ""):
            raise ValueError("later trace event requires a predecessor hash")

    @property
    def event_hash(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProcessingTraceV2:
    contract: str
    operation_id_hash: str
    config_hash: str
    owner_generation: int
    events: tuple[TraceEventV2, ...]
    previous_trace_hash: str | None = None

    @classmethod
    def create(
        cls,
        operation_id: str,
        config_hash: str,
        owner_generation: int,
        events: tuple[TraceEventV2, ...],
        *,
        previous_trace_hash: str | None = None,
    ) -> ProcessingTraceV2:
        _require_uuid(operation_id, "trace operation")
        if not _is_sha256(config_hash):
            raise ValueError("trace configuration hash is invalid")
        if (
            isinstance(owner_generation, bool)
            or not isinstance(owner_generation, int)
            or owner_generation < 0
        ):
            raise ValueError("trace owner generation is invalid")
        if previous_trace_hash is not None and not _is_sha256(previous_trace_hash):
            raise ValueError("previous trace hash is invalid")
        for expected_sequence, event in enumerate(events):
            if event.sequence != expected_sequence:
                raise ValueError("trace events must have contiguous sequence numbers")
            expected_previous = None if expected_sequence == 0 else events[expected_sequence - 1].event_hash
            if event.previous_event_hash != expected_previous:
                raise ValueError("trace event hash chain is invalid")
        return cls(
            contract=TRACE_V2_CONTRACT,
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            config_hash=config_hash,
            owner_generation=owner_generation,
            events=events,
            previous_trace_hash=previous_trace_hash,
        )

    @property
    def trace_hash(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} ID is invalid") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} ID is not canonical")
