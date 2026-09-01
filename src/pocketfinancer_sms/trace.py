"""Immutable processing traces for truthful local observability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
