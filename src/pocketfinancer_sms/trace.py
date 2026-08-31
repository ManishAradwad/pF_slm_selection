"""Immutable processing traces for truthful local observability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


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
