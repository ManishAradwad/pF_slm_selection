"""Append-only user feedback contract for future native-app integration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserFeedbackEvent:
    contract: str
    event_id: str
    operation_id_hash: str
    trace_hash: str
    revision: int
    action: str
    canonical_label_id: str | None
    created_at_epoch_ms: int
    actor_id_hash: str
    previous_event_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        operation_id: str,
        trace_hash: str,
        revision: int,
        action: str,
        canonical_label_id: str | None,
        created_at_epoch_ms: int,
        actor_id: str,
        previous_event_hash: str | None = None,
    ) -> UserFeedbackEvent:
        if revision < 1:
            raise ValueError("feedback revision must be positive")
        if action not in {"confirm", "correct", "reject", "request_review"}:
            raise ValueError("feedback action is unsupported")
        if action == "correct" and not canonical_label_id:
            raise ValueError("correct feedback requires a canonical label reference")
        return cls(
            contract="pocketfinancer.user-feedback/1",
            event_id=event_id,
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            trace_hash=trace_hash,
            revision=revision,
            action=action,
            canonical_label_id=canonical_label_id,
            created_at_epoch_ms=created_at_epoch_ms,
            actor_id_hash=hashlib.sha256(actor_id.encode()).hexdigest(),
            previous_event_hash=previous_event_hash,
        )
