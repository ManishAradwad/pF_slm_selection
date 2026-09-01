"""Append-only user feedback contract for future native-app integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class UserFeedbackEvent:
    contract: str
    event_id: str
    operation_id_hash: str
    trace_hash: str
    revision: int
    action: str
    canonical_label_id: str | None
    canonical_label_revision: int | None
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
        canonical_label_revision: int | None,
        created_at_epoch_ms: int,
        actor_id: str,
        previous_event_hash: str | None = None,
    ) -> UserFeedbackEvent:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("feedback revision must be positive")
        if action not in {"confirm", "correct", "reject", "request_review"}:
            raise ValueError("feedback action is unsupported")
        if not event_id or not operation_id or not actor_id or created_at_epoch_ms < 0:
            raise ValueError("feedback identity or timestamp is invalid")
        if not _is_sha256(trace_hash):
            raise ValueError("feedback trace hash is invalid")
        if revision == 1 and previous_event_hash is not None:
            raise ValueError("first feedback revision cannot reference a predecessor")
        if revision > 1 and not _is_sha256(previous_event_hash or ""):
            raise ValueError("later feedback revision requires a predecessor hash")
        has_label_id = bool(canonical_label_id)
        has_label_revision = (
            isinstance(canonical_label_revision, int)
            and not isinstance(canonical_label_revision, bool)
            and canonical_label_revision > 0
        )
        if has_label_id != has_label_revision:
            raise ValueError("feedback canonical label ID and revision must appear together")
        if action in {"confirm", "correct"} and not has_label_id:
            raise ValueError(f"{action} feedback requires a canonical label reference")
        return cls(
            contract="pocketfinancer.user-feedback/1",
            event_id=event_id,
            operation_id_hash=hashlib.sha256(operation_id.encode()).hexdigest(),
            trace_hash=trace_hash,
            revision=revision,
            action=action,
            canonical_label_id=canonical_label_id,
            canonical_label_revision=canonical_label_revision,
            created_at_epoch_ms=created_at_epoch_ms,
            actor_id_hash=hashlib.sha256(actor_id.encode()).hexdigest(),
            previous_event_hash=previous_event_hash,
        )

    @property
    def event_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
