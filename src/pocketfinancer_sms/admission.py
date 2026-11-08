"""Shared same-source admission semantics, independent of storage and inference.

Identity must be connector-scoped and reliable. A body hash is an integrity
check only, never an identity. Storage must serialize lookup/create per identity.
"""

from __future__ import annotations

from typing import Literal

AdmissionAction = Literal["new", "resume", "in_flight", "reuse", "conflict", "retry"]
ExistingState = Literal["ready", "in_flight", "settled", "interrupted"]


def admission_action(
    *,
    reliable_identity: bool,
    existing_state: ExistingState | None,
    same_evidence: bool = True,
    owner_retry: bool = False,
) -> AdmissionAction:
    """Decide before filtering, changing settings, analysis or model work.

    Interrupted work reuses its retryable outcome; only an explicit owner retry
    may create another extraction attempt. Ready work resumes its *stored*
    settings. Busy work is not another Review or permission to invoke a model.
    """
    if existing_state not in {None, "ready", "in_flight", "settled", "interrupted"}:
        raise ValueError("unsupported admission state")
    if owner_retry:
        if existing_state is None or not same_evidence:
            raise ValueError("retry requires an existing unchanged source")
        if existing_state == "in_flight":
            return "in_flight"
        return "retry"
    if not reliable_identity or existing_state is None:
        return "new"
    if not same_evidence:
        return "conflict"
    return {
        "ready": "resume",
        "in_flight": "in_flight",
        "settled": "reuse",
        "interrupted": "reuse",
    }[existing_state]
