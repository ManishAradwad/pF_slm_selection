"""Deterministic template-component assignment to non-leaking annotation pools."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


POOL_NAMES = (
    "regression_only",
    "legacy_review",
    "annotation_training",
    "annotation_development",
    "protected_test",
    "later_time_holdout",
)


@dataclass(frozen=True, slots=True)
class PoolInput:
    source_id: str
    timestamp: str
    exact_body_hash: str
    normalized_template_hash: str
    sender_template_group_hash: str


def assign_pools(
    rows: list[PoolInput],
    *,
    regression_template_hashes: set[str],
    legacy_template_hashes: set[str],
    later_time_cutoff: str,
) -> dict[str, str]:
    cutoff = _parse_datetime(later_time_cutoff)
    components: dict[str, list[PoolInput]] = defaultdict(list)
    for row in rows:
        components[row.normalized_template_hash].append(row)

    assignments: dict[str, str] = {}
    for template_hash, component in sorted(components.items()):
        if template_hash in regression_template_hashes:
            pool = "regression_only"
        elif template_hash in legacy_template_hashes:
            pool = "legacy_review"
        elif all(_parse_datetime(row.timestamp) >= cutoff for row in component):
            pool = "later_time_holdout"
        else:
            bucket = int(hashlib.sha256(f"pool-v1:{template_hash}".encode()).hexdigest()[:8], 16) % 100
            if bucket < 80:
                pool = "annotation_training"
            elif bucket < 90:
                pool = "annotation_development"
            else:
                pool = "protected_test"
        for row in component:
            assignments[row.source_id] = pool
    if len(assignments) != len(rows):
        raise ValueError("pool assignment did not cover every source row exactly once")
    return assignments


def leakage_audit(rows: list[PoolInput], assignments: dict[str, str]) -> dict[str, Any]:
    indexes: dict[str, dict[str, set[str]]] = {
        pool: {
            "source_id": set(),
            "exact_body": set(),
            "normalized_template": set(),
            "sender_template_group": set(),
        }
        for pool in POOL_NAMES
    }
    for row in rows:
        pool = assignments[row.source_id]
        indexes[pool]["source_id"].add(row.source_id)
        indexes[pool]["exact_body"].add(row.exact_body_hash)
        indexes[pool]["normalized_template"].add(row.normalized_template_hash)
        indexes[pool]["sender_template_group"].add(row.sender_template_group_hash)

    pairwise: list[dict[str, Any]] = []
    maxima = {field: 0 for field in next(iter(indexes.values()))}
    for left_index, left in enumerate(POOL_NAMES):
        for right in POOL_NAMES[left_index + 1 :]:
            overlaps = {
                field: len(indexes[left][field] & indexes[right][field]) for field in maxima
            }
            for field, count in overlaps.items():
                maxima[field] = max(maxima[field], count)
            pairwise.append({"left": left, "right": right, "overlaps": overlaps})
    return {"passed": not any(maxima.values()), "maximum_overlap": maxima, "pairwise": pairwise}


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
