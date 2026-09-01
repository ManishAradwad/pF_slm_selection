"""Aggregate-safe corpus reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..types import CandidateKind


def aggregate_reports(records: list[dict[str, Any]]) -> dict[str, Any]:
    pool_counts = Counter(record["pool"] for record in records)
    dispositions = Counter(record["weak_facets"]["disposition"] for record in records)
    selector_actions = Counter(record["weak_facets"]["selector_action"] for record in records)
    operational_classes = Counter(
        record["weak_facets"]["operational_class"] for record in records
    )
    candidate_fields = Counter()
    complete_event = 0
    for record in records:
        kinds = {
            candidate["kind"]
            for candidate in record["analysis"]["candidates"]
            if not candidate["explicit_absence"]
        }
        for kind in CandidateKind:
            if kind.value in kinds:
                candidate_fields[kind.value] += 1
        if CandidateKind.AMOUNT.value in kinds and CandidateKind.DIRECTION.value in kinds:
            complete_event += 1
    return {
        "row_count": len(records),
        "pool_counts": dict(sorted(pool_counts.items())),
        "prefilter_dispositions": dict(sorted(dispositions.items())),
        "selector_actions": dict(sorted(selector_actions.items())),
        "weak_operational_classes": dict(sorted(operational_classes.items())),
        "candidate_coverage": {
            "field_row_counts": dict(sorted(candidate_fields.items())),
            "complete_amount_and_direction_row_count": complete_event,
        },
    }
