"""High-recall triage policy over deterministic analysis."""

from __future__ import annotations

from collections import defaultdict

from .types import Analysis, CandidateKind, Disposition, SelectorAction, TriageDecision


def evaluate_triage(analysis: Analysis) -> TriageDecision:
    reasons = set(analysis.reason_codes)
    cues_by_clause: dict[str, set[str]] = defaultdict(set)
    for cue in analysis.cues:
        cues_by_clause[cue.clause_id].add(cue.kind)

    directions = analysis.candidates_of(CandidateKind.DIRECTION)
    amounts = analysis.candidates_of(CandidateKind.AMOUNT)
    completed_clauses = {candidate.clause_id for candidate in directions if candidate.clause_id}
    amount_clauses = {candidate.clause_id for candidate in amounts if candidate.clause_id}
    complete_supported_clauses = completed_clauses & amount_clauses

    if not analysis.metadata.get("input_valid", False):
        return _decision(Disposition.DISCARD, SelectorAction.SKIP, reasons, "discard_invalid_input")
    if analysis.metadata.get("is_outgoing") is True:
        return _decision(Disposition.DISCARD, SelectorAction.SKIP, reasons, "discard_reliable_outgoing")

    if not completed_clauses:
        if "credential_otp" in {cue.kind for cue in analysis.cues}:
            return _decision(
                Disposition.DISCARD,
                SelectorAction.SKIP,
                reasons,
                "discard_standalone_credential_otp",
            )
        if any(cue.kind == "request" for cue in analysis.cues):
            return _decision(
                Disposition.DISCARD,
                SelectorAction.SKIP,
                reasons,
                "discard_unapproved_request",
            )
        terminal_kinds = {"failure", "negation"}
        if any(cue.kind in terminal_kinds for cue in analysis.cues):
            return _decision(
                Disposition.DISCARD,
                SelectorAction.SKIP,
                reasons,
                "discard_explicit_non_posted_movement",
            )
        standalone_kinds = {"balance", "promotion", "administrative", "due"}
        cue_kinds = {cue.kind for cue in analysis.cues}
        if cue_kinds and cue_kinds <= standalone_kinds:
            return _decision(
                Disposition.DISCARD,
                SelectorAction.SKIP,
                reasons,
                "discard_unambiguous_standalone_non_event",
            )
        return _decision(
            Disposition.RETAIN_REVIEW,
            SelectorAction.SKIP,
            reasons,
            "review_no_completed_event_candidate",
        )

    if len(completed_clauses) > 1:
        action = SelectorAction.RUN_ASSISTIVE if complete_supported_clauses else SelectorAction.SKIP
        return _decision(
            Disposition.RETAIN_REVIEW,
            action,
            reasons,
            "review_multiple_completed_event_clauses",
        )

    if not complete_supported_clauses:
        return _decision(
            Disposition.RETAIN_REVIEW,
            SelectorAction.SKIP,
            reasons,
            "review_missing_core_candidate",
        )

    completed_clause = next(iter(completed_clauses))
    conflicting_non_posted = any(
        cue.kind in {"failure", "negation", "pending", "due", "request"}
        and cue.clause_id != completed_clause
        for cue in analysis.cues
    )
    same_clause_conflict = bool(
        cues_by_clause[completed_clause] & {"failure", "negation", "pending", "request"}
    )
    if "conflicting_currencies" in reasons or conflicting_non_posted or same_clause_conflict:
        return _decision(
            Disposition.RETAIN_REVIEW,
            SelectorAction.RUN_ASSISTIVE,
            reasons,
            "review_conflicting_or_ambiguous_context",
        )

    return _decision(
        Disposition.INVOKE,
        SelectorAction.RUN_NORMAL,
        reasons,
        "invoke_grounded_single_event",
    )


def _decision(
    disposition: Disposition,
    action: SelectorAction,
    reasons: set[str],
    terminal_reason: str,
) -> TriageDecision:
    reasons.add(terminal_reason)
    return TriageDecision(disposition, action, tuple(sorted(reasons)))
