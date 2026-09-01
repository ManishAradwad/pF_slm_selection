"""Shared immutable types for deterministic SMS processing.

These objects deliberately keep source evidence in host memory. The compact model
contract contains candidate IDs only; it never contains offsets or copied values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Disposition(StrEnum):
    INVOKE = "invoke"
    DISCARD = "discard"
    RETAIN_REVIEW = "retain_review"


class SelectorAction(StrEnum):
    RUN_NORMAL = "run_normal"
    RUN_ASSISTIVE = "run_assistive"
    SKIP = "skip"


class CandidateKind(StrEnum):
    AMOUNT = "amount"
    DIRECTION = "direction"
    ACCOUNT = "account"
    COUNTERPARTY = "counterparty"


class CurrencyProvenance(StrEnum):
    EXPLICIT_CODE = "explicit_code"
    EXPLICIT_UNAMBIGUOUS_MARKER = "explicit_unambiguous_symbol_or_marker"
    USER_PRIMARY_DEFAULT = "user_primary_default"


class Direction(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"


class AccountState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class CounterpartyState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class TimestampProvenance(StrEnum):
    PLATFORM_RECEIVED = "platform_received"
    MESSAGE_EXPLICIT = "message_explicit"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    start_char: int
    end_char: int
    start_utf8: int
    end_utf8: int
    text: str

    @classmethod
    def from_source(cls, source: str, start: int, end: int) -> EvidenceSpan:
        if start < 0 or end < start or end > len(source):
            raise ValueError("evidence span is outside the source message")
        return cls(
            start_char=start,
            end_char=end,
            start_utf8=len(source[:start].encode("utf-8")),
            end_utf8=len(source[:end].encode("utf-8")),
            text=source[start:end],
        )


@dataclass(frozen=True, slots=True)
class Clause:
    clause_id: str
    evidence: EvidenceSpan


@dataclass(frozen=True, slots=True)
class Cue:
    cue_id: str
    kind: str
    clause_id: str
    evidence: EvidenceSpan
    reason_code: str


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    kind: CandidateKind
    clause_id: str | None
    evidence: EvidenceSpan | None
    value: dict[str, Any]
    context: tuple[str, ...] = ()
    explicit_absence: bool = False


@dataclass(frozen=True, slots=True)
class Analysis:
    contract: str
    analysis_id: str
    source_fingerprint: str
    profile_id: str
    config_hash: str
    primary_currency: str
    source_length_chars: int
    source_length_utf8: int
    clauses: tuple[Clause, ...]
    candidates: tuple[Candidate, ...]
    cues: tuple[Cue, ...]
    reason_codes: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def candidates_of(self, kind: CandidateKind) -> tuple[Candidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.kind == kind)

    def candidate_map(self) -> dict[str, Candidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, source: str) -> Analysis:
        """Rehydrate one immutable, source-bound stored analysis.

        Workbench target previews must use the analysis frozen into the corpus run;
        regenerating it with newer analyzer code would break run reproducibility.
        """

        try:
            if value["contract"] != "pocketfinancer.sms-analysis/1":
                raise ValueError("stored analysis contract is unsupported")
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != value["source_fingerprint"]:
                raise ValueError("stored analysis source fingerprint does not match")
            if value["source_length_chars"] != len(source) or value["source_length_utf8"] != len(
                source.encode("utf-8")
            ):
                raise ValueError("stored analysis source length does not match")
            clauses = tuple(
                Clause(str(item["clause_id"]), _evidence_from_dict(item["evidence"], source))
                for item in value["clauses"]
            )
            candidates = tuple(
                _candidate_from_dict(item, source) for item in value["candidates"]
            )
            cues = tuple(
                Cue(
                    cue_id=str(item["cue_id"]),
                    kind=str(item["kind"]),
                    clause_id=str(item["clause_id"]),
                    evidence=_evidence_from_dict(item["evidence"], source),
                    reason_code=str(item["reason_code"]),
                )
                for item in value["cues"]
            )
            reason_codes = tuple(str(reason) for reason in value["reason_codes"])
            metadata = dict(value["metadata"])
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("stored analysis"):
                raise
            raise ValueError("stored analysis is malformed") from exc
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("stored analysis has duplicate candidate IDs")
        clause_ids = {clause.clause_id for clause in clauses}
        if any(
            candidate.clause_id is not None and candidate.clause_id not in clause_ids
            for candidate in candidates
        ) or any(
            cue.clause_id != "cl_unknown" and cue.clause_id not in clause_ids
            for cue in cues
        ):
            raise ValueError("stored analysis references an unknown clause")
        return cls(
            contract=value["contract"],
            analysis_id=str(value["analysis_id"]),
            source_fingerprint=str(value["source_fingerprint"]),
            profile_id=str(value["profile_id"]),
            config_hash=str(value["config_hash"]),
            primary_currency=str(value["primary_currency"]),
            source_length_chars=int(value["source_length_chars"]),
            source_length_utf8=int(value["source_length_utf8"]),
            clauses=clauses,
            candidates=candidates,
            cues=cues,
            reason_codes=reason_codes,
            metadata=metadata,
        )

    def to_dict(self, *, include_source_evidence: bool = True) -> dict[str, Any]:
        value = json.loads(json.dumps(asdict(self), ensure_ascii=False))
        if not include_source_evidence:
            for clause in value["clauses"]:
                clause["evidence"].pop("text", None)
            for candidate in value["candidates"]:
                if candidate["evidence"] is not None:
                    candidate["evidence"].pop("text", None)
            for cue in value["cues"]:
                cue["evidence"].pop("text", None)
        return value


@dataclass(frozen=True, slots=True)
class TriageDecision:
    disposition: Disposition
    selector_action: SelectorAction
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReconstructedTransaction:
    analysis_id: str
    amount_candidate_id: str
    direction_candidate_id: str
    account_candidate_id: str
    counterparty_candidate_id: str
    minor_units: int
    currency: str
    currency_provenance: CurrencyProvenance
    direction: Direction
    account_state: AccountState
    account_evidence: EvidenceSpan | None
    counterparty_state: CounterpartyState
    counterparty_evidence: EvidenceSpan | None
    amount_evidence: EvidenceSpan
    direction_evidence: EvidenceSpan


@dataclass(frozen=True, slots=True)
class SelectorResult:
    decision: str
    transaction: ReconstructedTransaction | None = None


@dataclass(frozen=True, slots=True)
class PersistenceContext:
    timestamp_epoch_ms: int | None
    timestamp_provenance: TimestampProvenance
    account_resolution_count: int
    approved_currency_provenance: frozenset[CurrencyProvenance]


@dataclass(frozen=True, slots=True)
class PersistenceDecision:
    safe_to_persist: bool
    reason_codes: tuple[str, ...]


def _evidence_from_dict(value: dict[str, Any], source: str) -> EvidenceSpan:
    if not isinstance(value, dict):
        raise ValueError("stored analysis evidence is malformed")
    start = value.get("start_char")
    end = value.get("end_char")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("stored analysis evidence is malformed")
    expected = EvidenceSpan.from_source(source, start, end)
    if (
        value.get("start_utf8") != expected.start_utf8
        or value.get("end_utf8") != expected.end_utf8
        or value.get("text") != expected.text
    ):
        raise ValueError("stored analysis evidence does not match source")
    return expected


def _candidate_from_dict(value: dict[str, Any], source: str) -> Candidate:
    if not isinstance(value, dict) or not isinstance(value.get("explicit_absence"), bool):
        raise ValueError("stored analysis candidate is malformed")
    context = value.get("context")
    if not isinstance(context, list) or not all(isinstance(item, str) for item in context):
        raise ValueError("stored analysis candidate is malformed")
    candidate_value = value.get("value")
    if not isinstance(candidate_value, dict):
        raise ValueError("stored analysis candidate is malformed")
    evidence_value = value.get("evidence")
    clause_value = value.get("clause_id")
    if clause_value is not None and not isinstance(clause_value, str):
        raise ValueError("stored analysis candidate is malformed")
    return Candidate(
        candidate_id=str(value["candidate_id"]),
        kind=CandidateKind(value["kind"]),
        clause_id=clause_value,
        evidence=(
            _evidence_from_dict(evidence_value, source)
            if evidence_value is not None
            else None
        ),
        value=dict(candidate_value),
        context=tuple(context),
        explicit_absence=value["explicit_absence"],
    )
