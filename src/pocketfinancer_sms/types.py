"""Shared immutable types for deterministic SMS processing.

These objects deliberately keep source evidence in host memory. The compact model
contract contains candidate IDs only; it never contains offsets or copied values.
"""

from __future__ import annotations

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

    def to_dict(self, *, include_source_evidence: bool = True) -> dict[str, Any]:
        value = asdict(self)
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
