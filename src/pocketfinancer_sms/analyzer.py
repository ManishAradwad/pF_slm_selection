"""Generalized deterministic SMS analyzer shared by triage and candidate selection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .currency import CurrencyContext, parse_money
from .profiles import AnalyzerProfile, resolve_profiles
from .structural_text import clause_for_span, split_clauses
from .types import (
    Analysis,
    Candidate,
    CandidateKind,
    Cue,
    CurrencyProvenance,
    Direction,
    EvidenceSpan,
)


ANALYSIS_CONTRACT = "pocketfinancer.sms-analysis/1"

_NUMBER = r"(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,3})?"
_EXPLICIT_CODE_AMOUNT = re.compile(
    rf"(?<![A-Z])(?P<currency>[A-Z]{{3}})\s*[:.-]?\s*(?P<number>{_NUMBER})(?!\d)",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(rf"(?<![\w])(?P<number>{_NUMBER})(?![\w])")

_DIRECTION_PATTERNS: tuple[tuple[Direction, re.Pattern[str]], ...] = (
    (
        Direction.DEBIT,
        re.compile(
            r"\b(?:has\s+been\s+|was\s+|is\s+)?(?:debited|deducted|withdrawn|spent|paid|charged)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Direction.CREDIT,
        re.compile(
            r"\b(?:has\s+been\s+|was\s+|is\s+)?(?:credited|deposited|received|refunded)\b",
            re.IGNORECASE,
        ),
    ),
)

_ACCOUNT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bank_account",
        re.compile(
            r"\b(?:a/?c|acct|account)\s*(?:no\.?\s*)?(?:ending\s*(?:in|with)?\s*)?"
            r"(?P<identifier>(?:[xX*•-]{2,}\s*)?\d{3,8})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "card",
        re.compile(
            r"\b(?:credit\s+|debit\s+)?card\s*(?:ending\s*(?:in|with)?\s*)?"
            r"(?P<identifier>(?:[xX*•-]{2,}\s*)?\d{3,8})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "vpa",
        re.compile(r"\b(?P<identifier>[A-Z0-9._-]{2,}@[A-Z][A-Z0-9.-]{1,})\b", re.IGNORECASE),
    ),
)

_COUNTERPARTY_PATTERN = re.compile(
    r"\b(?:at|to|from|by)\s+(?P<name>[A-Z0-9][A-Z0-9&._@/-]*(?:\s+[A-Z0-9][A-Z0-9&._@/-]*){0,4})",
    re.IGNORECASE,
)

_CUE_PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
    "failure": (
        "non_posted_failure",
        re.compile(r"\b(?:failed|declined|rejected|unsuccessful|could\s+not\s+be\s+processed)\b", re.I),
    ),
    "negation": (
        "negated_movement",
        re.compile(r"\b(?:not\s+(?:debited|credited|charged|processed)|no\s+money\s+(?:was\s+)?(?:debited|credited))\b", re.I),
    ),
    "pending": ("pending_event", re.compile(r"\b(?:pending|processing|in\s+progress)\b", re.I)),
    "due": ("amount_due", re.compile(r"\b(?:amount\s+due|payment\s+due|minimum\s+due|due\s+date)\b", re.I)),
    "request": (
        "request_or_authorization",
        re.compile(r"\b(?:collect\s+request|payment\s+request|approve|authorize|mandate\s+request)\b", re.I),
    ),
    "balance": (
        "balance_information",
        re.compile(r"\b(?:available|avail|avl|current|closing)\s+(?:a/?c\s+)?bal(?:ance)?\b", re.I),
    ),
    "promotion": (
        "promotion",
        re.compile(r"\b(?:offer|cashback\s+offer|discount|sale|apply\s+now|limited\s+time)\b", re.I),
    ),
    "administrative": (
        "administrative",
        re.compile(r"\b(?:statement\s+generated|kyc|profile\s+updated|nomination|registered)\b", re.I),
    ),
    "credential_otp": (
        "credential_otp",
        re.compile(r"\b(?:otp|one[- ]time\s+password|verification\s+code|login\s+code|passcode)\b", re.I),
    ),
}


class DeterministicSmsAnalyzer:
    """Enumerate source-backed cues and candidates without assigning human truth."""

    def __init__(self, currency_context: CurrencyContext) -> None:
        self.currency_context = currency_context
        self.profiles = resolve_profiles(currency_context.profile_ids)
        self.transaction_terms = tuple(
            dict.fromkeys(term for profile in self.profiles for term in profile.transaction_terms)
        )

    def analyze(
        self,
        source: str,
        *,
        operation_id: str,
        is_outgoing: bool | None = None,
        input_valid: bool = True,
    ) -> Analysis:
        if not isinstance(source, str):
            source = ""
            input_valid = False
        if not operation_id:
            raise ValueError("operation_id is required to scope candidate IDs")

        source_fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()
        analysis_id = hashlib.sha256(
            f"{operation_id}\0{source_fingerprint}\0{self.currency_context.config_hash}".encode()
        ).hexdigest()[:24]
        clauses = split_clauses(source)
        candidates: list[Candidate] = []
        cues: list[Cue] = []

        for kind, (reason_code, pattern) in _CUE_PATTERNS.items():
            for match in pattern.finditer(source):
                cues.append(self._cue(source, clauses, analysis_id, kind, reason_code, match))

        for profile in self.profiles:
            for rail, terms in profile.rails.items():
                rail_pattern = re.compile(
                    r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
                    re.IGNORECASE,
                )
                for match in rail_pattern.finditer(source):
                    cues.append(
                        self._cue(source, clauses, analysis_id, "payment_rail", f"rail_{rail}", match)
                    )

        candidates.extend(self._amount_candidates(source, clauses, analysis_id))
        candidates.extend(self._direction_candidates(source, clauses, analysis_id))
        candidates.extend(self._account_candidates(source, clauses, analysis_id))
        candidates.extend(self._counterparty_candidates(source, clauses, analysis_id))
        candidates.extend(self._absence_candidates(analysis_id))

        candidates = self._deduplicate_candidates(candidates)
        cues = self._deduplicate_cues(cues)
        reason_codes = self._aggregate_reason_codes(candidates, cues, input_valid, is_outgoing)
        metadata: dict[str, Any] = {
            "input_valid": bool(input_valid and source.strip()),
            "is_outgoing": is_outgoing,
            "completed_event_clause_count": len(
                {
                    candidate.clause_id
                    for candidate in candidates
                    if candidate.kind == CandidateKind.DIRECTION and candidate.clause_id is not None
                }
            ),
        }
        return Analysis(
            contract=ANALYSIS_CONTRACT,
            analysis_id=analysis_id,
            source_fingerprint=source_fingerprint,
            profile_id="+".join(profile.profile_id for profile in self.profiles),
            config_hash=self.currency_context.config_hash,
            primary_currency=self.currency_context.primary_currency,
            source_length_chars=len(source),
            source_length_utf8=len(source.encode("utf-8")),
            clauses=clauses,
            candidates=tuple(candidates),
            cues=tuple(cues),
            reason_codes=tuple(sorted(reason_codes)),
            metadata=metadata,
        )

    def _amount_candidates(
        self, source: str, clauses: tuple, analysis_id: str
    ) -> Iterable[Candidate]:
        covered: list[tuple[int, int]] = []
        for match in _EXPLICIT_CODE_AMOUNT.finditer(source):
            code = match.group("currency").upper()
            try:
                money = parse_money(
                    match.group("number"),
                    currency=code,
                    provenance=CurrencyProvenance.EXPLICIT_CODE,
                )
            except ValueError:
                continue
            covered.append(match.span())
            yield self._candidate(
                source,
                clauses,
                analysis_id,
                CandidateKind.AMOUNT,
                match.start(),
                match.end(),
                {
                    "minor_units": money.minor_units,
                    "currency": money.currency,
                    "currency_provenance": money.provenance.value,
                },
            )

        marker_patterns = self._marker_patterns()
        for currency, marker, pattern in marker_patterns:
            for match in pattern.finditer(source):
                if self._overlaps(match.span(), covered):
                    continue
                try:
                    money = parse_money(
                        match.group("number"),
                        currency=currency,
                        provenance=CurrencyProvenance.EXPLICIT_UNAMBIGUOUS_MARKER,
                    )
                except ValueError:
                    continue
                covered.append(match.span())
                yield self._candidate(
                    source,
                    clauses,
                    analysis_id,
                    CandidateKind.AMOUNT,
                    match.start(),
                    match.end(),
                    {
                        "minor_units": money.minor_units,
                        "currency": money.currency,
                        "currency_provenance": money.provenance.value,
                        "marker": marker,
                    },
                )

        lowered = source.casefold()
        has_transaction_context = any(term.casefold() in lowered for term in self.transaction_terms)
        if not has_transaction_context:
            return
        for match in _NUMBER_PATTERN.finditer(source):
            if self._overlaps(match.span(), covered):
                continue
            local_context = source[max(0, match.start() - 40) : min(len(source), match.end() + 40)]
            if not any(term.casefold() in local_context.casefold() for term in self.transaction_terms):
                continue
            try:
                money = parse_money(
                    match.group("number"),
                    currency=self.currency_context.primary_currency,
                    provenance=CurrencyProvenance.USER_PRIMARY_DEFAULT,
                )
            except ValueError:
                continue
            yield self._candidate(
                source,
                clauses,
                analysis_id,
                CandidateKind.AMOUNT,
                match.start(),
                match.end(),
                {
                    "minor_units": money.minor_units,
                    "currency": money.currency,
                    "currency_provenance": money.provenance.value,
                },
                context=("bare_amount_with_transaction_context",),
            )

    def _direction_candidates(
        self, source: str, clauses: tuple, analysis_id: str
    ) -> Iterable[Candidate]:
        for direction, pattern in _DIRECTION_PATTERNS:
            for match in pattern.finditer(source):
                yield self._candidate(
                    source,
                    clauses,
                    analysis_id,
                    CandidateKind.DIRECTION,
                    match.start(),
                    match.end(),
                    {"direction": direction.value},
                )

    def _account_candidates(
        self, source: str, clauses: tuple, analysis_id: str
    ) -> Iterable[Candidate]:
        for account_type, pattern in _ACCOUNT_PATTERNS:
            for match in pattern.finditer(source):
                identifier = match.group("identifier")
                yield self._candidate(
                    source,
                    clauses,
                    analysis_id,
                    CandidateKind.ACCOUNT,
                    match.start(),
                    match.end(),
                    {"account_type": account_type, "identifier": identifier},
                )

    def _counterparty_candidates(
        self, source: str, clauses: tuple, analysis_id: str
    ) -> Iterable[Candidate]:
        stop_words = {"your", "the", "a", "an", "account", "card", "bank"}
        for match in _COUNTERPARTY_PATTERN.finditer(source):
            name = match.group("name").strip(" .,:;-")
            words = name.casefold().split()
            if not name or all(word in stop_words for word in words):
                continue
            start = match.start("name")
            end = start + len(name)
            yield self._candidate(
                source,
                clauses,
                analysis_id,
                CandidateKind.COUNTERPARTY,
                start,
                end,
                {"surface": name},
            )

    def _absence_candidates(self, analysis_id: str) -> Iterable[Candidate]:
        for kind in (CandidateKind.ACCOUNT, CandidateKind.COUNTERPARTY):
            value = {"state": "absent"}
            yield Candidate(
                candidate_id=self._candidate_id(analysis_id, kind, None, value),
                kind=kind,
                clause_id=None,
                evidence=None,
                value=value,
                context=("explicit_absence",),
                explicit_absence=True,
            )

    def _marker_patterns(self) -> Iterable[tuple[str, str, re.Pattern[str]]]:
        seen: set[tuple[str, str]] = set()
        for profile in self.profiles:
            for currency, markers in profile.explicit_markers.items():
                for marker in markers:
                    key = (currency, marker.casefold())
                    if key in seen or marker.casefold() == currency.casefold():
                        continue
                    seen.add(key)
                    escaped = re.escape(marker)
                    pattern = re.compile(
                        rf"(?<!\w)(?:{escaped})\s*[:.-]?\s*(?P<number>{_NUMBER})(?!\d)",
                        re.IGNORECASE,
                    )
                    yield currency, marker, pattern

    def _candidate(
        self,
        source: str,
        clauses: tuple,
        analysis_id: str,
        kind: CandidateKind,
        start: int,
        end: int,
        value: dict[str, Any],
        *,
        context: tuple[str, ...] = (),
    ) -> Candidate:
        evidence = EvidenceSpan.from_source(source, start, end)
        clause_id = clause_for_span(clauses, start, end)
        return Candidate(
            candidate_id=self._candidate_id(analysis_id, kind, evidence, value),
            kind=kind,
            clause_id=clause_id,
            evidence=evidence,
            value=value,
            context=context,
        )

    def _cue(
        self,
        source: str,
        clauses: tuple,
        analysis_id: str,
        kind: str,
        reason_code: str,
        match: re.Match[str],
    ) -> Cue:
        evidence = EvidenceSpan.from_source(source, match.start(), match.end())
        clause_id = clause_for_span(clauses, match.start(), match.end()) or "cl_unknown"
        digest = hashlib.sha256(
            f"{analysis_id}|cue|{kind}|{match.start()}|{match.end()}".encode()
        ).hexdigest()[:12]
        return Cue(f"q_{digest}", kind, clause_id, evidence, reason_code)

    @staticmethod
    def _candidate_id(
        analysis_id: str,
        kind: CandidateKind,
        evidence: EvidenceSpan | None,
        value: dict[str, Any],
    ) -> str:
        span = "absent" if evidence is None else f"{evidence.start_char}:{evidence.end_char}"
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{analysis_id}|{kind.value}|{span}|{payload}".encode()).hexdigest()[:12]
        prefix = {
            CandidateKind.AMOUNT: "amt",
            CandidateKind.DIRECTION: "dir",
            CandidateKind.ACCOUNT: "acc",
            CandidateKind.COUNTERPARTY: "cp",
        }[kind]
        return f"{prefix}_{digest}"

    @staticmethod
    def _overlaps(span: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
        return any(span[0] < other_end and span[1] > other_start for other_start, other_end in covered)

    @staticmethod
    def _deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
        return list({candidate.candidate_id: candidate for candidate in candidates}.values())

    @staticmethod
    def _deduplicate_cues(cues: list[Cue]) -> list[Cue]:
        return list({cue.cue_id: cue for cue in cues}.values())

    @staticmethod
    def _aggregate_reason_codes(
        candidates: list[Candidate], cues: list[Cue], input_valid: bool, is_outgoing: bool | None
    ) -> set[str]:
        reasons = {cue.reason_code for cue in cues}
        if not input_valid:
            reasons.add("invalid_input")
        if is_outgoing is True:
            reasons.add("reliable_outgoing_metadata")
        currencies = {
            candidate.value["currency"]
            for candidate in candidates
            if candidate.kind == CandidateKind.AMOUNT
        }
        if len(currencies) > 1:
            reasons.add("conflicting_currencies")
        if any(candidate.kind == CandidateKind.AMOUNT for candidate in candidates):
            reasons.add("amount_candidate_present")
        if any(candidate.kind == CandidateKind.DIRECTION for candidate in candidates):
            reasons.add("completed_direction_candidate_present")
        return reasons
