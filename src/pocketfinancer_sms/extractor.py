"""Strict, source-grounded direct SMS extraction.

This module is deliberately separate from :mod:`selector`. The historical
candidate protocol remains readable while new operations use source spans and
host-owned normalization.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .currency import ISO_4217_CURRENT_CODES, ISO_MINOR_UNITS, CurrencyProvenance, parse_money
from .profiles import resolve_profiles
from .types import Analysis, Candidate, Cue, Direction, EvidenceSpan


EXTRACTOR_INPUT_CONTRACT = "pocketfinancer.sms-extractor-input/1"
EXTRACTOR_CONTRACT = "pocketfinancer.sms-extractor/1"
EXTRACTOR_VALIDATION_PROFILE = "pocketfinancer.extractor-validation-profile/1"
EXTRACTOR_PROMPT = "pocketfinancer.extractor-prompt/1"
RAW_OUTPUT_UTF8_BYTE_LIMIT = 16_384

_DECLARED_MONEY = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?\Z")
_MONEY_NUMBER = re.compile(
    r"(?<![\w,])(?:\d{1,3}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w,])"
)
_ACCOUNT_VPA = re.compile(r"[A-Z0-9._-]{2,}@[A-Z][A-Z0-9.-]{1,}", re.IGNORECASE)
_ACCOUNT_MASKED = re.compile(r"(?:[xX*\u2022-]{2,}\s*)?\d{3,8}")


class ExtractionValidationError(ValueError):
    """A stable, display-independent extractor validation failure."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A zero-based, half-open Unicode-scalar source selection."""

    start_scalar: int
    end_scalar: int
    text: str

    @classmethod
    def from_source(cls, source: str, start_scalar: int, end_scalar: int) -> "SourceSpan":
        if (
            isinstance(start_scalar, bool)
            or isinstance(end_scalar, bool)
            or not isinstance(start_scalar, int)
            or not isinstance(end_scalar, int)
            or start_scalar < 0
            or end_scalar <= start_scalar
            or end_scalar > len(source)
        ):
            raise ExtractionValidationError("extractor_evidence_out_of_bounds")
        text = source[start_scalar:end_scalar]
        if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
            raise ExtractionValidationError("extractor_evidence_out_of_bounds")
        return cls(start_scalar, end_scalar, text)

    @classmethod
    def from_payload(cls, value: Any, source: str) -> "SourceSpan":
        if not isinstance(value, dict) or set(value) != {
            "start_scalar",
            "end_scalar",
            "text",
        }:
            raise ExtractionValidationError("extractor_evidence_invalid")
        expected = cls.from_source(source, value["start_scalar"], value["end_scalar"])
        if not isinstance(value["text"], str) or value["text"] != expected.text:
            raise ExtractionValidationError("extractor_evidence_mismatch")
        return expected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizedExtraction:
    minor_units: int
    currency: str
    direction: Direction
    account_reference: str
    counterparty: str | None
    amount_span: SourceSpan
    direction_span: SourceSpan
    account_span: SourceSpan
    counterparty_span: SourceSpan | None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    decision: str
    transaction: NormalizedExtraction | None = None

    def __post_init__(self) -> None:
        if self.decision not in {"none", "abstain", "posted"}:
            raise ValueError("unsupported extraction decision")
        if (self.decision == "posted") != (self.transaction is not None):
            raise ValueError("extraction decision and transaction are inconsistent")


def build_extractor_input(
    source: str,
    analysis: Analysis,
    *,
    sender_family: str,
    primary_currency: str,
    enabled_profile_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build one direct request containing optional advisory analyzer evidence."""

    if not isinstance(source, str) or not source:
        raise ValueError("extractor source must be a nonempty string")
    if analysis.contract != "pocketfinancer.sms-analysis/2":
        raise ValueError("extractor requires sms-analysis/2 advisory provenance")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != analysis.source_fingerprint:
        raise ValueError("extractor source does not match the current analysis")
    if primary_currency not in ISO_MINOR_UNITS:
        raise ValueError("extractor primary currency is unsupported")
    profile_ids = tuple(enabled_profile_ids)
    resolve_profiles(profile_ids)
    if not isinstance(sender_family, str) or not sender_family.strip():
        raise ValueError("extractor sender family is required")

    evidence = [
        _candidate_advisory(item, analysis.contract) for item in analysis.candidates
    ]
    evidence.extend(_cue_advisory(item, analysis.contract) for item in analysis.cues)
    return {
        "contract": EXTRACTOR_INPUT_CONTRACT,
        "output_contract": EXTRACTOR_CONTRACT,
        "message": source,
        "sender_family": sender_family,
        "primary_currency": primary_currency,
        "enabled_profile_ids": list(profile_ids),
        "advisory_evidence": evidence,
        "output_rules": {
            "one_json_document": True,
            "decisions": ["none", "abstain", "posted"],
            "posted_required_fields": ["amount", "direction", "account", "counterparty"],
            "source_spans": "zero_based_half_open_unicode_scalars",
            "transaction_time_forbidden": True,
        },
    }


def parse_and_normalize_extraction(
    raw_output: str,
    source: str,
    *,
    primary_currency: str,
    enabled_profile_ids: tuple[str, ...] | list[str],
) -> ExtractionResult:
    """Parse exactly one output document, validate spans, then normalize semantics."""

    if not isinstance(raw_output, str):
        raise ExtractionValidationError("extractor_malformed_json")
    try:
        raw_output_size = len(raw_output.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ExtractionValidationError("extractor_malformed_json") from exc
    if raw_output_size > RAW_OUTPUT_UTF8_BYTE_LIMIT:
        raise ExtractionValidationError("runtime_output_truncated")
    decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
    )
    document_start = len(raw_output) - len(raw_output.lstrip())
    try:
        payload, document_end = decoder.raw_decode(raw_output, document_start)
    except _DuplicateJsonKeyError as exc:
        raise ExtractionValidationError("extractor_duplicate_json_key") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExtractionValidationError("extractor_malformed_json") from exc
    if raw_output[document_end:].strip():
        raise ExtractionValidationError("extractor_extra_content")
    if not isinstance(payload, dict):
        raise ExtractionValidationError("extractor_output_not_object")
    decision = payload.get("decision")
    if not isinstance(decision, str):
        raise ExtractionValidationError("extractor_decision_type_invalid")
    if decision in {"none", "abstain"}:
        if set(payload) != {"decision"}:
            raise ExtractionValidationError("extractor_non_posted_extra_fields")
        return ExtractionResult(decision)
    if decision != "posted":
        raise ExtractionValidationError("extractor_unknown_decision")

    expected = {"decision", "amount", "direction", "account", "counterparty"}
    if set(payload) != expected:
        missing = expected - set(payload)
        if "amount" in missing:
            raise ExtractionValidationError("extractor_missing_amount")
        if "direction" in missing:
            raise ExtractionValidationError("extractor_missing_direction")
        if "account" in missing:
            raise ExtractionValidationError("extractor_missing_account")
        raise ExtractionValidationError("extractor_posted_field_set_invalid")

    amount_value, amount_span = _field(payload["amount"], "value", source, "amount")
    amount_currency = _required_string(payload["amount"], "currency", "extractor_currency_invalid")
    direction_value, direction_span = _field(payload["direction"], "value", source, "direction")
    account_value, account_span = _field(payload["account"], "reference", source, "account")
    counterparty_value, counterparty_span = _nullable_counterparty(payload["counterparty"], source)

    profile_ids = tuple(enabled_profile_ids)
    profiles = resolve_profiles(profile_ids)
    if primary_currency not in ISO_MINOR_UNITS:
        raise ExtractionValidationError("extractor_currency_invalid")
    currency = amount_currency.upper()
    if amount_currency != currency or currency not in ISO_MINOR_UNITS:
        raise ExtractionValidationError("extractor_currency_invalid")
    _validate_currency_grounding(amount_span.text, currency, primary_currency, profiles)
    minor_units = _normalize_money(amount_span.text, amount_value, currency)
    try:
        direction = Direction(direction_value)
    except ValueError as exc:
        raise ExtractionValidationError("extractor_direction_invalid") from exc
    if not _direction_is_grounded(direction, direction_span.text):
        raise ExtractionValidationError("extractor_direction_invalid")
    normalized_account = normalize_account_reference(account_span.text)
    if not normalized_account or normalize_account_reference(account_value) != normalized_account:
        raise ExtractionValidationError("extractor_account_reference_invalid")
    normalized_counterparty = None
    if counterparty_span is not None:
        normalized_counterparty = normalize_counterparty(counterparty_span.text)
        if (
            not normalized_counterparty
            or normalize_counterparty(counterparty_value or "") != normalized_counterparty
        ):
            raise ExtractionValidationError("extractor_counterparty_invalid")

    return ExtractionResult(
        "posted",
        NormalizedExtraction(
            minor_units=minor_units,
            currency=currency,
            direction=direction,
            account_reference=normalized_account,
            counterparty=normalized_counterparty,
            amount_span=amount_span,
            direction_span=direction_span,
            account_span=account_span,
            counterparty_span=counterparty_span,
        ),
    )


def normalize_account_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    vpa = _ACCOUNT_VPA.search(normalized)
    if vpa:
        return vpa.group(0)
    matches = list(_ACCOUNT_MASKED.finditer(normalized))
    if len(matches) != 1:
        return ""
    digits = "".join(char for char in matches[0].group(0) if char.isdecimal())
    return digits if 3 <= len(digits) <= 8 else ""


def normalize_counterparty(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = " ".join(normalized.split())
    return normalized if normalized and len(normalized) <= 256 else ""


def _candidate_advisory(candidate: Candidate, analyzer_version: str) -> dict[str, Any]:
    return {
        "kind": candidate.kind.value,
        "source_span": _legacy_span(candidate.evidence),
        "clause": candidate.clause_id,
        "suggested_interpretation": candidate.value,
        "provenance": {
            "candidate_id": candidate.candidate_id,
            "analyzer_kind": "candidate",
        },
        "analyzer_version": analyzer_version,
    }


def _cue_advisory(cue: Cue, analyzer_version: str) -> dict[str, Any]:
    return {
        "kind": cue.kind,
        "source_span": _legacy_span(cue.evidence),
        "clause": cue.clause_id,
        "suggested_interpretation": {"reason_code": cue.reason_code},
        "provenance": {"cue_id": cue.cue_id, "analyzer_kind": "cue"},
        "analyzer_version": analyzer_version,
    }


def _legacy_span(evidence: EvidenceSpan | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return SourceSpan(evidence.start_char, evidence.end_char, evidence.text).to_dict()


def _field(value: Any, value_key: str, source: str, field: str) -> tuple[str, SourceSpan]:
    expected = {value_key, "evidence"}
    if field == "amount":
        expected.add("currency")
    if not isinstance(value, dict) or set(value) != expected:
        raise ExtractionValidationError(f"extractor_{field}_invalid")
    field_value = _required_string(value, value_key, f"extractor_{field}_invalid")
    return field_value, SourceSpan.from_payload(value["evidence"], source)


def _required_string(value: Any, key: str, reason: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get(key), str) or not value[key]:
        raise ExtractionValidationError(reason)
    return value[key]


def _nullable_counterparty(value: Any, source: str) -> tuple[str | None, SourceSpan | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict) or set(value) != {"value", "evidence"}:
        raise ExtractionValidationError("extractor_counterparty_invalid")
    text = _required_string(value, "value", "extractor_counterparty_invalid")
    return text, SourceSpan.from_payload(value["evidence"], source)


def _validate_currency_grounding(
    evidence: str, currency: str, primary_currency: str, profiles: tuple[Any, ...]
) -> None:
    normalized = unicodedata.normalize("NFKC", evidence).casefold()
    codes = {match.group(0).upper() for match in re.finditer(r"\b[A-Za-z]{3}\b", normalized)}
    real_codes = codes & ISO_4217_CURRENT_CODES
    if real_codes and real_codes != {currency}:
        raise ExtractionValidationError("extractor_currency_invalid")
    marker_currencies = {
        code
        for profile in profiles
        for code, markers in profile.explicit_markers.items()
        if any(marker.casefold() in normalized for marker in markers)
    }
    if marker_currencies and marker_currencies != {currency}:
        raise ExtractionValidationError("extractor_currency_invalid")
    if not real_codes and not marker_currencies and currency != primary_currency:
        raise ExtractionValidationError("extractor_currency_invalid")


def _normalize_money(evidence: str, declared_value: str, currency: str) -> int:
    if _DECLARED_MONEY.fullmatch(declared_value) is None:
        raise ExtractionValidationError("extractor_amount_invalid")
    normalized = unicodedata.normalize("NFKC", evidence)
    numbers = [match.group(0) for match in _MONEY_NUMBER.finditer(normalized)]
    if len(numbers) != 1:
        raise ExtractionValidationError("extractor_amount_invalid")
    try:
        grounded = parse_money(
            numbers[0], currency=currency, provenance=CurrencyProvenance.EXPLICIT_CODE
        )
        declared = parse_money(
            declared_value, currency=currency, provenance=CurrencyProvenance.EXPLICIT_CODE
        )
    except ValueError as exc:
        raise ExtractionValidationError("extractor_amount_invalid") from exc
    if grounded.minor_units != declared.minor_units:
        raise ExtractionValidationError("extractor_amount_value_disagreement")
    return grounded.minor_units


def _direction_is_grounded(direction: Direction, evidence: str) -> bool:
    normalized = unicodedata.normalize("NFKC", evidence).casefold()
    terms = {
        Direction.DEBIT: (
            "debit",
            "debited",
            "deducted",
            "withdrawn",
            "spent",
            "paid",
            "charged",
        ),
        Direction.CREDIT: ("credit", "credited", "deposited", "received", "refunded"),
    }
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms[direction])
