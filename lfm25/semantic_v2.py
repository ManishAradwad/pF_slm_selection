"""Platform-neutral Semantic V2 reference validation and eligibility projection.

Semantic V2 is a host semantic contract, not a universal model prompt or output
protocol.  Model-family profiles added in later phases may map their own safely
parsed output into this contract.  In particular, source timestamps are injected
by the host and never accepted as a model field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping


SEMANTIC_CONTRACT_ID = "pocketfinancer_semantic_v2"
SEMANTIC_CONTRACT_VERSION = 2
EVIDENCE_OFFSET_CONVENTION = "zero_based_half_open_utf8_bytes"
PORTABLE_SIGNED_INT64_MAX = (1 << 63) - 1
_PORTABLE_SIGNED_INT64_MAX_TEXT = str(PORTABLE_SIGNED_INT64_MAX)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_V2_SCHEMA_PATH = _REPOSITORY_ROOT / "configs/contracts/pocketfinancer-semantic-v2.schema.json"
_DECIMAL_TEXT_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_MONEY_NUMBER_RE = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)(?:\.[0-9]+)?")
_UPPERCASE_CURRENCY_CODE_RE = re.compile(r"(?<![A-Z])[A-Z]{3}(?![A-Z])")
_UNAMBIGUOUS_CURRENCY_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INR", re.compile(r"₹|\bRs\.?(?![A-Za-z])", re.IGNORECASE)),
    ("EUR", re.compile(r"€")),
    ("GBP", re.compile(r"£")),
)
_CURRENCY_MINOR_EXPONENTS = {"INR": 2, "USD": 2, "EUR": 2, "GBP": 2}


class SemanticV2Error(ValueError):
    """The supplied host semantic record violates the versioned contract."""


class Scope(str, Enum):
    BANK_CARD = "bank_card"
    WALLET_BNPL = "wallet_bnpl"
    OTHER = "other"


class PostingStatus(str, Enum):
    POSTED = "posted"
    NOT_POSTED = "not_posted"


class EventCardinality(str, Enum):
    NONE = "none"
    SINGLE = "single"
    MULTIPLE = "multiple"


class Direction(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


_DIRECTION_EVIDENCE_LEXICON_VERSION = 1
_DIRECTION_EVIDENCE_LEXEMES = {
    Direction.DEBIT: frozenset(
        {"debit", "debited", "deducted", "withdrawal", "withdrawn", "spent", "paid", "charged"}
    ),
    Direction.CREDIT: frozenset(
        {"credit", "credited", "deposit", "deposited", "received", "refunded"}
    ),
}


class CounterpartyState(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class TimestampProvenance(str, Enum):
    ANDROID_SMS_MESSAGE_DATE = "android_sms_message_date"
    IOS_INBOX_ALERT_RECEIVED_AT_SUPPLIED = "ios_inbox_alert_received_at_supplied"
    IOS_INBOX_ALERT_RECEIVED_AT_ASSIGNED_DURING_INGESTION = (
        "ios_inbox_alert_received_at_assigned_during_ingestion"
    )
    SYNTHETIC_SYSTEM_TIMESTAMP_INJECTED = "synthetic_system_timestamp_injected"


class IneligibilityReason(str, Enum):
    NOT_POSTED = "not_posted"
    EVENT_CARDINALITY_NOT_SINGLE = "event_cardinality_not_single"
    SCOPE_NOT_BANK_CARD = "scope_not_bank_card"
    MISSING_AMOUNT = "missing_amount"
    CURRENCY_NOT_INR = "currency_not_inr"
    EXACT_MINOR_UNITS_UNAVAILABLE = "exact_minor_units_unavailable"
    MISSING_DIRECTION = "missing_direction"
    MISSING_ACCOUNT = "missing_account"


@dataclass(frozen=True)
class EvidenceSpan:
    """A zero-based, half-open UTF-8 byte range into the original message."""

    start_utf8_byte: int
    end_utf8_byte: int


@dataclass(frozen=True)
class SourceMetadata:
    """Trusted host timestamp metadata; excluded from every model protocol."""

    received_at_epoch_ms: int
    received_at_provenance: TimestampProvenance


@dataclass(frozen=True)
class ExactAmount:
    """Source-grounded exact decimal money and its non-rounding minor-unit derivative."""

    decimal_text: str
    currency: str
    minor_units: int | None
    evidence: EvidenceSpan


@dataclass(frozen=True)
class DirectionEvidence:
    value: Direction
    evidence: EvidenceSpan


@dataclass(frozen=True)
class TextEvidence:
    value: str
    evidence: EvidenceSpan


@dataclass(frozen=True)
class Counterparty:
    state: CounterpartyState
    value: str | None
    evidence: EvidenceSpan | None


@dataclass(frozen=True)
class SemanticEvent:
    event_id: str
    amount: ExactAmount | None
    direction: DirectionEvidence | None
    account: TextEvidence | None
    counterparty: Counterparty


@dataclass(frozen=True)
class SemanticRecord:
    source_metadata: SourceMetadata
    scope: Scope
    posting_status: PostingStatus
    event_cardinality: EventCardinality
    events: tuple[SemanticEvent, ...]


@dataclass(frozen=True)
class RuntimeTransactionProjection:
    """Host-only data that an initial auto-post implementation may consume."""

    event_id: str
    amount_decimal_text: str
    currency: str
    minor_units: int
    direction: Direction
    account_value: str
    counterparty_state: CounterpartyState
    counterparty_value: str | None
    received_at_epoch_ms: int
    received_at_provenance: TimestampProvenance


@dataclass(frozen=True)
class RuntimeEligibilityProjection:
    eligible: bool
    reasons: tuple[IneligibilityReason, ...]
    transaction: RuntimeTransactionProjection | None


def semantic_v2_schema() -> dict[str, Any]:
    """Load and pin-check the committed JSON Schema resource.

    The executable checks below are deliberately dependency-free so the
    lightweight CI environment does not need a model or an additional package.
    They implement the schema's closed-object/type/cardinality constraints and
    then add the source-grounded rules that JSON Schema cannot express.
    """

    try:
        value = json.loads(SEMANTIC_V2_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticV2Error("Semantic V2 JSON Schema is unavailable or invalid") from error
    if not isinstance(value, dict):
        raise SemanticV2Error("Semantic V2 JSON Schema root must be an object")
    properties = value.get("properties")
    if (
        value.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or not isinstance(properties, dict)
        or properties.get("semantic_contract_id", {}).get("const") != SEMANTIC_CONTRACT_ID
        or properties.get("semantic_contract_version", {}).get("const")
        != SEMANTIC_CONTRACT_VERSION
    ):
        raise SemanticV2Error("Semantic V2 JSON Schema identity does not match the reference")
    return value


def slice_utf8_evidence(message: str, span: EvidenceSpan) -> str:
    """Return a range only when both byte positions land on UTF-8 boundaries."""

    if not isinstance(message, str):
        raise SemanticV2Error("message must be text")
    start = span.start_utf8_byte
    end = span.end_utf8_byte
    encoded = message.encode("utf-8")
    if start < 0 or end <= start or end > len(encoded):
        raise SemanticV2Error("evidence range is empty or outside the UTF-8 message")
    try:
        return encoded[start:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SemanticV2Error("evidence range splits a UTF-8 code point") from error


def derive_currency_from_amount_evidence(evidence_text: str) -> str:
    """Derive one unambiguous currency from the selected amount bytes."""

    if not isinstance(evidence_text, str) or not evidence_text:
        raise SemanticV2Error("amount evidence must be non-empty text")
    matches = set(_UPPERCASE_CURRENCY_CODE_RE.findall(evidence_text))
    for currency, pattern in _UNAMBIGUOUS_CURRENCY_MARKERS:
        if pattern.search(evidence_text):
            matches.add(currency)
    if "$" in evidence_text and "USD" not in matches:
        raise SemanticV2Error("a bare dollar sign is ambiguous currency evidence")
    if len(matches) != 1:
        raise SemanticV2Error("amount evidence must identify exactly one currency")
    return next(iter(matches))


def derive_decimal_text_from_amount_evidence(evidence_text: str) -> str:
    """Preserve the source fractional precision while removing only grouping commas."""

    values = [match.group(0) for match in _MONEY_NUMBER_RE.finditer(evidence_text)]
    if len(values) != 1:
        raise SemanticV2Error("amount evidence must contain exactly one numeric amount")
    decimal_text = values[0].replace(",", "")
    if not _DECIMAL_TEXT_RE.fullmatch(decimal_text):
        raise SemanticV2Error("amount evidence has a non-canonical decimal")
    return decimal_text


def derive_direction_from_evidence(evidence_text: str) -> Direction:
    """Map one complete source lexeme using the Semantic V2 direction policy."""

    if not isinstance(evidence_text, str) or not evidence_text:
        raise SemanticV2Error("direction evidence must be non-empty text")
    lexeme = evidence_text.strip().casefold()
    for direction, lexemes in _DIRECTION_EVIDENCE_LEXEMES.items():
        if lexeme in lexemes:
            return direction
    raise SemanticV2Error(
        "direction evidence is not in the "
        f"v{_DIRECTION_EVIDENCE_LEXICON_VERSION} direction lexicon"
    )


def derive_minor_units(decimal_text: str, currency: str) -> int | None:
    """Return exact currency minor units only when no rounding is necessary."""

    if not isinstance(decimal_text, str) or not _DECIMAL_TEXT_RE.fullmatch(decimal_text):
        raise SemanticV2Error("decimal_text is not canonical")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise SemanticV2Error("currency must be an uppercase ISO-style code")
    exponent = _CURRENCY_MINOR_EXPONENTS.get(currency)
    if exponent is None:
        return None

    whole, separator, fraction = decimal_text.partition(".")
    if separator and any(digit != "0" for digit in fraction[exponent:]):
        return None
    scaled_fraction = (fraction[:exponent] + ("0" * exponent))[:exponent]
    minor_units_text = (whole + scaled_fraction).lstrip("0") or "0"
    if (
        len(minor_units_text) > len(_PORTABLE_SIGNED_INT64_MAX_TEXT)
        or len(minor_units_text) == len(_PORTABLE_SIGNED_INT64_MAX_TEXT)
        and minor_units_text > _PORTABLE_SIGNED_INT64_MAX_TEXT
    ):
        raise SemanticV2Error("minor_units is outside the portable signed 64-bit range")
    return int(minor_units_text)


def inject_source_timestamp(
    semantic_core: Mapping[str, Any],
    *,
    received_at_epoch_ms: int,
    received_at_provenance: str | TimestampProvenance,
) -> dict[str, Any]:
    """Attach trusted source time after model interpretation, never before it."""

    if not isinstance(semantic_core, Mapping):
        raise SemanticV2Error("semantic core must be an object")
    if "source_metadata" in semantic_core:
        raise SemanticV2Error("source metadata must be injected exactly once")
    return {
        "source_metadata": {
            "received_at_epoch_ms": received_at_epoch_ms,
            "received_at_provenance": (
                received_at_provenance.value
                if isinstance(received_at_provenance, TimestampProvenance)
                else received_at_provenance
            ),
        },
        **dict(semantic_core),
    }


def validate_semantic_v2(value: Any, *, message: str) -> SemanticRecord:
    """Validate a complete Semantic V2 record against schema and source evidence."""

    semantic_v2_schema()
    root = _object(value, "record")
    _exact_keys(
        root,
        {
            "semantic_contract_id",
            "semantic_contract_version",
            "source_metadata",
            "scope",
            "posting_status",
            "event_cardinality",
            "events",
        },
        "record",
    )
    if root["semantic_contract_id"] != SEMANTIC_CONTRACT_ID:
        raise SemanticV2Error("record has an unsupported semantic_contract_id")
    if root["semantic_contract_version"] != SEMANTIC_CONTRACT_VERSION:
        raise SemanticV2Error("record has an unsupported semantic_contract_version")

    source_metadata = _parse_source_metadata(root["source_metadata"])
    scope = _enum(Scope, root["scope"], "scope")
    posting_status = _enum(PostingStatus, root["posting_status"], "posting_status")
    cardinality = _enum(EventCardinality, root["event_cardinality"], "event_cardinality")
    events_value = root["events"]
    if not isinstance(events_value, list):
        raise SemanticV2Error("events must be an array")
    events = tuple(_parse_event(item, message=message, index=index) for index, item in enumerate(events_value))
    _validate_cardinality(cardinality, posting_status, events)
    return SemanticRecord(source_metadata, scope, posting_status, cardinality, events)


def project_initial_auto_post(record: SemanticRecord) -> RuntimeEligibilityProjection:
    """Fail closed to the deliberately narrow initial automatic-post boundary."""

    reasons: list[IneligibilityReason] = []
    if record.posting_status is not PostingStatus.POSTED:
        reasons.append(IneligibilityReason.NOT_POSTED)
    if record.event_cardinality is not EventCardinality.SINGLE:
        reasons.append(IneligibilityReason.EVENT_CARDINALITY_NOT_SINGLE)
    if record.scope is not Scope.BANK_CARD:
        reasons.append(IneligibilityReason.SCOPE_NOT_BANK_CARD)

    event = record.events[0] if record.event_cardinality is EventCardinality.SINGLE else None
    if event is not None:
        if event.amount is None:
            reasons.append(IneligibilityReason.MISSING_AMOUNT)
        elif event.amount.currency != "INR":
            reasons.append(IneligibilityReason.CURRENCY_NOT_INR)
        elif event.amount.minor_units is None:
            reasons.append(IneligibilityReason.EXACT_MINOR_UNITS_UNAVAILABLE)
        if event.direction is None:
            reasons.append(IneligibilityReason.MISSING_DIRECTION)
        if event.account is None:
            reasons.append(IneligibilityReason.MISSING_ACCOUNT)
    if reasons:
        return RuntimeEligibilityProjection(False, tuple(reasons), None)

    assert event is not None
    assert event.amount is not None
    assert event.amount.minor_units is not None
    assert event.direction is not None
    assert event.account is not None
    return RuntimeEligibilityProjection(
        True,
        (),
        RuntimeTransactionProjection(
            event_id=event.event_id,
            amount_decimal_text=event.amount.decimal_text,
            currency=event.amount.currency,
            minor_units=event.amount.minor_units,
            direction=event.direction.value,
            account_value=event.account.value,
            counterparty_state=event.counterparty.state,
            counterparty_value=event.counterparty.value,
            received_at_epoch_ms=record.source_metadata.received_at_epoch_ms,
            received_at_provenance=record.source_metadata.received_at_provenance,
        ),
    )


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticV2Error(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    actual = set(value)
    missing = required - actual
    unexpected = actual - required
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise SemanticV2Error(f"{path} has invalid keys: {', '.join(details)}")


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticV2Error(f"{path} must be an integer")
    if value < minimum or value > PORTABLE_SIGNED_INT64_MAX:
        raise SemanticV2Error(f"{path} is outside the portable signed 64-bit range")
    return value


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise SemanticV2Error(f"{path} has an unsupported value") from error


def _parse_span(value: Any, path: str) -> EvidenceSpan:
    mapping = _object(value, path)
    _exact_keys(mapping, {"start_utf8_byte", "end_utf8_byte"}, path)
    start = _integer(mapping["start_utf8_byte"], f"{path}.start_utf8_byte")
    end = _integer(mapping["end_utf8_byte"], f"{path}.end_utf8_byte")
    if end <= start:
        raise SemanticV2Error(f"{path} must be a non-empty range")
    return EvidenceSpan(start, end)


def _parse_source_metadata(value: Any) -> SourceMetadata:
    mapping = _object(value, "source_metadata")
    _exact_keys(mapping, {"received_at_epoch_ms", "received_at_provenance"}, "source_metadata")
    return SourceMetadata(
        _integer(mapping["received_at_epoch_ms"], "source_metadata.received_at_epoch_ms"),
        _enum(
            TimestampProvenance,
            mapping["received_at_provenance"],
            "source_metadata.received_at_provenance",
        ),
    )


def _parse_amount(value: Any, *, message: str, path: str) -> ExactAmount | None:
    if value is None:
        return None
    mapping = _object(value, path)
    _exact_keys(mapping, {"decimal_text", "currency", "minor_units", "evidence"}, path)
    decimal_text = mapping["decimal_text"]
    currency = mapping["currency"]
    if not isinstance(decimal_text, str) or not _DECIMAL_TEXT_RE.fullmatch(decimal_text):
        raise SemanticV2Error(f"{path}.decimal_text must be canonical exact decimal text")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise SemanticV2Error(f"{path}.currency must be an uppercase ISO-style code")
    evidence = _parse_span(mapping["evidence"], f"{path}.evidence")
    evidence_text = slice_utf8_evidence(message, evidence)
    if derive_currency_from_amount_evidence(evidence_text) != currency:
        raise SemanticV2Error(f"{path}.currency does not match amount evidence")
    if derive_decimal_text_from_amount_evidence(evidence_text) != decimal_text:
        raise SemanticV2Error(f"{path}.decimal_text does not match amount evidence")
    expected_minor_units = derive_minor_units(decimal_text, currency)
    minor_units = mapping["minor_units"]
    if minor_units is not None:
        minor_units = _integer(minor_units, f"{path}.minor_units")
    if minor_units != expected_minor_units:
        raise SemanticV2Error(f"{path}.minor_units is not the exact deterministic derivative")
    return ExactAmount(decimal_text, currency, minor_units, evidence)


def _parse_direction(value: Any, *, message: str, path: str) -> DirectionEvidence | None:
    if value is None:
        return None
    mapping = _object(value, path)
    _exact_keys(mapping, {"value", "evidence"}, path)
    direction = _enum(Direction, mapping["value"], f"{path}.value")
    evidence = _parse_span(mapping["evidence"], f"{path}.evidence")
    evidence_text = slice_utf8_evidence(message, evidence)
    if derive_direction_from_evidence(evidence_text) is not direction:
        raise SemanticV2Error(f"{path}.value does not match direction evidence")
    return DirectionEvidence(direction, evidence)


def _parse_text_evidence(value: Any, *, message: str, path: str) -> TextEvidence | None:
    if value is None:
        return None
    mapping = _object(value, path)
    _exact_keys(mapping, {"value", "evidence"}, path)
    text = mapping["value"]
    if not isinstance(text, str) or not text:
        raise SemanticV2Error(f"{path}.value must be non-empty text")
    evidence = _parse_span(mapping["evidence"], f"{path}.evidence")
    if slice_utf8_evidence(message, evidence) != text:
        raise SemanticV2Error(f"{path}.value does not exactly match evidence")
    return TextEvidence(text, evidence)


def _parse_counterparty(value: Any, *, message: str, path: str) -> Counterparty:
    mapping = _object(value, path)
    state = _enum(CounterpartyState, mapping.get("state"), f"{path}.state")
    if state is CounterpartyState.ABSENT:
        _exact_keys(mapping, {"state"}, path)
        return Counterparty(state, None, None)
    _exact_keys(mapping, {"state", "value", "evidence"}, path)
    text = mapping["value"]
    if not isinstance(text, str) or not text:
        raise SemanticV2Error(f"{path}.value must be non-empty text")
    evidence = _parse_span(mapping["evidence"], f"{path}.evidence")
    if slice_utf8_evidence(message, evidence) != text:
        raise SemanticV2Error(f"{path}.value does not exactly match evidence")
    return Counterparty(state, text, evidence)


def _parse_event(value: Any, *, message: str, index: int) -> SemanticEvent:
    path = f"events[{index}]"
    mapping = _object(value, path)
    _exact_keys(mapping, {"event_id", "amount", "direction", "account", "counterparty"}, path)
    event_id = mapping["event_id"]
    expected_event_id = f"event_{index + 1}"
    if not isinstance(event_id, str) or event_id != expected_event_id:
        raise SemanticV2Error(f"{path}.event_id must be {expected_event_id}")
    return SemanticEvent(
        event_id,
        _parse_amount(mapping["amount"], message=message, path=f"{path}.amount"),
        _parse_direction(mapping["direction"], message=message, path=f"{path}.direction"),
        _parse_text_evidence(mapping["account"], message=message, path=f"{path}.account"),
        _parse_counterparty(mapping["counterparty"], message=message, path=f"{path}.counterparty"),
    )


def _validate_cardinality(
    cardinality: EventCardinality,
    posting_status: PostingStatus,
    events: tuple[SemanticEvent, ...],
) -> None:
    count = len(events)
    if cardinality is EventCardinality.NONE:
        if posting_status is not PostingStatus.NOT_POSTED or count != 0:
            raise SemanticV2Error("none cardinality requires not_posted and zero events")
    elif cardinality is EventCardinality.SINGLE and count != 1:
        raise SemanticV2Error("single cardinality requires exactly one event")
    elif cardinality is EventCardinality.MULTIPLE and count < 2:
        raise SemanticV2Error("multiple cardinality requires at least two events")
