"""Versioned, fail-closed host contract for grounded candidate selection.

Candidate Protocol V1 deliberately keeps protocol/version metadata, the message
timestamp, exact-money metadata, and local-account identity outside the model's
answer.  The model can only decide whether a transaction occurred and, for a
transaction, select source-backed candidate IDs.  This module is the host
reference for constructing that request, serializing supervised targets, parsing
the complete model output, and reconstructing PocketFinancer's existing
four-field transaction value.

The protocol is experimental and is not wire-compatible with the currently
locked PocketFinancer Android contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from lfm25.candidates import (
    Candidate,
    CandidateSet,
    canonical_amount_token,
    extract_protocol_candidates,
)
from lfm25.contract import (
    canonical_transaction,
    counterparty_matches,
    normalize_account,
    parse_gold,
)


PROTOCOL_VERSION = "candidate_protocol_v1"
PROTOCOL_REVISION = 1
NO_COUNTERPARTY_ID = "PN"
EVIDENCE_OFFSET_CONVENTION = "utf8_bytes"
PORTABLE_SIGNED_INT64_MAX = (1 << 63) - 1

NOT_TRANSACTION_KEYS = ("transaction",)
TRANSACTION_KEYS = ("transaction", "type", "amount", "account", "counterparty")
_GOLD_TRANSACTION_KEYS = frozenset(("amount", "counterparty", "type", "account"))

CANDIDATE_PROTOCOL_SYSTEM_PROMPT = """Classify one Indian financial SMS using only the candidates supplied by the user.
Return exactly one compact JSON object and no prose or markdown.
For everything other than a posted bank/card transaction return {"transaction":0}.
For a posted transaction return keys in this exact order: transaction=1, type D or C, amount ID, account ID, counterparty ID.
Counterparty ID PN explicitly means that the SMS contains no counterparty.
Never invent, copy, or modify a value. Select only IDs supplied for this SMS."""

_AMOUNT_ID_RE = re.compile(r"A(?:0|[1-9][0-9]*)\Z")
_ACCOUNT_ID_RE = re.compile(r"C(?:0|[1-9][0-9]*)\Z")
_COUNTERPARTY_ID_RE = re.compile(r"(?:PN|P(?:A|T|B|F|V|L|W|U|R|O)(?:0|[1-9][0-9]*))\Z")
_SOURCE_NUMBER_RE = re.compile(r"[+]?(?:[0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)(?:\.[0-9]+)?")
_ASCII_SEMANTIC_WHITESPACE = " \t\n\r\f\v"
_JSON_WHITESPACE = " \t\n\r"
_ASCII_SPACE_RE = re.compile(r"[ \t\n\r\f\v]+")
_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)

_SCHEMA_BINDING = {
    "not_transaction_keys": list(NOT_TRANSACTION_KEYS),
    "transaction_keys": list(TRANSACTION_KEYS),
    "transaction_flags": [0, 1],
    "type_codes": {"D": "debit", "C": "credit"},
    "no_counterparty_id": NO_COUNTERPARTY_ID,
    "strict_member_order": True,
    "surrounding_json_whitespace": True,
    "version_in_model_output": False,
    "message_timestamp_in_model_io": False,
    "amount_prompt_encoding": "source_precision_decimal_string",
    "portable_signed_int64_max": PORTABLE_SIGNED_INT64_MAX,
    "offset_convention": EVIDENCE_OFFSET_CONVENTION,
}


class CandidateProtocolError(ValueError):
    """A trusted caller supplied an invalid request, candidate set, or target."""


class OutcomeStatus(str, Enum):
    """Top-level disposition of a model answer."""

    NOT_TRANSACTION = "not_transaction"
    TRANSACTION = "transaction"
    REJECTED = "rejected"


class OutcomeReason(str, Enum):
    """Stable, aggregate-safe reason codes for accepted and rejected answers."""

    ACCEPTED_NOT_TRANSACTION = "accepted_not_transaction"
    ACCEPTED_TRANSACTION = "accepted_transaction"
    OUTPUT_NOT_TEXT = "output_not_text"
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    TRAILING_CONTENT = "trailing_content"
    OUTPUT_NOT_OBJECT = "output_not_object"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_TRANSACTION_FLAG = "invalid_transaction_flag"
    INVALID_TYPE_CODE = "invalid_type_code"
    UNKNOWN_AMOUNT_ID = "unknown_amount_id"
    UNKNOWN_ACCOUNT_ID = "unknown_account_id"
    UNKNOWN_COUNTERPARTY_ID = "unknown_counterparty_id"
    RECONSTRUCTION_FAILED = "reconstruction_failed"


class AccountHintState(str, Enum):
    """Cardinality of safe local-account matches for a selected source hint."""

    ZERO_MATCHES = "zero_matches"
    UNIQUE_MATCH = "unique_match"
    MULTIPLE_MATCHES = "multiple_matches"


@dataclass(frozen=True)
class ExactMoney:
    """Exact source amount alongside optional integer INR minor units.

    ``decimal_text`` removes grouping separators and a leading plus sign but
    preserves the source's fractional precision.  ``minor_units`` is integer
    paise only when that source precision is at most two decimal places.  The
    exact value therefore remains available even though the current app-facing
    transaction continues to use its existing floating-point amount field.
    """

    decimal_text: str
    minor_units: int | None

    @property
    def decimal(self) -> Decimal:
        return Decimal(self.decimal_text)

    def app_amount(self) -> float:
        """Return the compatibility value used by the existing app contract."""

        try:
            value = float(self.decimal)
        except (OverflowError, ValueError) as error:
            raise CandidateProtocolError("amount has no finite app projection") from error
        if not math.isfinite(value):
            raise CandidateProtocolError("amount has no finite app projection")
        return value


@dataclass(frozen=True)
class ExactAmountCandidate:
    """An amount candidate paired with its exact, source-derived value."""

    id: str
    money: ExactMoney


@dataclass(frozen=True)
class CandidateEvidence:
    """Platform-neutral source evidence for one non-PN candidate.

    Offsets are half-open UTF-8 byte offsets into ProtocolRequest.sms. They
    are not Python character indexes and are never projected into model input.
    """

    id: str
    candidate_kind: str
    source_text: str
    start_utf8_byte: int
    end_utf8_byte: int
    offset_convention: str = field(
        default=EVIDENCE_OFFSET_CONVENTION,
        init=False,
    )


@dataclass(frozen=True)
class ProtocolRequest:
    """Trusted host request; timestamp and version are never model-visible."""

    sender: str
    sms: str
    message_timestamp_epoch_ms: int | None
    candidates: CandidateSet
    exact_amounts: tuple[ExactAmountCandidate, ...]
    candidate_evidence: tuple[CandidateEvidence, ...]
    protocol_version: str = field(default=PROTOCOL_VERSION, init=False)

    def prompt_payload(self) -> dict[str, Any]:
        """Return the deterministic candidate payload projected into the prompt."""

        exact_by_id = {item.id: item.money.decimal_text for item in self.exact_amounts}
        return {
            "amounts": {item.id: exact_by_id[item.id] for item in self.candidates.amounts},
            "accounts": {item.id: item.value for item in self.candidates.accounts},
            "counterparties": {item.id: item.value for item in self.candidates.counterparties},
            "type_hints": list(self.candidates.type_hints),
        }

    def exact_money(self, candidate_id: str) -> ExactMoney | None:
        """Look up exact amount metadata without exposing a mutable mapping."""

        return next(
            (item.money for item in self.exact_amounts if item.id == candidate_id),
            None,
        )

    def evidence_for(self, candidate_id: str) -> CandidateEvidence | None:
        """Look up host evidence; PN intentionally has no evidence range."""

        return next(
            (item for item in self.candidate_evidence if item.id == candidate_id),
            None,
        )


@dataclass(frozen=True)
class OracleCoverage:
    """Whether an existing gold value is fully selectable from this request."""

    covered: bool
    is_transaction: bool
    amount_id: str | None
    account_id: str | None
    counterparty_id: str | None
    type_code: str | None
    missing_fields: tuple[str, ...]

    @property
    def selection(self) -> dict[str, Any] | None:
        """Return the canonical selector mapping when fully covered."""

        if not self.covered:
            return None
        if not self.is_transaction:
            return {"transaction": 0}
        return {
            "transaction": 1,
            "type": self.type_code,
            "amount": self.amount_id,
            "account": self.account_id,
            "counterparty": self.counterparty_id,
        }


@dataclass(frozen=True)
class LocalAccount:
    """A local app identity and a display hint comparable to an SMS account."""

    account_id: str
    account_hint: str


@dataclass(frozen=True)
class AccountHintResolution:
    """Fail-closed cardinality result for resolving one selected account hint."""

    state: AccountHintState
    hint: str
    matching_account_ids: tuple[str, ...]

    @property
    def unique_account_id(self) -> str | None:
        if self.state is AccountHintState.UNIQUE_MATCH:
            return self.matching_account_ids[0]
        return None


@dataclass(frozen=True)
class ProtocolOutcome:
    """Structured parse result with host-only reconstruction metadata."""

    status: OutcomeStatus
    reason: OutcomeReason
    selection: dict[str, Any] | None
    transaction: dict[str, Any] | None
    exact_amount: ExactMoney | None
    message_timestamp_epoch_ms: int | None
    account_resolution: AccountHintResolution | None

    @property
    def accepted(self) -> bool:
        return self.status is not OutcomeStatus.REJECTED

    @property
    def persistence_ready(self) -> bool:
        """Whether host-only data is sufficient for unambiguous persistence."""

        timestamp = self.message_timestamp_epoch_ms
        minor_units = None if self.exact_amount is None else self.exact_amount.minor_units
        return bool(
            self.status is OutcomeStatus.TRANSACTION
            and isinstance(timestamp, int)
            and not isinstance(timestamp, bool)
            and 0 <= timestamp <= PORTABLE_SIGNED_INT64_MAX
            and isinstance(minor_units, int)
            and not isinstance(minor_units, bool)
            and 0 <= minor_units <= PORTABLE_SIGNED_INT64_MAX
            and self.account_resolution is not None
            and self.account_resolution.state is AccountHintState.UNIQUE_MATCH
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_provenance() -> dict[str, Any]:
    """Return JSON-safe hashes binding the executable V1 host contract."""

    schema_json = json.dumps(
        _SCHEMA_BINDING,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    module_path = Path(__file__)
    return {
        "name": PROTOCOL_VERSION,
        "version": PROTOCOL_REVISION,
        "offset_convention": EVIDENCE_OFFSET_CONVENTION,
        "protocol_module_sha256": _file_sha256(module_path),
        "candidate_extractor_sha256": _file_sha256(module_path.with_name("candidates.py")),
        "system_prompt_utf8_sha256": hashlib.sha256(
            CANDIDATE_PROTOCOL_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "selector_schema_sha256": hashlib.sha256(schema_json).hexdigest(),
    }


def _exact_money(candidate: Candidate) -> ExactMoney:
    source = candidate.source_text
    if not isinstance(source, str):
        raise CandidateProtocolError(f"amount candidate {candidate.id} has no source decimal")
    matches = _SOURCE_NUMBER_RE.findall(source)
    if len(matches) != 1:
        raise CandidateProtocolError(
            f"amount candidate {candidate.id} must contain one source decimal"
        )
    try:
        decimal_text = canonical_amount_token(matches[0])
    except ValueError as error:
        raise CandidateProtocolError(
            f"amount candidate {candidate.id} is not source-derived money"
        ) from error
    decimal_value = Decimal(decimal_text)

    candidate_value = candidate.value
    if isinstance(candidate_value, bool) or not isinstance(candidate_value, (int, float, str)):
        raise CandidateProtocolError(
            f"amount candidate {candidate.id} has inconsistent numeric metadata"
        )
    try:
        if isinstance(candidate_value, str):
            candidate_decimal = Decimal(canonical_amount_token(candidate_value))
        else:
            candidate_decimal = Decimal(str(candidate_value))
    except (InvalidOperation, ValueError) as error:
        raise CandidateProtocolError(
            f"amount candidate {candidate.id} has inconsistent numeric metadata"
        ) from error
    if not candidate_decimal.is_finite() or decimal_value != candidate_decimal:
        raise CandidateProtocolError(
            f"amount candidate {candidate.id} disagrees with its source span"
        )

    _whole, dot, fraction = decimal_text.partition(".")
    minor_units: int | None = None
    if not dot or len(fraction) <= 2:
        scaled_decimal = decimal_value * 100
        if scaled_decimal <= PORTABLE_SIGNED_INT64_MAX:
            minor_units = int(scaled_decimal)
    return ExactMoney(decimal_text=decimal_text, minor_units=minor_units)


def _validate_ids(
    items: Sequence[Candidate],
    pattern: re.Pattern[str],
    field_name: str,
) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise CandidateProtocolError(f"duplicate {field_name} candidate ID")
    if any(
        not isinstance(candidate_id, str) or not pattern.fullmatch(candidate_id)
        for candidate_id in ids
    ):
        raise CandidateProtocolError(f"malformed {field_name} candidate ID")


def _validate_source_text_value(
    candidate: Candidate,
    candidate_kind: str,
    *,
    strip_edge_punctuation: bool = False,
) -> None:
    source = candidate.source_text
    value = candidate.value
    if not isinstance(source, str) or not isinstance(value, str):
        raise CandidateProtocolError(f"{candidate_kind} candidate {candidate.id} has no text value")
    derived = _normal_text(source)
    if strip_edge_punctuation:
        derived = derived.strip(" -:,.\n\t")
    if not derived or value != derived:
        raise CandidateProtocolError(
            f"{candidate_kind} candidate {candidate.id} disagrees with its source span"
        )


def _validate_candidate_set(candidates: CandidateSet) -> tuple[ExactAmountCandidate, ...]:
    if not isinstance(candidates, CandidateSet):
        raise TypeError("candidates must be a CandidateSet")
    _validate_ids(candidates.amounts, _AMOUNT_ID_RE, "amount")
    _validate_ids(candidates.accounts, _ACCOUNT_ID_RE, "account")
    _validate_ids(candidates.counterparties, _COUNTERPARTY_ID_RE, "counterparty")

    for item in candidates.accounts:
        _validate_source_text_value(item, "account")

    no_counterparty = [item for item in candidates.counterparties if item.id == NO_COUNTERPARTY_ID]
    if (
        len(no_counterparty) != 1
        or no_counterparty[0].value is not None
        or no_counterparty[0].source_text is not None
        or no_counterparty[0].start is not None
        or no_counterparty[0].end is not None
    ):
        raise CandidateProtocolError("candidate set must contain exactly one null PN")
    for item in candidates.counterparties:
        if item.id == NO_COUNTERPARTY_ID:
            continue
        if item.value is None:
            raise CandidateProtocolError("PN must be the only null counterparty candidate")
        _validate_source_text_value(item, "counterparty", strip_edge_punctuation=True)
    if any(not isinstance(hint, str) or hint not in ("D", "C") for hint in candidates.type_hints):
        raise CandidateProtocolError("candidate type hints must be D or C")

    return tuple(ExactAmountCandidate(item.id, _exact_money(item)) for item in candidates.amounts)


def _candidate_evidence(
    sms: str,
    candidates: CandidateSet,
) -> tuple[CandidateEvidence, ...]:
    evidence: list[CandidateEvidence] = []
    for candidate_kind, items in (
        ("amount", candidates.amounts),
        ("account", candidates.accounts),
        ("counterparty", candidates.counterparties),
    ):
        for item in items:
            if item.id == NO_COUNTERPARTY_ID:
                continue
            if (
                not isinstance(item.source_text, str)
                or not isinstance(item.start, int)
                or isinstance(item.start, bool)
                or not isinstance(item.end, int)
                or isinstance(item.end, bool)
                or item.start < 0
                or item.end < item.start
                or item.end > len(sms)
                or sms[item.start : item.end] != item.source_text
            ):
                raise CandidateProtocolError(
                    f"{candidate_kind} candidate {item.id} has invalid source evidence"
                )
            start_utf8_byte = len(sms[: item.start].encode("utf-8"))
            end_utf8_byte = len(sms[: item.end].encode("utf-8"))
            encoded_span = sms.encode("utf-8")[start_utf8_byte:end_utf8_byte]
            if encoded_span.decode("utf-8") != item.source_text:
                raise CandidateProtocolError(
                    f"{candidate_kind} candidate {item.id} has invalid UTF-8 evidence"
                )
            evidence.append(
                CandidateEvidence(
                    id=item.id,
                    candidate_kind=candidate_kind,
                    source_text=item.source_text,
                    start_utf8_byte=start_utf8_byte,
                    end_utf8_byte=end_utf8_byte,
                )
            )
    return tuple(evidence)


def _require_unicode_scalar_text(value: str, field_name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must contain only Unicode scalar values") from error


def build_protocol_request(
    sender: str,
    sms: str,
    message_timestamp_epoch_ms: int | None = None,
    *,
    candidates: CandidateSet | None = None,
) -> ProtocolRequest:
    """Construct a validated V1 request from one message.

    ``message_timestamp_epoch_ms`` may be ``None`` for historical/offline rows.
    It is carried by the returned host object and deliberately excluded from the
    model prompt and selector target.
    """

    if not isinstance(sender, str):
        raise TypeError("sender must be text")
    if not isinstance(sms, str):
        raise TypeError("sms must be text")
    _require_unicode_scalar_text(sender, "sender")
    _require_unicode_scalar_text(sms, "sms")
    if message_timestamp_epoch_ms is not None and (
        isinstance(message_timestamp_epoch_ms, bool)
        or not isinstance(message_timestamp_epoch_ms, int)
        or message_timestamp_epoch_ms < 0
        or message_timestamp_epoch_ms > PORTABLE_SIGNED_INT64_MAX
    ):
        raise ValueError("message_timestamp_epoch_ms must be a nonnegative signed Int64 or None")
    supplied_candidates = candidates is not None
    selected_candidates = extract_protocol_candidates(sms) if candidates is None else candidates
    exact_amounts = _validate_candidate_set(selected_candidates)
    candidate_evidence = _candidate_evidence(sms, selected_candidates)
    if supplied_candidates and selected_candidates != extract_protocol_candidates(sms):
        raise CandidateProtocolError(
            "supplied candidate set differs from deterministic protocol extraction"
        )
    return ProtocolRequest(
        sender=sender,
        sms=sms,
        message_timestamp_epoch_ms=message_timestamp_epoch_ms,
        candidates=selected_candidates,
        exact_amounts=exact_amounts,
        candidate_evidence=candidate_evidence,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_protocol_request(request: ProtocolRequest) -> None:
    if not isinstance(request, ProtocolRequest):
        raise TypeError("request must be a ProtocolRequest")


def canonical_request_bytes(request: ProtocolRequest) -> bytes:
    """Serialize the exact cross-platform host request schema as UTF-8."""

    _require_protocol_request(request)
    return _canonical_json_bytes(
        {
            "sender": request.sender,
            "sms": request.sms,
            "message_timestamp_epoch_ms": request.message_timestamp_epoch_ms,
        }
    )


def canonical_candidate_payload_bytes(request: ProtocolRequest) -> bytes:
    """Serialize the exact model-visible candidate payload as UTF-8."""

    _require_protocol_request(request)
    return _canonical_json_bytes(request.prompt_payload())


def canonical_candidate_evidence_bytes(request: ProtocolRequest) -> bytes:
    """Serialize exact UTF-8 evidence ranges without PN or host-only extras."""

    _require_protocol_request(request)
    return _canonical_json_bytes(
        [
            {
                "id": item.id,
                "candidate_kind": item.candidate_kind,
                "source_text": item.source_text,
                "start_utf8_byte": item.start_utf8_byte,
                "end_utf8_byte": item.end_utf8_byte,
                "offset_convention": item.offset_convention,
            }
            for item in request.candidate_evidence
        ]
    )


def candidate_protocol_messages(request: ProtocolRequest) -> list[dict[str, str]]:
    """Project a request into the two deterministic model messages."""

    _require_protocol_request(request)
    payload = canonical_candidate_payload_bytes(request).decode("utf-8")
    return [
        {"role": "system", "content": CANDIDATE_PROTOCOL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sender: {request.sender}\nSMS: {request.sms}\nCandidates: {payload}\nOutput:"
            ),
        },
    ]


def canonical_model_messages_bytes(request: ProtocolRequest) -> bytes:
    """Serialize the exact role/content message array as compact UTF-8 JSON."""

    return _canonical_json_bytes(candidate_protocol_messages(request))


def _normal_text(value: str) -> str:
    return _ASCII_SPACE_RE.sub(" ", value).strip(_ASCII_SEMANTIC_WHITESPACE)


def _ascii_lower(value: str) -> str:
    """Lowercase ASCII letters without host Unicode-version dependencies."""

    return value.translate(_ASCII_LOWER_TRANSLATION)


def _reject_nonfinite_json_number(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_protocol_gold(
    gold: Any,
) -> tuple[dict[str, Any], int | float | Decimal] | None:
    """Validate V1 gold while preserving its exact amount representation.

    The legacy four-field parser remains authoritative for type, account, and
    counterparty normalization. Its binary-float amount projection is bypassed:
    JSON decimals are decoded directly as ``Decimal`` and integer values remain
    integers.
    """

    raw = gold
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip(_JSON_WHITESPACE)
        try:
            raw = json.loads(
                stripped,
                parse_float=Decimal,
                parse_constant=_reject_nonfinite_json_number,
                object_pairs_hook=_unique_object,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("gold target is not valid JSON") from error
        if raw is None:
            return None
    if not isinstance(raw, dict) or frozenset(raw) != _GOLD_TRANSACTION_KEYS:
        raise ValueError("gold target violates the four-field contract")

    exact_amount = raw["amount"]
    if isinstance(exact_amount, bool) or not isinstance(
        exact_amount,
        (int, float, Decimal),
    ):
        raise ValueError("gold target violates the four-field contract")
    if isinstance(exact_amount, float) and not math.isfinite(exact_amount):
        raise ValueError("gold target violates the four-field contract")
    if isinstance(exact_amount, Decimal) and not exact_amount.is_finite():
        raise ValueError("gold target violates the four-field contract")
    if exact_amount < 0:
        raise ValueError("gold target violates the four-field contract")

    validation_value = dict(raw)
    validation_value["amount"] = 0
    canonical = parse_gold(validation_value)
    if canonical is None:  # A non-null object cannot become a null gold target.
        raise ValueError("gold target violates the four-field contract")
    return canonical, exact_amount


def _gold_amount_id(
    gold_amount: int | float | Decimal,
    request: ProtocolRequest,
) -> str | None:
    if isinstance(gold_amount, float):
        projected_matches: list[ExactAmountCandidate] = []
        for item in request.exact_amounts:
            try:
                projection = item.money.app_amount()
            except CandidateProtocolError:
                continue
            if projection == gold_amount:
                projected_matches.append(item)
        return projected_matches[0].id if len(projected_matches) == 1 else None

    exact_decimal = gold_amount if isinstance(gold_amount, Decimal) else Decimal(gold_amount)
    return next(
        (item.id for item in request.exact_amounts if item.money.decimal == exact_decimal),
        None,
    )


def oracle_coverage(gold: Any, request: ProtocolRequest) -> OracleCoverage:
    """Map a gold value to V1 IDs and report uncovered source fields."""

    if not isinstance(request, ProtocolRequest):
        raise TypeError("request must be a ProtocolRequest")
    parsed_gold = _parse_protocol_gold(gold)
    if parsed_gold is None:
        return OracleCoverage(True, False, None, None, None, None, ())

    parsed, gold_amount = parsed_gold
    amount_id = _gold_amount_id(gold_amount, request)
    gold_account = normalize_account(parsed["account"])
    account_id = next(
        (
            item.id
            for item in request.candidates.accounts
            if gold_account is not None and normalize_account(item.value) == gold_account
        ),
        None,
    )

    matching_counterparties = [
        item
        for item in request.candidates.counterparties
        if counterparty_matches(parsed["counterparty"], item.value)
    ]
    if parsed["counterparty"] is None:
        counterparty_id = next(
            (item.id for item in matching_counterparties if item.id == NO_COUNTERPARTY_ID),
            None,
        )
    else:
        gold_counterparty = _ascii_lower(_normal_text(str(parsed["counterparty"])))
        matching_counterparties.sort(
            key=lambda item: (
                _ascii_lower(_normal_text(str(item.value))) != gold_counterparty,
                abs(len(_normal_text(str(item.value))) - len(gold_counterparty)),
                item.id,
            )
        )
        counterparty_id = matching_counterparties[0].id if matching_counterparties else None

    type_code = "D" if parsed["type"] == "debit" else "C"
    missing_fields = tuple(
        field_name
        for field_name, candidate_id in (
            ("amount", amount_id),
            ("account", account_id),
            ("counterparty", counterparty_id),
        )
        if candidate_id is None
    )
    return OracleCoverage(
        covered=not missing_fields,
        is_transaction=True,
        amount_id=amount_id,
        account_id=account_id,
        counterparty_id=counterparty_id,
        type_code=type_code,
        missing_fields=missing_fields,
    )


def selector_target_mapping(gold: Any, request: ProtocolRequest) -> dict[str, Any]:
    """Return the validated, canonically ordered V1 selector mapping."""

    coverage = oracle_coverage(gold, request)
    if not coverage.covered:
        missing = ", ".join(coverage.missing_fields)
        raise CandidateProtocolError(f"gold fields absent from candidate set: {missing}")
    selection = coverage.selection
    if selection is None:  # Defensive: covered outcomes always have a selection.
        raise CandidateProtocolError("covered oracle did not produce a selector target")
    return selection


def serialize_selector_target(gold: Any, request: ProtocolRequest) -> str:
    """Serialize a supervised target with the V1 canonical member order."""

    return json.dumps(
        selector_target_mapping(gold, request),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _coerce_local_accounts(
    local_accounts: Iterable[LocalAccount] | Mapping[str, str],
) -> tuple[LocalAccount, ...]:
    if isinstance(local_accounts, Mapping):
        accounts = tuple(
            LocalAccount(account_id, account_hint)
            for account_id, account_hint in local_accounts.items()
        )
    else:
        accounts = tuple(local_accounts)
    if any(not isinstance(item, LocalAccount) for item in accounts):
        raise TypeError("local_accounts must contain LocalAccount values")
    if any(
        not isinstance(item.account_id, str)
        or not item.account_id
        or not isinstance(item.account_hint, str)
        or not item.account_hint
        for item in accounts
    ):
        raise ValueError("local account IDs and hints must be nonempty text")
    account_ids = [item.account_id for item in accounts]
    if len(account_ids) != len(set(account_ids)):
        raise ValueError("local account IDs must be unique")
    return accounts


def resolve_account_hint(
    hint: str,
    local_accounts: Iterable[LocalAccount] | Mapping[str, str],
) -> AccountHintResolution:
    """Resolve an SMS account hint without guessing across zero/many matches."""

    if not isinstance(hint, str):
        raise TypeError("account hint must be text")
    accounts = _coerce_local_accounts(local_accounts)
    normalized_hint = normalize_account(hint)
    matching_ids = tuple(
        item.account_id
        for item in accounts
        if normalized_hint is not None and normalize_account(item.account_hint) == normalized_hint
    )
    if not matching_ids:
        state = AccountHintState.ZERO_MATCHES
    elif len(matching_ids) == 1:
        state = AccountHintState.UNIQUE_MATCH
    else:
        state = AccountHintState.MULTIPLE_MATCHES
    return AccountHintResolution(state, hint, matching_ids)


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _rejected(reason: OutcomeReason, request: ProtocolRequest) -> ProtocolOutcome:
    return ProtocolOutcome(
        status=OutcomeStatus.REJECTED,
        reason=reason,
        selection=None,
        transaction=None,
        exact_amount=None,
        message_timestamp_epoch_ms=request.message_timestamp_epoch_ms,
        account_resolution=None,
    )


def _candidate_by_id(items: Sequence[Candidate], candidate_id: str) -> Candidate | None:
    return next((item for item in items if item.id == candidate_id), None)


def parse_selector_output(
    text: Any,
    request: ProtocolRequest,
    *,
    local_accounts: Iterable[LocalAccount] | Mapping[str, str] = (),
) -> ProtocolOutcome:
    """Strictly parse the complete selector output and reconstruct source values.

    Surrounding JSON whitespace is accepted.  Prose, markdown fences, trailing
    content, duplicate keys, reordered keys, extra fields, invented IDs, and all
    other schema deviations produce a structured rejected outcome and no
    transaction.
    """

    if not isinstance(request, ProtocolRequest):
        raise TypeError("request must be a ProtocolRequest")
    if not isinstance(text, str):
        return _rejected(OutcomeReason.OUTPUT_NOT_TEXT, request)
    if not text.strip(_JSON_WHITESPACE):
        return _rejected(OutcomeReason.EMPTY_OUTPUT, request)

    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    start = len(text) - len(text.lstrip(_JSON_WHITESPACE))
    try:
        selection, end = decoder.raw_decode(text, idx=start)
    except _DuplicateKeyError:
        return _rejected(OutcomeReason.DUPLICATE_KEY, request)
    except (json.JSONDecodeError, ValueError):
        return _rejected(OutcomeReason.INVALID_JSON, request)
    if text[end:].strip(_JSON_WHITESPACE):
        return _rejected(OutcomeReason.TRAILING_CONTENT, request)
    if not isinstance(selection, dict):
        return _rejected(OutcomeReason.OUTPUT_NOT_OBJECT, request)

    decision = selection.get("transaction")
    if "transaction" not in selection:
        return _rejected(OutcomeReason.SCHEMA_MISMATCH, request)
    if isinstance(decision, bool) or not isinstance(decision, int) or decision not in {0, 1}:
        return _rejected(OutcomeReason.INVALID_TRANSACTION_FLAG, request)

    keys = tuple(selection)
    if decision == 0:
        if keys != NOT_TRANSACTION_KEYS:
            return _rejected(OutcomeReason.SCHEMA_MISMATCH, request)
        return ProtocolOutcome(
            status=OutcomeStatus.NOT_TRANSACTION,
            reason=OutcomeReason.ACCEPTED_NOT_TRANSACTION,
            selection={"transaction": 0},
            transaction=None,
            exact_amount=None,
            message_timestamp_epoch_ms=request.message_timestamp_epoch_ms,
            account_resolution=None,
        )

    if keys != TRANSACTION_KEYS:
        return _rejected(OutcomeReason.SCHEMA_MISMATCH, request)
    type_code = selection["type"]
    if not isinstance(type_code, str) or type_code not in {"D", "C"}:
        return _rejected(OutcomeReason.INVALID_TYPE_CODE, request)

    amount_id = selection["amount"]
    exact_amount = request.exact_money(amount_id) if isinstance(amount_id, str) else None
    if exact_amount is None:
        return _rejected(OutcomeReason.UNKNOWN_AMOUNT_ID, request)

    account_id = selection["account"]
    account = (
        _candidate_by_id(request.candidates.accounts, account_id)
        if isinstance(account_id, str)
        else None
    )
    if account is None:
        return _rejected(OutcomeReason.UNKNOWN_ACCOUNT_ID, request)

    counterparty_id = selection["counterparty"]
    counterparty = (
        _candidate_by_id(request.candidates.counterparties, counterparty_id)
        if isinstance(counterparty_id, str)
        else None
    )
    if counterparty is None:
        return _rejected(OutcomeReason.UNKNOWN_COUNTERPARTY_ID, request)

    try:
        app_amount = exact_amount.app_amount()
    except CandidateProtocolError:
        return _rejected(OutcomeReason.RECONSTRUCTION_FAILED, request)
    canonical = canonical_transaction(
        {
            "amount": app_amount,
            "counterparty": (None if counterparty.id == NO_COUNTERPARTY_ID else counterparty.value),
            "type": "debit" if type_code == "D" else "credit",
            "account": account.value,
        }
    )
    if canonical is None:
        return _rejected(OutcomeReason.RECONSTRUCTION_FAILED, request)

    account_resolution = resolve_account_hint(str(account.value), local_accounts)
    return ProtocolOutcome(
        status=OutcomeStatus.TRANSACTION,
        reason=OutcomeReason.ACCEPTED_TRANSACTION,
        selection={key: selection[key] for key in TRANSACTION_KEYS},
        transaction=canonical,
        exact_amount=exact_amount,
        message_timestamp_epoch_ms=request.message_timestamp_epoch_ms,
        account_resolution=account_resolution,
    )
