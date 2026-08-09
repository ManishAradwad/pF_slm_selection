"""Shared policy and validation for the strictly local annotation workbench.

The module contains no HTTP code and never logs source rows.  It deliberately
keeps workflow-specific loading behind two explicit entry points so a blinded
test session cannot accidentally opt into training-only proposal or provenance
views.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .private_data import (
    PrivateDataError,
)


WORKBENCH_CONTRACT = "pocketfinancer-local-annotation-workbench-v1"
WORKBENCH_SCHEMA_VERSION = 1
SOURCE_PREFILL_OFF = "off"
SOURCE_PREFILL_UNAMBIGUOUS = "unambiguous"
SOURCE_PREFILL_POLICIES = (SOURCE_PREFILL_OFF, SOURCE_PREFILL_UNAMBIGUOUS)
SOURCE_PREFILL_POLICY_VERSION = 1
HUMAN_VERIFIED_METHODOLOGY = "human_verified"
ASSISTED_METHODOLOGY = "human_verified_candidate_assisted"
BLINDED_MODE = "blinded_test"
TRAINING_MODE = "training_curation"
WORKBENCH_MODES = (BLINDED_MODE, TRAINING_MODE)
ANNOTATION_STATUSES = (
    "pending",
    "draft",
    "completed",
    "uncertain",
    "needs_adjudication",
)
FILTER_NAMES = (
    "pending",
    "completed",
    "uncertain",
    "active_learning",
    "noted",
    "transaction",
    "null",
    "qc",
)
ACTIVE_LEARNING_QUEUE_TAGS = (
    "model_disagreement",
    "low_confidence_output",
    "candidate_coverage_miss",
    "hard_negative_with_amount",
    "otp_or_security",
    "pending_failed_declined_or_hold",
    "payment_request_or_reminder",
    "refund_or_reversal",
    "multiple_entities",
    "rare_sender_or_template",
)
ACTIVE_LEARNING_QUEUE_POLICY_VERSION = 1
PRIVATE_ROOT = Path("PRIVATE_DATA/lfm25")
DEFAULT_WORKBENCH_DIR = PRIVATE_ROOT / "annotation_workbench"
DEFAULT_BLINDED_DB = DEFAULT_WORKBENCH_DIR / "blinded_test.sqlite3"
DEFAULT_TRAINING_DB = DEFAULT_WORKBENCH_DIR / "training_curation.sqlite3"
DEFAULT_ASSISTED_BLINDED_DB = DEFAULT_WORKBENCH_DIR / "blinded_test_candidate_assisted.sqlite3"
DEFAULT_ASSISTED_TRAINING_DB = (
    DEFAULT_WORKBENCH_DIR / "training_curation_candidate_assisted.sqlite3"
)
DEFAULT_TRAINING_EXPORT = PRIVATE_ROOT / "training_curation_human_reviewed.jsonl"
DEFAULT_TRAINING_REPORT = PRIVATE_ROOT / "training_curation_export_report.json"
DEFAULT_ASSISTED_TRAINING_EXPORT = (
    PRIVATE_ROOT / "training_curation_candidate_assisted_human_reviewed.jsonl"
)
DEFAULT_ASSISTED_TRAINING_REPORT = (
    PRIVATE_ROOT / "training_curation_candidate_assisted_export_report.json"
)
SPAN_KEYS = ("text", "start", "end")
ANNOTATION_KEYS = (
    "decision",
    "amount_decimal",
    "amount_span",
    "type",
    "account_span",
    "counterparty_span",
    "counterparty_absent",
    "notes",
    "uncertain",
)
DECISIONS = ("transaction", "not_transaction")
TYPES = ("debit", "credit")
MAX_NOTES_LENGTH = 2_000
MAX_SOURCE_LENGTH = 100_000
QC_SAMPLE_NUMERATOR = 1
QC_SAMPLE_DENOMINATOR = 10

_AMOUNT_TOKEN_RE = re.compile(r"(?<![\d,])[+]?\d[\d,]*(?:\.\d+)?(?![\d,])")
_AMOUNT_SIGNAL_RE = re.compile(r"(?:\u20b9|\brs\.?|\binr\b)\s*[+]?\d", re.IGNORECASE)
_OTP_RE = re.compile(
    r"\b(?:otp|one[- ]time password|verification code|authenticate|do not share)\b",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"\b(?:pending|failed|declined|blocked|unsuccessful|on hold|pre[- ]?authori[sz])\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\b(?:request(?:ed)? to pay|collect request|payment due|amount due|bill due|"
    r"reminder|pay by)\b",
    re.IGNORECASE,
)
_REFUND_RE = re.compile(r"\b(?:refund(?:ed)?|reversal|reversed)\b", re.IGNORECASE)


class WorkbenchError(PrivateDataError):
    """A workbench failure whose text is safe to display locally or on stderr."""


class WorkbenchConflict(WorkbenchError):
    """An optimistic-revision or writer-lock conflict."""


@dataclass(frozen=True)
class ValidationProblem:
    """One row-local validation problem safe to return to the browser."""

    field: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code, "message": self.message}


class AnnotationValidationError(WorkbenchError):
    """A collection of actionable row-local validation problems."""

    def __init__(self, problems: Sequence[ValidationProblem]) -> None:
        if not problems:
            raise ValueError("annotation validation errors require at least one problem")
        self.problems = tuple(problems)
        super().__init__("the annotation has one or more validation errors")


@dataclass(frozen=True)
class WorkbenchSourceRow:
    """One stable source row prepared for storage without exposing private metadata."""

    row_id: str
    position: int
    sender: str
    sms: str
    source_json: str | None
    split: str | None
    queue_tags: tuple[str, ...]
    source_prefill: dict[str, Any] | None = None
    initial_annotation: dict[str, Any] | None = None
    initial_reviewer: str | None = None
    initial_reviewed_at: str | None = None

    def store_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "position": self.position,
            "sender": self.sender,
            "sms": self.sms,
            "source_json": self.source_json,
            "split": self.split,
            "queue_tags": list(self.queue_tags),
            "source_prefill": self.source_prefill,
            "initial_annotation": self.initial_annotation,
            "initial_reviewer": self.initial_reviewer,
            "initial_reviewed_at": self.initial_reviewed_at,
        }


@dataclass(frozen=True)
class WorkspaceDefinition:
    """A frozen workflow input and its private binding metadata."""

    mode: str
    rows: tuple[WorkbenchSourceRow, ...]
    binding: dict[str, Any]
    metadata: dict[str, Any]
    private_paths: tuple[tuple[str, Path], ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.rows)


def validate_source_prefill_policy(value: str) -> str:
    """Validate the immutable annotation assistance policy."""

    if value not in SOURCE_PREFILL_POLICIES:
        raise WorkbenchError("the source-prefill policy is invalid")
    return value


def source_prefill_methodology(value: str) -> str:
    """Name the human-review methodology bound to imports and exports."""

    policy = validate_source_prefill_policy(value)
    if policy == SOURCE_PREFILL_UNAMBIGUOUS:
        return ASSISTED_METHODOLOGY
    return HUMAN_VERIFIED_METHODOLOGY


def _reject_nonfinite(_value: str) -> Any:
    raise ValueError("non-finite JSON numbers are not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def exact_json_loads(value: str) -> Any:
    """Parse JSON while preserving every fractional number as ``Decimal``."""

    return json.loads(
        value,
        parse_float=Decimal,
        parse_constant=_reject_nonfinite,
        object_pairs_hook=_unique_object,
    )


def _encode_exact_json(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimals are not valid JSON")
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid JSON")
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        members: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            members.append(f"{json.dumps(key, ensure_ascii=False)}:{_encode_exact_json(item)}")
        return "{" + ",".join(members) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_encode_exact_json(item) for item in value) + "]"
    raise TypeError("value is not exact-JSON compatible")


def exact_json_dumps(value: Any) -> str:
    """Serialize compact UTF-8 JSON without projecting ``Decimal`` through float."""

    return _encode_exact_json(value)


def canonical_decimal_text(value: Any) -> str:
    """Validate and canonicalize a finite positive base-10 decimal string."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("amount must be an exact decimal string")
    if not re.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d+)?", value):
        raise ValueError("amount must use ungrouped base-10 notation")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("amount must be an exact decimal string") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError("amount must be positive")
    whole, dot, fraction = value.partition(".")
    canonical_whole = whole.lstrip("0") or "0"
    return canonical_whole + (dot + fraction if dot else "")


def _utf8_slice(text: str, start: int, end: int) -> str:
    encoded = text.encode("utf-8")
    if start < 0 or end <= start or end > len(encoded):
        raise ValueError("span offsets are outside the source SMS")
    try:
        return encoded[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("span offsets split a UTF-8 code point") from exc


def validate_source_span(value: Any, sms: str, *, field: str) -> dict[str, Any]:
    """Validate an exact zero-based half-open UTF-8 byte span."""

    if not isinstance(value, Mapping) or set(value) != set(SPAN_KEYS):
        raise ValueError(f"{field} must be an exact source span")
    text = value.get("text")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field} source text must be nonblank")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise ValueError(f"{field} offsets must be integers")
    if _utf8_slice(sms, start, end) != text:
        raise ValueError(f"{field} must exactly match the source SMS")
    return {"text": text, "start": start, "end": end}


def _amount_is_grounded(amount_text: str, span_text: str) -> bool:
    observed: list[str] = []
    for match in _AMOUNT_TOKEN_RE.finditer(span_text):
        token = match.group(0).replace(",", "").lstrip("+")
        try:
            value = Decimal(token)
        except InvalidOperation:
            continue
        if value.is_finite() and value > 0:
            whole, dot, fraction = token.partition(".")
            canonical_whole = whole.lstrip("0") or "0"
            observed.append(canonical_whole + (dot + fraction if dot else ""))
    return len(observed) == 1 and observed[0] == amount_text


def _problem(field: str, code: str, message: str) -> ValidationProblem:
    return ValidationProblem(field=field, code=code, message=message)


def validate_annotation(
    value: Any,
    sms: str,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate a human annotation or autosaved draft without float conversion."""

    if not isinstance(sms, str) or not sms or len(sms) > MAX_SOURCE_LENGTH:
        raise WorkbenchError("the source row has an invalid SMS field")
    if not isinstance(value, Mapping) or set(value) != set(ANNOTATION_KEYS):
        raise AnnotationValidationError(
            (_problem("annotation", "schema", "Annotation fields do not match V1."),)
        )

    problems: list[ValidationProblem] = []
    decision = value.get("decision")
    amount_decimal = value.get("amount_decimal")
    amount_span = value.get("amount_span")
    transaction_type = value.get("type")
    account_span = value.get("account_span")
    counterparty_span = value.get("counterparty_span")
    counterparty_absent = value.get("counterparty_absent")
    notes = value.get("notes")
    uncertain = value.get("uncertain")

    if decision is not None and decision not in DECISIONS:
        problems.append(_problem("decision", "invalid", "Choose transaction or not transaction."))
    if not isinstance(counterparty_absent, bool):
        problems.append(
            _problem("counterparty_absent", "invalid", "No-counterparty must be explicit.")
        )
    if not isinstance(uncertain, bool):
        problems.append(_problem("uncertain", "invalid", "Uncertain must be true or false."))
    if notes is not None and (
        not isinstance(notes, str) or len(notes) > MAX_NOTES_LENGTH
    ):
        problems.append(
            _problem("notes", "invalid", f"Notes must be at most {MAX_NOTES_LENGTH} characters.")
        )

    canonical_amount: str | None = None
    canonical_amount_span: dict[str, Any] | None = None
    canonical_account: dict[str, Any] | None = None
    canonical_counterparty: dict[str, Any] | None = None

    if amount_decimal is not None:
        try:
            canonical_amount = canonical_decimal_text(amount_decimal)
        except ValueError:
            problems.append(
                _problem("amount_decimal", "invalid", "Enter a positive exact decimal amount.")
            )
    if amount_span is not None:
        try:
            canonical_amount_span = validate_source_span(amount_span, sms, field="amount")
        except ValueError:
            problems.append(
                _problem("amount_span", "not_grounded", "Select the amount from the SMS.")
            )
    if account_span is not None:
        try:
            canonical_account = validate_source_span(account_span, sms, field="account")
        except ValueError:
            problems.append(
                _problem("account_span", "not_grounded", "Select the account from the SMS.")
            )
    if counterparty_span is not None:
        try:
            canonical_counterparty = validate_source_span(
                counterparty_span, sms, field="counterparty"
            )
        except ValueError:
            problems.append(
                _problem(
                    "counterparty_span",
                    "not_grounded",
                    "Select the counterparty from the SMS.",
                )
            )

    if decision == "not_transaction":
        extraction_values = (
            amount_decimal,
            amount_span,
            transaction_type,
            account_span,
            counterparty_span,
        )
        if any(item is not None for item in extraction_values) or counterparty_absent is True:
            problems.append(
                _problem(
                    "decision",
                    "non_transaction_fields",
                    "A non-transaction must leave every extraction field empty.",
                )
            )
    elif decision == "transaction":
        if require_complete and amount_decimal is None:
            problems.append(_problem("amount_decimal", "required", "Amount is required."))
        if require_complete and amount_span is None:
            problems.append(_problem("amount_span", "required", "Select the amount source span."))
        if (
            canonical_amount is not None
            and canonical_amount_span is not None
            and not _amount_is_grounded(canonical_amount, canonical_amount_span["text"])
        ):
            problems.append(
                _problem(
                    "amount_span",
                    "amount_mismatch",
                    "The selected source amount does not match the exact decimal.",
                )
            )
        if require_complete and transaction_type is None:
            problems.append(_problem("type", "required", "Choose debit or credit."))
        elif transaction_type is not None and transaction_type not in TYPES:
            problems.append(_problem("type", "invalid", "Choose debit or credit."))
        if require_complete and account_span is None:
            problems.append(_problem("account_span", "required", "Select the account source span."))
        if counterparty_absent is True and counterparty_span is not None:
            problems.append(
                _problem(
                    "counterparty_span",
                    "exclusive",
                    "Choose a source counterparty or explicit no-counterparty, not both.",
                )
            )
        if require_complete and counterparty_absent is False and counterparty_span is None:
            problems.append(
                _problem(
                    "counterparty_span",
                    "required",
                    "Select a counterparty or mark it explicitly absent.",
                )
            )
    elif require_complete and uncertain is not True:
        problems.append(_problem("decision", "required", "Choose a transaction decision."))

    if require_complete and uncertain is True:
        problems.append(
            _problem(
                "uncertain",
                "unresolved",
                "Resolve uncertainty before completing this row.",
            )
        )
    if problems:
        raise AnnotationValidationError(problems)

    return {
        "decision": decision,
        "amount_decimal": canonical_amount,
        "amount_span": canonical_amount_span,
        "type": transaction_type,
        "account_span": canonical_account,
        "counterparty_span": canonical_counterparty,
        "counterparty_absent": counterparty_absent,
        "notes": notes.strip() if isinstance(notes, str) and notes.strip() else None,
        "uncertain": uncertain,
    }


def _digest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _digest_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_digest_value(item) for item in value]
    return value


def annotation_digest(annotation: Mapping[str, Any]) -> str:
    canonical = exact_json_dumps(_digest_value(annotation))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def annotations_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return annotation_digest(left) == annotation_digest(right)


def empty_annotation() -> dict[str, Any]:
    return {
        "decision": None,
        "amount_decimal": None,
        "amount_span": None,
        "type": None,
        "account_span": None,
        "counterparty_span": None,
        "counterparty_absent": False,
        "notes": None,
        "uncertain": False,
    }


def annotation_to_legacy_fields(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Project a complete workbench annotation into the frozen V1 review schema."""

    if annotation.get("uncertain") is True:
        raise WorkbenchError("an uncertain annotation cannot be projected as completed")
    decision = annotation.get("decision")
    if decision == "not_transaction":
        return {
            "decision": decision,
            "amount": None,
            "counterparty": None,
            "type": None,
            "account": None,
            "notes": annotation.get("notes"),
        }
    if decision != "transaction":
        raise WorkbenchError("a pending annotation cannot be projected as completed")
    amount = canonical_decimal_text(annotation.get("amount_decimal"))
    account = annotation.get("account_span")
    counterparty = annotation.get("counterparty_span")
    if not isinstance(account, Mapping):
        raise WorkbenchError("a transaction annotation has no grounded account")
    if annotation.get("counterparty_absent") is True:
        counterparty_text = None
    elif isinstance(counterparty, Mapping):
        counterparty_text = counterparty.get("text")
    else:
        raise WorkbenchError("a transaction annotation has no counterparty decision")
    return {
        "decision": decision,
        "amount": Decimal(amount),
        "counterparty": counterparty_text,
        "type": annotation.get("type"),
        "account": account.get("text"),
        "notes": annotation.get("notes"),
    }
