"""Aggregate-only evaluation for annotated pipeline component outputs.

The input contract is deliberately language neutral: each UTF-8 JSONL row
pairs one human annotation with one independently produced component-output
record.  IDs must be unique and aligned in file order.  Source evidence uses
half-open UTF-8 byte offsets, and money is represented only by exact decimal
strings.  This module never returns per-row material.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import json
from pathlib import Path
import re
from typing import Any

from lfm25.annotation_workbench import WorkbenchError, validate_annotation


CONTRACT_NAME = "annotation_component_evaluation"
CONTRACT_VERSION = 1
OFFSET_CONVENTION = "utf8_bytes"
PORTABLE_SIGNED_INT64_MAX = (1 << 63) - 1

PREFILTER_STAGES = (
    "personal_mobile_sender",
    "currency_amount",
    "masked_account_or_card",
    "transaction_verb",
    "otp",
    "collect_or_mandate_request",
)
TRANSACTION_KEYS = ("transaction", "type", "amount", "account", "counterparty")
NOT_TRANSACTION_KEYS = ("transaction",)
FIELDS = ("amount", "counterparty", "type", "account")

_DECIMAL_TEXT_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z", re.ASCII)
_SOURCE_NUMBER_RE = re.compile(
    r"[+]?(?:[0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)(?:\.[0-9]+)?",
    re.ASCII,
)
_ROW_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_AMOUNT_ID_RE = re.compile(r"A(?:0|[1-9][0-9]*)\Z", re.ASCII)
_ACCOUNT_ID_RE = re.compile(r"C(?:0|[1-9][0-9]*)\Z", re.ASCII)
_COUNTERPARTY_ID_RE = re.compile(
    r"P(?P<relation>[ATBFVLWURO])(?P<index>0|[1-9][0-9]*)\Z",
    re.ASCII,
)
_ASCII_SPACE_RE = re.compile(r"[ \t\n\r\f\v]+", re.ASCII)
_ASCII_DIGIT_RUN_RE = re.compile(r"[0-9]+", re.ASCII)
_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)
_JSON_WHITESPACE = " \t\n\r"
_RATE_QUANTUM = Decimal("0.000001")


class ComponentEvaluationError(ValueError):
    """Raised for a malformed contract, annotation, or component-output row."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start_utf8_byte: int
    end_utf8_byte: int


@dataclass(frozen=True)
class GoldTransaction:
    type_value: str
    amount_text: str
    amount: Decimal
    amount_span: SourceSpan
    account_value: str
    account_span: SourceSpan
    counterparty_value: str | None
    counterparty_span: SourceSpan | None


@dataclass(frozen=True)
class Annotation:
    row_id: str
    sender: str
    sms: str
    message_timestamp_epoch_ms: int | None
    gold: GoldTransaction | None


@dataclass(frozen=True)
class AmountCandidate:
    candidate_id: str
    decimal_text: str
    amount: Decimal
    span: SourceSpan


@dataclass(frozen=True)
class TextCandidate:
    candidate_id: str
    value: str
    span: SourceSpan


@dataclass(frozen=True)
class CandidateOutput:
    amounts: tuple[AmountCandidate, ...]
    accounts: tuple[TextCandidate, ...]
    counterparties: tuple[TextCandidate, ...]
    has_null_counterparty: bool
    type_hints: tuple[str, ...]


@dataclass(frozen=True)
class ComponentOutput:
    row_id: str
    model_invoked: bool
    rejection_stage: str | None
    candidates: CandidateOutput
    model_output: Any


@dataclass(frozen=True)
class ReconstructedTransaction:
    type_value: str
    amount_text: str
    amount: Decimal
    account_value: str
    counterparty_value: str | None


@dataclass(frozen=True)
class ParserOutcome:
    status: str
    reason: str
    transaction: ReconstructedTransaction | None
    message_timestamp_epoch_ms: int | None


def _error(path: str, message: str) -> ComponentEvaluationError:
    return ComponentEvaluationError(f"{path}: {message}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _loads_strict(text: str, path: str) -> Any:
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except _DuplicateKeyError as error:
        raise _error(path, "contains a duplicate object member") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise _error(path, "is not strict JSON") from error


def _object(value: Any, keys: Sequence[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise _error(path, "has an invalid object shape")
    if any(not isinstance(key, str) for key in value):
        raise _error(path, "has a non-text member name")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "must be an array")
    return value


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise _error(path, "must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _error(path, "must contain only Unicode scalar values") from error
    return value


def _row_id(value: Any, path: str) -> str:
    result = _text(value, path)
    if _ROW_ID_RE.fullmatch(result) is None:
        raise _error(path, "is not a portable row ID")
    return result


def _integer(value: Any, path: str, *, maximum: int = PORTABLE_SIGNED_INT64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise _error(path, "must be a nonnegative portable integer")
    return value


def _decimal_text(value: Any, path: str) -> tuple[str, Decimal]:
    if not isinstance(value, str) or _DECIMAL_TEXT_RE.fullmatch(value) is None:
        raise _error(path, "must be a canonical non-exponent decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:  # Defensive; the lexical check is stricter.
        raise _error(path, "must be a finite decimal string") from error
    if not parsed.is_finite() or parsed <= 0:
        raise _error(path, "must be a positive finite decimal string")
    return value, parsed


def _canonical_source_decimal(token: str, path: str) -> str:
    ungrouped = token.replace(",", "")
    if ungrouped.startswith("+"):
        ungrouped = ungrouped[1:]
    whole, dot, fraction = ungrouped.partition(".")
    canonical = (whole.lstrip("0") or "0") + (dot + fraction if dot else "")
    _decimal_text(canonical, path)
    return canonical


def _span(value: Any, sms: str, path: str) -> SourceSpan:
    item = _object(value, ("text", "start_utf8_byte", "end_utf8_byte"), path)
    text = _text(item["text"], f"{path}.text")
    encoded = sms.encode("utf-8")
    start = _integer(item["start_utf8_byte"], f"{path}.start_utf8_byte", maximum=len(encoded))
    end = _integer(item["end_utf8_byte"], f"{path}.end_utf8_byte", maximum=len(encoded))
    if end <= start:
        raise _error(path, "must be a nonempty half-open range")
    try:
        sliced = encoded[start:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise _error(path, "does not land on UTF-8 code-point boundaries") from error
    if sliced != text:
        raise _error(path, "does not exactly slice source.sms")
    return SourceSpan(text, start, end)


def _amount_span_matches(decimal_text: str, span: SourceSpan, path: str) -> None:
    tokens = _SOURCE_NUMBER_RE.findall(span.text)
    if len(tokens) != 1:
        raise _error(path, "must contain exactly one supported source decimal")
    if _canonical_source_decimal(tokens[0], path) != decimal_text:
        raise _error(path, "does not preserve the exact source decimal")


def _normalize_text(value: str) -> str:
    return _ASCII_SPACE_RE.sub(" ", value).strip(" \t\n\r\f\v")


def _ascii_lower(value: str) -> str:
    return value.translate(_ASCII_LOWER_TRANSLATION)


def _source_text_value(span: SourceSpan, *, counterparty: bool = False) -> str:
    value = _normalize_text(span.text)
    return value.strip(" -:,.\n\t") if counterparty else value


def _account_identity(value: str) -> tuple[str, str] | None:
    lowered = _ascii_lower(value)
    category = "card" if "card" in lowered else "account"
    runs = [run for run in _ASCII_DIGIT_RUN_RE.findall(value) if len(run) >= 3]
    return (category, runs[-1][-4:]) if runs else None


def _counterparty_matches(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    first = _ascii_lower(_normalize_text(left))
    second = _ascii_lower(_normalize_text(right))
    if first == second:
        return True
    shorter, longer = (first, second) if len(first) <= len(second) else (second, first)
    return len(shorter) >= 3 and shorter in longer


def _gold_field(value: Any, sms: str, path: str) -> tuple[str, SourceSpan]:
    field = _object(value, ("value", "span"), path)
    field_value = _text(field["value"], f"{path}.value")
    return field_value, _span(field["span"], sms, f"{path}.span")


def _annotation(value: Any, row_index: int) -> Annotation:
    path = f"annotation[{row_index}]"
    item = _object(value, ("id", "source", "gold"), path)
    row_id = _row_id(item["id"], f"{path}.id")
    source = _object(
        item["source"],
        ("sender", "sms", "message_timestamp_epoch_ms"),
        f"{path}.source",
    )
    sender = _text(source["sender"], f"{path}.source.sender", allow_empty=True)
    sms = _text(source["sms"], f"{path}.source.sms", allow_empty=True)
    raw_timestamp = source["message_timestamp_epoch_ms"]
    timestamp = (
        None
        if raw_timestamp is None
        else _integer(raw_timestamp, f"{path}.source.message_timestamp_epoch_ms")
    )

    raw_gold = item["gold"]
    if raw_gold is None:
        gold = None
    else:
        gold_item = _object(
            raw_gold,
            ("type", "amount", "account", "counterparty"),
            f"{path}.gold",
        )
        type_value = _text(gold_item["type"], f"{path}.gold.type")
        if type_value not in {"debit", "credit"}:
            raise _error(f"{path}.gold.type", "must be debit or credit")

        amount_item = _object(
            gold_item["amount"],
            ("decimal", "span"),
            f"{path}.gold.amount",
        )
        amount_text, amount = _decimal_text(amount_item["decimal"], f"{path}.gold.amount.decimal")
        amount_span = _span(amount_item["span"], sms, f"{path}.gold.amount.span")
        _amount_span_matches(amount_text, amount_span, f"{path}.gold.amount.span")

        account_value, account_span = _gold_field(gold_item["account"], sms, f"{path}.gold.account")
        if account_value != _source_text_value(account_span):
            raise _error(f"{path}.gold.account.value", "must be derived from its exact span")

        raw_counterparty = gold_item["counterparty"]
        if raw_counterparty is None:
            counterparty_value = None
            counterparty_span = None
        else:
            counterparty_value, counterparty_span = _gold_field(
                raw_counterparty, sms, f"{path}.gold.counterparty"
            )
            if counterparty_value != _source_text_value(counterparty_span, counterparty=True):
                raise _error(
                    f"{path}.gold.counterparty.value",
                    "must be derived from its exact span",
                )
        gold = GoldTransaction(
            type_value,
            amount_text,
            amount,
            amount_span,
            account_value,
            account_span,
            counterparty_value,
            counterparty_span,
        )
    return Annotation(row_id, sender, sms, timestamp, gold)


def _component_span_from_workbench(
    value: Any,
    sms: str,
    path: str,
) -> dict[str, Any]:
    item = _object(value, ("text", "start", "end"), path)
    translated = {
        "text": item["text"],
        "start_utf8_byte": item["start"],
        "end_utf8_byte": item["end"],
    }
    _span(translated, sms, path)
    return translated


def adapt_workbench_annotation(
    workbench_annotation: Mapping[str, Any],
    *,
    row_id: str,
    sender: str,
    sms: str,
    message_timestamp_epoch_ms: int | None,
) -> dict[str, Any]:
    """Convert one complete shared-workbench label to evaluator V1.

    The shared workbench remains authoritative for annotation validation.
    Exact decimal text and its existing half-open UTF-8 byte spans are copied
    without any binary-float or character-offset projection.
    """

    adapted_id = _row_id(row_id, "workbench.id")
    adapted_sender = _text(sender, "workbench.source.sender", allow_empty=True)
    adapted_sms = _text(sms, "workbench.source.sms", allow_empty=True)
    adapted_timestamp = (
        None
        if message_timestamp_epoch_ms is None
        else _integer(
            message_timestamp_epoch_ms,
            "workbench.source.message_timestamp_epoch_ms",
        )
    )
    try:
        canonical = validate_annotation(
            workbench_annotation,
            adapted_sms,
            require_complete=True,
        )
    except WorkbenchError as error:
        raise ComponentEvaluationError(
            "workbench annotation is not a complete valid human label"
        ) from error

    source = {
        "sender": adapted_sender,
        "sms": adapted_sms,
        "message_timestamp_epoch_ms": adapted_timestamp,
    }
    if canonical["decision"] == "not_transaction":
        result = {"id": adapted_id, "source": source, "gold": None}
        _annotation(result, 0)
        return result
    if canonical["decision"] != "transaction":
        raise ComponentEvaluationError("workbench annotation is not a complete valid human label")

    amount_text, _amount = _decimal_text(
        canonical["amount_decimal"],
        "workbench.amount_decimal",
    )
    amount_span = _component_span_from_workbench(
        canonical["amount_span"],
        adapted_sms,
        "workbench.amount_span",
    )
    account_span = _component_span_from_workbench(
        canonical["account_span"],
        adapted_sms,
        "workbench.account_span",
    )
    if canonical["counterparty_absent"]:
        counterparty = None
    else:
        counterparty_span = _component_span_from_workbench(
            canonical["counterparty_span"],
            adapted_sms,
            "workbench.counterparty_span",
        )
        counterparty = {
            "value": _source_text_value(
                SourceSpan(
                    counterparty_span["text"],
                    counterparty_span["start_utf8_byte"],
                    counterparty_span["end_utf8_byte"],
                ),
                counterparty=True,
            ),
            "span": counterparty_span,
        }

    result = {
        "id": adapted_id,
        "source": source,
        "gold": {
            "type": canonical["type"],
            "amount": {"decimal": amount_text, "span": amount_span},
            "account": {
                "value": _source_text_value(
                    SourceSpan(
                        account_span["text"],
                        account_span["start_utf8_byte"],
                        account_span["end_utf8_byte"],
                    )
                ),
                "span": account_span,
            },
            "counterparty": counterparty,
        },
    }
    _annotation(result, 0)
    return result


def _require_source_order(spans: Sequence[SourceSpan], path: str) -> None:
    coordinates = [(span.start_utf8_byte, span.end_utf8_byte) for span in spans]
    if coordinates != sorted(coordinates):
        raise _error(path, "must be in ascending source-span order")


def _candidates(value: Any, sms: str, path: str) -> CandidateOutput:
    item = _object(
        value,
        ("amounts", "accounts", "counterparties", "type_hints"),
        path,
    )

    amounts: list[AmountCandidate] = []
    for index, raw in enumerate(_array(item["amounts"], f"{path}.amounts")):
        candidate_path = f"{path}.amounts[{index}]"
        candidate = _object(raw, ("id", "decimal", "span"), candidate_path)
        candidate_id = _text(candidate["id"], f"{candidate_path}.id")
        if _AMOUNT_ID_RE.fullmatch(candidate_id) is None or candidate_id != f"A{index}":
            raise _error(f"{candidate_path}.id", "is not aligned with array order")
        decimal_text, amount = _decimal_text(candidate["decimal"], f"{candidate_path}.decimal")
        span = _span(candidate["span"], sms, f"{candidate_path}.span")
        _amount_span_matches(decimal_text, span, f"{candidate_path}.span")
        amounts.append(AmountCandidate(candidate_id, decimal_text, amount, span))
    _require_source_order([candidate.span for candidate in amounts], f"{path}.amounts")

    accounts: list[TextCandidate] = []
    for index, raw in enumerate(_array(item["accounts"], f"{path}.accounts")):
        candidate_path = f"{path}.accounts[{index}]"
        candidate = _object(raw, ("id", "value", "span"), candidate_path)
        candidate_id = _text(candidate["id"], f"{candidate_path}.id")
        if _ACCOUNT_ID_RE.fullmatch(candidate_id) is None or candidate_id != f"C{index}":
            raise _error(f"{candidate_path}.id", "is not aligned with array order")
        candidate_value = _text(candidate["value"], f"{candidate_path}.value")
        span = _span(candidate["span"], sms, f"{candidate_path}.span")
        if candidate_value != _source_text_value(span):
            raise _error(f"{candidate_path}.value", "must be derived from its exact span")
        if _account_identity(candidate_value) is None:
            raise _error(f"{candidate_path}.value", "has no portable account identity")
        accounts.append(TextCandidate(candidate_id, candidate_value, span))
    _require_source_order([candidate.span for candidate in accounts], f"{path}.accounts")

    raw_counterparties = _array(item["counterparties"], f"{path}.counterparties")
    if not raw_counterparties:
        raise _error(f"{path}.counterparties", "must end with one PN candidate")
    counterparties: list[TextCandidate] = []
    prefix_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    has_null_counterparty = False
    for index, raw in enumerate(raw_counterparties):
        candidate_path = f"{path}.counterparties[{index}]"
        candidate = _object(raw, ("id", "value", "span"), candidate_path)
        candidate_id = _text(candidate["id"], f"{candidate_path}.id")
        if candidate_id in seen_ids:
            raise _error(f"{candidate_path}.id", "duplicates a candidate ID")
        seen_ids.add(candidate_id)
        if candidate_id == "PN":
            if index != len(raw_counterparties) - 1 or candidate["value"] is not None:
                raise _error(candidate_path, "PN must be the final null candidate")
            if candidate["span"] is not None:
                raise _error(candidate_path, "PN must not carry source evidence")
            has_null_counterparty = True
            continue
        match = _COUNTERPARTY_ID_RE.fullmatch(candidate_id)
        if match is None:
            raise _error(f"{candidate_path}.id", "is not a protocol counterparty ID")
        prefix = candidate_id[:2]
        if match.group("index") != str(prefix_counts[prefix]):
            raise _error(f"{candidate_path}.id", "is not aligned with prefix order")
        prefix_counts[prefix] += 1
        candidate_value = _text(candidate["value"], f"{candidate_path}.value")
        span = _span(candidate["span"], sms, f"{candidate_path}.span")
        if candidate_value != _source_text_value(span, counterparty=True):
            raise _error(f"{candidate_path}.value", "must be derived from its exact span")
        counterparties.append(TextCandidate(candidate_id, candidate_value, span))
    if not has_null_counterparty:
        raise _error(f"{path}.counterparties", "must end with one PN candidate")
    _require_source_order(
        [candidate.span for candidate in counterparties],
        f"{path}.counterparties",
    )

    type_hints = tuple(
        _text(raw, f"{path}.type_hints[{index}]")
        for index, raw in enumerate(_array(item["type_hints"], f"{path}.type_hints"))
    )
    if len(type_hints) != len(set(type_hints)) or any(
        hint not in {"D", "C"} for hint in type_hints
    ):
        raise _error(f"{path}.type_hints", "must contain unique D/C values")
    if type_hints != tuple(hint for hint in ("D", "C") if hint in type_hints):
        raise _error(f"{path}.type_hints", "must follow D then C order")
    return CandidateOutput(
        tuple(amounts),
        tuple(accounts),
        tuple(counterparties),
        has_null_counterparty,
        type_hints,
    )


def _component(value: Any, annotation: Annotation, row_index: int) -> ComponentOutput:
    path = f"component_output[{row_index}]"
    item = _object(value, ("id", "prefilter", "candidates", "parser"), path)
    row_id = _row_id(item["id"], f"{path}.id")
    prefilter = _object(
        item["prefilter"],
        ("model_invoked", "rejection_stage"),
        f"{path}.prefilter",
    )
    model_invoked = prefilter["model_invoked"]
    if not isinstance(model_invoked, bool):
        raise _error(f"{path}.prefilter.model_invoked", "must be boolean")
    rejection_stage = prefilter["rejection_stage"]
    if model_invoked:
        if rejection_stage is not None:
            raise _error(f"{path}.prefilter", "an invocation cannot have a rejection stage")
    elif rejection_stage not in PREFILTER_STAGES:
        raise _error(f"{path}.prefilter.rejection_stage", "is not a versioned stage")

    candidates = _candidates(item["candidates"], annotation.sms, f"{path}.candidates")
    parser = _object(item["parser"], ("model_output",), f"{path}.parser")
    model_output = parser["model_output"]
    if not model_invoked and model_output is not None:
        raise _error(f"{path}.parser.model_output", "must be null when the model was not invoked")
    return ComponentOutput(row_id, model_invoked, rejection_stage, candidates, model_output)


def _candidate_maps(
    candidates: CandidateOutput,
) -> tuple[dict[str, AmountCandidate], dict[str, TextCandidate], dict[str, TextCandidate | None]]:
    amounts = {candidate.candidate_id: candidate for candidate in candidates.amounts}
    accounts = {candidate.candidate_id: candidate for candidate in candidates.accounts}
    counterparties: dict[str, TextCandidate | None] = {
        candidate.candidate_id: candidate for candidate in candidates.counterparties
    }
    if candidates.has_null_counterparty:
        counterparties["PN"] = None
    return amounts, accounts, counterparties


def _rejected(reason: str, timestamp: int | None) -> ParserOutcome:
    return ParserOutcome("rejected", reason, None, timestamp)


def _strict_parse(
    model_output: Any,
    candidates: CandidateOutput,
    timestamp: int | None,
) -> ParserOutcome:
    if not isinstance(model_output, str):
        return _rejected("output_not_text", timestamp)
    if not model_output.strip(_JSON_WHITESPACE):
        return _rejected("empty_output", timestamp)
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_float=Decimal,
        parse_constant=_reject_nonfinite,
    )
    start = len(model_output) - len(model_output.lstrip(_JSON_WHITESPACE))
    try:
        selection, end = decoder.raw_decode(model_output, idx=start)
    except _DuplicateKeyError:
        return _rejected("duplicate_key", timestamp)
    except (json.JSONDecodeError, ValueError):
        return _rejected("invalid_json", timestamp)
    if model_output[end:].strip(_JSON_WHITESPACE):
        return _rejected("trailing_content", timestamp)
    if not isinstance(selection, dict):
        return _rejected("output_not_object", timestamp)
    if "transaction" not in selection:
        return _rejected("schema_mismatch", timestamp)
    decision = selection["transaction"]
    if isinstance(decision, bool) or not isinstance(decision, int) or decision not in {0, 1}:
        return _rejected("invalid_transaction_flag", timestamp)
    keys = tuple(selection)
    if decision == 0:
        if keys != NOT_TRANSACTION_KEYS:
            return _rejected("schema_mismatch", timestamp)
        return ParserOutcome("not_transaction", "accepted_not_transaction", None, timestamp)
    if keys != TRANSACTION_KEYS:
        return _rejected("schema_mismatch", timestamp)
    type_code = selection["type"]
    if not isinstance(type_code, str) or type_code not in {"D", "C"}:
        return _rejected("invalid_type_code", timestamp)

    amounts, accounts, counterparties = _candidate_maps(candidates)
    amount_id = selection["amount"]
    if not isinstance(amount_id, str) or amount_id not in amounts:
        return _rejected("unknown_amount_id", timestamp)
    account_id = selection["account"]
    if not isinstance(account_id, str) or account_id not in accounts:
        return _rejected("unknown_account_id", timestamp)
    counterparty_id = selection["counterparty"]
    if not isinstance(counterparty_id, str) or counterparty_id not in counterparties:
        return _rejected("unknown_counterparty_id", timestamp)

    amount = amounts[amount_id]
    account = accounts[account_id]
    counterparty = counterparties[counterparty_id]
    transaction = ReconstructedTransaction(
        "debit" if type_code == "D" else "credit",
        amount.decimal_text,
        amount.amount,
        account.value,
        None if counterparty is None else counterparty.value,
    )
    return ParserOutcome("transaction", "accepted_transaction", transaction, timestamp)


def _strict_cases(model_output: Any, candidates: CandidateOutput) -> set[str]:
    if not isinstance(model_output, str) or not model_output.strip(_JSON_WHITESPACE):
        return set()
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_float=Decimal,
        parse_constant=_reject_nonfinite,
    )
    start = len(model_output) - len(model_output.lstrip(_JSON_WHITESPACE))
    try:
        selection, end = decoder.raw_decode(model_output, idx=start)
    except _DuplicateKeyError:
        return {"duplicate_key"}
    except (json.JSONDecodeError, ValueError):
        return set()
    if model_output[end:].strip(_JSON_WHITESPACE) or not isinstance(selection, dict):
        return set()
    decision = selection.get("transaction")
    expected = NOT_TRANSACTION_KEYS if decision == 0 else TRANSACTION_KEYS
    cases: set[str] = set()
    if set(selection) == set(expected) and tuple(selection) != expected:
        cases.add("reordered_members")
    if decision == 1:
        amounts, accounts, counterparties = _candidate_maps(candidates)
        unknown = (
            (isinstance(selection.get("amount"), str) and selection["amount"] not in amounts)
            or (isinstance(selection.get("account"), str) and selection["account"] not in accounts)
            or (
                isinstance(selection.get("counterparty"), str)
                and selection["counterparty"] not in counterparties
            )
        )
        if unknown:
            cases.add("unknown_id")
    return cases


def _rate(numerator: int, denominator: int) -> str | None:
    if denominator == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        value = (Decimal(numerator) / Decimal(denominator)).quantize(
            _RATE_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
    return format(value, "f")


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {"count": numerator, "total": denominator, "rate": _rate(numerator, denominator)}


def _candidate_coverage(
    gold: GoldTransaction,
    candidates: CandidateOutput,
) -> tuple[dict[str, bool], dict[str, bool]]:
    amount_matches = [item for item in candidates.amounts if item.amount == gold.amount]
    account_identity = _account_identity(gold.account_value)
    account_matches = [
        item for item in candidates.accounts if _account_identity(item.value) == account_identity
    ]
    if gold.counterparty_value is None:
        counterparty_covered = candidates.has_null_counterparty
        counterparty_grounded = candidates.has_null_counterparty
    else:
        counterparty_matches = [
            item
            for item in candidates.counterparties
            if _counterparty_matches(gold.counterparty_value, item.value)
        ]
        counterparty_covered = bool(counterparty_matches)
        counterparty_grounded = any(
            item.span == gold.counterparty_span for item in counterparty_matches
        )
    covered = {
        "amount": bool(amount_matches),
        "account": bool(account_matches),
        "counterparty": counterparty_covered,
    }
    grounded = {
        "amount": any(item.span == gold.amount_span for item in amount_matches),
        "account": any(item.span == gold.account_span for item in account_matches),
        "counterparty": counterparty_grounded,
    }
    covered["joint"] = all(covered.values())
    grounded["joint"] = all(grounded.values())
    return covered, grounded


def _field_hits(
    gold: GoldTransaction,
    transaction: ReconstructedTransaction | None,
) -> dict[str, bool]:
    if transaction is None:
        return {field: False for field in FIELDS}
    return {
        "amount": transaction.amount == gold.amount and transaction.amount_text == gold.amount_text,
        "counterparty": transaction.counterparty_value == gold.counterparty_value,
        "type": transaction.type_value == gold.type_value,
        "account": transaction.account_value == gold.account_value,
    }


def evaluate_component_rows(
    annotations: Sequence[Mapping[str, Any]],
    component_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate aligned rows and return aggregate metrics only."""

    if len(annotations) != len(component_outputs):
        raise ComponentEvaluationError("annotation and component-output row counts differ")
    annotation_ids: set[str] = set()
    component_ids: set[str] = set()
    validated: list[tuple[Annotation, ComponentOutput]] = []
    for index, (raw_annotation, raw_component) in enumerate(
        zip(annotations, component_outputs, strict=True)
    ):
        annotation = _annotation(raw_annotation, index)
        if annotation.row_id in annotation_ids:
            raise _error(f"annotation[{index}].id", "duplicates an earlier annotation ID")
        annotation_ids.add(annotation.row_id)
        component = _component(raw_component, annotation, index)
        if component.row_id in component_ids:
            raise _error(
                f"component_output[{index}].id",
                "duplicates an earlier component-output ID",
            )
        component_ids.add(component.row_id)
        if annotation.row_id != component.row_id:
            raise _error(f"row[{index}]", "annotation and component IDs are not order-aligned")
        validated.append((annotation, component))

    prefilter: Counter[str] = Counter()
    prefilter_stages: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    grounding_counts: Counter[str] = Counter()
    parser_counts: Counter[str] = Counter()
    parser_reasons: Counter[str] = Counter()
    strict_cases: Counter[str] = Counter()
    strict_rejected: Counter[str] = Counter()
    strict_reason: Counter[str] = Counter()
    pipeline: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()

    for annotation, component in validated:
        gold_transaction = annotation.gold is not None
        prefilter["rows"] += 1
        prefilter["gold_transactions"] += int(gold_transaction)
        prefilter["gold_nulls"] += int(not gold_transaction)
        prefilter["model_invocations"] += int(component.model_invoked)
        prefilter["transactions_invoked"] += int(gold_transaction and component.model_invoked)
        prefilter["false_rejections"] += int(gold_transaction and not component.model_invoked)
        prefilter["nulls_rejected"] += int(not gold_transaction and not component.model_invoked)
        if component.rejection_stage is not None:
            prefilter_stages[component.rejection_stage] += 1

        if annotation.gold is not None:
            covered, grounded = _candidate_coverage(annotation.gold, component.candidates)
            for field in ("amount", "account", "counterparty", "joint"):
                coverage_counts[field] += int(covered[field])
                grounding_counts[field] += int(grounded[field])

        outcome: ParserOutcome | None = None
        if component.model_invoked:
            outcome = _strict_parse(
                component.model_output,
                component.candidates,
                annotation.message_timestamp_epoch_ms,
            )
            parser_counts["evaluated"] += 1
            parser_counts[outcome.status] += 1
            parser_reasons[outcome.reason] += 1
            parser_counts["timestamp_preserved"] += int(
                outcome.message_timestamp_epoch_ms == annotation.message_timestamp_epoch_ms
            )
            for case in _strict_cases(component.model_output, component.candidates):
                strict_cases[case] += 1
                strict_rejected[case] += int(outcome.status == "rejected")
                if case == "duplicate_key":
                    strict_reason[case] += int(outcome.reason == "duplicate_key")
                elif case == "reordered_members":
                    strict_reason[case] += int(outcome.reason == "schema_mismatch")
                elif case == "unknown_id":
                    strict_reason[case] += int(outcome.reason.startswith("unknown_"))

        final_transaction = (
            outcome.transaction if outcome is not None and outcome.status == "transaction" else None
        )
        pipeline["rows"] += 1
        pipeline["gold_transactions"] += int(gold_transaction)
        pipeline["gold_nulls"] += int(not gold_transaction)
        pipeline["predicted_transactions"] += int(final_transaction is not None)
        pipeline["ghosts"] += int(not gold_transaction and final_transaction is not None)
        pipeline["misses"] += int(gold_transaction and final_transaction is None)

        if annotation.gold is None:
            exact = final_transaction is None
            parser_exact = False
        else:
            hits = _field_hits(annotation.gold, final_transaction)
            for field, hit in hits.items():
                field_counts[field] += int(hit)
            exact = all(hits.values())
            parser_exact = exact and outcome is not None and outcome.status == "transaction"
        pipeline["transaction_exact"] += int(gold_transaction and exact)
        pipeline["whole_pipeline_exact"] += int(exact)
        if outcome is not None and outcome.status == "transaction":
            parser_counts["reconstruction_total"] += 1
            parser_counts["exact_reconstruction"] += int(parser_exact)

    rows = prefilter["rows"]
    gold_transactions = prefilter["gold_transactions"]
    gold_nulls = prefilter["gold_nulls"]
    parser_evaluated = parser_counts["evaluated"]
    report = {
        "contract": {
            "name": CONTRACT_NAME,
            "version": CONTRACT_VERSION,
            "offset_convention": OFFSET_CONVENTION,
            "aggregate_only": True,
        },
        "rows": rows,
        "prefilter": {
            "gold_transactions": gold_transactions,
            "gold_nulls": gold_nulls,
            "transactions_invoked": prefilter["transactions_invoked"],
            "transaction_recall": _rate(prefilter["transactions_invoked"], gold_transactions),
            "false_rejection_count": prefilter["false_rejections"],
            "nulls_rejected": prefilter["nulls_rejected"],
            "null_rejection_rate": _rate(prefilter["nulls_rejected"], gold_nulls),
            "model_invocations": prefilter["model_invocations"],
            "model_invocation_rate": _rate(prefilter["model_invocations"], rows),
            "rejection_counts_by_stage": {
                stage: prefilter_stages[stage] for stage in PREFILTER_STAGES
            },
        },
        "candidates": {
            "transactions": gold_transactions,
            "oracle_coverage": {
                field: _metric(coverage_counts[field], gold_transactions)
                for field in ("amount", "account", "counterparty", "joint")
            },
            "exact_span_grounding": {
                field: _metric(grounding_counts[field], gold_transactions)
                for field in ("amount", "account", "counterparty", "joint")
            },
        },
        "parser": {
            "evaluated": parser_evaluated,
            "accepted": parser_counts["transaction"] + parser_counts["not_transaction"],
            "rejected": parser_counts["rejected"],
            "status_counts": {
                status: parser_counts[status]
                for status in ("transaction", "not_transaction", "rejected")
            },
            "rejection_counts": {
                reason: count
                for reason, count in sorted(parser_reasons.items())
                if reason not in {"accepted_transaction", "accepted_not_transaction"}
            },
            "strict_behavior": {
                case: {
                    "cases": strict_cases[case],
                    "rejected": strict_rejected[case],
                    "correct_reason": strict_reason[case],
                    "rejection_rate": _rate(strict_rejected[case], strict_cases[case]),
                }
                for case in ("duplicate_key", "unknown_id", "reordered_members")
            },
            "exact_reconstruction": _metric(
                parser_counts["exact_reconstruction"],
                parser_counts["reconstruction_total"],
            ),
            "timestamp_preservation": _metric(
                parser_counts["timestamp_preserved"], parser_evaluated
            ),
        },
        "pipeline": {
            "gold_transactions": gold_transactions,
            "gold_nulls": gold_nulls,
            "predicted_transactions": pipeline["predicted_transactions"],
            "transaction_exact": pipeline["transaction_exact"],
            "transaction_exact_rate": _rate(pipeline["transaction_exact"], gold_transactions),
            "whole_pipeline_exact": pipeline["whole_pipeline_exact"],
            "whole_pipeline_exact_rate": _rate(pipeline["whole_pipeline_exact"], rows),
            "ghosts": pipeline["ghosts"],
            "misses": pipeline["misses"],
            "field_accuracy": {
                field: _metric(field_counts[field], gold_transactions) for field in FIELDS
            },
        },
    }
    return report


def read_paired_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Read paired rows without logging or retaining any reportable row detail."""

    annotations: list[Mapping[str, Any]] = []
    components: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ComponentEvaluationError("input is not a readable UTF-8 JSONL file") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _loads_strict(line, f"line {line_number}")
        row = _object(
            value,
            ("contract", "annotation", "component_output"),
            f"line {line_number}",
        )
        contract = _object(row["contract"], ("name", "version"), f"line {line_number}.contract")
        if (
            contract["name"] != CONTRACT_NAME
            or type(contract["version"]) is not int
            or contract["version"] != CONTRACT_VERSION
        ):
            raise _error(f"line {line_number}.contract", "does not select the supported version")
        if not isinstance(row["annotation"], Mapping) or not isinstance(
            row["component_output"], Mapping
        ):
            raise _error(f"line {line_number}", "must pair two objects")
        annotations.append(row["annotation"])
        components.append(row["component_output"])
    if not annotations:
        raise ComponentEvaluationError("input contains no paired rows")
    return annotations, components


def load_contract(path: Path) -> Mapping[str, Any]:
    """Validate the executable binding fields of the versioned JSON contract."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ComponentEvaluationError("contract is not a readable UTF-8 JSON file") from error
    value = _loads_strict(text, "contract")
    if not isinstance(value, Mapping):
        raise ComponentEvaluationError("contract must contain one JSON object")
    expected = {
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "record_format": "utf8_jsonl_paired_rows",
        "offset_convention": OFFSET_CONVENTION,
        "rate_representation": "fixed_6_decimal_string_or_null",
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _error(f"contract.{key}", "does not match the executable evaluator")
    return value


def evaluate_jsonl(path: Path, *, contract_path: Path | None = None) -> dict[str, Any]:
    """Evaluate one local paired JSONL file and return no per-row artifacts."""

    if contract_path is not None:
        load_contract(contract_path)
    annotations, components = read_paired_jsonl(path)
    return evaluate_component_rows(annotations, components)
