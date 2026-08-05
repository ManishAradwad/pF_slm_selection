"""Strict four-field extraction contract and repository-compatible matching."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Literal

FIELDS = ("amount", "counterparty", "type", "account")
REQUIRED_FIELDS = ("amount", "type", "account")
TRANSACTION_TYPES = frozenset({"debit", "credit"})


@dataclass(frozen=True)
class ParsedOutput:
    status: Literal["null", "transaction", "invalid"]
    value: dict[str, Any] | None
    extracted: str
    error: str | None = None


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def canonical_transaction(
    value: Any,
    *,
    allow_legacy_fields: bool = False,
) -> dict[str, Any] | None:
    """Return a validated four-field object, or None when invalid.

    Legacy gold targets may contain date; predictions may never contain extras.
    """
    if not isinstance(value, dict):
        return None
    keys = set(value)
    if allow_legacy_fields:
        if not set(FIELDS).issubset(keys):
            return None
    elif keys != set(FIELDS):
        return None

    amount = value.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return None
    amount = float(amount)
    if not math.isfinite(amount) or amount < 0:
        return None
    tx_type = value.get("type")
    if not isinstance(tx_type, str) or tx_type.strip().lower() not in TRANSACTION_TYPES:
        return None
    account = value.get("account")
    if not isinstance(account, str) or not account.strip() or account.strip().lower() == "null":
        return None
    counterparty = value.get("counterparty")
    if counterparty is not None and not isinstance(counterparty, str):
        return None
    if isinstance(counterparty, str):
        counterparty = counterparty.strip()
        if counterparty.lower() == "null":
            counterparty = None

    return {
        "amount": amount,
        "counterparty": counterparty,
        "type": tx_type.strip().lower(),
        "account": account.strip(),
    }


def parse_gold(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "null":
            return None
        try:
            value = json.loads(stripped)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("gold target is not valid JSON") from exc
    canonical = canonical_transaction(value, allow_legacy_fields=True)
    if canonical is None:
        raise ValueError("gold target violates the four-field contract")
    return canonical


def parse_prediction(text: str) -> ParsedOutput:
    if not isinstance(text, str):
        return ParsedOutput("invalid", None, "", "prediction is not text")
    cleaned = _strip_thinking(text)
    if re.fullmatch(r"null(?:\s*)", cleaned, flags=re.IGNORECASE):
        return ParsedOutput("null", None, "null")
    candidate = _first_balanced_object(cleaned)
    if candidate is None:
        return ParsedOutput("invalid", None, cleaned, "no JSON object or literal null")
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError:
        return ParsedOutput("invalid", None, candidate, "invalid JSON")
    canonical = canonical_transaction(raw)
    if canonical is None:
        return ParsedOutput("invalid", None, candidate, "schema violation")
    canonical_text = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return ParsedOutput("transaction", canonical, canonical_text)


def normalize_account(value: Any) -> tuple[str, str] | None:
    """Match the repository rule: card/account category plus final four digits."""
    if not isinstance(value, str):
        return None
    category = "card" if "card" in value.lower() else "account"
    runs = [run for run in re.findall(r"\d+", value) if len(run) >= 3]
    if not runs:
        return None
    return category, runs[-1][-4:]


def amount_matches(gold: Any, predicted: Any) -> bool:
    try:
        return abs(float(gold) - float(predicted)) < 0.01
    except (TypeError, ValueError):
        return False


def counterparty_matches(gold: Any, predicted: Any) -> bool:
    """Case/space-insensitive repository rule with guarded substring matching."""
    if gold is None and predicted is None:
        return True
    if gold is None or predicted is None:
        return False
    left = re.sub(r"\s+", " ", str(gold).strip().lower())
    right = re.sub(r"\s+", " ", str(predicted).strip().lower())
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return len(shorter) >= 3 and shorter in longer


def field_matches(field: str, gold: dict[str, Any], predicted: dict[str, Any]) -> bool:
    if field == "amount":
        return amount_matches(gold.get(field), predicted.get(field))
    if field == "counterparty":
        return counterparty_matches(gold.get(field), predicted.get(field))
    if field == "type":
        return str(gold.get(field, "")).strip().lower() == str(predicted.get(field, "")).strip().lower()
    if field == "account":
        gold_account = normalize_account(gold.get(field))
        predicted_account = normalize_account(predicted.get(field))
        return gold_account is not None and gold_account == predicted_account
    raise KeyError(field)


def transaction_matches(gold: dict[str, Any], predicted: dict[str, Any]) -> bool:
    return all(field_matches(field, gold, predicted) for field in FIELDS)
