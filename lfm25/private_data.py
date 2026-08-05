"""Local-only preparation and materialization for the private LFM2.5 corpus.

The module deliberately keeps raw messages inside ``PRIVATE_DATA/lfm25`` and
returns/prints aggregate diagnostics only.  It does not modify the source export
or the frozen regression dataset.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from DATA.utils import SYSTEM_PROMPT


SCHEMA_VERSION = 2
PREPARER_VERSION = "lfm25-private-data-v2"
LABEL_FIELDS = ("amount", "counterparty", "type", "account")
SPLIT_NAMES = ("train", "dev", "test")
SAFE_REVIEW_STATUSES = {
    "pending",
    "required",
    "human_approved",
    "human_rejected",
}


class PrivateDataError(RuntimeError):
    """An error whose message is safe to show without exposing source values."""


_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_OR_VPA_RE = re.compile(
    r"\b[a-z0-9][a-z0-9._%+\-]{0,63}@[a-z0-9][a-z0-9.\-]{1,63}\b",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"\b(?:upi\s*)?(?:ref(?:erence)?(?:\s*(?:no|number))?|utr|rrn|"
    r"txn(?:\s*(?:id|no|number))?|transaction\s*(?:id|no|number))"
    r"\s*[:#\-.]?\s*[a-z0-9][a-z0-9\-/]{3,}\b",
    re.IGNORECASE,
)
_ACCOUNT_RE = re.compile(
    r"\b(?:a\s*/?\s*c|acct|account|credit\s+card|debit\s+card|card)"
    r"(?:\s*(?:no\.?|number|ending(?:\s+in)?|linked))?"
    r"\s*[:#\-.]?\s*(?:x{2,}|\*{1,}|[x*]*\d)[x*\d\-]{2,}\b",
    re.IGNORECASE,
)
_MASKED_ACCOUNT_RE = re.compile(r"\b(?:x{2,}|\*{2,})\d{2,6}\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(?:₹|\brs\.?|\binr\b)\s*[-+]?\s*\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:inr|rupees?)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?"
    r"|\d{1,2}[-\s](?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"(?:[a-z]*[-\s]\d{2,4})?"
    r")\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)
_OTP_RE = re.compile(r"\b(?:otp|one[- ]time password)\s*(?:is|:)?\s*\d{4,8}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\w)")
_LONG_IDENTIFIER_RE = re.compile(r"(?<!\w)\d{6,}(?!\w)")
_COUNTERPARTY_RE = re.compile(
    r"\b(to|from|by|at)\s+(?:vpa\s+)?"
    r"[a-z][a-z0-9@._&*'\- ]{2,48}?"
    r"(?=\s+(?:on|via|ref|reference|dated|using|for|with|avbl|available|balance)\b"
    r"|[,;.!()]|$)",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")

_HARD_NEGATIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "otp_or_authentication",
        re.compile(
            r"\b(?:otp|one[- ]time password|verification code|authenticate|do not share)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pending_failed_or_blocked",
        re.compile(
            r"\b(?:failed|declined|pending|blocked|on hold|could not be processed|"
            r"unsuccessful|pre[- ]?authori[sz])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "payment_request_or_due",
        re.compile(
            r"\b(?:amount due|payment due|pay by|bill due|emi due|request(?:ed)? to pay|"
            r"collect request|autopay|e[- ]?mandate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "balance_or_statement",
        re.compile(
            r"\b(?:available balance|avbl bal|current balance|mini statement|statement|"
            r"investment summary|account summary)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "wallet_or_bnpl",
        re.compile(
            r"\b(?:wallet|bnpl|pay later)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "card_payment_received_notice",
        re.compile(
            r"\b(?:payment received towards (?:your )?credit card|"
            r"received (?:towards|for) (?:your )?credit card payment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "promotional",
        re.compile(
            r"\b(?:offer|cashback|discount|recharge|reward|coupon|promo|loan offer|"
            r"apply now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "action_or_link",
        re.compile(r"\b(?:click|tap|visit|download|register|verify now)\b", re.IGNORECASE),
    ),
)

_POSTED_TRANSACTION_RE = re.compile(
    r"\b(?:debited|credited|spent|withdrawn|paid|sent|received|refunded|deposited|"
    r"reversal|reversed|purchase(?:d)?|transferred)\b",
    re.IGNORECASE,
)
_DEBIT_RE = re.compile(
    r"\b(?:debited|spent|withdrawn|paid|sent|used|purchase(?:d)?|drawn)\b",
    re.IGNORECASE,
)
_CREDIT_RE = re.compile(
    r"\b(?:credited|received|refunded|deposited|reversal|reversed|added)\b",
    re.IGNORECASE,
)
_AMOUNT_CAPTURE_RE = re.compile(
    r"(?:₹|\brs\.?|\binr\b)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_ACCOUNT_CAPTURE_PATTERNS = (
    re.compile(
        r"\b((?:credit\s+card|debit\s+card|card)"
        r"(?:\s*(?:no\.?|number|ending(?:\s+in)?))?\s*[:#\-.]?\s*"
        r"(?:x{2,}|\*{1,}|[x*]*\d)[x*\d\-]{2,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:a\s*/?\s*c|acct|account)(?:\s*(?:no\.?|number))?"
        r"\s*[:#\-.]?\s*(?:x{2,}|\*{1,}|[x*]*\d)[x*\d\-]{2,})\b",
        re.IGNORECASE,
    ),
)
_COUNTERPARTY_CAPTURE_PATTERNS = (
    re.compile(
        r"\b(?:to|by|from\s+vpa|linked\s+to\s+vpa)\s+"
        r"([a-z0-9][a-z0-9@._&*'\- ]{1,64}?)"
        r"(?=\s+(?:on|via|ref|reference|dated|using|for|with)\b|[,;.!()]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bat\s+([a-z0-9][a-z0-9@._&*'\- ]{1,64}?)"
        r"(?=\s+(?:on|via|ref|reference|dated|using|for|with)\b|[,;.!()]|$)",
        re.IGNORECASE,
    ),
)


def normalize_unicode(value: str) -> str:
    """Return stable NFKC text with controls removed and whitespace collapsed."""

    normalized = unicodedata.normalize("NFKC", value)
    cleaned = "".join(
        " " if character.isspace() else character
        for character in normalized
        if character.isspace() or unicodedata.category(character) not in {"Cc", "Cf"}
    )
    return _SPACE_RE.sub(" ", cleaned).strip()


def canonical_exact_text(value: str) -> str:
    """Canonical text used in the case-insensitive exact-overlap check."""

    return normalize_unicode(value).casefold()


def normalize_template(value: str) -> str:
    """Normalize a message into a privacy-reduced deterministic template family."""

    text = canonical_exact_text(value)
    text = _URL_RE.sub(" <PII> ", text)
    text = _EMAIL_OR_VPA_RE.sub(" <PII> ", text)
    text = _REFERENCE_RE.sub(" <REFERENCE> ", text)
    text = _ACCOUNT_RE.sub(" <ACCOUNT> ", text)
    text = _MASKED_ACCOUNT_RE.sub(" <ACCOUNT> ", text)
    text = _AMOUNT_RE.sub(" <AMOUNT> ", text)
    text = _DATE_RE.sub(" <DATE> ", text)
    text = _TIME_RE.sub(" <DATE> ", text)
    text = _OTP_RE.sub(" <PII> ", text)
    text = _PHONE_RE.sub(" <PII> ", text)
    text = _LONG_IDENTIFIER_RE.sub(" <PII> ", text)
    text = _COUNTERPARTY_RE.sub(lambda match: f"{match.group(1)} <PII>", text)
    return _SPACE_RE.sub(" ", text).strip()


def character_ngrams(value: str, size: int) -> frozenset[str]:
    """Return the character shingles used by the configurable near-overlap check."""

    if size < 1:
        raise PrivateDataError("character n-gram size must be positive")
    compact = _SPACE_RE.sub(" ", value)
    if len(compact) <= size:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + size] for index in range(len(compact) - size + 1))


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Calculate a bounded Jaccard score without retaining matched source text."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_hash(key: bytes, namespace: str, value: str) -> str:
    payload = namespace.encode("utf-8") + b"\0" + value.encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def hash_key_identifier(key: bytes) -> str:
    return hashlib.sha256(b"lfm25-key-id\0" + key).hexdigest()[:16]


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateDataError("the private-data config could not be read") from exc
    if not isinstance(value, dict):
        raise PrivateDataError("the private-data config must be a JSON object")
    if tuple(value.get("label_fields", ())) != LABEL_FIELDS:
        raise PrivateDataError("the config label_fields must be the canonical four-field schema")
    source_fields = value.get("source_fields")
    required_source_fields = {"id", "date", "sender", "text", "is_from_me"}
    if not isinstance(source_fields, Mapping) or set(source_fields) != required_source_fields:
        raise PrivateDataError(
            "the config source_fields must map id, date, sender, text, and is_from_me"
        )
    configured_field_names = tuple(source_fields.values())
    if not all(isinstance(name, str) and name for name in configured_field_names):
        raise PrivateDataError("configured source field names must be non-empty strings")
    if len(set(configured_field_names)) != len(configured_field_names):
        raise PrivateDataError("configured source field names must be distinct")
    policy = value.get("consensus_policy")
    if not isinstance(policy, Mapping):
        raise PrivateDataError("the config must define a consensus acceptance policy")
    if policy.get("test_requires_human_review") is not True:
        raise PrivateDataError("the consensus policy must require human review for test rows")
    if policy.get("require_exact_four_fields") is not True:
        raise PrivateDataError(
            "the consensus policy must require the exact four-field schema"
        )
    return value


def _parse_source_boolean(value: Any, row_number: int) -> bool:
    """Normalize the JSON boolean and common lossless CSV encodings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"false", "0"}:
            return False
        if normalized in {"true", "1"}:
            return True
    raise PrivateDataError(
        f"source row {row_number} has an invalid is_from_me direction flag"
    )


def _validate_source_row(
    row: Mapping[str, Any], row_number: int, fields: Mapping[str, str]
) -> dict[str, Any]:
    missing = [name for name in fields.values() if name not in row]
    if missing:
        raise PrivateDataError(f"source row {row_number} is missing required fields")
    raw_id = row[fields["id"]]
    if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool):
        raise PrivateDataError(f"source row {row_number} has an unsupported ID type")
    text = row[fields["text"]]
    sender = row[fields["sender"]]
    date = row[fields["date"]]
    is_from_me = _parse_source_boolean(row[fields["is_from_me"]], row_number)
    if not isinstance(text, str) or not text.strip():
        raise PrivateDataError(f"source row {row_number} has no message text")
    if not isinstance(sender, str) or not isinstance(date, str):
        raise PrivateDataError(f"source row {row_number} has invalid metadata types")
    return {
        "source_id": str(raw_id),
        "date": date,
        "sender": sender,
        "sms": text,
        "is_from_me": is_from_me,
        "source_row": row_number,
    }


def read_source_rows(path: Path, fields: Mapping[str, str]) -> tuple[list[dict[str, Any]], str]:
    """Read all source rows, normalizing JSON/CSV direction without filtering."""

    before = file_sha256(path)
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("messages", payload.get("data"))
            if not isinstance(payload, list):
                raise PrivateDataError("source JSON must contain a row array")
            for index, raw_row in enumerate(payload, start=1):
                if not isinstance(raw_row, dict):
                    raise PrivateDataError(f"source row {index} is not an object")
                rows.append(_validate_source_row(raw_row, index, fields))
        elif path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = set(fields.values())
                if not required.issubset(set(reader.fieldnames or ())):
                    raise PrivateDataError("source CSV is missing required columns")
                for index, raw_row in enumerate(reader, start=2):
                    rows.append(_validate_source_row(raw_row, index, fields))
        else:
            raise PrivateDataError("source must be a local all_sms JSON or CSV file")
    except (OSError, csv.Error, json.JSONDecodeError) as exc:
        raise PrivateDataError("the source export could not be parsed") from exc
    after = file_sha256(path)
    if not hmac.compare_digest(before, after):
        raise PrivateDataError("the source export changed while it was being read")
    return rows, before


def read_regression_rows(path: Path, expected_rows: int | None = None) -> tuple[list[str], str]:
    """Read only SMS text from the frozen JSONL regression set."""

    before = file_sha256(path)
    messages: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("sms"), str):
                    raise PrivateDataError(f"regression row {row_number} has no SMS text")
                messages.append(row["sms"])
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateDataError("the regression dataset could not be parsed") from exc
    after = file_sha256(path)
    if not hmac.compare_digest(before, after):
        raise PrivateDataError("the regression dataset changed while it was being read")
    if expected_rows is not None and len(messages) != expected_rows:
        raise PrivateDataError("the frozen regression row count does not match the config")
    return messages, before


class RegressionIndex:
    """In-memory fingerprints for all regression exclusion levels."""

    def __init__(self, messages: Sequence[str], ngram_size: int) -> None:
        self.raw_exact = frozenset(messages)
        self.canonical_exact = frozenset(canonical_exact_text(message) for message in messages)
        self.templates = tuple(normalize_template(message) for message in messages)
        self.template_exact = frozenset(self.templates)
        self.shingles = tuple(character_ngrams(template, ngram_size) for template in self.templates)
        self.ngram_size = ngram_size

    def exclusion_reason(
        self,
        message: str,
        template: str,
        threshold: float,
        minimum_characters: int,
    ) -> str | None:
        if message in self.raw_exact or canonical_exact_text(message) in self.canonical_exact:
            return "exact"
        if template in self.template_exact:
            return "normalized_template"
        if len(template) < minimum_characters:
            return None
        candidate = character_ngrams(template, self.ngram_size)
        for regression_shingles in self.shingles:
            if jaccard_similarity(candidate, regression_shingles) >= threshold:
                return "near_relative"
        return None


def _parse_timestamp(value: str) -> float | None:
    cleaned = normalize_unicode(value)
    if not cleaned:
        return None
    candidates = (cleaned, cleaned.replace("Z", "+00:00"))
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, date_format).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _split_group_counts(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if group_count < 0:
        raise PrivateDataError("group count cannot be negative")
    values = [float(ratios[name]) for name in SPLIT_NAMES]
    if any(value < 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise PrivateDataError("split ratios must be non-negative and sum to one")
    raw = [group_count * value for value in values]
    counts = [math.floor(value) for value in raw]
    remainder = group_count - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    if group_count >= 3:
        for index, value in enumerate(values):
            if value <= 0 or counts[index] > 0:
                continue
            donor = max(range(3), key=lambda candidate: counts[candidate])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts, strict=True))


def _sender_template_components(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[tuple[str, ...]]:
    """Return template components connected by a private sender hash."""

    parent = {template_group: template_group for template_group in grouped}

    def find(template_group: str) -> str:
        while parent[template_group] != template_group:
            parent[template_group] = parent[parent[template_group]]
            template_group = parent[template_group]
        return template_group

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    first_template_by_sender: dict[str, str] = {}
    for template_group in sorted(grouped):
        sender_hashes = sorted(
            {
                str(row["private_hashes"]["sender"])
                for row in grouped[template_group]
            }
        )
        for sender_hash in sender_hashes:
            previous = first_template_by_sender.setdefault(sender_hash, template_group)
            union(previous, template_group)

    components: dict[str, list[str]] = defaultdict(list)
    for template_group in sorted(grouped):
        components[find(template_group)].append(template_group)
    return [tuple(groups) for _, groups in sorted(components.items())]


def _contiguous_weighted_split_assignments(
    weights: Sequence[int], ratios: Mapping[str, float]
) -> list[str]:
    """Partition chronological units near requested row ratios without splitting units."""

    if not weights:
        return []
    active_splits = [name for name in SPLIT_NAMES if float(ratios[name]) > 0]
    if not active_splits:
        raise PrivateDataError("at least one split ratio must be positive")
    if len(weights) < len(active_splits):
        counts = _split_group_counts(len(weights), ratios)
        return [name for name in SPLIT_NAMES for _ in range(counts[name])]

    prefix_weights = [0]
    for weight in weights:
        if weight < 1:
            raise PrivateDataError("split assignment units must contain at least one row")
        prefix_weights.append(prefix_weights[-1] + weight)

    assignments: list[str] = []
    previous_end = 0
    cumulative_ratio = 0.0
    total_weight = prefix_weights[-1]
    for active_index, split in enumerate(active_splits[:-1]):
        cumulative_ratio += float(ratios[split])
        target_weight = total_weight * cumulative_ratio
        remaining_splits = len(active_splits) - active_index - 1
        minimum_end = previous_end + 1
        maximum_end = len(weights) - remaining_splits
        end = min(
            range(minimum_end, maximum_end + 1),
            key=lambda candidate: (
                abs(prefix_weights[candidate] - target_weight),
                candidate,
            ),
        )
        assignments.extend([split] * (end - previous_end))
        previous_end = end
    assignments.extend([active_splits[-1]] * (len(weights) - previous_end))
    return assignments


def assign_splits(
    records: list[dict[str, Any]],
    ratios: Mapping[str, float],
    rule_version: str,
    key: bytes,
) -> dict[str, Any]:
    """Assign chronological sender/template components, with a truthful fallback."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["template_group"]].append(record)

    active_split_count = sum(float(ratios[name]) > 0 for name in SPLIT_NAMES)
    sender_components = _sender_template_components(grouped)
    sender_component_lock = len(sender_components) >= active_split_count
    assignment_unit = (
        "sender_template_component" if sender_component_lock else "template_family"
    )
    units = (
        sender_components
        if sender_component_lock
        else [(template_group,) for template_group in sorted(grouped)]
    )

    sortable_units: list[tuple[float, str, tuple[str, ...], int]] = []
    for template_groups in units:
        members = [row for group in template_groups for row in grouped[group]]
        timestamps = [
            timestamp
            for row in members
            if (timestamp := _parse_timestamp(row["date"])) is not None
        ]
        timestamps.sort()
        time_anchor = (
            timestamps[len(timestamps) // 2] if timestamps else float("-inf")
        )
        tie_break = private_hash(
            key,
            "split-unit-tie",
            "\0".join(template_groups),
        )
        sortable_units.append((time_anchor, tie_break, template_groups, len(members)))
    sortable_units.sort(key=lambda unit: (unit[0], unit[1]))

    assignments: dict[str, str] = {}
    unit_assignments = _contiguous_weighted_split_assignments(
        [unit[3] for unit in sortable_units],
        ratios,
    )
    unit_counts: Counter[str] = Counter()
    for (_, _, template_groups, _), split in zip(
        sortable_units, unit_assignments, strict=True
    ):
        unit_counts[split] += 1
        for template_group in template_groups:
            assignments[template_group] = split

    for record in records:
        split = assignments[record["template_group"]]
        record["split"] = split
        record["human_review_required"] = split == "test"
        record["review_status"] = "required" if split == "test" else "pending"
        record["split_provenance"] = {
            "rule_version": rule_version,
            "template_family_locked": True,
            "assignment_unit": assignment_unit,
            "sender_component_locked": sender_component_lock,
            "chronological_assignment_unit_order": True,
            "time_anchor": "median_parseable_source_timestamp",
            "assigned_before_labeling": True,
        }
    group_counts = Counter(assignments.values())
    return {
        "assignment_unit": assignment_unit,
        "assignment_unit_count": len(sortable_units),
        "sender_component_lock_applied": sender_component_lock,
        "split_assignment_unit_counts": {
            name: unit_counts.get(name, 0) for name in SPLIT_NAMES
        },
        "split_group_counts": {
            name: group_counts.get(name, 0) for name in SPLIT_NAMES
        },
    }


def categorize_hard_negative(message: str) -> str | None:
    """Classify high-value non-transaction examples without producing a gold label."""

    posted_transaction = bool(_POSTED_TRANSACTION_RE.search(message))
    for category, pattern in _HARD_NEGATIVE_PATTERNS:
        if not pattern.search(message):
            continue
        # Balance snippets, security links, or offer words commonly trail a real
        # posted transaction. They are negatives only when no posting verb exists.
        if posted_transaction and category in {
            "balance_or_statement",
            "promotional",
            "action_or_link",
        }:
            continue
        return category
    if not posted_transaction:
        return "non_transaction_or_ambiguous"
    return None


def _extract_amount(message: str) -> float | None:
    match = _AMOUNT_CAPTURE_RE.search(message)
    if not match:
        return None
    try:
        amount = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


def _extract_account(message: str) -> str | None:
    for pattern in _ACCOUNT_CAPTURE_PATTERNS:
        match = pattern.search(message)
        if match:
            return normalize_unicode(match.group(1))
    return None


def _extract_counterparty(message: str) -> str | None:
    for pattern in _COUNTERPARTY_CAPTURE_PATTERNS:
        match = pattern.search(message)
        if match:
            candidate = normalize_unicode(match.group(1)).strip(" -:,.\n\t")
            if candidate and candidate.casefold() not in {"your account", "your a/c", "your card"}:
                return candidate
    return None


def propose_silver_label(message: str, hard_negative: str | None) -> tuple[Any, float, list[str]]:
    """Return a heuristic proposal; it is always silver and never silently gold."""

    if hard_negative is not None:
        confidence = 0.92 if hard_negative != "non_transaction_or_ambiguous" else 0.62
        return None, confidence, [f"hard_negative:{hard_negative}"]

    amount = _extract_amount(message)
    account = _extract_account(message)
    debit = bool(_DEBIT_RE.search(message))
    credit = bool(_CREDIT_RE.search(message))
    transaction_type = (
        "debit" if debit and not credit else "credit" if credit and not debit else None
    )
    if amount is None or account is None or transaction_type is None:
        return None, 0.5, ["incomplete_transaction_evidence"]
    label = {
        "amount": amount,
        "counterparty": _extract_counterparty(message),
        "type": transaction_type,
        "account": account,
    }
    confidence = 0.86 + (0.04 if label["counterparty"] is not None else 0.0)
    return label, confidence, ["posted_transaction_pattern", "required_fields_extracted"]


def validate_label(label: Any, allow_null: bool = True) -> bool:
    if label is None:
        return allow_null
    if not isinstance(label, dict) or tuple(label) != LABEL_FIELDS:
        return False
    amount = label["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return False
    if not math.isfinite(float(amount)) or float(amount) < 0:
        return False
    if label["type"] not in {"debit", "credit"}:
        return False
    if not isinstance(label["account"], str) or not label["account"].strip():
        return False
    counterparty = label["counterparty"]
    return counterparty is None or (isinstance(counterparty, str) and bool(counterparty.strip()))


def canonical_label(label: Any) -> Any:
    if label is None:
        return None
    if not isinstance(label, Mapping) or set(label) != set(LABEL_FIELDS):
        raise PrivateDataError("a label does not have exactly the four canonical fields")
    ordered = {field: label[field] for field in LABEL_FIELDS}
    if not validate_label(ordered):
        raise PrivateDataError("a label does not satisfy the four-field schema")
    ordered["amount"] = float(ordered["amount"])
    return ordered


def empty_consensus(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_version": policy["policy_version"],
        "status": "not_evaluated",
        "accepted": False,
        "accepted_label": None,
        "valid_proposal_count": 0,
        "agreeing_model_count": 0,
        "independent_model_family_count": 0,
        "required_proposal_fields": [
            "model_id",
            "model_family",
            "label",
            "confidence",
            "inference_config_hash",
        ],
    }


def _safe_source_name(path: Path, allowed_names: Sequence[str]) -> None:
    if path.name not in allowed_names or path.suffix.casefold() not in {".json", ".csv"}:
        raise PrivateDataError("source must be an allowed local all_sms JSON or CSV export")


def _filter_incoming_candidates(
    source_rows: Sequence[dict[str, Any]],
    regression_index: RegressionIndex,
    threshold: float,
    minimum_template_characters: int,
) -> tuple[list[tuple[dict[str, Any], str]], Counter[str], dict[str, int]]:
    """Apply the incoming-only boundary before regression decontamination."""

    eligible: list[tuple[dict[str, Any], str]] = []
    excluded: Counter[str] = Counter()
    incoming_count = 0
    outgoing_count = 0
    for source_row in source_rows:
        if source_row["is_from_me"]:
            outgoing_count += 1
            continue
        incoming_count += 1
        template = normalize_template(source_row["sms"])
        reason = regression_index.exclusion_reason(
            source_row["sms"],
            template,
            threshold,
            minimum_template_characters,
        )
        if reason is not None:
            excluded[reason] += 1
            continue
        eligible.append((source_row, template))

    boundary_counts = {
        "total": len(source_rows),
        "incoming": incoming_count,
        "outgoing_excluded": outgoing_count,
    }
    if boundary_counts["total"] != incoming_count + outgoing_count:
        raise PrivateDataError("source direction boundary counts are inconsistent")
    if incoming_count != len(eligible) + sum(excluded.values()):
        raise PrivateDataError("incoming decontamination counts are inconsistent")
    return eligible, excluded, boundary_counts


def prepare_private_dataset(
    source_path: Path,
    regression_path: Path,
    config: Mapping[str, Any],
    key: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a decontaminated manifest and queue entirely in memory."""

    _safe_source_name(source_path, config["allowed_source_names"])
    all_source_rows, source_fingerprint = read_source_rows(
        source_path, config["source_fields"]
    )
    regression_rows, regression_fingerprint = read_regression_rows(
        regression_path,
        int(config["expected_regression_rows"]),
    )
    near_config = config["near_relative"]
    threshold = float(near_config["threshold"])
    if not 0 < threshold <= 1:
        raise PrivateDataError("near-relative threshold must be in (0, 1]")
    ngram_size = int(near_config["character_ngram_size"])
    minimum_characters = int(near_config["minimum_template_characters"])
    regression_index = RegressionIndex(regression_rows, ngram_size)

    eligible_candidates, excluded, source_boundary_counts = _filter_incoming_candidates(
        all_source_rows,
        regression_index,
        threshold,
        minimum_characters,
    )

    records: list[dict[str, Any]] = []
    key_id = hash_key_identifier(key)
    for source_row, template in eligible_candidates:
        source_id = source_row["source_id"]
        date = source_row["date"]
        sender = source_row["sender"]
        sms = source_row["sms"]
        record_hash = private_hash(key, "record", "\0".join((source_id, date, sender, sms)))
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_hash": record_hash,
                "source_id": source_id,
                "date": date,
                "sender": sender,
                "sms": sms,
                "is_from_me": False,
                "template_group": "tg_" + private_hash(key, "template", template)[:32],
                "private_hashes": {
                    "record": record_hash,
                    "source_id": private_hash(key, "source-id", source_id),
                    "sender": private_hash(key, "sender", sender),
                    "text": private_hash(key, "text", canonical_exact_text(sms)),
                    "template": private_hash(key, "template", template),
                    "key_id": key_id,
                },
                "provenance": {
                    "preparer_version": PREPARER_VERSION,
                    "source_file": source_path.name,
                    "source_format": source_path.suffix.casefold().lstrip("."),
                    "source_direction_filter": "is_from_me == false",
                    "source_row": source_row["source_row"],
                    "source_file_sha256": source_fingerprint,
                    "regression_file": regression_path.name,
                    "regression_file_sha256": regression_fingerprint,
                },
            }
        )

    split_config = config["splits"]
    split_assignment = assign_splits(
        records,
        {name: float(split_config[name]) for name in SPLIT_NAMES},
        str(split_config["rule_version"]),
        key,
    )

    # This phase intentionally occurs only after split assignment.
    for record in records:
        hard_negative = categorize_hard_negative(record["sms"])
        label, confidence, reason_codes = propose_silver_label(record["sms"], hard_negative)
        record.update(
            {
                "hard_negative_category": hard_negative,
                "label_tier": "silver",
                "silver_label": canonical_label(label),
                "confidence": confidence,
                "heuristic_reason_codes": reason_codes,
                "human_label": None,
                "human_reviewer": None,
                "human_reviewed_at": None,
                "local_model_proposals": [],
                "consensus_acceptance": empty_consensus(config["consensus_policy"]),
            }
        )

    review_queue = []
    for record in records:
        queue_record = dict(record)
        queue_record["review_priority"] = (
            0
            if record["split"] == "test"
            else 1
            if record["hard_negative_category"] is not None
            else 2
        )
        review_queue.append(queue_record)
    review_queue.sort(
        key=lambda row: (row["review_priority"], row["confidence"], row["record_hash"])
    )
    records.sort(key=lambda row: row["record_hash"])

    report = aggregate_validation_report(
        records,
        regression_index=regression_index,
        near_threshold=threshold,
        minimum_template_characters=minimum_characters,
    )
    source_immutable = file_sha256(source_path) == source_fingerprint
    regression_immutable = file_sha256(regression_path) == regression_fingerprint
    report.update(
        {
            "preparer_version": PREPARER_VERSION,
            "source_row_count": source_boundary_counts["total"],
            "source_boundary_counts": source_boundary_counts,
            "regression_row_count": len(regression_rows),
            "eligible_row_count": len(records),
            "excluded_row_count": sum(excluded.values()),
            "exclusions_by_reason": dict(sorted(excluded.items())),
            "split_group_counts": split_assignment["split_group_counts"],
            "split_assignment_unit_counts": split_assignment[
                "split_assignment_unit_counts"
            ],
            "split_assignment_unit": split_assignment["assignment_unit"],
            "split_assignment_unit_count": split_assignment["assignment_unit_count"],
            "sender_component_lock_applied": split_assignment[
                "sender_component_lock_applied"
            ],
            "source_immutable": source_immutable,
            "regression_immutable": regression_immutable,
        }
    )
    if not source_immutable or not regression_immutable:
        error_counts = dict(report["error_counts"])
        if not source_immutable:
            error_counts["source_changed_after_read"] = 1
        if not regression_immutable:
            error_counts["regression_changed_after_read"] = 1
        report["error_counts"] = error_counts
        report["valid"] = False
    report["invariants"]["source_export_mutated"] = not source_immutable
    report["invariants"]["regression_dataset_mutated"] = not regression_immutable
    report["invariants"]["source_boundary_count_mismatch"] = False
    report["invariants"]["incoming_preparation_count_mismatch"] = False
    return records, review_queue, report


def aggregate_validation_report(
    records: Sequence[Mapping[str, Any]],
    regression_index: RegressionIndex | None = None,
    near_threshold: float = 1.0,
    minimum_template_characters: int = 12,
) -> dict[str, Any]:
    """Validate invariants and expose counts only, never offending record values."""

    errors: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    label_tier_counts: Counter[str] = Counter()
    silver_label_counts: Counter[str] = Counter()
    template_splits: dict[str, set[str]] = defaultdict(set)
    sender_splits: dict[str, set[str]] = defaultdict(set)
    split_timestamps: dict[str, list[float]] = {name: [] for name in SPLIT_NAMES}
    unparseable_timestamp_count = 0
    sender_component_lock_claims: set[bool] = set()
    record_hashes: set[str] = set()
    for record in records:
        split = record.get("split")
        if split not in SPLIT_NAMES:
            errors["invalid_split"] += 1
        else:
            split_counts[str(split)] += 1
            date = record.get("date")
            timestamp = _parse_timestamp(date) if isinstance(date, str) else None
            if timestamp is None:
                errors["unparseable_source_timestamp"] += 1
                unparseable_timestamp_count += 1
            else:
                split_timestamps[str(split)].append(timestamp)
        if record.get("is_from_me") is not False:
            errors["outgoing_or_unknown_source_direction"] += 1
        template_group = record.get("template_group")
        if not isinstance(template_group, str):
            errors["missing_template_group"] += 1
        elif split in SPLIT_NAMES:
            template_splits[template_group].add(str(split))
        record_hash = record.get("record_hash")
        if not isinstance(record_hash, str) or not record_hash:
            errors["missing_record_hash"] += 1
        elif record_hash in record_hashes:
            errors["duplicate_record_hash"] += 1
        else:
            record_hashes.add(record_hash)
        if not all(field in record for field in ("source_id", "date", "sender")):
            errors["missing_source_metadata"] += 1
        private_hashes = record.get("private_hashes")
        if not isinstance(private_hashes, Mapping):
            errors["missing_private_hashes"] += 1
        else:
            sender_hash = private_hashes.get("sender")
            if not isinstance(sender_hash, str) or not sender_hash:
                errors["missing_private_sender_hash"] += 1
            elif split in SPLIT_NAMES:
                sender_splits[sender_hash].add(str(split))
        if not isinstance(record.get("provenance"), Mapping):
            errors["missing_provenance"] += 1
        split_provenance = record.get("split_provenance")
        if not isinstance(split_provenance, Mapping):
            errors["missing_split_provenance"] += 1
        else:
            if split_provenance.get("template_family_locked") is not True:
                errors["template_lock_not_declared"] += 1
            if split_provenance.get("assigned_before_labeling") is not True:
                errors["split_not_declared_before_labeling"] += 1
            if split_provenance.get("chronological_assignment_unit_order") is not True:
                errors["chronological_assignment_not_declared"] += 1
            sender_component_locked = split_provenance.get("sender_component_locked")
            if not isinstance(sender_component_locked, bool):
                errors["invalid_sender_component_lock_claim"] += 1
            else:
                sender_component_lock_claims.add(sender_component_locked)
        status = record.get("review_status")
        if status not in SAFE_REVIEW_STATUSES:
            errors["invalid_review_status"] += 1
        else:
            review_counts[str(status)] += 1
        if split == "test" and not record.get("human_review_required"):
            errors["test_not_marked_for_human_review"] += 1
        if status == "human_approved":
            reviewer = record.get("human_reviewer")
            reviewed_at = record.get("human_reviewed_at")
            if not isinstance(reviewer, str) or not reviewer.strip():
                errors["approved_row_missing_reviewer"] += 1
            if not isinstance(reviewed_at, str) or _parse_timestamp(reviewed_at) is None:
                errors["approved_row_missing_review_time"] += 1
        if record.get("label_tier") != "silver":
            errors["non_silver_heuristic_label"] += 1
        if not validate_label(record.get("silver_label")):
            errors["invalid_silver_label"] += 1
        label_tier_counts[str(record.get("label_tier"))] += 1
        silver_kind = "null" if record.get("silver_label") is None else "transaction"
        silver_label_counts[silver_kind] += 1
        hard_negative = record.get("hard_negative_category")
        if hard_negative is not None:
            hard_negative_counts[str(hard_negative)] += 1
        if regression_index is not None:
            sms = record.get("sms")
            if not isinstance(sms, str):
                errors["missing_sms"] += 1
            else:
                template = normalize_template(sms)
                if regression_index.exclusion_reason(
                    sms,
                    template,
                    near_threshold,
                    minimum_template_characters,
                ) is not None:
                    errors["regression_overlap"] += 1
    errors["template_split_leakage"] += sum(len(splits) > 1 for splits in template_splits.values())
    if len(sender_component_lock_claims) > 1:
        errors["inconsistent_sender_component_lock_claim"] += 1

    split_sender_sets = {
        split: {sender for sender, splits in sender_splits.items() if split in splits}
        for split in SPLIT_NAMES
    }
    sender_overlap_count = sum(len(splits) > 1 for splits in sender_splits.values())
    sender_component_lock_claimed = sender_component_lock_claims == {True}
    if sender_component_lock_claimed and sender_overlap_count:
        errors["sender_component_lock_violation"] += sender_overlap_count
    pairwise_sender_overlap_counts = {
        f"{left}_{right}": len(split_sender_sets[left] & split_sender_sets[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    sender_diagnostics = {
        "distinct_sender_count": len(sender_splits),
        "split_distinct_sender_counts": {
            split: len(split_sender_sets[split]) for split in SPLIT_NAMES
        },
        "cross_split_sender_count": sender_overlap_count,
        "pairwise_overlap_counts": pairwise_sender_overlap_counts,
        "sender_held_out": sender_overlap_count == 0,
        "sender_component_lock_claimed": sender_component_lock_claimed,
    }

    chronology_boundaries: dict[str, dict[str, Any]] = {}
    for earlier, later in (("train", "dev"), ("dev", "test")):
        earlier_times = split_timestamps[earlier]
        later_times = split_timestamps[later]
        diagnostic: dict[str, Any] = {
            "comparable": bool(earlier_times and later_times),
            "ordered_without_overlap": False,
            "earlier_rows_after_later_start": 0,
            "later_rows_before_earlier_end": 0,
        }
        if earlier_times and later_times:
            later_start = min(later_times)
            earlier_end = max(earlier_times)
            diagnostic.update(
                {
                    "ordered_without_overlap": earlier_end <= later_start,
                    "earlier_rows_after_later_start": sum(
                        timestamp > later_start for timestamp in earlier_times
                    ),
                    "later_rows_before_earlier_end": sum(
                        timestamp < earlier_end for timestamp in later_times
                    ),
                }
            )
        chronology_boundaries[f"{earlier}_to_{later}"] = diagnostic
    chronology_diagnostics = {
        "split_parseable_timestamp_row_counts": {
            split: len(split_timestamps[split]) for split in SPLIT_NAMES
        },
        "unparseable_timestamp_row_count": unparseable_timestamp_count,
        "boundaries": chronology_boundaries,
        "all_boundaries_ordered_without_overlap": all(
            boundary["comparable"] and boundary["ordered_without_overlap"]
            for boundary in chronology_boundaries.values()
        ),
    }
    error_counts = {name: count for name, count in sorted(errors.items()) if count}
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not error_counts,
        "error_counts": error_counts,
        "row_count": len(records),
        "split_row_counts": {name: split_counts.get(name, 0) for name in SPLIT_NAMES},
        "template_group_count": len(template_splits),
        "split_group_counts": {
            split: sum(split in splits for splits in template_splits.values())
            for split in SPLIT_NAMES
        },
        "sender_split_diagnostics": sender_diagnostics,
        "chronology_diagnostics": chronology_diagnostics,
        "review_status_counts": dict(sorted(review_counts.items())),
        "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
        "label_tier_counts": dict(sorted(label_tier_counts.items())),
        "silver_label_counts": dict(sorted(silver_label_counts.items())),
        "invariants": {
            "raw_values_emitted_to_stdout": False,
            "regression_dataset_mutated": False,
            "outgoing_rows_present": bool(
                errors.get("outgoing_or_unknown_source_direction")
            ),
            "template_families_cross_splits": bool(errors.get("template_split_leakage")),
            "sender_components_cross_splits": sender_overlap_count > 0,
            "sender_component_lock_violated": bool(
                errors.get("sender_component_lock_violation")
            ),
            "test_requires_human_review": True,
            "heuristics_are_silver_only": True,
        },
    }


def evaluate_consensus(record: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate later local-model proposals against the persisted acceptance policy."""

    base = empty_consensus(policy)
    split = record.get("split")
    if split not in policy["eligible_splits"]:
        base["status"] = "split_ineligible"
        return base
    proposals = record.get("local_model_proposals", [])
    if not isinstance(proposals, list):
        base["status"] = "invalid_proposals"
        return base

    valid: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    required = set(base["required_proposal_fields"])
    minimum_confidence = float(policy["minimum_proposal_confidence"])
    for proposal in proposals:
        if not isinstance(proposal, dict) or not required.issubset(proposal):
            continue
        model_id = proposal["model_id"]
        family = proposal["model_family"]
        config_hash = proposal["inference_config_hash"]
        confidence = proposal["confidence"]
        if not all(isinstance(value, str) and value for value in (model_id, family, config_hash)):
            continue
        if model_id in seen_models or isinstance(confidence, bool):
            continue
        if not isinstance(confidence, (int, float)) or confidence < minimum_confidence:
            continue
        try:
            label = canonical_label(proposal["label"])
        except PrivateDataError:
            continue
        seen_models.add(model_id)
        valid.append({"model_id": model_id, "model_family": family, "label": label})

    base["valid_proposal_count"] = len(valid)
    if len(valid) < int(policy["minimum_proposals"]):
        base["status"] = "insufficient_valid_proposals"
        return base

    agreement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in valid:
        encoded = json.dumps(proposal["label"], ensure_ascii=False, separators=(",", ":"))
        agreement[encoded].append(proposal)
    _, winners = max(agreement.items(), key=lambda item: (len(item[1]), item[0]))
    agreeing_models = len(winners)
    independent_families = len({proposal["model_family"] for proposal in winners})
    base["agreeing_model_count"] = agreeing_models
    base["independent_model_family_count"] = independent_families
    if agreeing_models < int(policy["minimum_agreeing_models"]):
        base["status"] = "insufficient_agreement"
        return base
    if independent_families < int(policy["minimum_independent_model_families"]):
        base["status"] = "insufficient_model_family_diversity"
        return base
    base.update(
        {
            "status": "accepted",
            "accepted": True,
            "accepted_label": winners[0]["label"],
        }
    )
    return base


def accepted_training_label(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[bool, Any, str | None]:
    """Select only human gold or policy-accepted silver for materialization."""

    if record.get("review_status") == "human_approved":
        reviewer = record.get("human_reviewer")
        reviewed_at = record.get("human_reviewed_at")
        if not isinstance(reviewer, str) or not reviewer.strip():
            return False, None, None
        if not isinstance(reviewed_at, str) or _parse_timestamp(reviewed_at) is None:
            return False, None, None
        try:
            return True, canonical_label(record.get("human_label")), "gold"
        except PrivateDataError:
            return False, None, None
    if record.get("split") == "test":
        return False, None, None
    consensus = evaluate_consensus(record, policy)
    if consensus["accepted"]:
        return True, consensus["accepted_label"], "silver"
    return False, None, None


def build_sft_conversation(record: Mapping[str, Any], label: Any) -> dict[str, Any]:
    """Build an assistant-last, completion-only compatible messages example."""

    canonical = canonical_label(label)
    assistant = (
        "null"
        if canonical is None
        else json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sender: {record['sender']}\nSMS: {record['sms']}",
            },
            {"role": "assistant", "content": assistant},
        ]
    }


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise PrivateDataError(f"private manifest row {row_number} is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateDataError("the private manifest could not be parsed") from exc
    return rows


def ensure_within(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PrivateDataError("a requested output is outside the configured private root") from exc
    return resolved_path


def is_git_ignored(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def require_private_ignore(repo_root: Path, private_root: Path) -> None:
    probes = (
        private_root / ".privacy-boundary-probe.jsonl",
        private_root / ".privacy-boundary-probe.json",
        private_root / ".privacy-boundary-secret",
    )
    if not all(is_git_ignored(repo_root, probe) for probe in probes):
        raise PrivateDataError("PRIVATE_DATA is not protected by the repository ignore rules")


def load_or_create_hash_key(private_root: Path) -> bytes:
    key_path = private_root / ".hash_key"
    private_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_root, 0o700)
    try:
        key = key_path.read_bytes()
    except FileNotFoundError:
        key = secrets.token_bytes(32)
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = key_path.read_bytes()
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
    os.chmod(key_path, 0o600)
    if len(key) < 32:
        raise PrivateDataError("the private hash key is invalid")
    return key


def _resolve_repo_input(repo_root: Path, configured: str | Path) -> Path:
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise PrivateDataError("input paths must remain inside the local repository") from exc
    if not resolved.is_file():
        raise PrivateDataError("a configured local input file does not exist")
    return resolved


def _configured_paths(
    repo_root: Path, config: Mapping[str, Any]
) -> tuple[Path, Path, dict[str, Path]]:
    required_private_root = (repo_root / "PRIVATE_DATA" / "lfm25").resolve()
    private_root = ensure_within(repo_root / config["private_root"], required_private_root)
    outputs = {
        name: ensure_within(private_root / filename, private_root)
        for name, filename in config["outputs"].items()
        if name != "sft_prefix"
    }
    return private_root, _resolve_repo_input(repo_root, config["regression_dataset"]), outputs


def run_prepare(
    repo_root: Path,
    config: Mapping[str, Any],
    source_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    private_root, regression_path, outputs = _configured_paths(repo_root, config)
    require_private_ignore(repo_root, private_root)
    source_path = _resolve_repo_input(repo_root, source_override or config["source"])
    _safe_source_name(source_path, config["allowed_source_names"])
    destinations = (outputs["manifest"], outputs["review_queue"], outputs["validation_report"])
    if not force and any(path.exists() for path in destinations):
        raise PrivateDataError("private preparation outputs already exist; pass --force to replace")
    if (
        source_path == regression_path
        or source_path in destinations
        or regression_path in destinations
    ):
        raise PrivateDataError("inputs and outputs must be distinct")
    key = load_or_create_hash_key(private_root)
    manifest, review_queue, report = prepare_private_dataset(
        source_path,
        regression_path,
        config,
        key,
    )
    if not report["valid"]:
        raise PrivateDataError("aggregate validation rejected the prepared dataset")
    _atomic_write_jsonl(outputs["manifest"], manifest)
    _atomic_write_jsonl(outputs["review_queue"], review_queue)
    _atomic_write_json(outputs["validation_report"], report)
    return report


def run_validate(
    repo_root: Path,
    config: Mapping[str, Any],
    manifest_override: str | Path | None = None,
    source_override: str | Path | None = None,
) -> dict[str, Any]:
    private_root, regression_path, outputs = _configured_paths(repo_root, config)
    require_private_ignore(repo_root, private_root)
    source_path = _resolve_repo_input(repo_root, source_override or config["source"])
    _safe_source_name(source_path, config["allowed_source_names"])
    manifest_path = (
        ensure_within(Path(manifest_override), private_root)
        if manifest_override is not None
        else outputs["manifest"]
    )
    records = read_jsonl(manifest_path)
    all_source_rows, source_fingerprint = read_source_rows(
        source_path,
        config["source_fields"],
    )
    regression_rows, regression_fingerprint = read_regression_rows(
        regression_path,
        int(config["expected_regression_rows"]),
    )
    near_config = config["near_relative"]
    threshold = float(near_config["threshold"])
    minimum_template_characters = int(near_config["minimum_template_characters"])
    regression_index = RegressionIndex(
        regression_rows,
        int(near_config["character_ngram_size"]),
    )
    eligible_candidates, excluded, source_boundary_counts = _filter_incoming_candidates(
        all_source_rows,
        regression_index,
        threshold,
        minimum_template_characters,
    )
    report = aggregate_validation_report(
        records,
        regression_index=regression_index,
        near_threshold=threshold,
        minimum_template_characters=minimum_template_characters,
    )

    additional_errors: Counter[str] = Counter()
    expected_source_ids = Counter(
        source_row["source_id"] for source_row, _ in eligible_candidates
    )
    manifest_source_ids = Counter(
        str(record["source_id"])
        for record in records
        if isinstance(record.get("source_id"), (str, int))
        and not isinstance(record.get("source_id"), bool)
    )
    additional_errors["eligible_source_rows_missing_from_manifest"] = sum(
        (expected_source_ids - manifest_source_ids).values()
    )
    additional_errors["ineligible_source_rows_present_in_manifest"] = sum(
        (manifest_source_ids - expected_source_ids).values()
    )

    source_provenance_mismatches = 0
    regression_provenance_mismatches = 0
    direction_filter_mismatches = 0
    for record in records:
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        if provenance.get("source_file_sha256") != source_fingerprint:
            source_provenance_mismatches += 1
        if provenance.get("regression_file_sha256") != regression_fingerprint:
            regression_provenance_mismatches += 1
        if provenance.get("source_direction_filter") != "is_from_me == false":
            direction_filter_mismatches += 1
    additional_errors["source_provenance_fingerprint_mismatch"] = (
        source_provenance_mismatches
    )
    additional_errors["regression_provenance_fingerprint_mismatch"] = (
        regression_provenance_mismatches
    )
    additional_errors["source_direction_provenance_mismatch"] = (
        direction_filter_mismatches
    )

    sender_component_lock_claimed = report["sender_split_diagnostics"][
        "sender_component_lock_claimed"
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        template_group = record.get("template_group")
        private_hashes = record.get("private_hashes")
        if (
            isinstance(template_group, str)
            and isinstance(private_hashes, Mapping)
            and isinstance(private_hashes.get("sender"), str)
            and bool(private_hashes.get("sender"))
        ):
            grouped[template_group].append(record)
    assignment_units = (
        _sender_template_components(grouped)
        if sender_component_lock_claimed
        else [(template_group,) for template_group in sorted(grouped)]
    )
    split_assignment_unit_counts: Counter[str] = Counter()
    for template_groups in assignment_units:
        unit_splits = {
            str(record["split"])
            for template_group in template_groups
            for record in grouped[template_group]
            if record.get("split") in SPLIT_NAMES
        }
        if len(unit_splits) != 1:
            additional_errors["split_assignment_unit_leakage"] += 1
        else:
            split_assignment_unit_counts[next(iter(unit_splits))] += 1

    source_immutable = file_sha256(source_path) == source_fingerprint
    regression_immutable = file_sha256(regression_path) == regression_fingerprint
    if not source_immutable:
        additional_errors["source_changed_after_read"] += 1
    if not regression_immutable:
        additional_errors["regression_changed_after_read"] += 1
    merged_errors = Counter(report["error_counts"])
    merged_errors.update(
        {name: count for name, count in additional_errors.items() if count}
    )
    report.update(
        {
            "preparer_version": PREPARER_VERSION,
            "valid": not merged_errors,
            "error_counts": dict(sorted(merged_errors.items())),
            "source_row_count": source_boundary_counts["total"],
            "source_boundary_counts": source_boundary_counts,
            "regression_row_count": len(regression_rows),
            "eligible_row_count": len(eligible_candidates),
            "excluded_row_count": sum(excluded.values()),
            "exclusions_by_reason": dict(sorted(excluded.items())),
            "split_assignment_unit": (
                "sender_template_component"
                if sender_component_lock_claimed
                else "template_family"
            ),
            "split_assignment_unit_count": len(assignment_units),
            "split_assignment_unit_counts": {
                split: split_assignment_unit_counts.get(split, 0)
                for split in SPLIT_NAMES
            },
            "sender_component_lock_applied": sender_component_lock_claimed,
            "source_immutable": source_immutable,
            "regression_immutable": regression_immutable,
        }
    )
    report["invariants"].update(
        {
            "source_export_mutated": not source_immutable,
            "regression_dataset_mutated": not regression_immutable,
            "source_boundary_count_mismatch": False,
            "incoming_preparation_count_mismatch": False,
            "manifest_membership_mismatch": bool(
                additional_errors["eligible_source_rows_missing_from_manifest"]
                or additional_errors["ineligible_source_rows_present_in_manifest"]
            ),
            "preparation_provenance_mismatch": bool(
                source_provenance_mismatches
                or regression_provenance_mismatches
                or direction_filter_mismatches
            ),
        }
    )
    _atomic_write_json(outputs["validation_report"], report)
    return report


def run_materialize(
    repo_root: Path,
    config: Mapping[str, Any],
    manifest_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    private_root, _, outputs = _configured_paths(repo_root, config)
    require_private_ignore(repo_root, private_root)
    manifest_path = (
        ensure_within(Path(manifest_override), private_root)
        if manifest_override is not None
        else outputs["manifest"]
    )
    output_dir = (
        ensure_within(Path(output_dir_override), private_root)
        if output_dir_override is not None
        else private_root
    )
    prefix = str(config["outputs"]["sft_prefix"])
    paths = {split: output_dir / f"{prefix}_{split}.jsonl" for split in SPLIT_NAMES}
    paths = {split: ensure_within(path, private_root) for split, path in paths.items()}
    report_path = outputs["materialization_report"]
    if not force and (report_path.exists() or any(path.exists() for path in paths.values())):
        raise PrivateDataError("SFT outputs already exist; pass --force to replace")

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    tier_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    records = read_jsonl(manifest_path)
    for record in sorted(records, key=lambda row: str(row.get("record_hash", ""))):
        split = record.get("split")
        if split not in SPLIT_NAMES:
            skipped_counts["invalid_split"] += 1
            continue
        accepted, label, tier = accepted_training_label(record, config["consensus_policy"])
        if not accepted:
            skipped_counts["not_accepted"] += 1
            continue
        if split == "test" and tier != "gold":
            skipped_counts["test_without_human_gold"] += 1
            continue
        rows_by_split[str(split)].append(build_sft_conversation(record, label))
        tier_counts[str(tier)] += 1

    for split, path in paths.items():
        _atomic_write_jsonl(path, rows_by_split[split])
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "input_row_count": len(records),
        "materialized_row_count": sum(len(rows) for rows in rows_by_split.values()),
        "split_row_counts": {split: len(rows_by_split[split]) for split in SPLIT_NAMES},
        "accepted_tier_counts": dict(sorted(tier_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "completion_only_compatible": True,
        "assistant_schema_fields": list(LABEL_FIELDS),
        "test_requires_human_gold": True,
    }
    _atomic_write_json(report_path, report)
    return report


def safe_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only report subset suitable for console output."""

    allowed = (
        "valid",
        "source_row_count",
        "source_boundary_counts",
        "regression_row_count",
        "eligible_row_count",
        "excluded_row_count",
        "exclusions_by_reason",
        "row_count",
        "split_row_counts",
        "split_group_counts",
        "split_assignment_unit_counts",
        "split_assignment_unit",
        "split_assignment_unit_count",
        "sender_component_lock_applied",
        "sender_split_diagnostics",
        "chronology_diagnostics",
        "review_status_counts",
        "hard_negative_counts",
        "label_tier_counts",
        "silver_label_counts",
        "error_counts",
        "invariants",
        "input_row_count",
        "materialized_row_count",
        "accepted_tier_counts",
        "skipped_counts",
    )
    return {field: report[field] for field in allowed if field in report}
