"""Generate and audit an unreleased, fully synthetic LFM2.5 SMS candidate.

Private texts may be read in memory by the audit functions. They are never returned,
logged, mapped, or copied into an output artifact.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import random
import re
import sqlite3
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
    from rapidfuzz import process as _rapidfuzz_process
except ImportError:  # pragma: no cover - exercised in dependency-minimal environments
    _rapidfuzz_fuzz = None
    _rapidfuzz_process = None


SCHEMA_VERSION = "lfm25-public-candidate-v1"
GENERATOR_NAME = "lfm25_fully_synthetic_sms_v1"
DEFAULT_SEED = 25_052_026
DEFAULT_ROW_COUNT = 120
DEFAULT_SIMILARITY_THRESHOLD = 0.72
DEFAULT_NGRAM_SIZE = 4
LABEL_FIELDS = ("amount", "counterparty", "type", "account")
ROW_FIELDS = {
    "public_id",
    "sender",
    "sms",
    "expected",
    "class",
    "template_family",
    "synthetic_provenance",
    "manual_review",
}
FORBIDDEN_LINKAGE_KEYS = {
    "source_id",
    "source_ids",
    "mapping",
    "mappings",
    "raw_hash",
    "raw_hashes",
    "linkage_id",
    "linkage_ids",
}
MANUAL_REVIEW_STATUS = "pending"


@dataclass(frozen=True)
class TemplateSpec:
    family: str
    class_name: str
    transaction_type: str | None
    channel: str


@dataclass(frozen=True)
class AuditBundle:
    """Safe aggregate audit results and accepted candidate rows."""

    accepted_rows: tuple[dict[str, Any], ...]
    report: dict[str, Any]
    memorization_probe_manifest: dict[str, Any]


TRANSACTION_TEMPLATES = (
    TemplateSpec("upi_sent", "upi_debit", "debit", "upi"),
    TemplateSpec("upi_received", "upi_credit", "credit", "upi"),
    TemplateSpec("card_purchase", "card_debit", "debit", "card"),
    TemplateSpec("atm_cash", "cash_withdrawal", "debit", "account"),
    TemplateSpec("merchant_refund", "refund_credit", "credit", "card"),
    TemplateSpec("utility_bill", "bill_payment_debit", "debit", "account"),
    TemplateSpec("scheduled_debit", "standing_instruction_debit", "debit", "account"),
    TemplateSpec("cash_deposit", "cash_deposit_credit", "credit", "account"),
    TemplateSpec("wallet_load", "wallet_load_debit", "debit", "card"),
    TemplateSpec("merchant_reversal", "reversal_credit", "credit", "card"),
)
HARD_NEGATIVE_TEMPLATES = (
    TemplateSpec("otp_only", "hard_negative_otp", None, "none"),
    TemplateSpec("spending_limit", "hard_negative_limit", None, "none"),
    TemplateSpec("promotional_offer", "hard_negative_promotion", None, "none"),
    TemplateSpec("bill_due", "hard_negative_due_notice", None, "none"),
    TemplateSpec("failed_payment", "hard_negative_failed_payment", None, "none"),
    TemplateSpec("security_notice", "hard_negative_security", None, "none"),
    TemplateSpec("delivery_update", "hard_negative_delivery", None, "none"),
    TemplateSpec("profile_update", "hard_negative_profile", None, "none"),
)
ALL_TEMPLATES = TRANSACTION_TEMPLATES + HARD_NEGATIVE_TEMPLATES

BANK_IDENTITIES = (
    ("Dakshin Cooperative Bank", "VK-DSYNBK"),
    ("Narmada Digital Bank", "JM-NDSYNB"),
    ("Sahyadri Test Bank", "AD-SHTSTB"),
    ("Konkan Sample Bank", "CP-KNSYNB"),
    ("Malabar Demo Bank", "TM-MLDMBK"),
)
COUNTERPARTIES = (
    "Mango Leaf Grocers",
    "Riverstone Books",
    "Copper Cup Cafe",
    "Blue Kite Mobility",
    "Monsoon Test Utilities",
    "Lotus Pixel Services",
    "Cedar Basket Market",
    "Paper Boat Supplies",
    "Sunbird Demo Pharmacy",
    "Coconut Grove Telecom",
    "Amber Trail Insurance",
    "Mint Lane Fashions",
)
SYNTHETIC_LOCATIONS = (
    "Test Nagar",
    "Sample Junction",
    "Demo Bazaar",
    "Mock Colony",
)
SENDER_IDS = tuple(item[1] for item in BANK_IDENTITIES) + (
    "VM-SYNUPI",
    "BP-SYNCARD",
    "AX-DEMOPAY",
    "JD-TSTALRT",
)
SYNTHETIC_PROVENANCE = {
    "kind": "fully_programmatic_synthesis",
    "generator": GENERATOR_NAME,
    "inputs": "curated_fictional_token_sets_only",
}


def _format_indian_amount(amount: float) -> str:
    whole, fraction = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        tail = whole[-3:]
        head = whole[:-3]
        groups: list[str] = []
        while head:
            groups.append(head[-2:])
            head = head[:-2]
        whole = ",".join(reversed(groups)) + "," + tail
    return f"{whole}.{fraction}"


def _new_public_id(rng: random.Random) -> str:
    """Return a deterministic UUID4-shaped ID unrelated to record contents."""

    value = rng.getrandbits(128)
    value &= ~(0xF << 76)
    value |= 0x4 << 76
    value &= ~(0x3 << 62)
    value |= 0x2 << 62
    return f"LFM25-PUB-{uuid.UUID(int=value)}"


def _amount_for(index: int, rng: random.Random) -> float:
    rupees = 10 + ((index * 977 + rng.randrange(10, 80_000)) % 79_990)
    paise = rng.choice((0, 0, 0, 25, 40, 50, 75, 90, 99))
    return float(f"{rupees}.{paise:02d}")


def _date_for(index: int, rng: random.Random) -> str:
    year = 2032 + (index % 5)
    month = 1 + ((index * 5 + rng.randrange(12)) % 12)
    day = 1 + ((index * 7 + rng.randrange(28)) % 28)
    return f"{day:02d}-{month:02d}-{year}"


def _reference_for(index: int, rng: random.Random) -> str:
    value = (index * 1_000_003 + rng.randrange(1_000_000_000_000)) % 1_000_000_000_000
    return f"SYNREF-{value:012d}"


def _render_transaction(
    spec: TemplateSpec,
    *,
    bank: str,
    amount_text: str,
    account: str,
    counterparty: str,
    date: str,
    reference: str,
    location: str,
    vpa: str,
) -> str:
    renderers = {
        "upi_sent": (
            "{bank} synthetic alert: INR {amount} sent from {account} to "
            "{counterparty} ({vpa}) on {date}. Ref {reference}."
        ),
        "upi_received": (
            "Credit simulation from {bank}: {account} received INR {amount} from "
            "{counterparty} via {vpa} on {date}. Reference {reference}."
        ),
        "card_purchase": (
            "Test purchase notice | {account} | INR {amount} | {counterparty} | "
            "{location} | {date} | {reference}."
        ),
        "atm_cash": (
            "Synthetic ATM event: INR {amount} withdrawn from {account} at "
            "Demo ATM, {location}, on {date}. Ref {reference}."
        ),
        "merchant_refund": (
            "Refund simulation: {counterparty} credited INR {amount} to "
            "{account} on {date}; tracking ref {reference}."
        ),
        "utility_bill": (
            "{bank}: INR {amount} debited from {account} for {counterparty} "
            "test bill on {date}. Ref {reference}."
        ),
        "scheduled_debit": (
            "Scheduled synthetic payment completed: {account} debited INR {amount} "
            "for {counterparty} on {date}. Mandate ref {reference}."
        ),
        "cash_deposit": (
            "Deposit simulation: INR {amount} credited to {account} at "
            "{bank}, {location}, on {date}. Ref {reference}."
        ),
        "wallet_load": (
            "Demo wallet load: INR {amount} charged to {account} for "
            "{counterparty} on {date}. Ref {reference}."
        ),
        "merchant_reversal": (
            "Reversal test notice: INR {amount} restored to {account} by "
            "{counterparty} on {date}. Ref {reference}."
        ),
    }
    return renderers[spec.family].format(
        bank=bank,
        amount=amount_text,
        account=account,
        counterparty=counterparty,
        date=date,
        reference=reference,
        location=location,
        vpa=vpa,
    )


def _render_hard_negative(
    spec: TemplateSpec,
    *,
    bank: str,
    amount_text: str,
    account: str,
    date: str,
    reference: str,
    location: str,
    otp: str,
    url_token: str,
) -> str:
    renderers = {
        "otp_only": (
            "Synthetic verification code {otp} authorizes a possible INR {amount} "
            "request on {account}; no payment has occurred. Never share this code."
        ),
        "spending_limit": (
            "{bank} test reminder: the monthly limit on {account} is INR {amount}. "
            "This is not a debit or credit notification."
        ),
        "promotional_offer": (
            "Demo offer: save INR {amount} on an eligible future purchase before "
            "{date}. Details: https://notice.example/{url_token}"
        ),
        "bill_due": (
            "Synthetic bill reminder: INR {amount} may be due for {account} on "
            "{date}. No payment is recorded by this message."
        ),
        "failed_payment": (
            "Test-only status: the attempted INR {amount} payment from {account} "
            "failed. No debit was completed. Case {reference}."
        ),
        "security_notice": (
            "{bank} synthetic security notice for {account}: review demo settings at "
            "https://notice.example/{url_token} or call +91 00000 00000."
        ),
        "delivery_update": (
            "Sample delivery is planned for {location} on {date}. Tracking "
            "{reference}. This message contains no financial transaction."
        ),
        "profile_update": (
            "Test profile update for {account} is complete. Confirmation sent to "
            "alerts+{otp}@notify.example; no funds moved."
        ),
    }
    return renderers[spec.family].format(
        bank=bank,
        amount=amount_text,
        account=account,
        date=date,
        reference=reference,
        location=location,
        otp=otp,
        url_token=url_token,
    )


def generate_candidate_rows(
    count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate deterministic rows using fictional, generator-owned values only."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    suffixes = list(range(9000, 10_000))
    rng.shuffle(suffixes)
    rows: list[dict[str, Any]] = []

    for index in range(count):
        spec = ALL_TEMPLATES[index % len(ALL_TEMPLATES)]
        bank, bank_sender = BANK_IDENTITIES[index % len(BANK_IDENTITIES)]
        sender = bank_sender if index % 3 else SENDER_IDS[index % len(SENDER_IDS)]
        suffix = suffixes[index % len(suffixes)]
        instrument = "Card" if spec.channel == "card" else "A/c"
        account = f"{instrument} XX{suffix}"
        party_offset = (index * 5 + rng.randrange(len(COUNTERPARTIES))) % len(COUNTERPARTIES)
        counterparty = COUNTERPARTIES[party_offset]
        location = SYNTHETIC_LOCATIONS[index % len(SYNTHETIC_LOCATIONS)]
        amount = _amount_for(index, rng)
        amount_text = _format_indian_amount(amount)
        date = _date_for(index, rng)
        reference = _reference_for(index, rng)
        vpa_stem = re.sub(r"[^a-z]", "", counterparty.lower())[:16]
        vpa = f"{vpa_stem}.{(index * 17 + 11) % 100:02d}@synthupi"
        otp = f"{100000 + ((index * 7919 + rng.randrange(900000)) % 900000):06d}"
        url_token = f"SYN-{index + 1:05d}-{rng.randrange(1000, 10000)}"

        if spec.transaction_type is None:
            sms = _render_hard_negative(
                spec,
                bank=bank,
                amount_text=amount_text,
                account=account,
                date=date,
                reference=reference,
                location=location,
                otp=otp,
                url_token=url_token,
            )
            label = None
        else:
            sms = _render_transaction(
                spec,
                bank=bank,
                amount_text=amount_text,
                account=account,
                counterparty=counterparty,
                date=date,
                reference=reference,
                location=location,
                vpa=vpa,
            )
            label = {
                "amount": amount,
                "counterparty": counterparty,
                "type": spec.transaction_type,
                "account": account,
            }

        rows.append(
            {
                "public_id": _new_public_id(rng),
                "sender": sender,
                "sms": sms,
                "expected": json.dumps(label, ensure_ascii=False, sort_keys=True),
                "class": spec.class_name,
                "template_family": spec.family,
                "synthetic_provenance": dict(SYNTHETIC_PROVENANCE),
                "manual_review": MANUAL_REVIEW_STATUS,
            }
        )
    return rows


def _nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _nested_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_keys(child)


_INVALID_LABEL = object()


def _parse_label(row: Mapping[str, Any]) -> dict[str, Any] | None | object:
    expected = row.get("expected")
    if not isinstance(expected, str):
        return _INVALID_LABEL
    try:
        label = json.loads(expected)
    except (TypeError, json.JSONDecodeError):
        return _INVALID_LABEL
    return label if label is None or isinstance(label, dict) else _INVALID_LABEL


def validate_candidate_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return machine-safe error codes without echoing row content."""

    errors: list[str] = []
    if set(row) != ROW_FIELDS:
        errors.append("row_fields")
    if any(key.lower() in FORBIDDEN_LINKAGE_KEYS for key in _nested_keys(row)):
        errors.append("forbidden_linkage_key")
    public_id = row.get("public_id")
    if not isinstance(public_id, str) or not public_id.startswith("LFM25-PUB-"):
        errors.append("public_id")
    if not isinstance(row.get("sender"), str) or not row.get("sender"):
        errors.append("sender")
    if not isinstance(row.get("sms"), str) or not row.get("sms"):
        errors.append("sms")
    if row.get("manual_review") != MANUAL_REVIEW_STATUS:
        errors.append("manual_review")
    if row.get("synthetic_provenance") != SYNTHETIC_PROVENANCE:
        errors.append("synthetic_provenance")

    label = _parse_label(row)
    if label is _INVALID_LABEL:
        errors.append("label_fields")
        return tuple(dict.fromkeys(errors))

    is_hard_negative = str(row.get("class", "")).startswith("hard_negative_")
    if is_hard_negative:
        if label is not None:
            errors.append("hard_negative_label")
        return tuple(dict.fromkeys(errors))
    if not isinstance(label, dict) or set(label) != set(LABEL_FIELDS):
        errors.append("label_fields")
        return tuple(dict.fromkeys(errors))
    amount = label["amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
        errors.append("amount")
    if label["type"] not in {"debit", "credit"}:
        errors.append("type")
    if not isinstance(label["counterparty"], str) or not label["counterparty"]:
        errors.append("counterparty")
    if not isinstance(label["account"], str) or not label["account"]:
        errors.append("account")
    return tuple(dict.fromkeys(errors))


_SCAN_PATTERNS: dict[str, re.Pattern[str]] = {
    "pii": re.compile(
        r"\b(?:[A-Z]{5}[0-9]{4}[A-Z]|[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4})\b",
        re.IGNORECASE,
    ),
    "quasi_id": re.compile(r"\bSYNQID-[A-Z0-9-]+\b", re.IGNORECASE),
    "url": re.compile(r"https?://[^\s]+|\bwww\.[^\s]+", re.IGNORECASE),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "vpa": re.compile(
        r"\b[A-Z0-9._-]{2,}@[A-Z][A-Z0-9_-]{2,}\b(?!\.)",
        re.IGNORECASE,
    ),
    "phone": re.compile(r"(?<![0-9])(?:\+91[ -]?)?[0-9]{5}[ -]?[0-9]{5}(?![0-9])"),
    "reference": re.compile(r"\bSYNREF-[0-9]{12}\b", re.IGNORECASE),
    "account": re.compile(
        r"\b(?:A/c|account|card)[ :#-]*(?:ending[ ]*)?(?:XX|X{2,})?[0-9]{4,}\b",
        re.IGNORECASE,
    ),
    "date": re.compile(r"\b(?:0?[1-9]|[12][0-9]|3[01])[-/](?:0?[1-9]|1[0-2])[-/](?:20)?[0-9]{2}\b"),
    "location": re.compile(
        r"\b(?:Test Nagar|Sample Junction|Demo Bazaar|Mock Colony|Mumbai|Delhi|"
        r"Bengaluru|Bangalore|Chennai|Kolkata|Hyderabad|Pune)\b",
        re.IGNORECASE,
    ),
    "secret": re.compile(
        r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|"
        r"\b(?:api[_ -]?key|secret|bearer)[ :=]+[A-Z0-9_./+-]{12,})",
        re.IGNORECASE,
    ),
}


def _is_owned_synthetic_row(row: Mapping[str, Any]) -> bool:
    return row.get("synthetic_provenance") == SYNTHETIC_PROVENANCE


def _is_safe_owned_token(category: str, token: str, row: Mapping[str, Any]) -> bool:
    """Allow only narrow generator-owned reserved formats."""

    if not _is_owned_synthetic_row(row):
        return False
    lowered = token.lower().rstrip(".,;:)")
    digits = re.sub(r"\D", "", token)
    if category == "pii":
        return len(digits) == 12 and f"SYNREF-{digits}" in str(row.get("sms", ""))
    if category == "url":
        return lowered.startswith("https://notice.example/")
    if category == "email":
        return lowered.endswith("@notify.example")
    if category == "vpa":
        return lowered.endswith("@synthupi")
    if category == "phone":
        return digits in {"0000000000", "910000000000"}
    if category == "reference":
        return bool(re.fullmatch(r"SYNREF-[0-9]{12}", token, re.IGNORECASE))
    if category == "account":
        suffix_match = re.search(r"([0-9]{4})\b", token)
        return bool(suffix_match and suffix_match.group(1).startswith("9"))
    if category == "date":
        year_match = re.search(r"(?:20)?([0-9]{2})$", token)
        return bool(year_match and 32 <= int(year_match.group(1)) <= 36)
    if category == "location":
        return lowered in {location.lower() for location in SYNTHETIC_LOCATIONS}
    if category == "quasi_id":
        return lowered.startswith("synqid-")
    return False


def scan_sensitive_tokens(row: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Return counts only; matched token values never leave this function."""

    sms = row.get("sms")
    if not isinstance(sms, str):
        sms = ""
    results: dict[str, dict[str, int]] = {}
    for category, pattern in _SCAN_PATTERNS.items():
        safe = 0
        blocked = 0
        for match in pattern.finditer(sms):
            if _is_safe_owned_token(category, match.group(0), row):
                safe += 1
            else:
                blocked += 1
        results[category] = {"safe_synthetic": safe, "blocked": blocked}
    return results


def _normalize_for_comparison(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _word_ngrams(normalized: str, size: int) -> set[tuple[str, ...]]:
    words = normalized.split()
    if not words:
        return set()
    actual_size = min(size, len(words))
    return {
        tuple(words[index : index + actual_size]) for index in range(len(words) - actual_size + 1)
    }


def _prepare_private_texts(
    private_texts: Sequence[str],
    ngram_size: int,
) -> tuple[tuple[str, frozenset[tuple[str, ...]]], ...]:
    """Normalize and deduplicate private texts once without persisting an index."""

    normalized_texts = dict.fromkeys(
        normalized for text in private_texts if (normalized := _normalize_for_comparison(text))
    )
    return tuple(
        (normalized, frozenset(_word_ngrams(normalized, ngram_size)))
        for normalized in normalized_texts
    )


def _maximum_private_similarity(
    text: str,
    prepared_private_texts: Sequence[tuple[str, frozenset[tuple[str, ...]]]],
    ngram_size: int,
) -> tuple[float, float, bool]:
    maximum_ngram = 0.0
    maximum_near_duplicate = 0.0
    exact_match = False
    normalized = _normalize_for_comparison(text)
    if not normalized:
        return maximum_ngram, maximum_near_duplicate, exact_match
    candidate_ngrams = _word_ngrams(normalized, ngram_size)
    normalized_choices: list[str] = []
    for private_normalized, private_ngrams in prepared_private_texts:
        normalized_choices.append(private_normalized)
        exact_match = exact_match or normalized == private_normalized
        intersection_size = len(candidate_ngrams.intersection(private_ngrams))
        union_size = len(candidate_ngrams) + len(private_ngrams) - intersection_size
        ngram_score = intersection_size / union_size if union_size else 0.0
        maximum_ngram = max(maximum_ngram, ngram_score)

    if _rapidfuzz_process is not None and _rapidfuzz_fuzz is not None:
        closest = _rapidfuzz_process.extractOne(
            normalized,
            normalized_choices,
            scorer=_rapidfuzz_fuzz.ratio,
            processor=None,
        )
        if closest is not None:
            maximum_near_duplicate = float(closest[1]) / 100.0
        return maximum_ngram, maximum_near_duplicate, exact_match

    for private_normalized in normalized_choices:
        matcher = SequenceMatcher(
            None,
            normalized,
            private_normalized,
            autojunk=False,
        )
        if matcher.real_quick_ratio() <= maximum_near_duplicate:
            continue
        if matcher.quick_ratio() <= maximum_near_duplicate:
            continue
        maximum_near_duplicate = max(maximum_near_duplicate, matcher.ratio())
    return maximum_ngram, maximum_near_duplicate, exact_match


def _rewrite_for_private_similarity(row: Mapping[str, Any], attempt: int) -> dict[str, Any]:
    """Produce a label-preserving rewrite without consulting private text."""

    rewritten = dict(row)
    label = _parse_label(row)
    if label is _INVALID_LABEL:
        return rewritten
    public_token = str(row.get("public_id", ""))[-8:]
    if str(row.get("class", "")).startswith("hard_negative_"):
        rewritten["sms"] = (
            f"Synthetic non-transaction service note {public_token}-{attempt}: no completed "
            "debit or credit occurred, and there is no transaction to extract."
        )
    else:
        if not isinstance(label, dict):
            return rewritten
        rewritten["sms"] = (
            f"Synthetic ledger event {public_token}-{attempt}: direction={label['type']}; "
            f"value=INR {_format_indian_amount(float(label['amount']))}; "
            f"instrument={label['account']}; party={label['counterparty']}. "
            "This generated notice describes no real person or account."
        )
    rewritten["template_family"] = f"similarity_rewrite_{attempt}"
    return rewritten


def _empty_privacy_counts() -> dict[str, dict[str, int]]:
    return {category: {"safe_synthetic": 0, "blocked": 0} for category in _SCAN_PATTERNS}


def _merge_privacy_counts(
    aggregate: dict[str, dict[str, int]],
    findings: Mapping[str, Mapping[str, int]],
) -> None:
    for category, counts in findings.items():
        aggregate[category]["safe_synthetic"] += counts["safe_synthetic"]
        aggregate[category]["blocked"] += counts["blocked"]


def audit_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    private_texts: Sequence[str],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    rewrite_attempts: int = 2,
) -> AuditBundle:
    """Audit rows while retaining aggregate private-comparison results only."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    if rewrite_attempts < 0:
        raise ValueError("rewrite_attempts must be non-negative")
    if any(not isinstance(text, str) for text in private_texts):
        raise TypeError("private_texts must contain strings")

    prepared_private_texts = _prepare_private_texts(private_texts, ngram_size)
    rejection_counts: Counter[str] = Counter()
    schema_error_counts: Counter[str] = Counter()
    privacy_counts = _empty_privacy_counts()
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    rewritten_rows = 0
    exact_private_matches = 0
    threshold_hits = 0
    maximum_ngram_seen = 0.0
    maximum_near_duplicate_seen = 0.0
    maximum_accepted_score = 0.0

    for original_row in rows:
        row = dict(original_row)
        schema_errors = validate_candidate_row(row)
        if schema_errors:
            rejection_counts["schema_or_label"] += 1
            schema_error_counts.update(schema_errors)
            continue

        public_id = str(row["public_id"])
        normalized_text = _normalize_for_comparison(str(row["sms"]))
        if public_id in seen_ids:
            rejection_counts["duplicate_public_id"] += 1
            continue
        if normalized_text in seen_texts:
            rejection_counts["duplicate_sms"] += 1
            continue

        ngram_score, near_duplicate_score, exact_match = _maximum_private_similarity(
            str(row["sms"]),
            prepared_private_texts,
            ngram_size,
        )
        maximum_ngram_seen = max(maximum_ngram_seen, ngram_score)
        maximum_near_duplicate_seen = max(maximum_near_duplicate_seen, near_duplicate_score)
        if exact_match:
            exact_private_matches += 1

        attempts_used = 0
        score = max(ngram_score, near_duplicate_score)
        while (exact_match or score >= similarity_threshold) and attempts_used < rewrite_attempts:
            threshold_hits += 1
            attempts_used += 1
            row = _rewrite_for_private_similarity(row, attempts_used)
            ngram_score, near_duplicate_score, exact_match = _maximum_private_similarity(
                str(row["sms"]),
                prepared_private_texts,
                ngram_size,
            )
            maximum_ngram_seen = max(maximum_ngram_seen, ngram_score)
            maximum_near_duplicate_seen = max(maximum_near_duplicate_seen, near_duplicate_score)
            score = max(ngram_score, near_duplicate_score)

        if attempts_used:
            rewritten_rows += 1
        if exact_match or score >= similarity_threshold:
            rejection_counts["private_similarity"] += 1
            continue

        findings = scan_sensitive_tokens(row)
        _merge_privacy_counts(privacy_counts, findings)
        if any(counts["blocked"] for counts in findings.values()):
            rejection_counts["sensitive_token"] += 1
            continue

        normalized_text = _normalize_for_comparison(str(row["sms"]))
        if normalized_text in seen_texts:
            rejection_counts["duplicate_after_rewrite"] += 1
            continue

        seen_ids.add(public_id)
        seen_texts.add(normalized_text)
        maximum_accepted_score = max(maximum_accepted_score, score)
        accepted.append(row)

    classes = Counter(str(row["class"]) for row in accepted)
    templates = Counter(str(row["template_family"]) for row in accepted)
    senders = Counter(str(row["sender"]) for row in accepted)
    pending_count = sum(row.get("manual_review") == MANUAL_REVIEW_STATUS for row in accepted)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_state": "unreleased",
        "input_candidate_rows": len(rows),
        "accepted_candidate_rows": len(accepted),
        "rejected_candidate_rows": len(rows) - len(accepted),
        "rejected_counts": dict(sorted(rejection_counts.items())),
        "schema_and_label_audit": {
            "required_label_fields": list(LABEL_FIELDS),
            "error_counts": dict(sorted(schema_error_counts.items())),
            "accepted_rows_valid": all(not validate_candidate_row(row) for row in accepted),
        },
        "duplicate_audit": {
            "unique_public_ids": len({row["public_id"] for row in accepted}),
            "unique_normalized_sms": len(
                {_normalize_for_comparison(str(row["sms"])) for row in accepted}
            ),
        },
        "private_similarity_audit": {
            "private_documents_compared_in_memory": len(private_texts),
            "unique_normalized_documents_compared": len(prepared_private_texts),
            "private_text_or_hash_persisted": False,
            "ngram_size": ngram_size,
            "near_duplicate_engine": (
                "rapidfuzz_ratio" if _rapidfuzz_process is not None else "difflib_sequence_matcher"
            ),
            "similarity_threshold": similarity_threshold,
            "maximum_ngram_jaccard_observed": round(maximum_ngram_seen, 6),
            "maximum_near_duplicate_ratio_observed": round(
                maximum_near_duplicate_seen,
                6,
            ),
            "maximum_accepted_similarity": round(maximum_accepted_score, 6),
            "exact_matches_before_rewrite": exact_private_matches,
            "threshold_events": threshold_hits,
            "rows_rewritten": rewritten_rows,
            "rows_rejected_after_rewrite": rejection_counts["private_similarity"],
        },
        "sensitive_data_scan": privacy_counts,
        "coverage": {
            "classes": dict(sorted(classes.items())),
            "template_families": dict(sorted(templates.items())),
            "senders": dict(sorted(senders.items())),
        },
        "review": {
            "required_status": MANUAL_REVIEW_STATUS,
            "pending_rows": pending_count,
            "other_status_rows": len(accepted) - pending_count,
            "human_review_completed": False,
        },
        "release_decision": "not_made",
        "license_decision": "not_made",
    }
    memorization_manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate_state": "unreleased_manual_review_pending",
        "contains_private_text": False,
        "contains_private_hashes": False,
        "data_probes": [
            {
                "name": "normalized_exact_match",
                "status": "executed",
                "result_count": exact_private_matches,
                "retention": "aggregate_count_only",
            },
            {
                "name": "word_ngram_jaccard",
                "status": "executed",
                "ngram_size": ngram_size,
                "threshold": similarity_threshold,
                "retention": "aggregate_scores_only",
            },
            {
                "name": "near_duplicate_sequence_ratio",
                "status": "executed",
                "threshold": similarity_threshold,
                "retention": "aggregate_scores_only",
            },
        ],
        "model_memorization_probes": [
            {
                "name": "verbatim_completion_probe",
                "status": "not_run",
                "required_before_release": True,
            },
            {
                "name": "rare_ngram_completion_probe",
                "status": "not_run",
                "required_before_release": True,
            },
            {
                "name": "membership_inference_review",
                "status": "not_run",
                "required_before_release": True,
            },
        ],
        "release_decision": "not_made",
    }
    return AuditBundle(tuple(accepted), report, memorization_manifest)


def read_candidate_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read candidate JSONL with errors that never echo row text."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def load_private_texts(paths: Sequence[Path], text_field: str = "sms") -> list[str]:
    """Read private JSONL texts into memory without copying metadata or logging values."""

    texts: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid private JSON at {path}:{line_number}") from error
                if not isinstance(value, dict) or not isinstance(value.get(text_field), str):
                    raise ValueError(f"missing private text field at {path}:{line_number}")
                texts.append(value[text_field])
    return texts


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _validated_export_text(
    row: Mapping[str, Any],
    *,
    path: Path,
    row_number: int,
    text_field: str,
) -> str:
    if text_field not in row:
        raise ValueError(f"private export row {row_number} is missing its text field: {path}")
    value = row[text_field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"private export row {row_number} has invalid text: {path}")
    return value


def load_private_texts_export(
    paths: Sequence[Path],
    *,
    text_field: str = "text",
) -> list[str]:
    """Read every text from JSON-array/CSV exports without retaining metadata.

    Failures identify only the source and row position. Private values are never
    placed in exceptions, logs, hashes returned to callers, or output artifacts.
    Each source is hashed before and after its read so concurrent mutation cannot
    silently produce a partial or internally inconsistent audit corpus.
    """

    if not isinstance(text_field, str) or not text_field:
        raise ValueError("private export text field must be a non-empty string")

    texts: list[str] = []
    for path in paths:
        resolved = path.resolve(strict=True)
        before = _file_digest(resolved)
        source_texts: list[str] = []
        suffix = resolved.suffix.casefold()
        try:
            if suffix == ".json":
                with resolved.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, list):
                    raise ValueError(f"private export JSON must be a row array: {path}")
                for row_number, row in enumerate(payload, start=1):
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"private export row {row_number} is not an object: {path}"
                        )
                    source_texts.append(
                        _validated_export_text(
                            row,
                            path=path,
                            row_number=row_number,
                            text_field=text_field,
                        )
                    )
            elif suffix == ".csv":
                with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    try:
                        header = next(reader)
                    except StopIteration as error:
                        raise ValueError(f"private export CSV has no header: {path}") from error
                    matching_columns = [
                        index for index, name in enumerate(header) if name == text_field
                    ]
                    if len(matching_columns) != 1:
                        raise ValueError(
                            f"private export CSV must contain one configured text field: {path}"
                        )
                    text_index = matching_columns[0]
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(
                                f"private export CSV row {row_number} is malformed: {path}"
                            )
                        value = row[text_index]
                        if not value.strip():
                            raise ValueError(
                                f"private export row {row_number} has invalid text: {path}"
                            )
                        source_texts.append(value)
            else:
                raise ValueError("private export must be a JSON or CSV file")
        except (csv.Error, json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"private export could not be parsed: {path}") from error

        after = _file_digest(resolved)
        if not hmac.compare_digest(before, after):
            raise RuntimeError(f"private export changed while it was read: {path}")
        texts.extend(source_texts)
    return texts


def load_private_texts_sqlite(
    paths: Sequence[Path],
    *,
    table: str = "message",
    text_column: str = "text",
) -> list[str]:
    """Read only one explicitly named text column from immutable SQLite archives."""

    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    if not identifier.fullmatch(table) or not identifier.fullmatch(text_column):
        raise ValueError("SQLite table and text column must be simple identifiers")

    texts: list[str] = []
    query = (
        f'SELECT "{text_column}" FROM "{table}" '
        f'WHERE "{text_column}" IS NOT NULL AND trim("{text_column}") <> \'\''
    )
    for path in paths:
        resolved = path.resolve(strict=True)
        uri = f"{resolved.as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
            try:
                cursor = connection.execute(query)
            except sqlite3.DatabaseError as error:
                raise ValueError(
                    f"private SQLite source lacks {table}.{text_column}: {path}"
                ) from error
            for (value,) in cursor:
                if not isinstance(value, str):
                    raise ValueError(
                        f"private SQLite text column contains a non-string value: {path}"
                    )
                texts.append(value)
        finally:
            connection.close()

    return texts


def load_private_text_sources(
    *,
    jsonl_paths: Sequence[Path] = (),
    export_paths: Sequence[Path] = (),
    sqlite_paths: Sequence[Path] = (),
    jsonl_text_field: str = "sms",
    export_text_field: str = "text",
    sqlite_table: str = "message",
    sqlite_text_column: str = "text",
) -> list[str]:
    """Aggregate compatible private sources while retaining text values only."""

    texts = load_private_texts(jsonl_paths, text_field=jsonl_text_field)
    texts.extend(load_private_texts_export(export_paths, text_field=export_text_field))
    texts.extend(
        load_private_texts_sqlite(
            sqlite_paths,
            table=sqlite_table,
            text_column=sqlite_text_column,
        )
    )
    return texts


def ensure_ignored_output_tree(repo_root: Path, output_dir: Path) -> None:
    """Fail closed unless output is inside the ignored public-candidate tree."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    required_root = (repo_root / "PUBLIC_CANDIDATE" / "lfm25").resolve()
    try:
        output_dir.relative_to(required_root)
    except ValueError as error:
        raise RuntimeError("output must stay under PUBLIC_CANDIDATE/lfm25") from error

    probe = output_dir / "candidate.jsonl"
    relative_probe = probe.relative_to(repo_root)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            relative_probe.as_posix(),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError("PUBLIC_CANDIDATE/lfm25 is not protected by .gitignore")


def _atomic_write_text(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_generation_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    output_dir: Path,
    seed: int,
    force: bool = False,
) -> tuple[Path, ...]:
    """Write only the unreviewed synthetic pool and a non-linkage manifest."""

    ensure_ignored_output_tree(repo_root, output_dir)
    candidate_path = output_dir / "candidate_unreviewed.jsonl"
    manifest_path = output_dir / "generation_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "deterministic_seed": seed,
        "row_count": len(rows),
        "provenance": SYNTHETIC_PROVENANCE,
        "private_records_used_for_generation": False,
        "contains_source_ids_or_linkage": False,
        "manual_review": MANUAL_REVIEW_STATUS,
        "candidate_state": "unreleased",
        "release_decision": "not_made",
        "license_decision": "not_made",
    }
    _atomic_write_text(candidate_path, _jsonl_text(rows), force=force)
    _atomic_write_text(manifest_path, _json_text(manifest), force=force)
    return candidate_path, manifest_path


def _render_coverage_table(title: str, values: Mapping[str, Any]) -> str:
    if not isinstance(values, Mapping):
        raise ValueError(f"dataset-card {title.casefold()} coverage must be a mapping")

    rows: list[str] = []
    for raw_name, raw_count in sorted(values.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str):
            raise ValueError(f"dataset-card {title.casefold()} name must be a string")
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
            raise ValueError(
                f"dataset-card {title.casefold()} row count must be a non-negative integer"
            )
        safe_name = raw_name.replace("`", "'").replace("|", r"\|").replace("\n", " ")
        rows.append(f"| `{safe_name}` | {raw_count} |")
    if not rows:
        rows.append("| _(none)_ | 0 |")
    return "\n".join(
        [
            f"### {title}",
            "",
            "| Name | Accepted rows |",
            "| --- | ---: |",
            *rows,
        ]
    )


def render_dataset_card(report: Mapping[str, Any]) -> str:
    accepted = int(report["accepted_candidate_rows"])
    rejected = int(report["rejected_candidate_rows"])
    coverage = report["coverage"]
    if not isinstance(coverage, Mapping):
        raise ValueError("dataset-card coverage must be a mapping")
    class_table = _render_coverage_table("Classes", coverage["classes"])
    template_table = _render_coverage_table("Template families", coverage["template_families"])
    return f"""# LFM2.5 synthetic SMS public candidate

Status: **unreleased; manual review pending**

This local candidate contains {accepted} fully programmatic synthetic rows. The audit
rejected {rejected} rows. It was generated from curated fictional token sets; private
messages were not generation inputs. Private texts were used only for in-memory
similarity checks, and neither their text nor hashes were retained.

## Row contract

Each row has a fresh public ID, fictional sender, synthetic SMS, class and template
family, synthetic provenance, and `manual_review=pending`. `expected` is a JSON string
containing literal `null` or exactly `amount`, `counterparty`, `type`, and `account`.

## Safety gates

The audit checks schema and labels, exact duplicates, normalized exact matches,
{report["private_similarity_audit"]["ngram_size"]}-gram Jaccard similarity,
near-duplicate sequence similarity, sensitive tokens, coverage, and review status.
Rows at or above the configured private-similarity threshold are rewritten without
consulting private text and rechecked, then rejected if they still exceed it.

## Detailed audited coverage

These counts come directly from accepted rows in the audit report. They describe this
generated candidate's composition, not the prevalence of classes in real SMS traffic.

{class_table}

{template_table}

## Intended use

Until manual review is complete, use is limited to local engineering inspection and
privacy auditing. If all release gates are later cleared, the candidate may support
local research on the four-field extraction contract, parser behavior, prompt/model
comparison, and synthetic hard-negative handling. It is not production validation.

## Prohibited uses

- Do not deploy these rows or models trained on them to make financial decisions,
  initiate transactions, contact customers, or automate actions on real accounts.
- Do not use them for identity verification, credit or insurance eligibility, fraud
  adjudication, employment, law enforcement, or any other high-impact decision.
- Do not present synthetic messages, labels, institutions, accounts, or transactions
  as authentic records or use them to attempt reconstruction or re-identification of
  the private archive.
- Do not claim production accuracy, demographic fairness, multilingual coverage, or
  population representativeness from results on this candidate.
- Do not publish, redistribute, sublicense, or externally upload the candidate before
  manual, privacy, memorization, data-rights, trademark, and license reviews are
  complete and an explicit release decision has been recorded.

## Known biases and limitations

- Rows are deterministic templates populated from small curated fictional token sets;
  wording and class balance therefore reflect generator design rather than real use.
- Coverage is English-language and India-focused (for example INR, UPI, and local bank
  alert conventions). It does not adequately cover Indian-language or code-switched
  text, other regions, accessibility needs, spelling noise, or evolving scam patterns.
- The transaction taxonomy and hard-negative families are narrow. Unlisted message
  types, ambiguous events, multi-transaction messages, and adversarial inputs remain
  underrepresented.
- Regex token scans and exact, n-gram, and near-duplicate checks have blind spots. A
  passing aggregate audit does not prove absence of personal data, semantic similarity,
  re-identification risk, or model memorization.
- All accepted rows still require manual review. Synthetic-only results cannot establish
  real-world accuracy, robustness, safety, calibration, or demographic fairness.

## Release and rights

No release decision and no license decision have been made. The accompanying rights
review note is a checklist, not a legal conclusion. A qualified reviewer must complete
manual content, data-rights, trademark, privacy, and model-memorization review before
any publication or redistribution.
"""


def render_license_data_rights_note(report: Mapping[str, Any]) -> str:
    return f"""# License and data-rights review note

Status: **open review; no legal conclusion**

- Candidate state: unreleased.
- Accepted synthetic rows awaiting manual review: {report["accepted_candidate_rows"]}.
- Generation basis: curated fictional token sets and deterministic templates only.
- Private material: used locally for in-memory similarity comparison only; no private
  text, identifiers, mappings, or hashes are persisted by the audit.
- Source archive handling: read-only; the tools neither mutate nor copy it.
- License selection: not made.
- Release approval: not made.

Before any release, a qualified reviewer should assess ownership of generator code,
third-party marks and confusing similarity, privacy and re-identification risk,
applicable contractual restrictions, jurisdiction-specific database rights, the
memorization-probe results, and the proposed distribution license. This note records
engineering facts and open questions only; it does not determine rights or grant a
license.
"""


def write_audit_artifacts(
    bundle: AuditBundle,
    *,
    repo_root: Path,
    output_dir: Path,
    preview_count: int = 12,
    force: bool = False,
) -> tuple[Path, ...]:
    """Write audited candidate artifacts; no rejected/private text is serialized."""

    if preview_count < 0:
        raise ValueError("preview_count must be non-negative")
    ensure_ignored_output_tree(repo_root, output_dir)
    paths_and_content = (
        (output_dir / "candidate.jsonl", _jsonl_text(bundle.accepted_rows)),
        (
            output_dir / "safe_preview.jsonl",
            _jsonl_text(bundle.accepted_rows[:preview_count]),
        ),
        (output_dir / "audit_report.json", _json_text(bundle.report)),
        (
            output_dir / "memorization_probe_manifest.json",
            _json_text(bundle.memorization_probe_manifest),
        ),
        (output_dir / "dataset_card.md", render_dataset_card(bundle.report)),
        (
            output_dir / "license_data_rights_review.md",
            render_license_data_rights_note(bundle.report),
        ),
    )
    for path, content in paths_and_content:
        _atomic_write_text(path, content, force=force)
    return tuple(path for path, _ in paths_and_content)
