"""Privacy-bound exact, template, sender-template, and time grouping."""

from __future__ import annotations

import hashlib
import hmac
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime


_SPACE = re.compile(r"\s+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL_VPA = re.compile(r"\b[a-z0-9][a-z0-9._%+-]*@[a-z0-9][a-z0-9.-]*\b", re.I)
_CURRENCY_AMOUNT = re.compile(
    r"(?:\b[A-Z]{3}\b|₹|\$|€|£|¥|\brs\.?)\s*[:.-]?\s*\d[\d,]*(?:\.\d+)?",
    re.I,
)
_DATE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?)\b"
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.I)
_MASKED = re.compile(r"(?:[xX*•-]{2,}\s*)?\d{3,}")
_NUMBER = re.compile(r"\d+")
_SENDER_PREFIX = re.compile(r"^[A-Z]{2}-", re.I)


@dataclass(frozen=True, slots=True)
class Grouping:
    exact_body_hash: str
    normalized_template_hash: str
    sender_hash: str
    sender_family_hash: str
    sender_template_group_hash: str
    time_group: str


def canonical_exact_body(body: str) -> str:
    normalized = unicodedata.normalize("NFKC", body)
    return _SPACE.sub(" ", normalized).strip().casefold()


def normalized_template(body: str) -> str:
    value = canonical_exact_body(body)
    value = _URL.sub(" <URL> ", value)
    value = _EMAIL_VPA.sub(" <ADDRESS> ", value)
    value = _CURRENCY_AMOUNT.sub(" <AMOUNT> ", value)
    value = _DATE.sub(" <DATE> ", value)
    value = _TIME.sub(" <TIME> ", value)
    value = _MASKED.sub(" <IDENTIFIER> ", value)
    value = _NUMBER.sub(" <NUMBER> ", value)
    return _SPACE.sub(" ", value).strip()


def sender_family(sender: str) -> str:
    value = unicodedata.normalize("NFKC", sender).strip().casefold()
    value = _SENDER_PREFIX.sub("", value)
    value = _NUMBER.sub("#", value)
    return value


def build_grouping(key: bytes, body: str, sender: str, timestamp: str) -> Grouping:
    exact = canonical_exact_body(body)
    template = normalized_template(body)
    family = sender_family(sender)
    return Grouping(
        exact_body_hash=_private_hash(key, "exact-body", exact),
        normalized_template_hash=_private_hash(key, "template", template),
        sender_hash=_private_hash(key, "sender", sender),
        sender_family_hash=_private_hash(key, "sender-family", family),
        sender_template_group_hash=_private_hash(key, "sender-template", f"{family}\0{template}"),
        time_group=_time_group(timestamp),
    )


def template_hash(key: bytes, body: str) -> str:
    return _private_hash(key, "template", normalized_template(body))


def _private_hash(key: bytes, namespace: str, value: str) -> str:
    return hmac.new(key, f"{namespace}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()


def _time_group(timestamp: str) -> str:
    normalized = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return "unknown"
    return f"{parsed.year:04d}-{parsed.month:02d}"
