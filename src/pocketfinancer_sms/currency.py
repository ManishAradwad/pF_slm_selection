"""Currency configuration and exact-money parsing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .types import CurrencyProvenance


ISO_MINOR_UNITS: dict[str, int] = {
    "AED": 2,
    "AUD": 2,
    "CAD": 2,
    "CHF": 2,
    "EUR": 2,
    "GBP": 2,
    "INR": 2,
    "JPY": 0,
    "SGD": 2,
    "USD": 2,
}


@dataclass(frozen=True, slots=True)
class CurrencyContext:
    primary_currency: str
    profile_ids: tuple[str, ...] = ("core-en",)

    def __post_init__(self) -> None:
        code = self.primary_currency.upper()
        if code not in ISO_MINOR_UNITS:
            raise ValueError("primary currency is not in the supported ISO-4217 table")
        object.__setattr__(self, "primary_currency", code)

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(
            {"primary_currency": self.primary_currency, "profile_ids": self.profile_ids},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedMoney:
    minor_units: int
    currency: str
    provenance: CurrencyProvenance


def parse_money(
    number_text: str,
    *,
    currency: str,
    provenance: CurrencyProvenance,
) -> ParsedMoney:
    currency = currency.upper()
    if currency not in ISO_MINOR_UNITS:
        raise ValueError("amount currency is unsupported")
    normalized = number_text.replace(",", "").replace(" ", "")
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError("amount is not valid decimal money") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError("amount must be finite and greater than zero")
    scale = ISO_MINOR_UNITS[currency]
    quantum = Decimal(1).scaleb(-scale)
    if value.quantize(quantum) != value:
        raise ValueError("amount has more precision than its currency permits")
    minor_units = int(value * (10**scale))
    return ParsedMoney(minor_units, currency, provenance)
