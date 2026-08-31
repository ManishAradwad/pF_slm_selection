"""Locale/profile declarations used by the shared analyzer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalyzerProfile:
    profile_id: str
    explicit_markers: dict[str, tuple[str, ...]]
    transaction_terms: tuple[str, ...]
    rails: dict[str, tuple[str, ...]]


CORE_EN = AnalyzerProfile(
    profile_id="core-en",
    explicit_markers={
        "USD": ("$",),
        "EUR": ("€",),
        "GBP": ("£",),
        "JPY": ("¥",),
    },
    transaction_terms=(
        "transaction",
        "payment",
        "purchase",
        "transfer",
        "debit",
        "credit",
        "paid",
        "spent",
        "received",
        "withdrawn",
        "deposited",
        "refunded",
    ),
    rails={},
)


INDIA = AnalyzerProfile(
    profile_id="india",
    explicit_markers={"INR": ("₹", "rs", "rs.", "inr")},
    transaction_terms=CORE_EN.transaction_terms
    + ("a/c", "acct", "account", "card", "upi", "imps", "neft", "rtgs", "nach"),
    rails={
        "upi": ("upi", "vpa"),
        "imps": ("imps",),
        "neft": ("neft",),
        "rtgs": ("rtgs",),
        "nach": ("nach", "e-mandate", "emandate"),
    },
)


PROFILES = {profile.profile_id: profile for profile in (CORE_EN, INDIA)}


def resolve_profiles(profile_ids: tuple[str, ...]) -> tuple[AnalyzerProfile, ...]:
    try:
        return tuple(PROFILES[profile_id] for profile_id in profile_ids)
    except KeyError as exc:
        raise ValueError(f"unknown analyzer profile: {exc.args[0]}") from exc
