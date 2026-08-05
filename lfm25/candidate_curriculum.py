"""Policy-valid programmatic curriculum for semantic candidate IDs."""

from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Any, Mapping, Sequence

from lfm25.android_contract import prefilter_sms
from lfm25.candidates import extract_candidates, selector_target
from lfm25.private_data import canonical_exact_text, normalize_template


CURRICULUM_VERSION = "lfm25-semantic-candidate-curriculum-v2"
RELATION_PREFIXES = (
    "PA",
    "PT",
    "PB",
    "PF",
    "PV",
    "PL",
    "PW",
    "PU",
    "PR",
    "PO",
    "PN",
)


def _account(index: int) -> str:
    return f"A/c XX{(7000 + index) % 10000:04d}"


def _card(index: int) -> str:
    return f"Credit Card ending XX{(8000 + index) % 10000:04d}"


def _amount(index: int) -> float:
    return round(17.0 + (index * 37.19) % 7900, 2)


def _party(index: int) -> str:
    first = ("AURELIS", "BRIGHTON", "CIRRUS", "DOVETAIL", "EMBERLY", "FARADAY")[index % 6]
    second = ("MARKET", "SERVICES", "TRADERS", "FOODS", "WORKS", "STORES")[(index // 6) % 6]
    return f"{first} {second} {index:03d}"


def _row(
    *,
    index: int,
    family: str,
    sender: str,
    sms: str,
    expected: Any,
    expected_prefix: str | None,
    sample_weight: float,
) -> dict[str, Any]:
    if not prefilter_sms(sender, sms).accepted:
        raise ValueError(f"synthetic curriculum family {family} does not pass Android prefilter")
    candidates = extract_candidates(sms)
    target = selector_target(expected, candidates)
    if expected_prefix is not None:
        counterparty_id = str(target["counterparty"])
        if not counterparty_id.startswith(expected_prefix):
            raise ValueError(
                f"synthetic family {family} expected {expected_prefix}, got {counterparty_id}"
            )
    stable_id = hashlib.sha256(
        f"{CURRICULUM_VERSION}\0{family}\0{index}".encode("ascii")
    ).hexdigest()
    return {
        "sender": sender,
        "sms": sms,
        "expected": expected,
        "sample_weight": sample_weight,
        "confidence": 1.0,
        "label_tier": "programmatic_synthetic_curriculum",
        "source": {
            "dataset": "programmatic_semantic_candidate_curriculum",
            "record_hash": stable_id,
            "original_split": "synthetic",
            "template_group": f"synthetic:{family}:{index}",
            "private_sender_hash": f"synthetic:{sender}:{index}",
        },
        "provenance": {
            "curriculum_version": CURRICULUM_VERSION,
            "family": family,
            "programmatic": True,
            "private_derived": False,
            "sample_weight_basis": "low_weight_format_and_relation_coverage",
        },
    }


def generate_candidate_curriculum(
    *,
    seed: int = 35_025,
    rows_per_relation: int = 20,
    rows_per_negative: int = 10,
    sample_weight: float = 0.2,
) -> list[dict[str, Any]]:
    if rows_per_relation < 1 or rows_per_negative < 1:
        raise ValueError("curriculum row counts must be positive")
    if not 0.0 < sample_weight <= 1.0:
        raise ValueError("sample weight must be in (0, 1]")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    index = 0

    def transaction(family: str, prefix: str, render) -> None:
        nonlocal index
        for _ in range(rows_per_relation):
            amount = _amount(index)
            party = _party(index)
            account = _account(index)
            sender = f"AX-FX{index % 97:04d}"
            sms, label_account, label_party = render(index, amount, party, account)
            expected = {
                "amount": amount,
                "counterparty": label_party,
                "type": "credit" if "credited" in sms.casefold() else "debit",
                "account": label_account,
            }
            rows.append(
                _row(
                    index=index,
                    family=family,
                    sender=sender,
                    sms=sms,
                    expected=expected,
                    expected_prefix=prefix,
                    sample_weight=sample_weight,
                )
            )
            index += 1

    transaction(
        "at_merchant",
        "PA",
        lambda i, a, p, _acct: (
            f"INR {a:.2f} spent on your {_card(i)} at {p} on 12-JAN. Avl limit INR 90000.00.",
            _card(i),
            p,
        ),
    )
    transaction(
        "to_person",
        "PT",
        lambda i, a, p, acct: (
            f"Amt Sent Rs.{a:.2f} From Fixture Bank {acct} To {p} On 12-JAN Ref {910000+i}.",
            acct,
            p,
        ),
    )
    transaction(
        "by_person",
        "PB",
        lambda i, a, p, acct: (
            f"Rs.{a:.2f} credited to {acct} on 12-JAN by {p}, Ref {920000+i}.",
            acct,
            p,
        ),
    )
    transaction(
        "transfer_from",
        "PF",
        lambda i, a, p, acct: (
            f"INR {a:.2f} credited to {acct} on 12-JAN transfer from {p} Ref No {930000+i}.",
            acct,
            p,
        ),
    )
    transaction(
        "from_vpa",
        "PV",
        lambda i, a, _p, acct: (
            f"INR {a:.2f} credited to {acct} from VPA fixture{i:03d}@okdemo on 12-JAN.",
            acct,
            f"fixture{i:03d}@okdemo",
        ),
    )
    transaction(
        "linked_mobile",
        "PL",
        lambda i, a, _p, acct: (
            f"Rs.{a:.2f} credited to {acct} by a/c linked to mobile FIXTURE{i:05d} on 12-JAN.",
            acct,
            f"FIXTURE{i:05d}",
        ),
    )
    transaction(
        "towards_party",
        "PW",
        lambda i, a, p, acct: (
            f"Rs.{a:.2f} debited from {acct} towards {p} on 12-JAN Ref {940000+i}.",
            acct,
            p,
        ),
    )
    transaction(
        "for_upi",
        "PU",
        lambda i, a, p, acct: (
            f"{acct} debited for INR {a:.2f} on 12-JAN for UPI {p} to dispute call 1800000000.",
            acct,
            p,
        ),
    )
    transaction(
        "for_party",
        "PR",
        lambda i, a, p, acct: (
            f"{acct} debited for INR {a:.2f} on 12-JAN for {p} to dispute call 1800000000.",
            acct,
            p,
        ),
    )
    transaction(
        "on_party_with_distractor",
        "PO",
        lambda i, a, p, _acct: (
            f"INR {a:.2f} spent using Fixture Bank {_card(i)} on 12-JAN on {p} avl limit INR 90000.00.",
            _card(i),
            p,
        ),
    )
    transaction(
        "atm_withdrawal_no_counterparty",
        "PN",
        lambda i, a, _p, acct: (
            f"INR {a:.2f} withdrawn from {acct} on 12-JAN. Avl balance INR 50000.00.",
            acct,
            None,
        ),
    )

    negative_renderers = {
        "null_wallet": lambda i, a, acct: (
            f"INR {a:.2f} received in FixtureWallet linked with {acct}. Wallet balance updated."
        ),
        "null_failed": lambda i, a, _acct: (
            f"INR {a:.2f} was spent using {_card(i)} but the transaction was declined and not processed."
        ),
        "null_pending": lambda i, a, _acct: (
            f"INR {a:.2f} spent using {_card(i)} is pending and has not posted."
        ),
        "null_card_payment_notice": lambda i, a, _acct: (
            f"Payment of INR {a:.2f} received towards your {_card(i)}. Thank you."
        ),
        "null_mandate_info": lambda i, a, acct: (
            f"Auto-debit mandate for INR {a:.2f} on {acct} has been registered successfully."
        ),
        "null_balance": lambda i, a, acct: (
            f"Available balance in {acct} is INR {a:.2f}; last amount received is informational."
        ),
    }
    for family, render in negative_renderers.items():
        for _ in range(rows_per_negative):
            amount = _amount(index)
            account = _account(index)
            sender = f"AX-NX{index % 97:04d}"
            sms = render(index, amount, account)
            rows.append(
                _row(
                    index=index,
                    family=family,
                    sender=sender,
                    sms=sms,
                    expected=None,
                    expected_prefix=None,
                    sample_weight=sample_weight,
                )
            )
            index += 1
    rng.shuffle(rows)
    return rows


def audit_curriculum_overlap(
    curriculum_rows: Sequence[Mapping[str, Any]],
    reference_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return aggregate-only exact/template overlap evidence for curriculum SMS."""

    curriculum_messages = [
        str(row["sms"]) for row in curriculum_rows if isinstance(row.get("sms"), str)
    ]
    exact = {canonical_exact_text(message) for message in curriculum_messages}
    templates = {normalize_template(message) for message in curriculum_messages}
    by_reference: dict[str, Any] = {}
    total_exact: set[str] = set()
    total_templates: set[str] = set()
    for name, rows in sorted(reference_rows.items()):
        messages = [str(row["sms"]) for row in rows if isinstance(row.get("sms"), str)]
        reference_exact = {canonical_exact_text(message) for message in messages}
        reference_templates = {normalize_template(message) for message in messages}
        exact_overlap = exact & reference_exact
        template_overlap = templates & reference_templates
        total_exact.update(exact_overlap)
        total_templates.update(template_overlap)
        by_reference[name] = {
            "rows": len(rows),
            "sms_rows": len(messages),
            "exact_overlap_count": len(exact_overlap),
            "normalized_template_overlap_count": len(template_overlap),
        }
    return {
        "curriculum_rows": len(curriculum_rows),
        "curriculum_unique_exact": len(exact),
        "curriculum_unique_normalized_templates": len(templates),
        "exact_overlap_count": len(total_exact),
        "normalized_template_overlap_count": len(total_templates),
        "by_reference": by_reference,
    }


def mix_private_and_curriculum(
    private_rows: Sequence[Mapping[str, Any]],
    curriculum_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 35_025,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private = [dict(row) for row in private_rows]
    curriculum = [dict(row) for row in curriculum_rows]
    private_hashes = {str(row["source"]["record_hash"]) for row in private}
    curriculum_hashes = {str(row["source"]["record_hash"]) for row in curriculum}
    if len(private_hashes) != len(private) or len(curriculum_hashes) != len(curriculum):
        raise ValueError("duplicate record hashes in mixed candidate training sources")
    if private_hashes & curriculum_hashes:
        raise ValueError("private and curriculum record hashes overlap")
    rows = private + curriculum
    random.Random(seed).shuffle(rows)
    kinds = Counter("transaction" if row.get("expected") is not None else "null" for row in rows)
    tiers = Counter(str(row.get("label_tier")) for row in rows)
    return rows, {
        "private_rows": len(private),
        "curriculum_rows": len(curriculum),
        "mixed_rows": len(rows),
        "label_kind_counts": dict(sorted(kinds.items())),
        "label_tier_counts": dict(sorted(tiers.items())),
    }
