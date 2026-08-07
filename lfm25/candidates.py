"""Grounded candidate selection for the small-model extraction track.

The normal PocketFinancer contract asks the language model to copy arbitrary
strings and numbers into JSON.  That is an unnecessarily difficult job for a
350M parameter model: a one-token generation error can turn a correct decision
into a fabricated amount or merchant.  This module moves copying into
deterministic code.  The model selects IDs and ``reconstruct_transaction``
copies the corresponding source spans into the existing four-field contract.

This is an experimental contract and requires a small app-side pre/postprocessor;
it does not pretend to be wire-compatible with the current Android prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import re
from typing import Any, Iterable, Literal, Mapping

from lfm25.contract import (
    ParsedOutput,
    amount_matches,
    canonical_transaction,
    counterparty_matches,
    normalize_account,
    parse_gold,
)


SELECTOR_SYSTEM_PROMPT = """Classify one Indian financial SMS using only the candidates supplied by the user.
Return exactly one compact JSON object and no prose.
For a posted bank/card transaction use transaction=1, type D or C, and select one amount, account, and counterparty ID.
Counterparty ID prefixes describe their source cue: PA=at, PT=to, PB=by, PF=from, PV=VPA, PL=linked mobile, PW=towards, PU=UPI/NEFT, PR=for, PO=on, PN=none.
For everything else use {\"transaction\":0}. Never invent or copy a value; output candidate IDs only."""


_AMOUNT_RE = re.compile(
    r"(?ix)(?:₹|\brs\.?|\binr\b)\s*[:=\-]?\s*"
    r"(?P<number>[+]?(?:[0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)(?:\.[0-9]+)?)"
    r"(?![0-9A-Za-z_]|[.,][0-9A-Za-z_.,])",
    flags=re.ASCII,
)
_ACCOUNT_PATTERNS = (
    re.compile(
        r"(?aix)\b(?P<value>"
        r"(?:credit\s+card|debit\s+card|card)"
        r"(?:\s*(?:no\.?|number|ending(?:\s+in)?|ended(?:\s+in)?))?"
        r"\s*[:#\-.]?\s*(?:x{2,}|\*{1,}|[x*]*\d)[x*\d\-]{2,})\b"
    ),
    re.compile(
        r"(?aix)\b(?P<value>"
        r"(?:a\s*/?\s*c|acct|account)"
        r"(?:\s*(?:no\.?|number))?\s*[:#\-.]?\s*"
        r"(?:x{2,}|\*{1,}|[x*]*\d)[x*\d\-]{2,})\b"
    ),
)
_CURRENCY_AMOUNT_MARKER_RE = re.compile(
    r"(?ai)(?:\b(?:inr|rs)\.?\s*[:=\-]?\s*[0-9]|\u20b9\s*[0-9])"
)
_CP_STOP = (
    r"(?=\s+(?:at|to|by|from|towards|on|via|ref(?:erence)?(?:\s+no\.?)?|dated|"
    r"using|for|with|info|upi\s+ref|avl|avbl|available|balance|to\s+dispute|"
    r"on\s+application)\b|[,;.!()]|$)"
)
_COUNTERPARTY_PATTERNS = (
    (
        "PL",
        re.compile(
            r"(?aix)\bby\s+(?:a\s*/?\s*c\s+)?linked\s+to\s+(?:mobile|vpa)\s+"
            r"(?P<value>[a-z0-9][a-z0-9@._*+\-]{1,64})"
        ),
    ),
    (
        "PV",
        re.compile(
            r"(?aix)\b(?:from\s+vpa|linked\s+to\s+vpa)\s+"
            r"(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,72}?)" + _CP_STOP
        ),
    ),
    (
        "PA",
        re.compile(r"(?aix)\bat\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,72}?)" + _CP_STOP),
    ),
    (
        "PT",
        re.compile(r"(?aix)\bto\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,72}?)" + _CP_STOP),
    ),
    (
        "PB",
        re.compile(r"(?aix)\bby\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,72}?)" + _CP_STOP),
    ),
    (
        "PF",
        re.compile(
            r"(?aix)\b(?:transfer\s+from|from)\s+"
            r"(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,72}?)" + _CP_STOP
        ),
    ),
    (
        "PW",
        re.compile(r"(?aix)\btowards\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,96}?)" + _CP_STOP),
    ),
    (
        "PU",
        re.compile(
            r"(?aix)\bfor\s+(?:upi|neft(?:\s+cr)?)(?:\s+to)?\s+"
            r"(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,96}?)" + _CP_STOP
        ),
    ),
    (
        "PR",
        re.compile(r"(?aix)\bfor\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,96}?)" + _CP_STOP),
    ),
    (
        "PO",
        re.compile(r"(?aix)\bon\s+(?P<value>[a-z0-9][a-z0-9@._&*'\-/ ]{1,96}?)" + _CP_STOP),
    ),
)
_VPA_RE = re.compile(
    r"(?ai)(?<![\w@])(?P<value>[a-z0-9._-]{2,}@[a-z][a-z0-9.-]{1,})(?![\w@])"
)
_TYPE_DEBIT_RE = re.compile(
    r"\b(?:spent|debited|withdrawn|paid|sent|used|purchase(?:d)?|drawn)\b",
    re.IGNORECASE | re.ASCII,
)
_TYPE_CREDIT_RE = re.compile(
    r"\b(?:credited|received|refunded|deposited|reversal|reversed|added)\b",
    re.IGNORECASE | re.ASCII,
)
_SPACE_RE = re.compile(r"[ \t\n\r\f\v]+")
_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)

_SOURCE_DECIMAL_TOKEN_RE = re.compile(r"[+]?(?:[0-9]{1,3}(?:,[0-9]{2,3})+|[0-9]+)(?:\.[0-9]+)?\Z")
COUNTERPARTY_SOURCE_TIE_BREAK_ORDER = tuple(prefix for prefix, _pattern in _COUNTERPARTY_PATTERNS)
_COUNTERPARTY_SOURCE_TIE_BREAK_RANK = {
    prefix: index for index, prefix in enumerate(COUNTERPARTY_SOURCE_TIE_BREAK_ORDER)
}


@dataclass(frozen=True)
class Candidate:
    """A model-selectable value backed by an exact span in the current SMS."""

    id: str
    value: str | float | None
    source_text: str | None
    start: int | None
    end: int | None

    def public_dict(self) -> dict[str, Any]:
        """Return the prompt-safe form; offsets make audits reproducible."""

        return asdict(self)


@dataclass(frozen=True)
class CandidateSet:
    amounts: tuple[Candidate, ...]
    accounts: tuple[Candidate, ...]
    counterparties: tuple[Candidate, ...]
    type_hints: tuple[Literal["D", "C"], ...]

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "amounts": {item.id: item.value for item in self.amounts},
            "accounts": {item.id: item.value for item in self.accounts},
            "counterparties": {item.id: item.value for item in self.counterparties},
            "type_hints": list(self.type_hints),
        }


@dataclass(frozen=True)
class OracleSelection:
    covered: bool
    amount_id: str | None
    account_id: str | None
    counterparty_id: str | None
    type_code: str | None
    missing_fields: tuple[str, ...]


def _normal_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" \t\n\r\f\v")


def _ascii_lower(value: str) -> str:
    """Lowercase ASCII letters without host Unicode-version dependencies."""

    return value.translate(_ASCII_LOWER_TRANSLATION)


def canonical_amount_token(token: str) -> str:
    """Return a positive source decimal without grouping or a leading plus.

    The integer part is canonicalized while fractional precision is preserved.
    This representation is suitable for exact Candidate Protocol identity and
    deliberately performs no binary floating-point conversion.
    """

    if not isinstance(token, str) or not _SOURCE_DECIMAL_TOKEN_RE.fullmatch(token):
        raise ValueError("amount token is not a supported source decimal")
    ungrouped = token.replace(",", "")
    if ungrouped.startswith("+"):
        ungrouped = ungrouped[1:]
    try:
        value = Decimal(ungrouped)
    except InvalidOperation as error:
        raise ValueError("amount token is not a finite base-10 decimal") from error
    if not value.is_finite() or value <= 0:
        raise ValueError("amount token must be finite and positive")
    whole, dot, fraction = ungrouped.partition(".")
    canonical_whole = whole.lstrip("0") or "0"
    return canonical_whole + (dot + fraction if dot else "")


def _dedupe(candidates: Iterable[tuple[Any, str, int, int]], prefix: str) -> tuple[Candidate, ...]:
    seen: set[tuple[str, str]] = set()
    answer: list[Candidate] = []
    for value, source_text, start, end in sorted(candidates, key=lambda item: (item[2], item[3])):
        if isinstance(value, float):
            key = ("number", f"{value:.12g}")
        else:
            key = ("text", _ascii_lower(_normal_text(str(value))))
        if key in seen:
            continue
        seen.add(key)
        answer.append(Candidate(f"{prefix}{len(answer)}", value, source_text, start, end))
    return tuple(answer)


def _dedupe_semantic_counterparties(
    candidates: Iterable[tuple[str, str, int, int, str]],
) -> tuple[Candidate, ...]:
    seen: set[str] = set()
    counts: dict[str, int] = {}
    answer: list[Candidate] = []
    for value, source_text, start, end, prefix in candidates:
        key = _ascii_lower(_normal_text(value))
        if key in seen:
            continue
        seen.add(key)
        index = counts.get(prefix, 0)
        counts[prefix] = index + 1
        answer.append(Candidate(f"{prefix}{index}", value, source_text, start, end))
    return tuple(answer)


def _dedupe_exact_amounts(
    candidates: Iterable[tuple[str, str, int, int]],
) -> tuple[Candidate, ...]:
    """Deduplicate source decimals exactly and assign source-ordered IDs."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[2],
            item[3],
            Decimal(item[0]),
            item[0],
            item[1],
        ),
    )
    seen: set[Decimal] = set()
    answer: list[Candidate] = []
    for decimal_text, source_text, start, end in ordered:
        identity = Decimal(decimal_text)
        if identity in seen:
            continue
        seen.add(identity)
        answer.append(Candidate(f"A{len(answer)}", decimal_text, source_text, start, end))
    return tuple(answer)


def _dedupe_source_text_candidates(
    candidates: Iterable[tuple[str, str, int, int]],
    prefix: str,
) -> tuple[Candidate, ...]:
    """Deduplicate normalized text after deterministic source-span sorting."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[2],
            item[3],
            _ascii_lower(_normal_text(item[0])),
            _normal_text(item[0]),
            item[1],
        ),
    )
    seen: set[str] = set()
    answer: list[Candidate] = []
    for value, source_text, start, end in ordered:
        identity = _ascii_lower(_normal_text(value))
        if identity in seen:
            continue
        seen.add(identity)
        answer.append(Candidate(f"{prefix}{len(answer)}", value, source_text, start, end))
    return tuple(answer)


def _dedupe_source_counterparties(
    candidates: Iterable[tuple[str, str, int, int, str]],
) -> tuple[Candidate, ...]:
    """Deduplicate counterparties by earliest span with explicit tie-breaks."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            item[2],
            item[3],
            _COUNTERPARTY_SOURCE_TIE_BREAK_RANK.get(
                item[4], len(_COUNTERPARTY_SOURCE_TIE_BREAK_RANK)
            ),
            item[4],
            _ascii_lower(_normal_text(item[0])),
            _normal_text(item[0]),
            item[1],
        ),
    )
    seen: set[str] = set()
    counts: dict[str, int] = {}
    answer: list[Candidate] = []
    for value, source_text, start, end, prefix in ordered:
        identity = _ascii_lower(_normal_text(value))
        if identity in seen:
            continue
        seen.add(identity)
        index = counts.get(prefix, 0)
        counts[prefix] = index + 1
        answer.append(Candidate(f"{prefix}{index}", value, source_text, start, end))
    return tuple(answer)


def _raw_accounts(sms: str) -> list[tuple[str, str, int, int]]:
    raw_accounts: list[tuple[str, str, int, int]] = []
    for pattern in _ACCOUNT_PATTERNS:
        for match in pattern.finditer(sms):
            value = _normal_text(match.group("value"))
            raw_accounts.append(
                (value, match.group("value"), match.start("value"), match.end("value"))
            )
    return raw_accounts


def _raw_counterparties(sms: str) -> list[tuple[str, str, int, int, str]]:
    raw_counterparties: list[tuple[str, str, int, int, str]] = []
    for prefix, pattern in _COUNTERPARTY_PATTERNS:
        for match in pattern.finditer(sms):
            source_text = match.group("value")
            currency_marker = _CURRENCY_AMOUNT_MARKER_RE.search(source_text)
            if currency_marker is not None:
                source_text = source_text[: currency_marker.start()]
            source_text = source_text.rstrip(" -:,.\n\t")
            value = _normal_text(source_text).strip(" -:,.\n\t")
            if value and _ascii_lower(value) not in {
                "your account",
                "your a/c",
                "your card",
            }:
                start = match.start("value")
                raw_counterparties.append(
                    (value, source_text, start, start + len(source_text), prefix)
                )
    for match in _VPA_RE.finditer(sms):
        source_text = match.group("value").rstrip(" -:,.\n\t")
        if source_text:
            start = match.start("value")
            raw_counterparties.append(
                (source_text, source_text, start, start + len(source_text), "PV")
            )
    return raw_counterparties


def _type_hints(sms: str) -> tuple[Literal["D", "C"], ...]:
    hints: list[Literal["D", "C"]] = []
    if _TYPE_DEBIT_RE.search(sms):
        hints.append("D")
    if _TYPE_CREDIT_RE.search(sms):
        hints.append("C")
    return tuple(hints)


def extract_candidates(sms: str) -> CandidateSet:
    """Enumerate model-selectable source values without using a gold label."""

    if not isinstance(sms, str):
        raise TypeError("sms must be text")

    raw_amounts: list[tuple[float, str, int, int]] = []
    for match in _AMOUNT_RE.finditer(sms):
        number = match.group("number")
        try:
            amount = float(number.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(amount) and amount > 0:
            raw_amounts.append((amount, match.group(0), match.start(), match.end()))

    raw_accounts: list[tuple[str, str, int, int]] = []
    for pattern in _ACCOUNT_PATTERNS:
        for match in pattern.finditer(sms):
            value = _normal_text(match.group("value"))
            raw_accounts.append(
                (value, match.group("value"), match.start("value"), match.end("value"))
            )

    raw_counterparties: list[tuple[str, str, int, int, str]] = []
    for prefix, pattern in _COUNTERPARTY_PATTERNS:
        for match in pattern.finditer(sms):
            source_text = match.group("value")
            currency_marker = _CURRENCY_AMOUNT_MARKER_RE.search(source_text)
            if currency_marker is not None:
                source_text = source_text[: currency_marker.start()]
            source_text = source_text.rstrip(" -:,.\n\t")
            value = _normal_text(source_text).strip(" -:,.\n\t")
            if value and _ascii_lower(value) not in {"your account", "your a/c", "your card"}:
                start = match.start("value")
                raw_counterparties.append(
                    (
                        value,
                        source_text,
                        start,
                        start + len(source_text),
                        prefix,
                    )
                )
    for match in _VPA_RE.finditer(sms):
        value = match.group("value")
        raw_counterparties.append((value, value, match.start("value"), match.end("value"), "PV"))

    counterparties = list(_dedupe_semantic_counterparties(raw_counterparties))
    # PN is a real choice: the current synthetic curriculum omitted it entirely.
    counterparties.append(Candidate("PN", None, None, None, None))

    hints: list[Literal["D", "C"]] = []
    if _TYPE_DEBIT_RE.search(sms):
        hints.append("D")
    if _TYPE_CREDIT_RE.search(sms):
        hints.append("C")
    return CandidateSet(
        amounts=_dedupe(raw_amounts, "A"),
        accounts=_dedupe(raw_accounts, "C"),
        counterparties=tuple(counterparties),
        type_hints=tuple(hints),
    )


def extract_protocol_candidates(sms: str) -> CandidateSet:
    """Enumerate source-ordered V1 candidates without binary-float loss."""

    if not isinstance(sms, str):
        raise TypeError("sms must be text")

    raw_amounts: list[tuple[str, str, int, int]] = []
    for match in _AMOUNT_RE.finditer(sms):
        try:
            decimal_text = canonical_amount_token(match.group("number"))
        except ValueError:
            continue
        raw_amounts.append((decimal_text, match.group(0), match.start(), match.end()))

    counterparties = list(_dedupe_source_counterparties(_raw_counterparties(sms)))
    counterparties.append(Candidate("PN", None, None, None, None))
    return CandidateSet(
        amounts=_dedupe_exact_amounts(raw_amounts),
        accounts=_dedupe_source_text_candidates(_raw_accounts(sms), "C"),
        counterparties=tuple(counterparties),
        type_hints=_type_hints(sms),
    )


def candidate_selector_messages(sender: str, sms: str) -> list[dict[str, str]]:
    """Build the compact experimental selector prompt."""

    candidates = extract_candidates(sms)
    payload = json.dumps(candidates.prompt_payload(), ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sender: {sender}\nSMS: {sms}\nCandidates: {payload}\nOutput:",
        },
    ]


def oracle_selection(gold: Any, candidates: CandidateSet) -> OracleSelection:
    """Map a gold transaction to candidate IDs and expose candidate recall."""

    parsed = parse_gold(gold)
    if parsed is None:
        return OracleSelection(True, None, None, None, None, ())

    amount = next(
        (item.id for item in candidates.amounts if amount_matches(parsed["amount"], item.value)),
        None,
    )
    gold_account = normalize_account(parsed["account"])
    account = next(
        (
            item.id
            for item in candidates.accounts
            if gold_account is not None and normalize_account(item.value) == gold_account
        ),
        None,
    )
    matching_counterparties = [
        item
        for item in candidates.counterparties
        if counterparty_matches(parsed["counterparty"], item.value)
    ]
    if parsed["counterparty"] is None:
        counterparty = next(
            (item.id for item in matching_counterparties if item.value is None),
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
        counterparty = matching_counterparties[0].id if matching_counterparties else None
    type_code = "D" if parsed["type"] == "debit" else "C"
    missing = tuple(
        name
        for name, value in (
            ("amount", amount),
            ("account", account),
            ("counterparty", counterparty),
        )
        if value is None
    )
    return OracleSelection(not missing, amount, account, counterparty, type_code, missing)


def selector_target(gold: Any, candidates: CandidateSet) -> dict[str, Any]:
    """Return the supervised compact target, failing on uncovered transactions."""

    selection = oracle_selection(gold, candidates)
    if parse_gold(gold) is None:
        return {"transaction": 0}
    if not selection.covered:
        raise ValueError(
            f"gold fields absent from candidate set: {', '.join(selection.missing_fields)}"
        )
    return {
        "transaction": 1,
        "type": selection.type_code,
        "amount": selection.amount_id,
        "account": selection.account_id,
        "counterparty": selection.counterparty_id,
    }


def reconstruct_transaction(
    selection: Mapping[str, Any], candidates: CandidateSet
) -> dict[str, Any] | None:
    """Resolve selector JSON to PocketFinancer's existing four-field object."""

    decision = selection.get("transaction")
    if (
        set(selection) == {"transaction"}
        and isinstance(decision, int)
        and not isinstance(decision, bool)
        and decision == 0
    ):
        return None
    required = {"transaction", "type", "amount", "account", "counterparty"}
    if (
        set(selection) != required
        or not isinstance(decision, int)
        or isinstance(decision, bool)
        or decision != 1
    ):
        raise ValueError("selector output has the wrong schema")
    if selection["type"] not in {"D", "C"}:
        raise ValueError("selector type must be D or C")

    def choose(items: tuple[Candidate, ...], candidate_id: Any, field: str) -> Any:
        match = next((item for item in items if item.id == candidate_id), None)
        if match is None:
            raise ValueError(f"unknown {field} candidate ID")
        return match.value

    value = {
        "amount": choose(candidates.amounts, selection["amount"], "amount"),
        "counterparty": choose(
            candidates.counterparties, selection["counterparty"], "counterparty"
        ),
        "type": "debit" if selection["type"] == "D" else "credit",
        "account": choose(candidates.accounts, selection["account"], "account"),
    }
    canonical = canonical_transaction(value)
    if canonical is None:
        raise ValueError("resolved candidates violate the transaction contract")
    return canonical


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def counterparty_quality_penalty(candidate: Candidate) -> int:
    """Penalize source spans that look like context/distractors, not a party."""

    if candidate.value is None:
        return 4
    value = _ascii_lower(_normal_text(str(candidate.value)))
    penalty = 0
    penalty += 5 * bool(re.search(r"\bbank\b", value))
    penalty += 5 * bool(re.search(r"\b(?:inr|rs)\b|\u20b9", value))
    penalty += 4 * bool(re.search(r"\b(?:a\s*/?\s*c|account|card)\b", value))
    penalty += 3 * bool(re.match(r"\d", value))
    penalty += 2 * bool(
        re.search(
            r"\b\d{1,2}[-/]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2})",
            value,
        )
    )
    penalty += 2 * (len(value) > 40)
    return penalty


def _counterparty_has_currency_marker(candidate: Candidate) -> bool:
    if candidate.value is None:
        return False
    value = _ascii_lower(_normal_text(str(candidate.value)))
    return bool(_CURRENCY_AMOUNT_MARKER_RE.search(value))


def _hybrid_selection(
    selection: Mapping[str, Any],
    candidates: CandidateSet,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    prepared = dict(selection)
    interventions: list[str] = []
    decision = prepared.get("transaction")

    # A passed Android-filter row with a source VPA, one amount/account, and one
    # unambiguous direction is a narrow high-precision transaction fallback.
    if decision == 0:
        vpa = [item for item in candidates.counterparties if item.id.startswith("PV")]
        if (
            vpa
            and len(candidates.amounts) == 1
            and len(candidates.accounts) == 1
            and len(candidates.type_hints) == 1
        ):
            party = min(vpa, key=lambda item: (counterparty_quality_penalty(item), item.id))
            prepared = {
                "transaction": 1,
                "type": candidates.type_hints[0],
                "amount": candidates.amounts[0].id,
                "account": candidates.accounts[0].id,
                "counterparty": party.id,
            }
            interventions.append("vpa_unambiguous_transaction_fallback")
        return prepared, tuple(interventions)

    if decision == 1:
        selected_id = prepared.get("counterparty")
        selected = next(
            (item for item in candidates.counterparties if item.id == selected_id),
            None,
        )
        # Keep this guard deliberately narrow.  Oracle stress tests found that
        # generic "bad-looking span" replacement corrupts valid counterparties.
        # A currency-bearing counterparty span, however, was never a valid gold
        # party across the locked, private, and curriculum audits.
        if selected is not None and _counterparty_has_currency_marker(selected):
            alternatives = [
                item
                for item in candidates.counterparties
                if item.id != selected.id and not _counterparty_has_currency_marker(item)
            ]
            if alternatives:
                ranked = min(
                    alternatives,
                    key=lambda item: (counterparty_quality_penalty(item), item.id),
                )
                prepared["counterparty"] = ranked.id
                interventions.append("counterparty_currency_contamination_override")
    return prepared, tuple(interventions)


def resolve_selector_prediction(
    text: str,
    candidates: CandidateSet,
    *,
    hybrid_safety: bool = False,
) -> tuple[ParsedOutput, tuple[str, ...]]:
    """Parse, optionally apply grounded safety rules, and reconstruct output."""

    if not isinstance(text, str):
        return ParsedOutput("invalid", None, "", "prediction is not text"), ()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    candidate = _first_balanced_json_object(cleaned)
    if candidate is None:
        return ParsedOutput("invalid", None, cleaned, "no selector JSON object"), ()
    try:
        selection = json.loads(candidate)
    except json.JSONDecodeError:
        return ParsedOutput("invalid", None, candidate, "invalid selector JSON"), ()
    if not isinstance(selection, Mapping):
        return (
            ParsedOutput("invalid", None, candidate, "selector JSON is not an object"),
            (),
        )
    interventions: tuple[str, ...] = ()
    if hybrid_safety:
        selection, interventions = _hybrid_selection(selection, candidates)
    try:
        resolved = reconstruct_transaction(selection, candidates)
    except ValueError as error:
        return ParsedOutput("invalid", None, candidate, str(error)), interventions
    if resolved is None:
        return ParsedOutput("null", None, "null"), interventions
    rendered = json.dumps(resolved, ensure_ascii=False, separators=(",", ":"))
    return ParsedOutput("transaction", resolved, rendered), interventions


def parse_selector_prediction(text: str, candidates: CandidateSet) -> ParsedOutput:
    """Strict selector parsing without hybrid interventions."""

    return resolve_selector_prediction(text, candidates, hybrid_safety=False)[0]
