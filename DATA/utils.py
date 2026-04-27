"""
Utilities for the sms_extraction evaluation task.
Provides: prompt formatting, output filtering, and custom metrics.
"""

import json
import re
import sys


# ── Prompt ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are extracting transaction details from an Indian bank/card SMS. Output either a single JSON object or the literal word null. Nothing else — no prose, no markdown fences, no explanations.

Field nullability: in a real bank/card transaction SMS, amount, type, and account are always present — banks always state them. merchant may be null when the SMS has no counterparty (e.g. SBI UPI, ATM withdrawal). date may be null when the SMS has no date, or only day-month with no year.

STEP 1 — Is this a real bank/card transaction?
Qualifying: money actually moved in or out of the user's bank account (savings or current), credit card, or debit card RIGHT NOW.
Not qualifying: mobile wallets (PayTM wallet, PhonePe wallet, Simpl, slice, or any similar wallet or BNPL service) — output null even if money genuinely moved inside a wallet.

Check the sender ID first — Indian commercial SMS use the format XX-YYYYYY (2-char operator prefix, dash, 6-char entity code). The entity code is the key signal:
- A real bank sender's entity code visibly resembles the bank's name or acronym (e.g. HDFCBK, ICICIB, SBIINB, AXISBK, KOTAKB, YESBNK, FEDBNK, INDUSB — and similarly shaped codes for banks not listed here).
- If the entity code reads like a brand, wallet provider, marketing word, travel/e-commerce/telecom name, or app name, the message is almost certainly not a bank/card transaction.
- A purely numeric entity code (e.g. a 6-digit number) is almost always promotional, not a bank transaction alert.

Then check the content — output null for:
- Any wallet or BNPL message (PayTM wallet, PhonePe, Simpl, slice, etc.) — real transaction or promotional
- Recharge offers, plan promos, "Recharge with Rs.X" messages
- Cashback ads, discount offers, "Get Rs.X off" messages
- Referral bonuses, spin/game rewards
- Balance alerts, mini-statements, account statements, investment summaries
- Account-opening ads, loan offers, credit card marketing, insurance promos
- Booking confirmations with "pay after" / "pay later" language
- OTP, verification, or security alert messages
- Any message whose main purpose is to get the user to click a link or take an action, even if it mentions an amount

If the sender looks like a real bank AND the content confirms money moved in/out of a bank account or card right now → go to STEP 2. Otherwise → null.

STEP 2 — Extract fields from the CURRENT SMS only
Never copy values from the few-shot examples below. Every field must come from the SMS you are given right now.

- amount: number from the SMS (e.g. 150.0). Parse "Rs.1,500" as 1500.0.
- type: use the verb in the SMS.
    debit  ← spent, debited, withdrawn, paid, sent, used, purchase, drawn
    credit ← credited, received, refunded, deposited, reversal, added
- merchant: who the user paid or got paid by. Look for these phrases:
    "at <MERCHANT>"              → e.g. "at MIDAS DAILY"
    "To <NAME>"                  → e.g. "To AJAY KUMAR YADAV"
    "by <NAME/COMPANY>"          → e.g. "by MULTIPL FINTECH SOLUTIONS"
    "from VPA <handle>"          → e.g. "from VPA user@okhdfc"
    "linked to VPA <handle>"     → the VPA is the merchant
    "by a/c linked to mobile X"  → the mobile/APIBANKING ID is the merchant
  This is NEVER the bank name. If the SMS genuinely has no counterparty, use null — but try hard first.
- date: always DD-MM-YYYY with numeric month.
    "01Mar24"      → "01-03-2024"
    "2024-03-01"   → "01-03-2024"
    "10-FEB-2024"  → "10-02-2024"   (JAN=01, FEB=02, ..., DEC=12)
  If the SMS has no date, use null. If the SMS only has day-month with no year anywhere, also use null. Never invent a placeholder date.
- account: the masked account or card label exactly as written in the SMS. Keep any "A/c", "Card", "Credit Card", "Debit Card" prefix. Do NOT use the bank name. Examples: "A/c XX6254", "Credit Card XX3782", "Debit Card xx4955"."""

FEW_SHOT_EXAMPLES = [
    # --- Transactions ---
    {
        "sender": "AX-HDFCBK",
        "sms": "HDFC Bank: Rs.500.00 credited to a/c XXXXXX0000 on 01-01-20 by a/c linked to VPA demouser000@examplebank (UPI Ref No 000000000000).",
        "answer": '{"amount": 500.0, "merchant": "demouser000@examplebank", "date": "01-01-2020", "type": "credit", "account": "a/c XXXXXX0000"}',
    },
    {
        "sender": "VM-HDFCBK",
        "sms": "Amt Sent Rs.75.00\nFrom HDFC Bank A/C *0000\nTo EXAMPLE DEMO USER\nOn 02-01\nRef 000000000000\nNot You? Call 18002586161",
        "answer": '{"amount": 75.0, "merchant": "EXAMPLE DEMO USER", "date": null, "type": "debit", "account": "A/C *0000"}',
    },
    {
        "sender": "VD-IDFCFB",
        "sms": "Transaction Successful! INR 23.00 spent on your IDFC FIRST Bank Credit Card ending XX0000 at DEMO SHOP DAILY on 03-JAN-2020 at 07:01 PM. Avbl limit: Rs.10000.00.",
        "answer": '{"amount": 23.0, "merchant": "DEMO SHOP DAILY", "date": "03-01-2020", "type": "debit", "account": "Credit Card ending XX0000"}',
    },
    # --- Rejections ---
    {
        "sender": "JK-000000",
        "sms": "Hurry! Recharge your number on ExampleApp & get rewards upto Rs.400. Recharge with Rs.239 plan. T&CA. Click https://example.app/xx",
        "answer": "null",
    },
    {
        "sender": "VM-OFFERZ",
        "sms": "CRICKET MANIA OFFER! Get Rs.75 OFF on Rs.3099 recharge! Enjoy 2GB/Day + more. 365 Days https://example.link/promo",
        "answer": "null",
    },
    {
        "sender": "VK-GOIBIB",
        "sms": "Dear Customer, we have reserved a cab for your pickup from Mumbai Airport starting @ just Rs.515. Pay after trip. Show OTP at kiosk. https://example.bo/abc",
        "answer": "null",
    },
]


def doc_to_text(doc: dict) -> str:
    """Build the full prompt with system instruction, few-shot examples, and query.

    Demos are wrapped in an explicit `### EXAMPLES` block and the actual query
    in `### YOUR TASK`. Without that delimiter, thinking-style models (Qwen3)
    treat the few-shot demos as in-flight chat turns and end up answering one
    of those instead of the real query — observed in practice on Qwen3-1.7B,
    where it produced the first few-shot answer verbatim for unrelated SMS.
    The delimiter is a domain-level fix (any model benefits), not a per-model
    workaround.
    """
    parts = [SYSTEM_PROMPT, ""]

    parts.append("### EXAMPLES (already labeled — for reference only, do NOT answer these)")
    parts.append("")
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Sender: {ex['sender']}")
        parts.append(f"SMS: {ex['sms']}")
        parts.append(f"Output: {ex['answer']}")
        parts.append("")

    parts.append("### YOUR TASK (answer this one SMS only — output JSON or null, nothing else)")
    parts.append("")
    parts.append(f"Sender: {doc['sender']}")
    parts.append(f"SMS: {doc['sms']}")
    parts.append("Output: ")

    return "\n".join(parts)


# ── Filters ─────────────────────────────────────────────────────────────────────

def extract_json_filter(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Extract the first JSON object or 'null' from each model response.
    Strips <think>...</think> blocks produced by chain-of-thought models (e.g. Qwen3)
    before searching for the answer."""
    results = []
    for resp_list in resps:
        filtered = []
        for resp in resp_list:
            resp = resp.strip()
            # Strip chain-of-thought thinking blocks (Qwen3 and similar models)
            resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
            # Check for null first
            if resp.lower().startswith("null"):
                filtered.append("null")
                continue
            # Try to find a JSON object
            match = re.search(r"\{.*?\}", resp, re.DOTALL)
            if match:
                filtered.append(match.group(0))
            else:
                filtered.append(resp)
        results.append(filtered)
    return results


# Matches the grammar: amount/type/account must always be non-null in a real
# txn; merchant and date may be null (merchant is absent in SBI UPI, ATM,
# and gateway SMS; date is resolved from SMS arrival timestamp on-device).
_REQUIRED_NONNULL_FIELDS = ("amount", "type", "account")


def _is_null_ish(v) -> bool:
    """True if v is JSON null or the literal 4-char string 'null'
    (grammar-constrained outputs can't emit real null on non-nullable fields
    and fall back to the string)."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() == "null":
        return True
    return False


def extract_json_nonnull_filter(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Same as extract_json_filter, but rejects any dict whose required fields
    (see _REQUIRED_NONNULL_FIELDS) contain null — the model's uncertainty signal."""
    results = extract_json_filter(resps, docs)
    for i, resp_list in enumerate(results):
        for j, resp in enumerate(resp_list):
            if resp.strip().lower() == "null":
                continue
            try:
                obj = json.loads(resp)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            if any(_is_null_ish(obj.get(f)) for f in _REQUIRED_NONNULL_FIELDS):
                results[i][j] = "null"
    return results


# ── Metrics ─────────────────────────────────────────────────────────────────────

def _parse(text: str):
    """Parse a JSON string or 'null' into a Python object."""
    text = text.strip()
    if text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return "PARSE_ERROR"


def json_validity(references: list[str], predictions: list[str]) -> float:
    """1.0 if the prediction is valid JSON or the literal 'null', else 0.0."""
    pred = predictions[0].strip()
    if pred.lower() == "null":
        return 1.0
    try:
        json.loads(pred)
        return 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0


def rejection_accuracy(references: list[str], predictions: list[str]) -> float:
    """1.0 if both ref and pred agree on null/non-null, else 0.0."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])

    if pred == "PARSE_ERROR":
        return 0.0

    ref_is_null = ref is None
    pred_is_null = pred is None

    return 1.0 if ref_is_null == pred_is_null else 0.0


def field_accuracy(references: list[str], predictions: list[str]) -> float:
    """Fraction of fields (merchant, date, type, account) that match exactly.
    Returns 1.0 for correct null-vs-null, 0.0 for mismatched null/non-null."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])

    if pred == "PARSE_ERROR":
        return 0.0

    # Both null — perfect
    if ref is None and pred is None:
        return 1.0
    # One null, one not — complete miss
    if ref is None or pred is None:
        return 0.0

    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0

    fields = ["merchant", "date", "type", "account"]
    matches = 0
    total = 0
    for f in fields:
        ref_val = ref.get(f)
        pred_val = pred.get(f)
        total += 1
        if ref_val is None and pred_val is None:
            matches += 1
        elif ref_val is not None and pred_val is not None:
            if str(ref_val).lower().strip() == str(pred_val).lower().strip():
                matches += 1

    return matches / total if total > 0 else 0.0


def amount_accuracy(references: list[str], predictions: list[str]) -> float:
    """1.0 if extracted amount matches reference amount, else 0.0.
    Returns 1.0 for correct null-vs-null, 0.0 for mismatched null/non-null."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])

    if pred == "PARSE_ERROR":
        return 0.0

    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0

    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0

    try:
        ref_amt = float(ref.get("amount", -1))
        pred_amt = float(pred.get("amount", -2))
        return 1.0 if abs(ref_amt - pred_amt) < 0.01 else 0.0
    except (ValueError, TypeError):
        return 0.0


# ── Normalizers ──────────────────────────────────────────────────────────────────

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _normalize_account(s: str | None) -> tuple | None:
    """Return (category, last-4-digits) for a masked account/card string.

    category is 'card' if the string contains 'card', else 'account'.
    digits are the last 4 significant digits of the last digit-run found.
    Returns None if no digit-run of 3+ digits exists.
    """
    if not isinstance(s, str):
        return None
    category = "card" if "card" in s.lower() else "account"
    runs = re.findall(r"\d+", s)
    long_runs = [r for r in runs if len(r) >= 3]
    if not long_runs:
        return None
    return (category, long_runs[-1][-4:])


def _normalize_date(s: str | None) -> tuple | None:
    """Parse a date string into (day, month, year_or_None).

    Handles: DD-MM-YYYY, DD-MM-YY, DD-MM, DD/MM/YYYY, YYYY-MM-DD, DD-MON-YYYY.
    Returns None if the string cannot be parsed.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    # DD-MM[-YYYY/-YY] or DD/MM[-YYYY/-YY]
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})(?:[-/](\d{2,4}))?$", s)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if 1 <= day <= 31 and 1 <= month <= 12:
            year = None
            if m.group(3):
                yr = int(m.group(3))
                year = yr if yr >= 100 else (2000 + yr if yr < 50 else 1900 + yr)
            return (day, month, year)
    # YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return (day, month, year)
    # DD-MON-YYYY or DD-MON-YY (e.g. 14-APR-2024)
    m = re.match(r"^(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{2,4})$", s)
    if m:
        day = int(m.group(1))
        mon_key = m.group(2).lower()
        if mon_key in _MONTH_ABBR and 1 <= day <= 31:
            yr = int(m.group(3))
            year = yr if yr >= 100 else (2000 + yr if yr < 50 else 1900 + yr)
            return (day, _MONTH_ABBR[mon_key], year)
    return None


def _normalize_type(s: str | None) -> str | None:
    return s.strip().lower() if isinstance(s, str) else None


def _dates_match(gd: tuple | None, pd: tuple | None) -> bool:
    """True if two normalised dates agree. Year is skipped when either side has None."""
    if gd is None and pd is None:
        return True
    if gd is None or pd is None:
        return False
    g_day, g_mon, g_year = gd
    p_day, p_mon, p_year = pd
    if g_day != p_day or g_mon != p_mon:
        return False
    if g_year is not None and p_year is not None and g_year != p_year:
        return False
    return True


def _amount_match(ref: dict, pred: dict) -> bool:
    try:
        return abs(float(ref.get("amount", -1)) - float(pred.get("amount", -2))) < 0.01
    except (ValueError, TypeError):
        return False


def _merchant_match(ref_m, pred_m) -> bool:
    """Case/whitespace-insensitive, allows substring either direction.

    Why loose: bank SMS wrap the same entity in varying boilerplate
    ("VPA x@y", "mobile 9XXXXXXX123-APIBANKING", "UPI-<ref>-Compass", trailing
    city names on gateway descriptors). Requiring byte-equal match punishes the
    model for cosmetic wrapping it has no way to strip.
    Stays strict on None vs non-None (over-extraction is still an error).
    3-char floor on the shorter side avoids pathological substrings.
    """
    if ref_m is None and pred_m is None:
        return True
    if ref_m is None or pred_m is None:
        return False
    r = re.sub(r"\s+", " ", str(ref_m).strip().lower())
    p = re.sub(r"\s+", " ", str(pred_m).strip().lower())
    if r == p:
        return True
    shorter = r if len(r) <= len(p) else p
    longer = p if shorter is r else r
    if len(shorter) < 3:
        return False
    return shorter in longer


# Pre-compute few-shot signatures used by few_shot_leakage_rate.
# Signature = (amount, merchant_lower, normalized_account).
_FEW_SHOT_SIGNATURES: list[tuple] = []
for _ex in FEW_SHOT_EXAMPLES:
    if _ex["answer"] == "null":
        continue
    try:
        _obj = json.loads(_ex["answer"])
        _FEW_SHOT_SIGNATURES.append((
            _obj.get("amount"),
            str(_obj.get("merchant", "")).lower().strip() if _obj.get("merchant") else None,
            _normalize_account(_obj.get("account")),
        ))
    except (json.JSONDecodeError, TypeError):
        pass


# ── New Metrics ──────────────────────────────────────────────────────────────────

def type_accuracy(references: list[str], predictions: list[str]) -> float:
    """1.0 if transaction type (debit/credit) matches exactly. Null-null = 1.0.
    Critical: wrong type flips balance direction in the app."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0
    ref_t = _normalize_type(ref.get("type"))
    pred_t = _normalize_type(pred.get("type"))
    if ref_t is None and pred_t is None:
        return 1.0
    if ref_t is None or pred_t is None:
        return 0.0
    return 1.0 if ref_t == pred_t else 0.0


def account_accuracy(references: list[str], predictions: list[str]) -> float:
    """Normalized account match: last-4-digits + card/account category. Null-null = 1.0.
    Tolerates prefix/masking-style variation — the app identifies accounts by trailing digits."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0
    ref_raw = ref.get("account")
    pred_raw = pred.get("account")
    if ref_raw is None and pred_raw is None:
        return 1.0
    if ref_raw is None or pred_raw is None:
        return 0.0
    ref_norm = _normalize_account(ref_raw)
    pred_norm = _normalize_account(pred_raw)
    if ref_norm is None or pred_norm is None:
        return 0.0
    return 1.0 if ref_norm == pred_norm else 0.0


def date_accuracy(references: list[str], predictions: list[str]) -> float:
    """Semantically-normalised date match. Handles format variation and partial dates.
    Year comparison skipped when either gold or pred omits it."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0
    ref_raw = ref.get("date")
    pred_raw = pred.get("date")
    if ref_raw is None and pred_raw is None:
        return 1.0
    if ref_raw is None or pred_raw is None:
        return 0.0
    return 1.0 if _dates_match(_normalize_date(ref_raw), _normalize_date(pred_raw)) else 0.0


def merchant_accuracy(references: list[str], predictions: list[str]) -> float:
    """Standalone merchant field accuracy (case-insensitive exact match). Null-null = 1.0.
    Extracted from field_accuracy for per-field visibility during model comparison."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0
    return 1.0 if _merchant_match(ref.get("merchant"), pred.get("merchant")) else 0.0


def ghost_transaction_rate(references: list[str], predictions: list[str]) -> float:
    """1.0 if gold=null but pred=transaction (fabricated ledger entry). Lower is better.
    Averaged over ALL samples — measures ghost-transaction risk for the app."""
    ref = _parse(references[0])
    if ref is not None:  # Only relevant on null-gold samples; others contribute 0
        return 0.0
    pred = _parse(predictions[0])
    if pred is None or pred == "PARSE_ERROR" or not isinstance(pred, dict):
        return 0.0
    return 1.0


def missed_transaction_rate(references: list[str], predictions: list[str]) -> float:
    """1.0 if gold=transaction but pred=null (extraction skipped). Lower is better.
    Averaged over ALL samples — measures stale-balance risk for the app."""
    ref = _parse(references[0])
    if ref is None:  # Only relevant on non-null-gold samples; others contribute 0
        return 0.0
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    return 1.0 if pred is None else 0.0


def full_match_accuracy(references: list[str], predictions: list[str]) -> float:
    """1.0 iff all 5 fields correct with normalisation (amount, type, account, date, merchant).
    The end-to-end 'this SMS would be handled correctly in the app' signal."""
    ref = _parse(references[0])
    pred = _parse(predictions[0])
    if pred == "PARSE_ERROR":
        return 0.0
    if ref is None and pred is None:
        return 1.0
    if ref is None or pred is None:
        return 0.0
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return 0.0
    if not _amount_match(ref, pred):
        return 0.0
    if _normalize_type(ref.get("type")) != _normalize_type(pred.get("type")):
        return 0.0
    # Account: both fields absent is ok; if present, normalised digits must match
    ref_raw_acc, pred_raw_acc = ref.get("account"), pred.get("account")
    if ref_raw_acc is None and pred_raw_acc is None:
        pass
    elif ref_raw_acc is None or pred_raw_acc is None:
        return 0.0
    else:
        ref_acc = _normalize_account(ref_raw_acc)
        pred_acc = _normalize_account(pred_raw_acc)
        if ref_acc is None or pred_acc is None or ref_acc != pred_acc:
            return 0.0
    # Date: semantic match with optional year
    if not _dates_match(_normalize_date(ref.get("date")), _normalize_date(pred.get("date"))):
        return 0.0
    if not _merchant_match(ref.get("merchant"), pred.get("merchant")):
        return 0.0
    return 1.0


def few_shot_leakage_rate(references: list[str], predictions: list[str]) -> float:
    """On null-gold samples: 1.0 if pred's (amount, merchant, account) matches a few-shot example.
    Detects models that copy few-shot answers instead of processing the input. Lower is better."""
    ref = _parse(references[0])
    if ref is not None:  # Only check on null-gold samples
        return 0.0
    pred = _parse(predictions[0])
    if pred is None or pred == "PARSE_ERROR" or not isinstance(pred, dict):
        return 0.0
    sig = (
        pred.get("amount"),
        str(pred.get("merchant", "")).lower().strip() if pred.get("merchant") else None,
        _normalize_account(pred.get("account")),
    )
    return 1.0 if sig in _FEW_SHOT_SIGNATURES else 0.0


# ── Sanity checks ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0

    def _check(label: str, got, expected):
        global passed, failed
        if got == expected:
            print(f"  PASS  {label}")
            passed += 1
        else:
            print(f"  FAIL  {label}")
            print(f"        got:      {got!r}")
            print(f"        expected: {expected!r}")
            failed += 1

    print("=== _normalize_account ===")
    _check("A/c X6254",              _normalize_account("A/c X6254"),              ("account", "6254"))
    _check("a/c no. XXXXXXXX6254",   _normalize_account("a/c no. XXXXXXXX6254"),  ("account", "6254"))
    _check("A/cX6254",               _normalize_account("A/cX6254"),               ("account", "6254"))
    _check("A/C XXXXX436254",        _normalize_account("A/C XXXXX436254"),        ("account", "6254"))
    _check("A/C *9141",              _normalize_account("A/C *9141"),              ("account", "9141"))
    _check("Credit Card XX3782",     _normalize_account("Credit Card XX3782"),     ("card", "3782"))
    _check("CREDIT CARD ENDING 3782",_normalize_account("CREDIT CARD ENDING 3782"),("card", "3782"))
    _check("Card 0816",              _normalize_account("Card 0816"),              ("card", "0816"))
    _check("card != account",
           _normalize_account("Credit Card XX3782") == _normalize_account("A/c X6254"), False)

    print("\n=== _normalize_date ===")
    _check("02-05-2022",   _normalize_date("02-05-2022"),  (2, 5, 2022))
    _check("02-05-22",     _normalize_date("02-05-22"),    (2, 5, 2022))
    _check("2022-05-02",   _normalize_date("2022-05-02"),  (2, 5, 2022))
    _check("31-5-2024",    _normalize_date("31-5-2024"),   (31, 5, 2024))
    _check("01-10 partial",_normalize_date("01-10"),       (1, 10, None))
    _check("14-APR-2024",  _normalize_date("14-APR-2024"), (14, 4, 2024))
    _check("01-FEB-2024",  _normalize_date("01-FEB-2024"), (1, 2, 2024))
    _check("19/12/2022",   _normalize_date("19/12/2022"),  (19, 12, 2022))
    _check("None → None",  _normalize_date(None),          None)

    print("\n=== _dates_match ===")
    _check("same full date",
           _dates_match(_normalize_date("02-05-2022"), _normalize_date("02-05-2022")), True)
    _check("4-digit vs 2-digit year",
           _dates_match(_normalize_date("02-05-2022"), _normalize_date("02-05-22")), True)
    _check("YYYY-MM-DD vs DD-MM-YYYY",
           _dates_match(_normalize_date("2022-05-02"), _normalize_date("02-05-2022")), True)
    _check("partial (no year) vs full — skip year",
           _dates_match(_normalize_date("01-10"), _normalize_date("01-10-2024")), True)
    _check("year mismatch",
           _dates_match(_normalize_date("01-10-2023"), _normalize_date("01-10-2024")), False)
    _check("day mismatch",
           _dates_match(_normalize_date("02-05-2022"), _normalize_date("03-05-2022")), False)
    _check("both None",
           _dates_match(None, None), True)
    _check("one None",
           _dates_match(None, _normalize_date("01-10-2024")), False)

    print("\n=== _merchant_match (loose) ===")
    _check("exact",                          _merchant_match("Zomato", "Zomato"), True)
    _check("case-insensitive",               _merchant_match("Zomato", "ZOMATO"), True)
    _check("double vs single space",         _merchant_match("OMA  RAM", "OMA RAM"), True)
    _check("prefix VPA ",                    _merchant_match("manisharadwad@oksbi", "VPA manisharadwad@oksbi"), True)
    _check("prefix mobile ",                 _merchant_match("1XXXXXX890-APIBANKING", "mobile 1XXXXXX890-APIBANKING"), True)
    _check("suffix UPI-ref",                 _merchant_match("Compass", "UPI-640971125815-Compass"), True)
    _check("trailing city noise",            _merchant_match("GOOGLEPLAY MUMBAI MAH", "GOOGLEPLAY"), True)
    _check("both None",                      _merchant_match(None, None), True)
    _check("gold None, pred set → mismatch", _merchant_match(None, "SBI UPI"), False)
    _check("gold set, pred None → mismatch", _merchant_match("Saanvi Star", None), False)
    _check("different VPA digit",            _merchant_match("q36252344@ybl", "q362523441@ybl"), False)
    _check("typo in VPA",                    _merchant_match("manisharadwadoksbi", "manishardwadoksbi"), False)
    _check("substring < 3 chars guard",      _merchant_match("A", "ABC Corp"), False)

    print("\n=== ghost_transaction_rate / missed_transaction_rate ===")
    null_gold = ["null"]
    txn_gold  = ['{"amount": 150.0, "merchant": "Zomato", "date": "01-03-2024", "type": "debit", "account": "A/c XX1234"}']
    txn_pred  = ['{"amount": 150.0, "merchant": "Zomato", "date": "01-03-2024", "type": "debit", "account": "A/c XX1234"}']
    null_pred = ["null"]

    _check("ghost: null gold + txn pred → 1.0",  ghost_transaction_rate(null_gold, txn_pred),  1.0)
    _check("ghost: null gold + null pred → 0.0", ghost_transaction_rate(null_gold, null_pred), 0.0)
    _check("ghost: txn gold + txn pred → 0.0",   ghost_transaction_rate(txn_gold, txn_pred),   0.0)
    _check("missed: txn gold + null pred → 1.0", missed_transaction_rate(txn_gold, null_pred), 1.0)
    _check("missed: txn gold + txn pred → 0.0",  missed_transaction_rate(txn_gold, txn_pred),  0.0)
    _check("missed: null gold + null pred → 0.0",missed_transaction_rate(null_gold, null_pred),0.0)

    print("\n=== few_shot_leakage_rate ===")
    vpa_leakage = ['{"amount": 500.0, "merchant": "demouser000@examplebank", "date": "01-01-2020", "type": "credit", "account": "a/c XXXXXX0000"}']
    diff_pred   = ['{"amount": 999.0, "merchant": "Foo", "date": "01-01-2024", "type": "debit", "account": "A/c XX9999"}']
    _check("leakage: null gold + few-shot answer → 1.0",
           few_shot_leakage_rate(null_gold, vpa_leakage), 1.0)
    _check("no leakage: null gold + different pred → 0.0",
           few_shot_leakage_rate(null_gold, diff_pred), 0.0)
    _check("no check: txn gold + few-shot answer → 0.0 (only null gold counts)",
           few_shot_leakage_rate(txn_gold, vpa_leakage), 0.0)

    print("\n=== full_match_accuracy ===")
    ref1 = ['{"amount": 5000.0, "merchant": null, "date": "02-05-2022", "type": "debit", "account": "a/c no. XXXXXXXX6254"}']
    # Normalised format, different date/account style — should match
    pred_ok = ['{"amount": 5000.0, "merchant": null, "date": "02-05-22", "type": "debit", "account": "A/c XXXXXXXX6254"}']
    _check("full match with normalized date+account",  full_match_accuracy(ref1, pred_ok), 1.0)
    pred_wrong_type = ['{"amount": 5000.0, "merchant": null, "date": "02-05-2022", "type": "credit", "account": "a/c no. XXXXXXXX6254"}']
    _check("wrong type → 0.0",    full_match_accuracy(ref1, pred_wrong_type), 0.0)
    pred_wrong_amt  = ['{"amount": 500.0,  "merchant": null, "date": "02-05-2022", "type": "debit",  "account": "a/c no. XXXXXXXX6254"}']
    _check("wrong amount → 0.0",  full_match_accuracy(ref1, pred_wrong_amt),  0.0)
    _check("null-null → 1.0",     full_match_accuracy(["null"], ["null"]),     1.0)
    _check("null gold, txn pred → 0.0", full_match_accuracy(["null"], pred_ok), 0.0)
    _check("txn gold, null pred → 0.0", full_match_accuracy(ref1, ["null"]),    0.0)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
