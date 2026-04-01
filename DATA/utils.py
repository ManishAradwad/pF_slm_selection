"""
Utilities for the sms_extraction evaluation task.
Provides: prompt formatting, output filtering, and custom metrics.
"""

import json
import re


# ── Prompt ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an on-device financial assistant. "
    "Given an SMS, extract transaction details as JSON with these fields: "
    "amount (number), merchant (string or null), date (DD-MM-YYYY or null), "
    "type (\"debit\", \"credit\", or \"refund\"), account (string). "
    "If the SMS is NOT a real bank/card transaction (e.g. spam, promo, "
    "balance info), respond with exactly: null"
)

FEW_SHOT_EXAMPLES = [
    {
        "sms": "Rs.150 debited from A/c XX1234 on 01Mar24 to Zomato. UPI Ref 123456789.",
        "answer": '{"amount": 150.0, "merchant": "Zomato", "date": "01-03-2024", "type": "debit", "account": "A/c XX1234"}',
    },
    {
        "sms": "Get 50% cashback up to Rs.200 on your next recharge! Use code SAVE50. T&C apply.",
        "answer": "null",
    },
]


def doc_to_text(doc: dict) -> str:
    """Build the full prompt with system instruction, few-shot examples, and query."""
    parts = [SYSTEM_PROMPT, ""]

    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"SMS: {ex['sms']}")
        parts.append(f"Output: {ex['answer']}")
        parts.append("")

    parts.append(f"SMS: {doc['sms']}")
    parts.append("Output: ")

    return "\n".join(parts)


# ── Filters ─────────────────────────────────────────────────────────────────────

def extract_json_filter(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Extract the first JSON object or 'null' from each model response."""
    results = []
    for resp_list in resps:
        filtered = []
        for resp in resp_list:
            resp = resp.strip()
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

    try:
        ref_amt = float(ref.get("amount", -1))
        pred_amt = float(pred.get("amount", -2))
        return 1.0 if abs(ref_amt - pred_amt) < 0.01 else 0.0
    except (ValueError, TypeError):
        return 0.0
