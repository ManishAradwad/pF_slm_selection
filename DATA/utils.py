"""
Utilities for the sms_extraction evaluation task.
Provides: prompt formatting, output filtering, and custom metrics.
"""

import json
import re


# ── Prompt ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Given the sender ID and SMS body, extract transaction details from an Indian bank/card SMS as a JSON object.

Rules:
- amount: number (e.g. 150.0). Parse "Rs.1,500" as 1500.0.
- merchant: the person, shop, or UPI ID that money was sent to or received from. This is NOT the bank name. If no recipient/sender is mentioned, use null.
- date: always DD-MM-YYYY with numeric month (e.g. 01-03-2024). Convert "01Mar24" to "01-03-2024", "2024-03-01" to "01-03-2024". If no date in the SMS, use null.
- type: "debit" if money left your account, "credit" if money came in.
- account: the masked account or card number exactly as shown in the SMS (e.g. "A/c X6254", "Credit Card XX3782", "Card 0816"). Do NOT use the bank name here.

If the SMS is NOT a real bank/card transaction output exactly: null
Reject these: promotional offers, cashback ads, wallet top-ups, balance reports, investment summaries, account-opening ads, game/spin rewards.
Hint: legitimate bank senders typically contain the bank name (e.g. SBIUPI, HDFCBK, IDFCFB, MAHABK)."""

FEW_SHOT_EXAMPLES = [
    {
        "sender": "AX-SBIUPI",
        "sms": "Rs.150 debited from A/c XX1234 on 01Mar24 to Zomato. UPI Ref 123456789.",
        "answer": '{"amount": 150.0, "merchant": "Zomato", "date": "01-03-2024", "type": "debit", "account": "A/c XX1234"}',
    },
    {
        "sender": "BW-SBIUPI",
        "sms": "Dear SBI UPI User, ur A/cX6254 credited by Rs296.25 on 05Sep23 by (Ref no 324835400880)",
        "answer": '{"amount": 296.25, "merchant": null, "date": "05-09-2023", "type": "credit", "account": "A/c X6254"}',
    },
    {
        "sender": "AD-HDFCBK",
        "sms": "Transaction Reversed!On HDFC Bank CREDIT Card xx6719 Amt: Rs.50 By CAFE MOCHA On 2024-03-17:01:17:08",
        "answer": '{"amount": 50.0, "merchant": "CAFE MOCHA", "date": "17-03-2024", "type": "refund", "account": "Credit Card xx6719"}',
    },
    {
        "sender": "VM-OFFERZ",
        "sms": "Get 50% cashback up to Rs.200 on your next recharge! Use code SAVE50. T&C apply.",
        "answer": "null",
    },
    {
        "sender": "VK-GMERMY",
        "sms": "Dear Customer, Your A/c XXXX921 is Credited With Rs.10,000 withdraw directly in your wallet a/c. Check Now: http://example.com RUM G",
        "answer": "null",
    },
]


def doc_to_text(doc: dict) -> str:
    """Build the full prompt with system instruction, few-shot examples, and query."""
    parts = [SYSTEM_PROMPT, ""]

    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"Sender: {ex['sender']}")
        parts.append(f"SMS: {ex['sms']}")
        parts.append(f"Output: {ex['answer']}")
        parts.append("")

    parts.append(f"Sender: {doc['sender']}")
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
