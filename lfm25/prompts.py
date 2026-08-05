"""Short production prompt used for tuned-model training and deployment."""

from __future__ import annotations

PRODUCTION_SYSTEM_PROMPT = """Extract a posted bank/card transaction from one SMS.
Return exactly null or JSON with only amount, counterparty, type, account.
amount is a number; type is debit or credit; account is the masked account/card text; counterparty may be null.
Return null for OTP/security messages, balances/statements, offers, bills or payment requests, mandates, pending/failed/declined/blocked transactions, and wallet/BNPL activity.
A reversal/refund qualifies only when the SMS explicitly confirms money was credited. Use only values in the current SMS. No prose."""


def extraction_messages(sender: str, sms: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PRODUCTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Sender: {sender}\nSMS: {sms}"},
    ]
