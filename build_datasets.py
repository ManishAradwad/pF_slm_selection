import json
import re

def is_likely_financial(text: str) -> bool:
    """Quick heuristic to pre-flag likely financial SMS."""
    patterns = [
        r"(?:Rs\.?|INR|₹)\s*[\d,]+",           # Amount patterns
        r"(?:debited|credited|debit|credit)",      # Transaction words
        r"(?:A/[Cc]|a/c|acct)\s*[Xx\d]+",         # Account numbers
        r"(?:UPI|NEFT|IMPS|RTGS|ATM)",             # Transfer modes
        r"(?:Avl\s*Bal|balance|bal)",               # Balance mentions
        r"(?:OTP|CVV|PIN)\s*\d+",                  # OTP messages (financial)
        r"(?:EMI|loan|premium|insurance)",          # Loan/insurance
        r"(?:SBI|HDFC|ICICI|Axis|Kotak|BOB|PNB|Yes Bank|IndusInd)",  # Bank names
        r"(?:Paytm|PhonePe|GPay|BHIM|Razorpay)",   # Payment apps
        r"(?:credit\s*card|debit\s*card)",          # Card mentions
        r"(?:mutual\s*fund|SIP|NAV|folio)",         # Investment
    ]
    text_lower = text.lower()
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def build_classification_candidates(raw_sms_path="all_sms.json"):
    """
    Pre-sort SMS into financial vs non-financial
    for manual review and dataset creation.
    """
    with open(raw_sms_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Filter: only incoming messages (not from you)
    incoming = [m for m in messages if not m["is_from_me"] and m["text"]]

    financial = []
    non_financial = []

    for msg in incoming:
        if is_likely_financial(msg["text"]):
            financial.append(msg)
        else:
            non_financial.append(msg)

    print(f"Total incoming SMS: {len(incoming)}")
    print(f"  Likely financial: {len(financial)}")
    print(f"  Likely non-financial: {len(non_financial)}")

    # Save for manual review
    with open("candidates_financial.json", "w", encoding="utf-8") as f:
        json.dump(financial, f, indent=2, ensure_ascii=False)

    with open("candidates_non_financial.json", "w", encoding="utf-8") as f:
        json.dump(non_financial, f, indent=2, ensure_ascii=False)

    return financial, non_financial


if __name__ == "__main__":
    build_classification_candidates()