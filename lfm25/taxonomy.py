"""
Comprehensive Indian SMS Financial & Non-Financial Taxonomy Engine.

Defines a 3-tier taxonomy specifically adapted for Indian SMS patterns:
1. TRANSACTIONAL (Posted financial movements: Debits & Credits)
2. EDGE_FINANCIAL (Financial hard negatives: Failed txns, collect requests, statements, etc.)
3. NON_TRANSACTIONAL (General traffic: OTPs, telecom, delivery, security alerts, promo, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ==============================================================================
# TAXONOMY DEFINITIONS
# ==============================================================================

CATEGORIES_METADATA: dict[str, dict[str, Any]] = {
    # 1. TRANSACTIONAL
    "tx.debit.card": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Credit Card / Debit Card POS swipe, online checkout, tap-to-pay, RuPay on UPI",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.upi": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Outward UPI transfer or merchant payment via QR/VPA/App",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.bank_account": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Outward NetBanking, NEFT, IMPS, RTGS or cheque debit from bank account",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.mandate_auto_debit_emi": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Recurring auto-debit, NACH mandate, loan EMI, or SIP installment deduction",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.atm_withdrawal": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Cash withdrawal from ATM / CDM",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.bank_charges_fees": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Bank service fee, annual maintenance charge, or SMS alert charge debit",
        "ground_truth_target": "transaction_json",
    },
    "tx.debit.bill_or_credit_card_paid": {
        "primary": "TRANSACTIONAL",
        "action_type": "debit",
        "description": "Receipt confirmation of utility bill or credit card payment settled",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.upi": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Inward money received via UPI from peer or customer",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.bank_inward": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Direct bank deposit, inward IMPS/NEFT/RTGS transfer, or CDM deposit",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.salary_payroll": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Corporate salary, stipend, or employer payroll deposit",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.refund_cashback_reversal": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Merchant refund, transaction reversal, or cashback credit",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.interest_dividend": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Bank savings account quarterly interest or equity/mutual fund dividend credit",
        "ground_truth_target": "transaction_json",
    },
    "tx.credit.investment_redemption": {
        "primary": "TRANSACTIONAL",
        "action_type": "credit",
        "description": "Mutual fund redemption payout or securities liquidation proceeds credited",
        "ground_truth_target": "transaction_json",
    },

    # 2. EDGE_FINANCIAL (HARD NEGATIVES)
    "edge.txn_failed_declined": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Transaction attempted but failed, declined, timed out, or cancelled",
        "ground_truth_target": "null",
    },
    "edge.payment_collect_request": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "UPI collect request, payment request, or mandate authorization request",
        "ground_truth_target": "null",
    },
    "edge.bill_generated_or_due": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Statement generated or bill due reminder without executed payment",
        "ground_truth_target": "null",
    },
    "edge.upcoming_sip_mandate_alert": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Upcoming SIP or auto-debit reminder asking to maintain bank balance",
        "ground_truth_target": "null",
    },
    "edge.balance_inquiry_or_update": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Balance inquiry result, missed-call balance alert, or credit limit status",
        "ground_truth_target": "null",
    },
    "edge.investment_demat_statement": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Regulatory depository & broker fund/securities balance statement (NSDL, CDSL, NSE)",
        "ground_truth_target": "null",
    },
    "edge.mandate_sip_created_or_cancelled": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Administrative notice of SIP/mandate setup, pause, or cancellation",
        "ground_truth_target": "null",
    },
    "edge.marketing_loan_credit_offer": {
        "primary": "EDGE_FINANCIAL",
        "action_type": "none",
        "description": "Pre-approved loan or credit card offer mentioning monetary amounts",
        "ground_truth_target": "null",
    },

    # 3. NON_TRANSACTIONAL
    "non_tx.otp_auth_2fa": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "One-Time Password, 2FA code, or authentication token",
        "ground_truth_target": "null",
    },
    "non_tx.sim_binding_registration": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "UPI SIM binding handshake or banking app device registration notice",
        "ground_truth_target": "null",
    },
    "non_tx.telecom_recharge_data": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "Telecom validity, data usage threshold, recharge reminder, or plan expiry",
        "ground_truth_target": "null",
    },
    "non_tx.delivery_service_logistics": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "E-commerce order tracking, food delivery, ride hailing, or travel PNR",
        "ground_truth_target": "null",
    },
    "non_tx.account_security_admin": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "NetBanking login alert, KYC update reminder, or card security advisory",
        "ground_truth_target": "null",
    },
    "non_tx.promotional_marketing": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "Commercial advertisement, retail discount, or festive offer",
        "ground_truth_target": "null",
    },
    "non_tx.education_recruitment": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "College/university admission alerts, job applications, or Chegg Q&A",
        "ground_truth_target": "null",
    },
    "non_tx.regulatory_telecom_advisories": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "TRAI / DoT / Police anti-fraud advisory notices",
        "ground_truth_target": "null",
    },
    "non_tx.regional_language_vas": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "Non-English regional language text (Marathi/Hindi caller tunes, VAS)",
        "ground_truth_target": "null",
    },
    "non_tx.personal_outgoing": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "Outgoing message sent by the user (chat or silent SMS payload)",
        "ground_truth_target": "null",
    },
    "non_tx.empty_or_corrupted": {
        "primary": "NON_TRANSACTIONAL",
        "action_type": "none",
        "description": "Empty text body or corrupted unicode/RBM media replacement character",
        "ground_truth_target": "null",
    },
    "unclassified_residual": {
        "primary": "UNCATEGORIZED",
        "action_type": "unknown",
        "description": "Unclassified residual message requiring manual review",
        "ground_truth_target": "null",
    },
}


@dataclass(frozen=True, slots=True)
class TaxonomyClassification:
    primary_category: str  # TRANSACTIONAL, EDGE_FINANCIAL, NON_TRANSACTIONAL, UNCATEGORIZED
    subcategory: str
    action_type: str  # debit, credit, none, unknown
    description: str
    ground_truth_target: str  # transaction_json or null


def classify_sms_record(sender: str, body: str, is_from_me: bool = False) -> TaxonomyClassification:
    """
    Deterministically classify an Indian SMS record into the 3-tier taxonomy.
    """
    text = str(body or "").strip()
    s = str(sender or "").strip()

    # 0. Empty / corrupted / non-text
    if not text or text in ("\ufffc", "￼") or len(text) == 0:
        meta = CATEGORIES_METADATA["non_tx.empty_or_corrupted"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.empty_or_corrupted",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 1. Outgoing / Personal
    if is_from_me or s.lower() == "me":
        meta = CATEGORIES_METADATA["non_tx.personal_outgoing"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.personal_outgoing",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 2. Hard Negatives & Edge Financial
    # Failed / Declined
    if re.search(
        r"(?:declined|failed|unsuccessful|cancelled|timed out|could not be processed|rejected|un-successful).*(?:due to|limit|insufficient|pin|bank|technical|exceeded)?",
        text,
        re.IGNORECASE,
    ) and re.search(r"(?:rs\.?|inr|₹|\btxn\b|amount)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["edge.txn_failed_declined"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.txn_failed_declined",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Mandate / SIP setup or cancellation
    if re.search(
        r"(?:mandate|sip|standing instruction).*(?:registered|cancelled|modified|setup|cancellation request|pause|is confirmed on groww)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["edge.mandate_sip_created_or_cancelled"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.mandate_sip_created_or_cancelled",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Collect / Payment Request
    if re.search(
        r"(?:requested money|has requested|request of rs|collect request|pay request|payment request|mandate request|approve the request)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["edge.payment_collect_request"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.payment_collect_request",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Upcoming SIP / Maintain balance alert
    if re.search(
        r"(?:next sip installment|sip installment will be deducted|ensure balance of rs|maintain balance of rs)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["edge.upcoming_sip_mandate_alert"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.upcoming_sip_mandate_alert",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Bill / Statement generated (without debit/payment settlement)
    if re.search(
        r"(?:bill generated|statement generated|total amount due|payment due date|min(?:imum)? amount due|bill of rs\.?|e-bill|outstanding of rs|is due on|bill reminder)",
        text,
        re.IGNORECASE,
    ) and not re.search(
        r"(?:thank you for (?:making )?payment|received payment of rs|debited|spent|paid rs)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["edge.bill_generated_or_due"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.bill_generated_or_due",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Demat / Depository / Brokerage Statement
    if re.search(
        r"(?:reported your fund balance|securities balance|cdsl|nsdl|nsetra|bsetra|equity derivative|segment of nse|settlement of fund|statement of account for recent txn|holding in demat)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["edge.investment_demat_statement"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.investment_demat_statement",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Pre-approved Loan / Credit Card Marketing Offer with Amounts
    if re.search(
        r"(?:pre-approved|eligible for|instant loan of|apply for card|credit limit increase|loan offer|pre-qualified|get up to rs|claim credit card|enjoy rs\..*yearly benefits|savings of rs\..*on hdfc|earn 200-2000 rupees|crypto journey with just rs).*(?:rs\.?|inr|₹|\blakh\b|\bcr\b|\bk\b|rupees)",
        text,
        re.IGNORECASE,
    ) and not re.search(r"(?:debited|credited|withdrawn|spent)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["edge.marketing_loan_credit_offer"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.marketing_loan_credit_offer",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # Balance inquiry / Avl Bal alert (without txn)
    if re.search(r"^(?:avl bal|available bal|balance in a/c|clear balance|your bal is|a/c balance)", text, re.IGNORECASE) or (
        re.search(r"(?:bal:|avl bal:|available balance:)", text, re.IGNORECASE)
        and not re.search(r"(?:debited|credited|spent|transferred|withdrawn|paid|txn rs|money transfer:rs)", text, re.IGNORECASE)
    ):
        meta = CATEGORIES_METADATA["edge.balance_inquiry_or_update"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="edge.balance_inquiry_or_update",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 3. OTP & Authentication Codes
    if re.search(
        r"(?:\botp\b|one time password|verification code|security code|login code|auth code|secret code|\b\d{4,6}\b\s+is\s+(?:your\s+)?(?:otp|secret|login|code)|is your verification code)",
        text,
        re.IGNORECASE,
    ) and not re.search(r"(?:debited by|credited to|spent rs)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["non_tx.otp_auth_2fa"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.otp_auth_2fa",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # ================== 4. TRANSACTIONAL (Posted Events) ==================
    # 4.1 Investment Redemption / Payout
    if re.search(
        r"(?:redemption payout|redemption transaction amounting|payout will be released|payout will be initiated).*(?:rs\.?|inr|₹|neft|rtgs)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["tx.credit.investment_redemption"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.investment_redemption",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.2 Refunds, Reversals, Cashback
    if re.search(
        r"(?:refund|refunded|reversal|reversed).*(?:rs\.?|inr|₹|\bdebited\b|\bcredited\b|successful)?",
        text,
        re.IGNORECASE,
    ) or re.search(r"(?:cashback|cash back).*(?:credited|received|rs\.?|₹)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["tx.credit.refund_cashback_reversal"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.refund_cashback_reversal",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.3 Salary & Payroll
    if re.search(r"(?:salary|payroll|stipend).*(?:credited|deposited|rs\.?|₹)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["tx.credit.salary_payroll"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.salary_payroll",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.4 Interest & Dividends
    if re.search(r"(?:interest|dividend).*(?:credited|rs\.?|₹)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["tx.credit.interest_dividend"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.interest_dividend",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.5 Inward UPI Credit
    if re.search(r"(?:credited|received).*(?:upi|vpa|@|gpay|phonepe|paytm)", text, re.IGNORECASE) and not re.search(
        r"debited", text, re.IGNORECASE
    ):
        meta = CATEGORIES_METADATA["tx.credit.upi"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.upi",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.6 Direct Bank Inward Credit
    if re.search(r"(?:credited|deposited|received into|deposited into).*(?:a/c|acct|account|bank)", text, re.IGNORECASE) or re.search(
        r"(?:a/c|acct|account).*(?:is credited|has been credited|credited with|credited for).*(?:rs|inr|₹)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["tx.credit.bank_inward"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.credit.bank_inward",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.7 ATM Cash Withdrawal
    if re.search(r"(?:atm|cash withdrawal|withdrawn at|cash wdl|wdl at atm)", text, re.IGNORECASE) and re.search(
        r"(?:debited|withdrawn|rs\.?|inr|₹)", text, re.IGNORECASE
    ):
        meta = CATEGORIES_METADATA["tx.debit.atm_withdrawal"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.atm_withdrawal",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.8 Mandates / Auto-Debit / NACH / EMI
    if re.search(
        r"(?:auto-debit|autodebit|mandate|standing instruction|si debited|emi debited|nach debit|nach\b)",
        text,
        re.IGNORECASE,
    ) and re.search(r"(?:debited|processed|successful|rs\.?|inr|₹)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["tx.debit.mandate_auto_debit_emi"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.mandate_auto_debit_emi",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.9 Outward UPI Debit
    if (
        re.search(r"money transfer:rs\s*[\d,.]+\s*from.*to.*upi", text, re.IGNORECASE)
        or re.search(
            r"(?:debited|paid|sent|transferred).*(?:via upi|by upi|using upi|upi ref|vpa|to vpa|@)",
            text,
            re.IGNORECASE,
        )
        or re.search(r"upi.*(?:debited|paid|transferred)", text, re.IGNORECASE)
    ):
        meta = CATEGORIES_METADATA["tx.debit.upi"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.upi",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.10 Card Debit
    if re.search(
        r"(?:txn rs\.|spent rs\.|on (?:hdfc|icici|sbi|axis|kotak|rbl|citi|bank)?\s*(?:debit\s*|credit\s*)?card|card ending|card xx\d+|card \d{4}|charge on card|block cc|you've spent rs)",
        text,
        re.IGNORECASE,
    ) and re.search(r"(?:spent|debited|txn|charge|paid|rs\.?|inr|₹)", text, re.IGNORECASE):
        meta = CATEGORIES_METADATA["tx.debit.card"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.card",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.11 Bank Service Fees
    if re.search(
        r"(?:annual fee|sms alert charge|consolidated charge|non maintenance fee|penalty charge).*(?:debited|rs\.?|₹)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["tx.debit.bank_charges_fees"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.bank_charges_fees",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.12 General Bank Outward Debit
    if (
        re.search(r"money transfer:rs\s*[\d,.]+\s*from", text, re.IGNORECASE)
        or re.search(r"(?:debited|withdrawn|transferred from|paid from).*(?:a/c|acct|account)", text, re.IGNORECASE)
        or re.search(
            r"(?:a/c|acct|account).*(?:is debited|has been debited|debited by|debited for|debited with).*(?:rs|inr|₹)",
            text,
            re.IGNORECASE,
        )
    ):
        meta = CATEGORIES_METADATA["tx.debit.bank_account"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.bank_account",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 4.13 Bill / Credit Card Paid
    if re.search(
        r"(?:thank you for (?:making )?payment|payment of rs\.?.*received toward|payment received for your (?:credit card|bill)|successfully paid)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["tx.debit.bill_or_credit_card_paid"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="tx.debit.bill_or_credit_card_paid",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # ================== 5. NON-TRANSACTIONAL UTILITIES ==================
    # 5.1 SIM Binding & Device Registration
    if re.search(
        r"successfully captured for UPI registration|UPI registration has started|registration for UPI|Device SMS count Exceeded|SIM binding|Google Pay has begun|successfully registered your device|set the upi pin|set your 4-digit login pin|enabled - biometric login",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.sim_binding_registration"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.sim_binding_registration",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.2 Telecom & Data Alerts
    if re.search(
        r"(?:daily data|data pack|100% daily|50% data|validity|recharge of rs|unlimited 5g|caller tune|jio|airtel|vi |vodafone|bsnl|pack expiry|talktime|recharge successful|dial 57373|callertune|dial \*121#|pack.*has expired|enjoy calling|vi care|jio prepaid)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.telecom_recharge_data"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.telecom_recharge_data",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.3 Delivery, Logistics & Travel PNR
    if re.search(
        r"(?:order confirmed|out for delivery|delivered|shipped|tracking|arriving today|pnr|booking confirmed|flight|boarding|swiggy|zomato|uber|ola|amazon|flipkart|myntra|blinkit|zepto|dunzo|shipment|doxper|prescription|food delivery|courier)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.delivery_service_logistics"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.delivery_service_logistics",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.4 Account Security, Admin & KYC
    if re.search(
        r"(?:kyc|pan linked|update kyc|block card|profile updated|password changed|welcome to|alert:|dear customer|netbanking login|login alert|statement for|nomination|registered email|admissions office|applicant|oneassist|wallet protection|block all your bank cards|registered for)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.account_security_admin"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.account_security_admin",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.5 Regulatory Telecom & Fraud Warnings
    if re.search(
        r"(?:trai|dotmah|antarrashtriya call|noc for installation|cyber police|department of telecom)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.regulatory_telecom_advisories"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.regulatory_telecom_advisories",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.6 Promotional & Marketing
    if re.search(
        r"(?:flat \d+%|discount|offer|coupon|hurry|shop now|deal of|festive|cashback upto|win|exclusive|sale is live|use code|avail now|special price|limited period|apply code|buy now|entertainment|movies|tv shows|vi movies|tataneu|nobroker|chggin)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.promotional_marketing"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.promotional_marketing",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.7 Regional Language Content
    if re.search(r"[\u0900-\u097F]", text):
        meta = CATEGORIES_METADATA["non_tx.regional_language_vas"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.regional_language_vas",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.8 Education, Job & Recruitment Alerts
    if re.search(
        r"(?:admission|m\.tech|iit|iisc|university|careers360|naukri|interview|placement|hiring|chegg|upsc|unacademy|civil services)",
        text,
        re.IGNORECASE,
    ):
        meta = CATEGORIES_METADATA["non_tx.education_recruitment"]
        return TaxonomyClassification(
            primary_category=meta["primary"],
            subcategory="non_tx.education_recruitment",
            action_type=meta["action_type"],
            description=meta["description"],
            ground_truth_target=meta["ground_truth_target"],
        )

    # 5.9 Residual fallback
    meta = CATEGORIES_METADATA["unclassified_residual"]
    return TaxonomyClassification(
        primary_category=meta["primary"],
        subcategory="unclassified_residual",
        action_type=meta["action_type"],
        description=meta["description"],
        ground_truth_target=meta["ground_truth_target"],
    )
