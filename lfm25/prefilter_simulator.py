"""
PocketFinancer Pre-SLM Filter Simulators for Android, iOS, and Unified Triage.

Accurately simulates:
1. Android SmsFilterPipeline.kt (6 stages)
2. iOS AlertFilter.swift (8 stages, rulesVersion: ios-eligibility.v2)
3. Unified Platform-Neutral Pre-SLM Pipeline (combining best of both)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ==============================================================================
# 1. ANDROID SMS FILTER SIMULATOR (SmsFilterPipeline.kt)
# ==============================================================================

ANDROID_PERSONAL_MOBILE_SENDER_RE = re.compile(r"^\+?[0-9]{10,15}$")
ANDROID_CURRENCY_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|\u20b9)\s*[\d,]+(?:\.\d{1,2})?|"
    r"[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|\u20b9)",
    re.IGNORECASE,
)
ANDROID_MASKED_ACCOUNT_OR_CARD_RE = re.compile(
    r"a/c\s*(?:no\.?\s*)?[X*x]+\d+|"
    r"a/?c\s*(?:no\.?\s*)?\*+\d+|"
    r"card\s*(?:no\.?\s*)?[Xx*]+\d+|"
    r"card\s+\d{4}\b|"
    r"card\s+ending\s+[Xx*]*\d+",
    re.IGNORECASE,
)
ANDROID_TRANSACTION_VERB_RE = re.compile(
    r"\b(?:debited|credited|deducted|spent|paid|received|transferred|sent|"
    r"reversed|refunded|used|withdrawn|deposited)(?=[^a-zA-Z]|$)|"
    r"\btxn\b|"
    r"\bhas\s+(?:a\s+)?debit\s+by\b|"
    r"\bhas\s+credit\s+for\b|"
    r"\bwithout\s+OTP\b|"
    r"\bauto.?debit\b|"
    r"\bDebit\s+in\s+a/c\b|"
    r"\btxn\s+of\s+Rs\b|"
    r"\bRedemption\s+payout\b|"
    r"\b(?:money\s+transfer|amt\s+sent|amt\s+received)\b|"
    r"you've\s+hand-?picked",
    re.IGNORECASE,
)
ANDROID_OTP_RE = re.compile(
    r"\botp\b|\bone.?time.?password\b|\bverification.?code\b",
    re.IGNORECASE,
)
ANDROID_COLLECT_OR_MANDATE_REQUEST_RE = re.compile(
    r"has\s+requested\s+money|"
    r"requested\s+Rs\.?|"
    r"collect\s+request|"
    r"mandate\s+request|"
    r"request\s+from\s+you",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AndroidPrefilterResult:
    passed: bool
    rejection_stage: str | None
    rejection_reason: str | None
    stage_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rejection_stage": self.rejection_stage,
            "rejection_reason": self.rejection_reason,
            "stage_index": self.stage_index,
        }


def run_android_prefilter(sender: str, body: str) -> AndroidPrefilterResult:
    """Simulate PocketFinancer Android SmsFilterPipeline.kt."""
    normalized_sender = str(sender or "").strip()
    sms_body = str(body or "")

    # Stage 1: Personal mobile sender
    if normalized_sender and ANDROID_PERSONAL_MOBILE_SENDER_RE.match(normalized_sender):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="reject_personal_mobile_sender",
            rejection_reason="Sender is a personal mobile number.",
            stage_index=1,
        )

    # Stage 2: Currency amount
    if not ANDROID_CURRENCY_AMOUNT_RE.search(sms_body):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="require_currency_amount",
            rejection_reason="No currency amount detected.",
            stage_index=2,
        )

    # Stage 3: Masked account or card
    if not ANDROID_MASKED_ACCOUNT_OR_CARD_RE.search(sms_body):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="require_masked_account_or_card",
            rejection_reason="No masked account or card found.",
            stage_index=3,
        )

    # Stage 4: Completed transaction verb
    if not ANDROID_TRANSACTION_VERB_RE.search(sms_body):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="require_completed_transaction_verb",
            rejection_reason="No transaction verb detected.",
            stage_index=4,
        )

    # Stage 5: OTP or verification
    if ANDROID_OTP_RE.search(sms_body):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="reject_otp_or_verification",
            rejection_reason="Contains OTP or verification code.",
            stage_index=5,
        )

    # Stage 6: Collect or mandate request
    if ANDROID_COLLECT_OR_MANDATE_REQUEST_RE.search(sms_body):
        return AndroidPrefilterResult(
            passed=False,
            rejection_stage="reject_collect_or_mandate_request",
            rejection_reason="Contains collect or mandate request.",
            stage_index=6,
        )

    return AndroidPrefilterResult(
        passed=True,
        rejection_stage=None,
        rejection_reason=None,
        stage_index=None,
    )


# ==============================================================================
# 2. IOS SMS FILTER SIMULATOR (AlertFilter.swift - ios-eligibility.v2)
# ==============================================================================

IOS_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|₹)",
    re.IGNORECASE,
)
IOS_ACCOUNT_RE = re.compile(
    r"a/c\s*(?:no\.?\s*)?[X*x]+\d+|a/?c\s*(?:no\.?\s*)?\*+\d+|card\s*(?:no\.?\s*)?[Xx*]+\d+|card\s+\d{4}\b|card\s+ending\s+[Xx*]*\d+",
    re.IGNORECASE,
)
IOS_TXN_VERB_RE = re.compile(
    r"\b(?:debited|credited|deducted|spent|paid|received|transferred|sent|reversed|refunded|used|withdrawn|deposited)(?=[^a-zA-Z]|$)|\btxn\b|\bhas\s+(?:a\s+)?debit\s+by\b|\bhas\s+credit\s+for\b|\bwithout\s+OTP\b|\bauto.?debit\b|\bDebit\s+in\s+a/c\b|\btxn\s+of\s+Rs\b|\bRedemption\s+payout\b|\b(?:money\s+transfer|amt\s+sent|amt\s+received)\b|you've\s+hand-?picked",
    re.IGNORECASE,
)
IOS_OTP_RE = re.compile(
    r"\botp\b|\bone.?time.?password\b|\bverification.?code\b",
    re.IGNORECASE,
)
IOS_NON_CREDENTIAL_OTP_RE = re.compile(
    r"\bwithout\s+otp\b",
    re.IGNORECASE,
)
IOS_COLLECT_REQUEST_RE = re.compile(
    r"has\s+requested\s+money|requested\s+Rs\.?|collect\s+request|mandate\s+request|request\s+from\s+you",
    re.IGNORECASE,
)
IOS_PROMOTION_RE = re.compile(
    r"\b(?:exclusive\s+|special\s+)?offers?\b|\bpre.?approved\s+(?:loan|credit)\b|\b(?:shop|apply)\s+now\b|\b(?:avail|get|enjoy)\s+(?:up\s+to\s+)?\d{1,3}%\s+off\b",
    re.IGNORECASE,
)
IOS_UNSUCCESSFUL_TXN_RE = re.compile(
    r"\b(?:transaction|txn|payment|transfer|purchase|withdrawal|debit|credit)\b[^.!?\n]{0,32}\b(?:failed|declined|unsuccessful|cancelled|canceled|rejected|not\s+(?:successful|completed|processed))\b|"
    r"\b(?:not|never)\s+(?:been\s+)?(?:debited|credited|deducted|paid|transferred|sent|received|withdrawn|deposited)\b|"
    r"\b(?:amount|a/c|account|card)\b[^.!?\n]{0,24}\b(?:not|never)\s+(?:been\s+)?(?:debited|credited|deducted)\b|"
    r"\breversed\s+before\s+(?:completion|processing)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class IOSPrefilterResult:
    passed: bool
    decision: str  # "eligible", "needsReview", "rejectAndErase"
    rejection_code: str | None
    completed_stages: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "rejection_code": self.rejection_code,
            "completed_stages": self.completed_stages,
        }


def run_ios_prefilter(sender: str, body: str) -> IOSPrefilterResult:
    """Simulate PocketFinancer iOS AlertFilter.swift."""
    sms_body = str(body or "")
    trimmed_body = sms_body.strip()

    # Stage 1: Non-empty body
    if not trimmed_body:
        return IOSPrefilterResult(
            passed=False,
            decision="rejectAndErase",
            rejection_code="empty_body",
            completed_stages=[],
        )

    completed_stages: list[str] = []

    # Prepare credentialBody (stripping "without otp")
    credential_body = IOS_NON_CREDENTIAL_OTP_RE.sub(" ", sms_body)

    # Check eligibility cues
    has_txn_eligibility_cues = (
        bool(IOS_AMOUNT_RE.search(sms_body))
        and bool(IOS_ACCOUNT_RE.search(sms_body))
        and bool(IOS_TXN_VERB_RE.search(sms_body))
    )

    # Stage 2: OTP / Verification
    if IOS_OTP_RE.search(credential_body):
        return IOSPrefilterResult(
            passed=False,
            decision="rejectAndErase",
            rejection_code="one_time_password",
            completed_stages=completed_stages,
        )
    completed_stages.append("otp")

    # Stage 3: Collect request
    if IOS_COLLECT_REQUEST_RE.search(sms_body):
        return IOSPrefilterResult(
            passed=False,
            decision="rejectAndErase",
            rejection_code="collect_request",
            completed_stages=completed_stages,
        )
    completed_stages.append("collect")

    # Stage 4: Unsuccessful transaction
    if IOS_UNSUCCESSFUL_TXN_RE.search(sms_body):
        return IOSPrefilterResult(
            passed=False,
            decision="rejectAndErase",
            rejection_code="unsuccessful_transaction",
            completed_stages=completed_stages,
        )
    completed_stages.append("outcome")

    # Stage 5: Promotion (unless strong transaction eligibility cues exist)
    if IOS_PROMOTION_RE.search(sms_body) and not has_txn_eligibility_cues:
        return IOSPrefilterResult(
            passed=False,
            decision="rejectAndErase",
            rejection_code="promotion",
            completed_stages=completed_stages,
        )
    completed_stages.append("promotion")

    # Stage 6: Amount
    if not IOS_AMOUNT_RE.search(sms_body):
        return IOSPrefilterResult(
            passed=False,
            decision="needsReview",
            rejection_code="missing_amount",
            completed_stages=completed_stages,
        )
    completed_stages.append("amount")

    # Stage 7: Account
    if not IOS_ACCOUNT_RE.search(sms_body):
        return IOSPrefilterResult(
            passed=False,
            decision="needsReview",
            rejection_code="missing_account",
            completed_stages=completed_stages,
        )
    completed_stages.append("account")

    # Stage 8: Transaction verb
    if not IOS_TXN_VERB_RE.search(sms_body):
        return IOSPrefilterResult(
            passed=False,
            decision="needsReview",
            rejection_code="missing_transaction_verb",
            completed_stages=completed_stages,
        )
    completed_stages.append("verb")

    return IOSPrefilterResult(
        passed=True,
        decision="eligible",
        rejection_code=None,
        completed_stages=completed_stages,
    )


# ==============================================================================
# 3. UNIFIED PLATFORM-NEUTRAL FILTER SIMULATOR
# ==============================================================================

UNIFIED_ACCOUNT_OR_SOURCE_RE = re.compile(
    r"a/c\s*(?:no\.?\s*)?[X*x]+\d+|"
    r"a/?c\s*(?:no\.?\s*)?\*+\d+|"
    r"card\s*(?:no\.?\s*)?[Xx*]+\d+|"
    r"card\s+\d{4}\b|"
    r"card\s+ending\s+[Xx*]*\d+|"
    r"vpa\s+[a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+|"
    r"\b(?:hdfc|sbi|icici|axis|kotak|pnb|bob|paytm|canara|union)\s+bank\s+(?:a/c|card)\b",
    re.IGNORECASE,
)

UNIFIED_UNSUCCESSFUL_TXN_RE = re.compile(
    r"\b(?:transaction|txn|payment|transfer|purchase|withdrawal|debit|credit)\b.{0,48}\b(?:failed|declined|unsuccessful|cancelled|canceled|rejected|timed out|could not be processed|not\s+(?:successful|completed|processed))\b|"
    r"\b(?:not|never)\s+(?:been\s+)?(?:debited|credited|deducted|paid|transferred|sent|received|withdrawn|deposited)\b|"
    r"\b(?:amount|a/c|account|card)\b.{0,36}\b(?:not|never)\s+(?:been\s+)?(?:debited|credited|deducted)\b|"
    r"\breversed\s+before\s+(?:completion|processing)\b|"
    r"\b(?:declined|failed|unsuccessful|cancelled|rejected)\s+due\s+to\b",
    re.IGNORECASE,
)

UNIFIED_BILL_OR_STATEMENT_RE = re.compile(
    r"(?:bill|statement)\s+generated\b|"
    r"(?:payment|bill)\s+due\s+date\b|"
    r"(?:min(?:imum)?|total)\s+amount\s+due\b|"
    r"is\s+due\s+on\b|"
    r"outstanding\s+of\s+Rs",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class UnifiedPrefilterResult:
    passed: bool
    rejection_stage: str | None
    rejection_reason: str | None
    stage_index: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rejection_stage": self.rejection_stage,
            "rejection_reason": self.rejection_reason,
            "stage_index": self.stage_index,
        }


def run_unified_prefilter(sender: str, body: str) -> UnifiedPrefilterResult:
    """
    Unified Platform-Neutral Pre-SLM Triage Pipeline.
    Combines sender validation, explicit hard-negative rejection (failed/declined,
    OTPs, collect requests, statements due, marketing), currency amount,
    account/source identifiers, and transaction verbs.
    """
    normalized_sender = str(sender or "").strip()
    sms_body = str(body or "")
    trimmed_body = sms_body.strip()

    # Stage 1: Non-empty check
    if not trimmed_body:
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="empty_body",
            rejection_reason="Empty or non-text message body.",
            stage_index=1,
        )

    # Stage 2: Reject personal mobile sender
    if normalized_sender and ANDROID_PERSONAL_MOBILE_SENDER_RE.match(normalized_sender):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_personal_sender",
            rejection_reason="Sender is a personal 10-15 digit phone number.",
            stage_index=2,
        )

    # Stage 3: Reject OTP / Authentication (preserving "without OTP" non-credential phrases)
    credential_body = IOS_NON_CREDENTIAL_OTP_RE.sub(" ", sms_body)
    if IOS_OTP_RE.search(credential_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_otp",
            rejection_reason="Message is an OTP or authentication verification code.",
            stage_index=3,
        )

    # Stage 4: Reject collect/mandate request
    if IOS_COLLECT_REQUEST_RE.search(sms_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_collect_request",
            rejection_reason="Message is an unapproved collect or mandate request.",
            stage_index=4,
        )

    # Stage 5: Reject unsuccessful / failed / declined transaction
    if UNIFIED_UNSUCCESSFUL_TXN_RE.search(sms_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_unsuccessful_transaction",
            rejection_reason="Transaction was declined, failed, or cancelled.",
            stage_index=5,
        )

    # Stage 6: Reject bill generated / statement due (without payment settlement)
    if UNIFIED_BILL_OR_STATEMENT_RE.search(sms_body) and not re.search(
        r"(?:thank you for (?:making )?payment|payment received|debited|spent)", sms_body, re.IGNORECASE
    ):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_bill_generated_reminder",
            rejection_reason="Bill or statement generated reminder without payment.",
            stage_index=6,
        )

    # Stage 7: Reject pure marketing / loan offers (unless actual txn cues exist)
    has_txn_cues = (
        bool(ANDROID_CURRENCY_AMOUNT_RE.search(sms_body))
        and bool(UNIFIED_ACCOUNT_OR_SOURCE_RE.search(sms_body))
        and bool(ANDROID_TRANSACTION_VERB_RE.search(sms_body))
    )
    if IOS_PROMOTION_RE.search(sms_body) and not has_txn_cues:
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="reject_promotion",
            rejection_reason="Standalone marketing or loan solicitation.",
            stage_index=7,
        )

    # Stage 8: Require currency amount
    if not ANDROID_CURRENCY_AMOUNT_RE.search(sms_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="require_currency_amount",
            rejection_reason="No monetary currency amount detected.",
            stage_index=8,
        )

    # Stage 9: Require account, card, or source reference
    if not UNIFIED_ACCOUNT_OR_SOURCE_RE.search(sms_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="require_account_or_card",
            rejection_reason="No masked account, card, or source reference found.",
            stage_index=9,
        )

    # Stage 10: Require completed transaction verb
    if not ANDROID_TRANSACTION_VERB_RE.search(sms_body):
        return UnifiedPrefilterResult(
            passed=False,
            rejection_stage="require_transaction_verb",
            rejection_reason="No completed transaction action verb found.",
            stage_index=10,
        )

    return UnifiedPrefilterResult(
        passed=True,
        rejection_stage=None,
        rejection_reason=None,
        stage_index=None,
    )
