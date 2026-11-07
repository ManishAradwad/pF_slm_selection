"""Pure, opt-in pilot policy; not connected to intake or evidence retention.

Contract: conservative-prefilter-v1. Only whole ASCII messages matching the
closed templates below are excluded. ASCII case and runs of space/tab/CR/LF
are ignored for matching only. Non-ASCII and all other messages pass. Sender,
analyzer, amount/account matches and model output are deliberately not inputs.
The caller retains the unchanged source; this module neither stores nor erases it.
The invented shared vectors specify mechanics, not real-data exclusion recall.
"""

from dataclasses import dataclass
import re
from typing import Literal

POLICY_ID = "conservative-prefilter-v1"

_VERIFICATION = re.compile(
    r"your (?:otp|login verification code) is [0-9]{4,8}\. do not share it\."
)
_PROMOTION = re.compile(
    r"advertisement: save [0-9]{1,2}% in our weekend sale\. shop now\."
)
_ASCII_WHITESPACE = re.compile(r"[ \t\r\n]+")


@dataclass(frozen=True)
class PrefilterResult:
    decision: Literal["pass", "exclude"]
    reason: Literal["uncertain", "standalone_verification", "standalone_promotion"]


def prefilter(body: str) -> PrefilterResult:
    """Return a safe reason only; pass is not a transaction classification."""
    if body.isascii():
        normalized = _ASCII_WHITESPACE.sub(" ", body.lower()).strip(" ")
        if _VERIFICATION.fullmatch(normalized):
            return PrefilterResult("exclude", "standalone_verification")
        if _PROMOTION.fullmatch(normalized):
            return PrefilterResult("exclude", "standalone_promotion")
    return PrefilterResult("pass", "uncertain")
