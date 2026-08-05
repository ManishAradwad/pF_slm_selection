"""PocketFinancer prompt contract and current Android runtime configuration.

The current app runs the ordered six-stage prefilter before inference. Models
without thinking support use one 256-token answer pass; thinking models use a
1,024-token unconstrained thinking pass and a 256-token answer pass. Both run
inside a 3,072-token CPU context. GBNF is an optional user preference and is
disabled by default.

The app sends a generic outer system message and places the long extraction
instruction, all seven demonstrations, and the current SMS in the user message.
Its native runtime applies the GGUF's built-in chat template when available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from DATA.utils import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, doc_to_text
from lfm25.contract import ParsedOutput, parse_gold


POCKETFINANCER_CONTRACT = "pocketfinancer"
POCKETFINANCER_CONTRACT_ALIASES = frozenset(
    {POCKETFINANCER_CONTRACT, "android", "android-current", "android-runtime"}
)
ANDROID_CONTRACT_NAME = POCKETFINANCER_CONTRACT
ANDROID_CONTRACT_VERSION = 3
ANDROID_OUTER_SYSTEM_PROMPT = "You are a helpful financial SMS extraction assistant."
ANDROID_SOURCE_REVISION = "a9b7df44be2183daac3a05cadbfd40b8f309cd4b"

# SHA-256 of canonical git blob bytes at ANDROID_SOURCE_REVISION. These are
# upstream evidence, not hashes of the Python approximation in this repository.
ANDROID_SOURCE_SHA256 = {
    "pipeline/src/main/java/com/pocketfinancer/pipeline/PromptBuilder.kt": (
        "316c9df493184c6ae67e38d837363f8f94ec470a3c867d34b67b7ed97896bdfa"
    ),
    "pipeline/src/main/java/com/pocketfinancer/pipeline/PipelineService.kt": (
        "1a2587be76f45263254696d021045e1d52acfab53f2ab555ddf1eeb84578109e"
    ),
    "pipeline/src/main/java/com/pocketfinancer/pipeline/SmsFilterPipeline.kt": (
        "2df07947474177729659cb2c4db0d4828fce63e02b5aae915f8972899ef51999"
    ),
    "pipeline/src/main/java/com/pocketfinancer/pipeline/SlmProcessingPreferences.kt": (
        "a795a4d90686bb37c402e63c134eb71906004de448061fa43dfc1d73a4041d7e"
    ),
    "pipeline/src/main/java/com/pocketfinancer/pipeline/ExtractionParser.kt": (
        "7d394a6cbd1097b1cecc71a463450b9489a88ca8b9a75a90edb064e32b10091c"
    ),
    "inference/src/main/java/com/pocketfinancer/inference/LlamaEngine.kt": (
        "7210c81268390fa261cd1b95d68b749e9344497c7f945035c49657c48055fee9"
    ),
    "inference/src/main/java/com/pocketfinancer/inference/SlmRuntime.kt": (
        "f4b1bbc2dd538fab79d8766f69639c5da89f5ac34aab76bc2f65742d1ba9fbf4"
    ),
    "inference/src/main/cpp/llama_jni.cpp": (
        "c80fe00d19f2005e57a87931be3f307d459775a05c129d1adfc85b8f6f2a7dde"
    ),
    "hardware/src/main/java/com/pocketfinancer/hardware/SlmSelector.kt": (
        "03c53546fab7848a77be951010fc573f40e547408c2adf30f18b974caf046e37"
    ),
    "app/src/main/java/com/pocketfinancer/SlmModelSpecs.kt": (
        "49fa500801b8e5ef0129244a6ef9863c985627fe42d762976cbedfd729f848c0"
    ),
    "inference/src/main/assets/system_prompt.txt": (
        "16e042a07a18165e1cd0b1c0d0cd3bcee67f64825df8adc74e568b3eadffd64a"
    ),
    "inference/src/main/assets/few_shot_examples.json": (
        "ea4e57c646f2232b5e1d24c1211b8ee6ac68cfef2f69c52ac9ae3765116749aa"
    ),
    "inference/src/main/assets/sms_extraction.gbnf": (
        "c321daca16ea3dbdf4269c6504f7cbab5e587d1ce849e3b79133e5449d1c7939"
    ),
}

ANDROID_DECODE_DEFAULTS: dict[str, Any] = {
    "n_ctx": 3072,
    "thinking_max_tokens": 1024,
    "answer_max_tokens": 256,
    "temperature": 0.0,
    "repeat_penalty": 1.0,
    "thinking_mode": "model_config",
    "grammar": False,
    "prefilter": True,
    "n_gpu_layers": 0,
    "max_cpu_threads": 4,
    "n_batch": 512,
    "n_ubatch": 256,
    "flash_attention": True,
    "use_mmap": False,
    "chat_template": "gguf_builtin_primary",
}

# Transformers cannot attach the app's llama.cpp GBNF grammar. This profile is
# useful for prompt/training diagnostics, but is not Android runtime parity.
ANDROID_PROMPT_PROXY_DECODE_DEFAULTS: dict[str, Any] = {
    "n_ctx": 3072,
    "max_tokens": 256,
    "repeat_penalty": 1.0,
    "grammar": False,
    "two_pass": False,
    "prefilter": True,
}

# Measured with the locked 203-row regression set and locally locked
# LFM2.5-350M tokenizer. Every prompt plus a direct answer budget fits n_ctx.
LFM25_350M_LOCKED_PROMPT_TOKENS: dict[str, Any] = {
    "dataset_rows": 203,
    "minimum": 1862,
    "maximum": 2017,
    "mean": 1899.832512,
    "android_n_ctx": 3072,
    "all_prompts_exceed_android_n_ctx": False,
    "all_prompts_and_direct_answers_fit_android_n_ctx": True,
}

# These are the evaluator settings used before the Android contract was added.
LEGACY_DECODE_DEFAULTS: dict[str, Any] = {
    "n_ctx": 2048,
    "max_tokens": 96,
    "repeat_penalty": 1.05,
    "grammar": True,
}


# This is the current Android SmsFilterPipeline, which runs before model
# inference. The one-based stage indices preserve this repository's API.
PERSONAL_MOBILE_SENDER_RE = re.compile(r"^\+?[0-9]{10,15}$")
CURRENCY_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|\u20b9)\s*[\d,]+(?:\.\d{1,2})?|"
    r"[\d,]+(?:\.\d{1,2})?\s*(?:rs\.?|inr|\u20b9)",
    re.IGNORECASE,
)
MASKED_ACCOUNT_OR_CARD_RE = re.compile(
    r"a/c\s*(?:no\.?\s*)?[X*x]+\d+|"
    r"a/?c\s*(?:no\.?\s*)?\*+\d+|"
    r"card\s*(?:no\.?\s*)?[Xx*]+\d+|"
    r"card\s+\d{4}\b|"
    r"card\s+ending\s+[Xx*]*\d+",
    re.IGNORECASE,
)
TRANSACTION_VERB_RE = re.compile(
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
OTP_RE = re.compile(
    r"\botp\b|\bone.?time.?password\b|\bverification.?code\b",
    re.IGNORECASE,
)
COLLECT_OR_MANDATE_REQUEST_RE = re.compile(
    r"has\s+requested\s+money|"
    r"requested\s+Rs\.?|"
    r"collect\s+request|"
    r"mandate\s+request|"
    r"request\s+from\s+you",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SelectionPrefilterResult:
    """Disposition from PocketFinancer's current six-stage prefilter."""

    accepted: bool
    rejection_stage: str | None = None
    rejection_reason: str | None = None
    stage_index: int | None = None

    @property
    def passed(self) -> bool:
        return self.accepted

    @property
    def stage(self) -> str | None:
        return self.rejection_stage

    @property
    def reason(self) -> str | None:
        return self.rejection_reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_stage": self.rejection_stage,
            "rejection_reason": self.rejection_reason,
            "stage_index": self.stage_index,
        }


def _rejected(index: int, stage: str, reason: str) -> SelectionPrefilterResult:
    return SelectionPrefilterResult(
        accepted=False,
        rejection_stage=stage,
        rejection_reason=reason,
        stage_index=index,
    )


def pocketfinancer_prefilter_sms(sender: str, sms: str) -> SelectionPrefilterResult:
    """Run the current PocketFinancer pre-inference SMS filter."""

    normalized_sender = str(sender).strip()
    body = str(sms)

    if PERSONAL_MOBILE_SENDER_RE.fullmatch(normalized_sender):
        return _rejected(1, "personal_mobile_sender", "sender is a personal 10-15 digit number")
    if CURRENCY_AMOUNT_RE.search(body) is None:
        return _rejected(2, "currency_amount", "SMS has no currency-denominated amount")
    if MASKED_ACCOUNT_OR_CARD_RE.search(body) is None:
        return _rejected(3, "masked_account_or_card", "SMS has no masked account or card")
    if TRANSACTION_VERB_RE.search(body) is None:
        return _rejected(4, "transaction_verb", "SMS has no completed-transaction verb")
    if OTP_RE.search(body) is not None:
        return _rejected(5, "otp", "SMS is an OTP or verification message")
    if COLLECT_OR_MANDATE_REQUEST_RE.search(body) is not None:
        return _rejected(
            6,
            "collect_or_mandate_request",
            "SMS is a collect or mandate request",
        )
    return SelectionPrefilterResult(accepted=True)


# Compatibility aliases for existing evaluator/curriculum code. New code should
# use the PocketFinancer-named entry point.
PrefilterResult = SelectionPrefilterResult


def selection_prefilter_sms(sender: str, sms: str) -> SelectionPrefilterResult:
    """Compatibility alias for :func:`pocketfinancer_prefilter_sms`."""

    return pocketfinancer_prefilter_sms(sender, sms)


def prefilter_sms(sender: str, sms: str) -> SelectionPrefilterResult:
    """Compatibility alias for :func:`pocketfinancer_prefilter_sms`."""

    return pocketfinancer_prefilter_sms(sender, sms)


def _pocketfinancer_nullish(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() == "null"
    )


def _pocketfinancer_opt_string(value: Any, default: str = "") -> str:
    """Approximate JSONObject.optString while preserving useful scalar values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def pocketfinancer_parse_prediction(raw_output: str) -> ParsedOutput:
    """Interpret model text exactly as the app's ExtractionParser does.

    Android deliberately collapses malformed output and a literal ``null`` to
    the same no-transaction result. The returned ``error`` preserves that
    distinction for diagnostics while ``status`` reflects what reaches the app.
    """

    if not isinstance(raw_output, str):
        return ParsedOutput("null", None, "", "app parser rejected non-text output")

    # Kotlin's Regex is case-sensitive here and removes complete think blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    if cleaned.lower() == "null":
        return ParsedOutput("null", None, "null")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return ParsedOutput(
            "null", None, cleaned, "app parser found no JSON object"
        )
    candidate = cleaned[start : end + 1]
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return ParsedOutput("null", None, candidate, "app parser rejected invalid JSON")
    if not isinstance(value, dict):
        return ParsedOutput("null", None, candidate, "app parser expected an object")

    for field in ("amount", "type", "account"):
        if _pocketfinancer_nullish(value.get(field)):
            return ParsedOutput(
                "null", None, candidate, f"app parser rejected nullish {field}"
            )

    raw_amount = value.get("amount")
    amount: float | None
    if isinstance(raw_amount, bool):
        amount = None
    elif isinstance(raw_amount, (int, float)):
        amount = float(raw_amount)
    elif isinstance(raw_amount, str):
        numeric = re.sub(r"[^0-9.]", "", raw_amount).lstrip(".")
        try:
            amount = float(numeric)
        except ValueError:
            amount = None
    else:
        amount = None
    if amount is None or not math.isfinite(amount) or amount <= 0:
        return ParsedOutput("null", None, candidate, "app parser rejected amount")

    tx_type = _pocketfinancer_opt_string(value.get("type")).strip().lower()
    if tx_type not in {"debit", "credit"}:
        return ParsedOutput("null", None, candidate, "app parser rejected type")

    account_text = _pocketfinancer_opt_string(value.get("account"))
    account = account_text if account_text.strip() else None
    counterparty_text = _pocketfinancer_opt_string(value.get("counterparty"))
    counterparty = (
        counterparty_text
        if counterparty_text.strip()
        and counterparty_text.strip().lower() != "null"
        else None
    )
    normalized = {
        "amount": amount,
        "counterparty": counterparty,
        "type": tx_type,
        "account": account,
    }
    return ParsedOutput(
        "transaction",
        normalized,
        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
    )


def pocketfinancer_normalize_prediction(raw_output: str) -> str:
    """Return the transaction JSON or ``null`` seen after Android parsing."""

    parsed = pocketfinancer_parse_prediction(raw_output)
    return parsed.extracted if parsed.status == "transaction" else "null"


def android_raw_prompt(sender: str, sms: str) -> str:
    """Return the exact raw user prompt built by Android PromptBuilder."""

    return doc_to_text({"sender": str(sender), "sms": str(sms)})


def android_extraction_messages(sender: str, sms: str) -> list[dict[str, str]]:
    """Return Android's two chat messages before model-template rendering."""

    return [
        {"role": "system", "content": ANDROID_OUTER_SYSTEM_PROMPT},
        {"role": "user", "content": android_raw_prompt(sender, sms)},
    ]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def android_prompt_sha256(sender: str, sms: str) -> str:
    """Fingerprint one raw Android prompt, including its sender and SMS."""

    return _sha256_text(android_raw_prompt(sender, sms))


def decode_defaults(contract: str) -> dict[str, Any]:
    """Return a mutable copy of the selected evaluator defaults."""

    if contract in POCKETFINANCER_CONTRACT_ALIASES:
        return dict(ANDROID_DECODE_DEFAULTS)
    if contract == "android-prompt-proxy":
        return dict(ANDROID_PROMPT_PROXY_DECODE_DEFAULTS)
    if contract == "legacy":
        return dict(LEGACY_DECODE_DEFAULTS)
    raise ValueError(f"unknown evaluation contract: {contract!r}")


def should_apply_prefilter(contract: str, requested: bool | None) -> bool:
    """Resolve PocketFinancer preprocessing or an explicit ablation.

    The current app applies it by default. ``--no-apply-prefilter`` is an
    explicit evaluator ablation; legacy decoding retains its historical default.
    """

    if requested is not None:
        return requested
    if contract in POCKETFINANCER_CONTRACT_ALIASES or contract == "android-prompt-proxy":
        return True
    if contract == "legacy":
        return False
    raise ValueError(f"unknown evaluation contract: {contract!r}")


def context_compatibility(
    prompt_tokens: int,
    *,
    n_ctx: int,
    generation_tokens: int,
) -> dict[str, Any]:
    """Describe whether one Android-style completion request fits its context."""

    required = prompt_tokens + generation_tokens
    return {
        "compatible": required <= n_ctx,
        "prompt_tokens": prompt_tokens,
        "generation_budget_tokens": generation_tokens,
        "required_tokens": required,
        "n_ctx": n_ctx,
        "overflow_tokens": max(0, required - n_ctx),
    }


def summarize_prefilter(
    records: Iterable[Mapping[str, Any]],
    *,
    enabled: bool,
    gold_key: str = "gold",
) -> dict[str, Any]:
    """Report selection-prefilter recall/rejections independently of model quality."""

    counts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    for record in records:
        gold_transaction = parse_gold(record.get(gold_key)) is not None
        passed = bool(record.get("prefilter_passed", True))
        counts["rows"] += 1
        counts["passed"] += int(passed)
        counts["rejected"] += int(not passed)
        counts["gold_transactions"] += int(gold_transaction)
        counts["gold_nulls"] += int(not gold_transaction)
        counts["transactions_passed"] += int(gold_transaction and passed)
        counts["transactions_rejected"] += int(gold_transaction and not passed)
        counts["nulls_passed"] += int(not gold_transaction and passed)
        counts["nulls_rejected"] += int(not gold_transaction and not passed)
        if not passed:
            stages[str(record.get("prefilter_rejection_stage") or "unknown")] += 1

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    return {
        "enabled": enabled,
        "n": counts["rows"],
        "model_invocations": counts["passed"],
        "rejected": counts["rejected"],
        "rejection_rate": ratio(counts["rejected"], counts["rows"]),
        "gold_transactions": counts["gold_transactions"],
        "transactions_passed": counts["transactions_passed"],
        "transactions_rejected": counts["transactions_rejected"],
        "transaction_recall": ratio(
            counts["transactions_passed"], counts["gold_transactions"]
        ),
        "gold_nulls": counts["gold_nulls"],
        "nulls_rejected": counts["nulls_rejected"],
        "null_rejection_rate": ratio(counts["nulls_rejected"], counts["gold_nulls"]),
        "rejections_by_stage": dict(sorted(stages.items())),
    }


def contract_provenance() -> dict[str, Any]:
    """Return prompt hashes and evidence for the current Android runtime."""

    examples_asset = json.dumps(FEW_SHOT_EXAMPLES, ensure_ascii=False, indent=2)
    template_prompt = android_raw_prompt("{sender}", "{sms}")
    stage_patterns = [
        PERSONAL_MOBILE_SENDER_RE.pattern,
        CURRENCY_AMOUNT_RE.pattern,
        MASKED_ACCOUNT_OR_CARD_RE.pattern,
        TRANSACTION_VERB_RE.pattern,
        OTP_RE.pattern,
        COLLECT_OR_MANDATE_REQUEST_RE.pattern,
    ]
    repo_root = Path(__file__).resolve().parents[1]
    selection_source_hashes = {}
    for relative in ("DATA/utils.py", "lfm25/android_contract.py"):
        selection_source_hashes[relative] = hashlib.sha256(
            (repo_root / relative).read_bytes()
        ).hexdigest()

    system_hash = _sha256_text(SYSTEM_PROMPT)
    examples_hash = _sha256_text(examples_asset)
    grammar_bytes = (repo_root / "DATA/sms_extraction.gbnf").read_bytes()
    return {
        "name": ANDROID_CONTRACT_NAME,
        "version": ANDROID_CONTRACT_VERSION,
        "profile": POCKETFINANCER_CONTRACT,
        "chat_roles": ["system", "user"],
        "few_shot_example_count": len(FEW_SHOT_EXAMPLES),
        "outer_system_prompt_sha256": _sha256_text(ANDROID_OUTER_SYSTEM_PROMPT),
        "system_prompt_sha256": system_hash,
        "few_shot_examples_sha256": examples_hash,
        "prompt_template_sha256": _sha256_text(template_prompt),
        "android_prompt_template_sha256": _sha256_text(template_prompt),
        "runtime_parity_scope": "prompt_messages_and_declared_android_runtime_configuration",
        "pipeline": {
            "queued_sms_prefilter": True,
            "n_gpu_layers": 0,
            "n_ctx": 3072,
            "n_batch": 512,
            "n_ubatch": 256,
            "max_cpu_threads": 4,
            "flash_attention": True,
            "memory_map": False,
            "chat_template": "gguf_builtin_primary_with_qwen3_fallback",
            "temperature": 0.0,
            "repeat_penalty": 1.0,
            "inference_passes": [
                {
                    "name": "thinking",
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "grammar_constrained": False,
                    "stop": "</think>",
                    "when": "model.hasThinkingMode",
                },
                {
                    "name": "answer",
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "grammar_constrained": "optional_preference_default_false",
                    "stop": None,
                    "when": "always; direct first pass for non-thinking models",
                },
            ],
        },
        "context_compatibility": {
            "known_lfm25_350m_locked_prompt_tokens": dict(
                LFM25_350M_LOCKED_PROMPT_TOKENS
            ),
            "note": (
                "The current 3072-token Android context holds the locked "
                "LFM2.5-350M prompts and its direct 256-token answer budget."
            ),
        },
        "selection_prefilter": {
            "name": "pocketfinancer_six_stage_prefilter",
            "profile": POCKETFINANCER_CONTRACT,
            "part_of_android_current": True,
            "sha256": _sha256_text(
                json.dumps(stage_patterns, ensure_ascii=False, separators=(",", ":"))
            ),
        },
        "assets": {
            "system_prompt.txt": {
                "bytes": len(SYSTEM_PROMPT.encode("utf-8")),
                "sha256": system_hash,
            },
            "few_shot_examples.json": {
                "bytes": len(examples_asset.encode("utf-8")),
                "sha256": examples_hash,
            },
            "sms_extraction.gbnf": {
                "bytes": len(grammar_bytes),
                "sha256": hashlib.sha256(grammar_bytes).hexdigest(),
            },
        },
        "decode_defaults": dict(ANDROID_DECODE_DEFAULTS),
        "android_source": {
            "repository": "https://github.com/ManishAradwad/pocket-financer-android",
            "revision": ANDROID_SOURCE_REVISION,
            "hash_basis": "sha256_of_git_blob_content_at_revision",
            "files": dict(ANDROID_SOURCE_SHA256),
            "source_sha256": dict(ANDROID_SOURCE_SHA256),
        },
        "selection_repo_source_sha256": selection_source_hashes,
    }
