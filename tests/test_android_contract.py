from __future__ import annotations

import json

import pytest

from DATA.utils import FEW_SHOT_EXAMPLES, doc_to_text
from lfm25.android_contract import (
    ANDROID_CONTRACT_VERSION,
    ANDROID_DECODE_DEFAULTS,
    ANDROID_OUTER_SYSTEM_PROMPT,
    ANDROID_PROMPT_PROXY_DECODE_DEFAULTS,
    ANDROID_SOURCE_REVISION,
    LFM25_350M_LOCKED_PROMPT_TOKENS,
    POCKETFINANCER_CONTRACT_ALIASES,
    android_extraction_messages,
    android_prompt_sha256,
    android_raw_prompt,
    context_compatibility,
    contract_provenance,
    decode_defaults,
    pocketfinancer_normalize_prediction,
    pocketfinancer_parse_prediction,
    pocketfinancer_prefilter_sms,
    prefilter_sms,
    selection_prefilter_sms,
    should_apply_prefilter,
    summarize_prefilter,
)


TXN_SMS = "INR 125.50 debited from A/c XX1234 at DEMO STORE."
TXN_GOLD = json.dumps(
    {
        "amount": 125.5,
        "counterparty": "DEMO STORE",
        "type": "debit",
        "account": "A/c XX1234",
    }
)


def test_android_messages_reproduce_the_raw_prompt_and_outer_roles_exactly() -> None:
    sender = "AX-TESTBK"
    sms = "Synthetic notice: \u20b91.25 credited to Card XX0000."

    raw_prompt = android_raw_prompt(sender, sms)
    messages = android_extraction_messages(sender, sms)

    assert raw_prompt == doc_to_text({"sender": sender, "sms": sms})
    assert messages == [
        {"role": "system", "content": ANDROID_OUTER_SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt},
    ]
    assert len(FEW_SHOT_EXAMPLES) == 7
    for example in FEW_SHOT_EXAMPLES:
        assert f"Sender: {example['sender']}" in raw_prompt
        assert f"Output: {example['answer']}" in raw_prompt
    assert raw_prompt.endswith(f"Sender: {sender}\nSMS: {sms}\nOutput: ")


def test_android_current_provenance_records_prompt_grammar_and_source_snapshot() -> None:
    provenance = contract_provenance()

    assert ANDROID_CONTRACT_VERSION == 3
    assert provenance["name"] == "pocketfinancer"
    assert provenance["version"] == 3
    assert provenance["profile"] == "pocketfinancer"
    assert provenance["android_prompt_template_sha256"] == (
        provenance["prompt_template_sha256"]
    )
    assert provenance["system_prompt_sha256"] == (
        "16e042a07a18165e1cd0b1c0d0cd3bcee67f64825df8adc74e568b3eadffd64a"
    )
    assert provenance["few_shot_examples_sha256"] == (
        "ea4e57c646f2232b5e1d24c1211b8ee6ac68cfef2f69c52ac9ae3765116749aa"
    )
    assert provenance["assets"]["sms_extraction.gbnf"] == {
        "bytes": 515,
        "sha256": "c321daca16ea3dbdf4269c6504f7cbab5e587d1ce849e3b79133e5449d1c7939",
    }
    assert ANDROID_SOURCE_REVISION == "a9b7df44be2183daac3a05cadbfd40b8f309cd4b"
    assert provenance["android_source"]["revision"] == ANDROID_SOURCE_REVISION
    assert provenance["android_source"]["hash_basis"] == (
        "sha256_of_git_blob_content_at_revision"
    )
    assert provenance["android_source"]["files"] == provenance["android_source"][
        "source_sha256"
    ]
    assert provenance["android_source"]["files"][
        "pipeline/src/main/java/com/pocketfinancer/pipeline/SmsFilterPipeline.kt"
    ] == "2df07947474177729659cb2c4db0d4828fce63e02b5aae915f8972899ef51999"
    assert len(android_prompt_sha256("AX-TESTBK", TXN_SMS)) == 64


def test_android_current_defaults_are_exact_and_proxy_is_distinct() -> None:
    expected = {
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
    assert ANDROID_DECODE_DEFAULTS == expected
    for profile in POCKETFINANCER_CONTRACT_ALIASES:
        assert decode_defaults(profile) == expected
    assert decode_defaults("android-prompt-proxy") == {
        "n_ctx": 3072,
        "max_tokens": 256,
        "repeat_penalty": 1.0,
        "grammar": False,
        "two_pass": False,
        "prefilter": True,
    }
    assert ANDROID_PROMPT_PROXY_DECODE_DEFAULTS != ANDROID_DECODE_DEFAULTS


def test_locked_lfm_prompt_range_and_direct_answer_fit_android_context() -> None:
    assert LFM25_350M_LOCKED_PROMPT_TOKENS["minimum"] == 1862
    assert LFM25_350M_LOCKED_PROMPT_TOKENS["maximum"] == 2017
    compatibility = context_compatibility(
        LFM25_350M_LOCKED_PROMPT_TOKENS["minimum"],
        n_ctx=ANDROID_DECODE_DEFAULTS["n_ctx"],
        generation_tokens=ANDROID_DECODE_DEFAULTS["answer_max_tokens"],
    )
    assert compatibility == {
        "compatible": True,
        "prompt_tokens": 1862,
        "generation_budget_tokens": 256,
        "required_tokens": 2118,
        "n_ctx": 3072,
        "overflow_tokens": 0,
    }
    evidence = contract_provenance()["context_compatibility"]
    assert evidence["known_lfm25_350m_locked_prompt_tokens"]["maximum"] == 2017
    assert evidence["known_lfm25_350m_locked_prompt_tokens"][
        "all_prompts_and_direct_answers_fit_android_n_ctx"
    ] is True


@pytest.mark.parametrize(
    ("sender", "sms", "stage", "index"),
    [
        ("+919876543210", TXN_SMS, "personal_mobile_sender", 1),
        ("AX-TESTBK", "A/c XX1234 was debited.", "currency_amount", 2),
        ("AX-TESTBK", "Rs.125.50 was debited.", "masked_account_or_card", 3),
        ("AX-TESTBK", "Rs.125.50 is available on Card XX1234.", "transaction_verb", 4),
        ("AX-TESTBK", "OTP 123456: Rs.125.50 debited from A/c XX1234.", "otp", 5),
        (
            "AX-TESTBK",
            "Rs.125.50 debited from A/c XX1234; a collect request from you is pending.",
            "collect_or_mandate_request",
            6,
        ),
    ],
)
def test_current_pocketfinancer_prefilter_rejects_in_stage_order(
    sender: str,
    sms: str,
    stage: str,
    index: int,
) -> None:
    result = pocketfinancer_prefilter_sms(sender, sms)

    assert not result.accepted
    assert result.rejection_stage == stage
    assert result.stage_index == index
    assert result.rejection_reason


@pytest.mark.parametrize(
    "sms",
    [
        TXN_SMS,
        "Your Card 0816 has credit for 1,250 INR.",
        "Txn Rs.75 used on card ending **9876.",
        "Rs 42 withdrawn from a/c no. *4321.",
        "\u20b942 withdrawn from a/c no. *4321.",
    ],
)
def test_current_pocketfinancer_prefilter_accepts_transaction_shapes(sms: str) -> None:
    result = pocketfinancer_prefilter_sms("AX-TESTBK", sms)

    assert result.accepted
    assert prefilter_sms("AX-TESTBK", sms) == result
    assert selection_prefilter_sms("AX-TESTBK", sms) == result


def test_pocketfinancer_parser_matches_app_json_coercion_and_think_stripping() -> None:
    raw = (
        '<think>{"decoy": true}</think> prefix '
        '{"amount":"INR 1,250.50","type":"CREDIT","account":"Card XX1234",'
        '"counterparty":"null","ignored":"extra fields are accepted"} suffix'
    )

    parsed = pocketfinancer_parse_prediction(raw)

    assert parsed.status == "transaction"
    assert parsed.error is None
    assert parsed.value == {
        "amount": 1250.5,
        "counterparty": None,
        "type": "credit",
        "account": "Card XX1234",
    }
    assert json.loads(parsed.extracted) == parsed.value
    assert pocketfinancer_normalize_prediction(raw) == parsed.extracted


@pytest.mark.parametrize("raw", ["null", " NULL\n", "\tNuLl "])
def test_pocketfinancer_parser_accepts_case_insensitive_literal_null(raw: str) -> None:
    parsed = pocketfinancer_parse_prediction(raw)

    assert parsed.status == "null"
    assert parsed.value is None
    assert pocketfinancer_normalize_prediction(raw) == "null"


@pytest.mark.parametrize(
    "raw",
    [
        '{"amount":0,"type":"debit","account":"A/c XX1234"}',
        '{"amount":10,"type":"transfer","account":"A/c XX1234"}',
        '{"amount":10,"type":"debit","account":null}',
        (
            '<THINK>{"decoy":true}</THINK>'
            '{"amount":10,"type":"debit","account":"A/c XX1234"}'
        ),
        "no JSON object here",
    ],
)
def test_pocketfinancer_parser_collapses_app_rejections_to_null(raw: str) -> None:
    parsed = pocketfinancer_parse_prediction(raw)

    assert parsed.status == "null"
    assert parsed.value is None
    assert parsed.error
    assert pocketfinancer_normalize_prediction(raw) == "null"


def test_pocketfinancer_profiles_enable_prefilter_by_default_and_allow_ablation() -> None:
    for profile in POCKETFINANCER_CONTRACT_ALIASES:
        assert should_apply_prefilter(profile, None)
    assert should_apply_prefilter("android-prompt-proxy", None)
    assert not should_apply_prefilter("legacy", None)
    assert not should_apply_prefilter("pocketfinancer", False)
    assert should_apply_prefilter("legacy", True)

    provenance = contract_provenance()
    assert provenance["pipeline"]["queued_sms_prefilter"]
    assert provenance["selection_prefilter"]["profile"] == "pocketfinancer"
    assert provenance["selection_prefilter"]["part_of_android_current"]


def test_prefilter_summary_reports_the_app_selection_boundary() -> None:
    records = [
        {"gold": TXN_GOLD, "prefilter_passed": True},
        {
            "gold": TXN_GOLD,
            "prefilter_passed": False,
            "prefilter_rejection_stage": "masked_account_or_card",
        },
        {
            "gold": "null",
            "prefilter_passed": False,
            "prefilter_rejection_stage": "otp",
        },
        {"gold": "null", "prefilter_passed": True},
    ]

    summary = summarize_prefilter(records, enabled=True)

    assert summary["n"] == 4
    assert summary["model_invocations"] == 2
    assert summary["transaction_recall"] == 0.5
    assert summary["null_rejection_rate"] == 0.5
