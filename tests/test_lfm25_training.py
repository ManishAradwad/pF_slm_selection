import json
import math

import pytest

from lfm25.android_contract import android_extraction_messages
from lfm25.candidates import candidate_selector_messages
from lfm25.training_loss import normalized_completion_cross_entropy
from scripts.train_lfm25_lora import (
    ANDROID_CONTEXT_LENGTH,
    LEGACY_MAX_LENGTH,
    CompletionCollator,
    CompletionDataset,
    OverlengthDatasetError,
    _messages,
    parse_args,
)


SELECTOR_SMS = (
    "INR 23.00 spent on your Credit Card ending XX0000 at DEMO SHOP DAILY "
    "on 03-JAN. Avbl limit: Rs.10000.00."
)
SELECTOR_GOLD = {
    "amount": 23.0,
    "counterparty": "DEMO SHOP DAILY",
    "type": "debit",
    "account": "Credit Card ending XX0000",
}


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
    ):
        assert tokenize is True
        assert return_tensors is None
        has_completion = bool(messages and messages[-1]["role"] == "assistant")
        prompt_messages = messages[:-1] if has_completion else messages
        prompt_width = 2 + sum(len(item["content"].split()) for item in prompt_messages)
        prompt = list(range(1, prompt_width + 1))
        if not has_completion:
            assert add_generation_prompt is True
            return prompt
        assert add_generation_prompt is False
        completion_width = max(1, len(messages[-1]["content"].split()))
        return prompt + list(range(100, 100 + completion_width)) + [199]


def _required_cli() -> list[str]:
    return ["--train", "train.jsonl", "--eval", "eval.jsonl", "--output-dir", "out"]


def test_profile_defaults_use_pocketfinancer_and_keep_historical_modes_explicit() -> None:
    default = parse_args(_required_cli())
    assert default.prompt_profile == "android"
    assert default.max_length == ANDROID_CONTEXT_LENGTH
    assert default.first_supervised_token_weight == 3.0
    android = parse_args([*_required_cli(), "--contract", "android"])
    assert android.prompt_profile == "android"
    assert android.max_length == ANDROID_CONTEXT_LENGTH
    selector = parse_args([*_required_cli(), "--contract", "candidate_selector"])
    assert selector.max_length == LEGACY_MAX_LENGTH
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_required_cli(),
                "--contract",
                "android",
                "--max-length",
                str(ANDROID_CONTEXT_LENGTH + 1),
            ]
        )


def test_android_and_selector_profiles_build_their_exact_messages() -> None:
    android_row = {"sender": "AX-BANKXX", "sms": "A message", "expected": None}
    android = _messages(android_row, "android")
    assert android[:2] == android_extraction_messages("AX-BANKXX", "A message")
    assert android[-1] == {"role": "assistant", "content": "null"}

    selector_row = {
        "sender": "VD-IDFCFB",
        "sms": SELECTOR_SMS,
        "expected": SELECTOR_GOLD,
    }
    selector = _messages(selector_row, "candidate_selector")
    assert selector[:2] == candidate_selector_messages("VD-IDFCFB", SELECTOR_SMS)
    target = json.loads(selector[-1]["content"])
    assert set(target) == {"transaction", "type", "amount", "account", "counterparty"}
    assert target["transaction"] == 1


def test_dataset_and_collator_mask_prompt_padding_and_carry_sample_weight(tmp_path) -> None:
    rows = [
        {"sender": "AX-BANKXX", "sms": "short text", "expected": None, "sample_weight": 2.5},
        {
            "sender": "AX-BANKXX",
            "sms": "a somewhat longer source message for padding",
            "expected": None,
        },
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = CompletionDataset(path, FakeTokenizer(), max_length=100)

    assert dataset.stats["overlength_rows"] == 0
    assert dataset.stats["sample_weight_explicit_rows"] == 1
    for feature in dataset.features:
        first_supervised = feature["labels"].index(next(x for x in feature["labels"] if x != -100))
        assert all(label == -100 for label in feature["labels"][:first_supervised])
        assert feature["labels"][first_supervised:] == feature["input_ids"][first_supervised:]

    pytest.importorskip("torch")
    batch = CompletionCollator(FakeTokenizer())(dataset.features)
    assert batch["sample_weight"].tolist() == [2.5, 1.0]
    short_length = len(dataset.features[0]["input_ids"])
    assert batch["attention_mask"][0, short_length:].eq(0).all()
    assert batch["labels"][0, short_length:].eq(-100).all()


def test_dataset_reports_all_overlength_rows_without_truncating(tmp_path) -> None:
    rows = [
        {"sender": "AX-BANKXX", "sms": "first message", "expected": None},
        {"sender": "AX-BANKXX", "sms": "second longer message", "expected": None},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(OverlengthDatasetError) as caught:
        CompletionDataset(path, FakeTokenizer(), max_length=2)
    assert caught.value.overlength_count == 2
    assert caught.value.max_observed_length > 2
    assert "2 tokenized row(s)" in str(caught.value)


def test_loss_normalizes_rows_then_applies_example_and_decision_weights() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 4, 2), dtype=torch.float64)
    logits[0, 1] = torch.tensor([2.0, 0.0], dtype=torch.float64)
    logits[1, 0] = torch.tensor([0.0, 2.0], dtype=torch.float64)
    labels = torch.tensor(
        [
            [-100, -100, 0, 1],
            [-100, 1, -100, -100],
        ]
    )
    loss = normalized_completion_cross_entropy(
        logits,
        labels,
        sample_weight=torch.tensor([1.0, 3.0], dtype=torch.float64),
        first_supervised_token_weight=3.0,
    )
    confident = math.log1p(math.exp(-2.0))
    row_zero = (3.0 * confident + math.log(2.0)) / 4.0
    expected = (row_zero + 3.0 * confident) / 4.0
    assert loss.item() == pytest.approx(expected)


def test_loss_ignores_zero_weight_examples_and_rejects_empty_supervision() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 3, 2), dtype=torch.float32)
    labels = torch.tensor([[-100, 0, -100], [-100, 1, -100]])
    logits[0, 0] = torch.tensor([4.0, 0.0])
    logits[1, 0] = torch.tensor([0.0, 0.0])
    weighted = normalized_completion_cross_entropy(
        logits,
        labels,
        sample_weight=torch.tensor([1.0, 0.0]),
    )
    assert weighted.item() == pytest.approx(math.log1p(math.exp(-4.0)), abs=1e-7)
    with pytest.raises(ValueError, match="at least one supervised"):
        normalized_completion_cross_entropy(logits, torch.full_like(labels, -100))
