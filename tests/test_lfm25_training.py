from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest

from lfm25.android_contract import android_extraction_messages
from lfm25.candidates import candidate_selector_messages
from lfm25.training_loss import normalized_completion_cross_entropy
from scripts.train_lfm25_lora import (
    ANDROID_CONTEXT_LENGTH,
    LEGACY_MAX_LENGTH,
    RESUME_PROVENANCE_FILENAME,
    CompletionCollator,
    CompletionDataset,
    OverlengthDatasetError,
    _local_hf_model_provenance,
    _messages,
    _resume_provenance,
    _training_arguments_provenance,
    _trainer_state_provenance,
    _validate_resume_checkpoint_provenance,
    _write_resume_provenance,
    parse_args,
    target_module_inventory,
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


def test_target_inventory_reports_aggregate_leaf_coverage_and_rejects_none() -> None:
    class FakeModel:
        @staticmethod
        def named_modules():
            return iter(
                (
                    ("", object()),
                    ("model.layers.0.in_proj", object()),
                    ("model.layers.1.in_proj", object()),
                    ("model.layers.1.q_proj", object()),
                    ("model.layers.1.unrelated", object()),
                )
            )

    inventory = target_module_inventory(FakeModel())

    assert inventory["matched_module_count"] == 3
    assert inventory["matched_leaf_counts"]["in_proj"] == 2
    assert inventory["matched_leaf_counts"]["q_proj"] == 1
    assert "model.layers.0.in_proj" not in str(inventory)

    with pytest.raises(RuntimeError, match="no matching modules"):
        target_module_inventory(SimpleNamespace(named_modules=lambda: iter((("", object()),))))


def test_local_hf_provenance_hashes_shards_index_config_and_tokenizer(tmp_path) -> None:
    assets = {
        "config.json": "{}",
        "generation_config.json": "{}",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "model-00001-of-00002.safetensors": "first",
        "model-00002-of-00002.safetensors": "second",
        "model.safetensors.index.json": "{}",
    }
    for name, content in assets.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    provenance = _local_hf_model_provenance(tmp_path)

    assert provenance["format"] == "local_hf_assets_v1"
    assert set(provenance["files"]) == set(assets)
    assert provenance["weight_files"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert provenance["weight_index_files"] == ["model.safetensors.index.json"]
    assert all("sha256" in item for item in provenance["files"].values())


def test_trainer_state_provenance_links_best_log_and_restoration_to_checkpoint() -> None:
    best_checkpoint = "/tmp/run/checkpoint-10"
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
        ),
        state=SimpleNamespace(
            best_model_checkpoint=best_checkpoint,
            best_metric=0.25,
            global_step=15,
            epoch=3.0,
            log_history=[
                {"eval_loss": 0.4, "epoch": 1.0, "step": 5},
                {"eval_loss": 0.25, "epoch": 2.0, "step": 10},
                {"eval_loss": 0.3, "epoch": 3.0, "step": 15},
                {"eval_loss": 0.25, "epoch": 3.0, "step": 15},
            ],
        ),
        _best_model_restoration_completed=True,
        _restored_best_model_checkpoint=best_checkpoint,
    )

    provenance = _trainer_state_provenance(trainer)

    assert provenance["best_model_checkpoint"] == best_checkpoint
    assert provenance["best_metric"] == 0.25
    assert provenance["best_eval_log"] == {
        "metric_name": "eval_loss",
        "metric_value": 0.25,
        "epoch": 2.0,
        "step": 10,
    }
    assert provenance["final_global_step"] == 15
    assert provenance["final_epoch"] == 3.0
    assert provenance["load_best_model_at_end_restored_best_checkpoint"] is True


def test_training_arguments_provenance_uses_constructed_values() -> None:
    training_args = SimpleNamespace(
        optim=SimpleNamespace(value="adafactor"),
        lr_scheduler_type=SimpleNamespace(value="linear"),
        max_grad_norm=0.75,
        bf16=False,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        full_determinism=False,
        eval_strategy=SimpleNamespace(value="steps"),
        save_strategy=SimpleNamespace(value="steps"),
        per_device_eval_batch_size=3,
    )

    assert _training_arguments_provenance(training_args) == {
        "optimizer": "adafactor",
        "lr_scheduler_type": "linear",
        "max_grad_norm": 0.75,
        "bf16": False,
        "tf32": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": True,
        "full_determinism": False,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "per_device_eval_batch_size": 3,
    }


def _resume_record() -> dict:
    args = parse_args(_required_cli())
    training_args = SimpleNamespace(
        warmup_steps=4,
        optim=SimpleNamespace(value="adamw_torch"),
        lr_scheduler_type=SimpleNamespace(value="cosine"),
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        full_determinism=True,
        eval_strategy=SimpleNamespace(value="epoch"),
        save_strategy=SimpleNamespace(value="epoch"),
        per_device_eval_batch_size=args.eval_batch_size,
    )
    return _resume_provenance(
        args,
        training_args,
        base_model_provenance={
            "format": "local_hf_assets_v1",
            "files": {
                "config.json": {
                    "bytes": 2,
                    "sha256": "base-config-sha256",
                }
            },
        },
        train_sha256="train-sha256",
        eval_sha256="eval-sha256",
        contract={
            "profile": "android",
            "contract": "pocketfinancer_android",
            "prompt_sha256": "prompt-sha256",
        },
    )


def test_resume_provenance_covers_model_data_contract_and_material_training_settings() -> None:
    provenance = _resume_record()

    assert provenance["base_model_provenance"]["files"]["config.json"]["sha256"] == (
        "base-config-sha256"
    )
    assert provenance["datasets"] == {
        "train_sha256": "train-sha256",
        "eval_sha256": "eval-sha256",
    }
    assert provenance["contract"]["prompt_sha256"] == "prompt-sha256"
    assert provenance["training"]["loss"]["first_supervised_token_weight"] == 3.0
    assert provenance["training"]["lora"] == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": [
            "in_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "w1",
            "w2",
            "w3",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    optimization = provenance["training"]["optimization"]
    assert optimization["learning_rate"] == 2e-4
    assert optimization["epochs_requested"] == 12.0
    assert optimization["warmup_steps"] == 4
    assert optimization["optimizer"] == "adamw_torch"


def test_resume_checkpoint_requires_provenance_and_accepts_exact_match(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    provenance = _resume_record()

    with pytest.raises(RuntimeError, match="missing required provenance"):
        _validate_resume_checkpoint_provenance(checkpoint, provenance)

    written = _write_resume_provenance(checkpoint, provenance)

    assert written == checkpoint / RESUME_PROVENANCE_FILENAME
    assert _validate_resume_checkpoint_provenance(checkpoint, provenance) == checkpoint.resolve()


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("base_model_provenance", "files", "config.json", "sha256"), "other-base"),
        (("datasets", "train_sha256"), "other-train"),
        (("datasets", "eval_sha256"), "other-eval"),
        (("contract", "prompt_sha256"), "other-prompt"),
        (("training", "lora", "rank"), 8),
        (("training", "optimization", "learning_rate"), 1e-4),
    ],
)
def test_resume_checkpoint_rejects_each_provenance_class(
    tmp_path,
    field_path,
    replacement,
) -> None:
    checkpoint = tmp_path / "checkpoint-5"
    checkpoint.mkdir()
    expected = _resume_record()
    observed = deepcopy(expected)
    target = observed
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = replacement
    (checkpoint / RESUME_PROVENANCE_FILENAME).write_text(
        json.dumps(observed),
        encoding="utf-8",
    )

    expected_path = ".".join(field_path)
    with pytest.raises(RuntimeError, match="does not match") as caught:
        _validate_resume_checkpoint_provenance(checkpoint, expected)
    assert expected_path in str(caught.value)


def test_resume_provenance_writer_refuses_to_replace_incompatible_identity(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    expected = _resume_record()
    _write_resume_provenance(output_dir, expected)
    incompatible = deepcopy(expected)
    incompatible["training"]["seed"] = 99

    with pytest.raises(RuntimeError, match="refusing to replace incompatible"):
        _write_resume_provenance(output_dir, incompatible)
