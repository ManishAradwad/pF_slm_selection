#!/usr/bin/env python3
"""Deterministic PocketFinancer-aligned BF16 LoRA SFT.

The supported happy path serializes every example exactly like the Android app
and computes loss only on the assistant completion. Historical profiles remain
available solely to reproduce earlier experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.contract import parse_gold, parse_prediction  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402
from lfm25.prompts import PRODUCTION_SYSTEM_PROMPT  # noqa: E402
from lfm25.training_loss import LOSS_MODE, normalized_completion_cross_entropy  # noqa: E402

LORA_TARGETS = ["in_proj", "q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"]
LEGACY_MAX_LENGTH = 512
ANDROID_CONTEXT_LENGTH = 3072
ANDROID_OUTER_SYSTEM_PROMPT = "You are a helpful financial SMS extraction assistant."
PROMPT_PROFILE_ALIASES = {
    "legacy": "legacy",
    "legacy_short": "legacy",
    "short": "legacy",
    "android": "android",
    "pocketfinancer": "android",
    "pocketfinancer_android": "android",
    "candidate_selector": "candidate_selector",
    "selector": "candidate_selector",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _prompt_profile(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    try:
        return PROMPT_PROFILE_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(PROMPT_PROFILE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown prompt profile {value!r}; choose one of: {choices}"
        ) from error


def _load_optional_contract(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return None


def _android_messages(sender: str, sms: str) -> list[dict[str, str]]:
    module = _load_optional_contract("lfm25.android_contract")
    if module is not None:
        helper = getattr(module, "android_extraction_messages", None)
        if not callable(helper):
            raise RuntimeError("lfm25.android_contract lacks android_extraction_messages")
        messages = helper(sender, sms)
    else:
        try:
            from DATA.utils import doc_to_text
        except ImportError as error:
            raise RuntimeError(
                "the Android prompt profile requires lfm25.android_contract or DATA.utils"
            ) from error
        messages = [
            {"role": "system", "content": ANDROID_OUTER_SYSTEM_PROMPT},
            {"role": "user", "content": doc_to_text({"sender": sender, "sms": sms})},
        ]
    return _validate_prompt_messages(messages, contract="Android")


def _selector_messages(sender: str, sms: str) -> list[dict[str, str]]:
    module = _load_optional_contract("lfm25.candidates")
    if module is None:
        raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
    helper = getattr(module, "candidate_selector_messages", None)
    if not callable(helper):
        raise RuntimeError("lfm25.candidates lacks candidate_selector_messages")
    return _validate_prompt_messages(helper(sender, sms), contract="candidate-selector")


def _validate_prompt_messages(messages: Any, *, contract: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != 2:
        raise RuntimeError(f"the {contract} contract must produce exactly system and user messages")
    prepared = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in messages
        if isinstance(item, dict)
    ]
    if len(prepared) != 2 or [item["role"] for item in prepared] != ["system", "user"]:
        raise RuntimeError(f"the {contract} contract must produce exactly system and user messages")
    return prepared


def _contract_provenance(prompt_profile: str) -> dict[str, Any]:
    if prompt_profile == "legacy":
        return {
            "profile": "legacy",
            "contract": "lfm25_short_extraction",
            "contract_version": 1,
            "prompt_sha256": hashlib.sha256(PRODUCTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        }
    if prompt_profile == "candidate_selector":
        module = _load_optional_contract("lfm25.candidates")
        if module is None:
            raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
        helper = getattr(module, "contract_provenance", None)
        provenance = helper() if callable(helper) else {}
        if not isinstance(provenance, dict):
            raise RuntimeError("lfm25.candidates.contract_provenance() must return a dict")
        selector_prompt = str(getattr(module, "SELECTOR_SYSTEM_PROMPT", ""))
        source_path = Path(str(module.__file__))
        return {
            "profile": "candidate_selector",
            "contract": "grounded_candidate_selector",
            "prompt_sha256": hashlib.sha256(selector_prompt.encode("utf-8")).hexdigest(),
            "source_sha256": _sha256(source_path),
            **provenance,
        }
    module = _load_optional_contract("lfm25.android_contract")
    if module is not None:
        helper = getattr(module, "contract_provenance", None)
        if not callable(helper):
            raise RuntimeError("lfm25.android_contract lacks contract_provenance")
        provenance = helper()
        if not isinstance(provenance, dict):
            raise RuntimeError("lfm25.android_contract.contract_provenance() must return a dict")
        return {"profile": "android", **provenance}
    source = REPO_ROOT / "DATA" / "utils.py"
    return {
        "profile": "android",
        "contract": "pocketfinancer_android",
        "contract_version": "DATA.utils fallback",
        "prompt_source": "DATA.utils.doc_to_text",
        "prompt_source_sha256": _sha256(source) if source.is_file() else None,
        "outer_system_prompt_sha256": hashlib.sha256(
            ANDROID_OUTER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }


def _assistant_text(value: Any) -> str:
    gold = parse_gold(value)
    if gold is None:
        return "null"
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))


def _row_target(row: dict[str, Any]) -> tuple[Any, bool]:
    for key in ("expected", "label"):
        if key in row:
            return row[key], True
    return None, False


def _existing_assistant_text(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    if (
        not messages
        or not isinstance(messages[-1], dict)
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("every SFT row must end with an assistant message")
    return str(messages[-1].get("content", ""))


def _selector_assistant_text(row: dict[str, Any], sms: str) -> str:
    module = _load_optional_contract("lfm25.candidates")
    if module is None:
        raise RuntimeError("the candidate-selector profile requires lfm25.candidates")
    target, has_target = _row_target(row)
    candidates = module.extract_candidates(sms)
    if has_target:
        selection = module.selector_target(target, candidates)
    else:
        existing = _existing_assistant_text(row)
        if existing is None:
            raise ValueError(
                "candidate-selector rows require expected, label, or assistant content"
            )
        try:
            selection = json.loads(existing)
            module.reconstruct_transaction(selection, candidates)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                "assistant completion violates the candidate-selector contract"
            ) from error
    return json.dumps(selection, ensure_ascii=False, separators=(",", ":"))


def _messages(row: dict[str, Any], prompt_profile: str = "legacy") -> list[dict[str, str]]:
    prompt_profile = _prompt_profile(prompt_profile)
    sender = str(row.get("sender", ""))
    sms = str(row.get("sms", ""))
    if prompt_profile == "legacy" and isinstance(row.get("messages"), list):
        messages = row["messages"]
    elif prompt_profile == "candidate_selector":
        messages = _selector_messages(sender, sms)
        messages.append({"role": "assistant", "content": _selector_assistant_text(row, sms)})
    else:
        target, has_target = _row_target(row)
        existing = _existing_assistant_text(row)
        if not has_target and existing is None:
            raise ValueError("SFT rows require expected, label, or assistant content")
        assistant = _assistant_text(target) if has_target else str(existing)
        messages = (
            _android_messages(sender, sms)
            if prompt_profile == "android"
            else extraction_messages(sender, sms)
        )
        messages.append({"role": "assistant", "content": assistant})
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("every SFT row must end with an assistant message")
    if prompt_profile == "candidate_selector":
        _selector_assistant_text({**row, "messages": messages}, sms)
    else:
        parsed = parse_prediction(str(messages[-1].get("content", "")))
        if parsed.status == "invalid":
            raise ValueError("assistant completion violates the strict extraction contract")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


class OverlengthDatasetError(ValueError):
    def __init__(self, path: Path, max_length: int, rows: list[tuple[int, int]]):
        self.path = path
        self.max_length = max_length
        self.overlength_count = len(rows)
        self.max_observed_length = max(length for _, length in rows)
        preview = ", ".join(f"{index}:{length}" for index, length in rows[:8])
        if len(rows) > 8:
            preview += ", ..."
        super().__init__(
            f"{len(rows)} tokenized row(s) in {path} exceed --max-length {max_length}; "
            f"maximum observed length is {self.max_observed_length}; row:length {preview}. "
            "Raise the limit instead of truncating."
        )


class CompletionDataset:
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        prompt_profile: str = "legacy",
    ):
        if max_length <= 1:
            raise ValueError("max_length must be greater than one")
        self.path = path
        self.prompt_profile = _prompt_profile(prompt_profile)
        self.features: list[dict[str, Any]] = []
        lengths: list[int] = []
        supervised_lengths: list[int] = []
        sample_weights: list[float] = []
        explicit_sample_weights = 0
        overlength: list[tuple[int, int]] = []
        for row_index, row in enumerate(_read_jsonl(path), start=1):
            raw_weight = row.get("sample_weight", 1.0)
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"row {row_index} sample_weight must be numeric")
            sample_weight = float(raw_weight)
            if not math.isfinite(sample_weight) or sample_weight <= 0:
                raise ValueError(f"row {row_index} sample_weight must be finite and positive")
            explicit_sample_weights += int("sample_weight" in row)

            messages = _messages(row, self.prompt_profile)
            prompt_encoding = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            full_encoding = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
            prompt_ids = _token_ids(prompt_encoding)
            input_ids = _token_ids(full_encoding)
            common = 0
            for left, right in zip(prompt_ids, input_ids):
                if left != right:
                    break
                common += 1
            if common == 0:
                raise ValueError(
                    f"row {row_index} chat template has no stable prompt/completion boundary"
                )
            if common == len(input_ids):
                raise ValueError(
                    f"row {row_index} chat template left no assistant completion tokens"
                )
            if len(input_ids) > max_length:
                overlength.append((row_index, len(input_ids)))
                continue
            labels = [-100] * common + input_ids[common:]
            supervised_length = sum(label != -100 for label in labels[1:])
            if supervised_length == 0:
                raise ValueError(f"row {row_index} has no causally supervised completion tokens")
            self.features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "sample_weight": sample_weight,
                }
            )
            lengths.append(len(input_ids))
            supervised_lengths.append(supervised_length)
            sample_weights.append(sample_weight)
        if overlength:
            raise OverlengthDatasetError(path, max_length, overlength)
        if not self.features:
            raise ValueError(f"no rows in {path}")
        ordered = sorted(lengths)
        p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        self.stats = {
            "rows": len(self.features),
            "tokens_min": min(lengths),
            "tokens_mean": round(statistics.fmean(lengths), 2),
            "tokens_p95": ordered[p95_index],
            "tokens_max": max(lengths),
            "completion_tokens_mean": round(statistics.fmean(supervised_lengths), 2),
            "completion_tokens_min": min(supervised_lengths),
            "max_length_limit": max_length,
            "overlength_rows": 0,
            "prompt_profile": self.prompt_profile,
            "sample_weight_field": "sample_weight",
            "sample_weight_explicit_rows": explicit_sample_weights,
            "sample_weight_min": min(sample_weights),
            "sample_weight_mean": round(statistics.fmean(sample_weights), 6),
            "sample_weight_max": max(sample_weights),
            "sample_weight_sum": round(sum(sample_weights), 6),
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def _token_ids(encoding: Any) -> list[int]:
    values = (
        encoding["input_ids"] if hasattr(encoding, "keys") and "input_ids" in encoding else encoding
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("chat template unexpectedly returned a batch")
        values = values[0]
    return [int(value) for value in values]


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer has no pad token")

    def __call__(self, features):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels, sample_weights = [], [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            sample_weights.append(float(feature.get("sample_weight", 1.0)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TRAINING_ARTIFACTS/base/LFM2.5-350M")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-profile",
        "--contract",
        dest="prompt_profile",
        type=_prompt_profile,
        default="pocketfinancer",
        help=(
            "prompt profile: pocketfinancer (default); legacy and "
            "candidate_selector are historical research profiles"
        ),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="defaults to 3072 for Android and 512 for legacy/selector profiles",
    )
    parser.add_argument("--loss-mode", choices=[LOSS_MODE], default=LOSS_MODE)
    parser.add_argument(
        "--first-supervised-token-weight",
        "--first-token-weight",
        "--decision-token-weight",
        dest="first_supervised_token_weight",
        type=float,
        default=3.0,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_length is None:
        args.max_length = (
            ANDROID_CONTEXT_LENGTH if args.prompt_profile == "android" else LEGACY_MAX_LENGTH
        )
    if args.max_length <= 1:
        parser.error("--max-length must be greater than one")
    if args.prompt_profile == "android" and args.max_length > ANDROID_CONTEXT_LENGTH:
        parser.error(
            f"Android contract context is {ANDROID_CONTEXT_LENGTH}; --max-length cannot exceed it"
        )
    if not math.isfinite(args.first_supervised_token_weight) or (
        args.first_supervised_token_weight <= 0
    ):
        parser.error("--first-supervised-token-weight must be finite and greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    import peft
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class CompletionNormalizedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            sample_weight = model_inputs.pop("sample_weight", None)
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = normalized_completion_cross_entropy(
                logits,
                labels,
                sample_weight=sample_weight,
                first_supervised_token_weight=args.first_supervised_token_weight,
            )
            return (loss, outputs) if return_outputs else loss

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BF16 LoRA training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    contract = _contract_provenance(args.prompt_profile)
    train_dataset = CompletionDataset(
        args.train,
        tokenizer,
        args.max_length,
        prompt_profile=args.prompt_profile,
    )
    eval_dataset = CompletionDataset(
        args.eval,
        tokenizer,
        args.max_length,
        prompt_profile=args.prompt_profile,
    )
    print(
        json.dumps(
            {
                "dataset_preflight": {
                    "contract": args.prompt_profile,
                    "max_length": args.max_length,
                    "train": train_dataset.stats,
                    "eval": eval_dataset.stats,
                }
            },
            sort_keys=True,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=LORA_TARGETS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("LoRA attached no trainable parameters")
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / args.gradient_accumulation))
    total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = round(total_updates * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cache=False,
    )
    trainer = CompletionNormalizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "base_model": str(Path(args.model).resolve()),
        "train_file": str(args.train.resolve()),
        "train_sha256": _sha256(args.train),
        "eval_file": str(args.eval.resolve()),
        "eval_sha256": _sha256(args.eval),
        "train_stats": train_dataset.stats,
        "eval_stats": eval_dataset.stats,
        "contract": contract,
        "loss": {
            "mode": args.loss_mode,
            "causal_shift": True,
            "ignore_index": -100,
            "token_reduction": "weighted_mean_per_example",
            "example_reduction": "sample_weighted_mean",
            "weights": {
                "sample_weight_field": "sample_weight",
                "default_sample_weight": 1.0,
                "first_supervised_token_weight": args.first_supervised_token_weight,
                "train_explicit_sample_weight_rows": train_dataset.stats[
                    "sample_weight_explicit_rows"
                ],
                "eval_explicit_sample_weight_rows": eval_dataset.stats[
                    "sample_weight_explicit_rows"
                ],
            },
        },
        "seed": args.seed,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": LORA_TARGETS,
            "trainable_parameters": trainable,
            "total_parameters_with_adapter": total,
            "trainable_percent": round(100 * trainable / total, 6),
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "max_length": args.max_length,
            "prompt_profile": args.prompt_profile,
            "loss_mode": args.loss_mode,
            "first_supervised_token_weight": args.first_supervised_token_weight,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "privacy": {
            "report_to": "none",
            "push_to_hub": False,
            "raw_examples_logged": False,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "adapter_dir": str(adapter_dir),
                "trainable_parameters": trainable,
                "eval_loss": eval_metrics.get("eval_loss"),
                "peak_vram_mib": manifest["peak_vram_mib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
