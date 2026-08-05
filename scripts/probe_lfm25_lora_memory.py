#!/usr/bin/env python3
"""Probe a real PocketFinancer LoRA backward pass without logging private rows.

The probe deliberately uses the longest examples in an existing local SFT file.
It attaches the same adapters and loss used by the trainer, performs one optimizer-
free forward/backward pass, and writes only aggregate diagnostics to an ignored
result file.  It is a capacity gate, not a training or quality result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.provenance import (  # noqa: E402
    code_fingerprints,
    fingerprint_file,
    fingerprint_named_files,
)
from lfm25.training_loss import normalized_completion_cross_entropy  # noqa: E402
from scripts.train_lfm25_lora import (  # noqa: E402
    ANDROID_CONTEXT_LENGTH,
    LORA_TARGETS,
    CompletionCollator,
    CompletionDataset,
    _contract_provenance,
    _prompt_profile,
)


QUANTIZATION_MODES = ("bf16", "qlora-nf4")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real completion-only LoRA backward pass and report only "
            "aggregate memory/module diagnostics."
        )
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument(
        "--prompt-profile",
        "--contract",
        dest="prompt_profile",
        type=_prompt_profile,
        default="pocketfinancer",
    )
    parser.add_argument("--quantization-mode", choices=QUANTIZATION_MODES, default="bf16")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--max-length", type=_positive_int, default=2304)
    parser.add_argument("--rank", type=_positive_int, default=16)
    parser.add_argument("--alpha", type=_positive_int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--first-supervised-token-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "RESULTS" / "pocketfinancer" / "probes" / "lora_memory.json",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompt_profile == "android" and args.max_length > ANDROID_CONTEXT_LENGTH:
        parser.error(
            f"Android contract context is {ANDROID_CONTEXT_LENGTH}; --max-length cannot exceed it"
        )
    if not math.isfinite(args.dropout) or not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be finite and in [0, 1)")
    if not math.isfinite(args.first_supervised_token_weight) or (
        args.first_supervised_token_weight <= 0
    ):
        parser.error("--first-supervised-token-weight must be finite and positive")
    return args


def target_module_inventory(model: Any) -> dict[str, Any]:
    """Return aggregate LoRA target coverage without parameter values."""

    matched = [
        name
        for name, _module in model.named_modules()
        if name and name.rsplit(".", 1)[-1] in LORA_TARGETS
    ]
    counts = Counter(name.rsplit(".", 1)[-1] for name in matched)
    return {
        "configured_leaf_names": list(LORA_TARGETS),
        "matched_module_count": len(matched),
        "matched_leaf_counts": {name: counts.get(name, 0) for name in LORA_TARGETS},
    }


def _model_fingerprint(model_root: Path) -> dict[str, Any]:
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
        "model.safetensors.index.json",
    ]
    names.extend(path.name for path in sorted(model_root.glob("model-*.safetensors")))
    return fingerprint_named_files(model_root, tuple(dict.fromkeys(names)))


def _output_path_is_ignored(output: Path, repo_root: Path = REPO_ROOT) -> bool:
    """Ask Git whether the resolved repository-relative output is ignored."""

    resolved_root = repo_root.resolve(strict=True)
    resolved_output = (
        output.resolve() if output.is_absolute() else (resolved_root / output).resolve()
    )
    try:
        relative_output = resolved_output.relative_to(resolved_root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_output.as_posix()],
        cwd=resolved_root,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("git check-ignore failed while validating probe output privacy")
    return result.returncode == 0


def _output_path_is_tracked(output: Path, repo_root: Path = REPO_ROOT) -> bool:
    """Return whether Git already tracks the resolved repository-relative path."""

    resolved_root = repo_root.resolve(strict=True)
    resolved_output = (
        output.resolve() if output.is_absolute() else (resolved_root / output).resolve()
    )
    try:
        relative_output = resolved_output.relative_to(resolved_root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_output.as_posix()],
        cwd=resolved_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("git ls-files failed while validating probe output privacy")
    return result.returncode == 0


def _resolve_safe_output_path(output: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve an ignored, untracked output path inside the repository."""

    resolved_root = repo_root.resolve(strict=True)
    resolved_output = (
        output.resolve() if output.is_absolute() else (resolved_root / output).resolve()
    )
    if _output_path_is_tracked(resolved_output, resolved_root):
        raise ValueError("probe output must not overwrite a tracked repository file")
    if not _output_path_is_ignored(resolved_output, resolved_root):
        raise ValueError("probe output must be an ignored path inside the repository")
    return resolved_output


def _input_provenance(train: Path, prompt_profile: str) -> dict[str, Any]:
    resolved_profile = _prompt_profile(prompt_profile)
    return {
        "train_file": fingerprint_file(train),
        "serialization": {
            "prompt_profile": resolved_profile,
            "contract": _contract_provenance(resolved_profile),
        },
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = _resolve_safe_output_path(args.output)
    input_provenance = _input_provenance(args.train, args.prompt_profile)
    os.environ.setdefault("WANDB_DISABLED", "true")

    import peft
    import torch
    import transformers
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the LoRA memory probe")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = CompletionDataset(
        args.train,
        tokenizer,
        args.max_length,
        prompt_profile=args.prompt_profile,
    )
    selected = sorted(
        dataset.features,
        key=lambda feature: len(feature["input_ids"]),
        reverse=True,
    )[: args.batch_size]
    if len(selected) != args.batch_size:
        raise RuntimeError("the requested probe batch exceeds the available training rows")
    batch = CompletionCollator(tokenizer)(selected)

    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    if args.quantization_mode == "qlora-nf4":
        load_kwargs.update(
            {
                "device_map": {"": 0},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
            }
        )
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    inventory = target_module_inventory(model)
    if inventory["matched_module_count"] <= 0:
        raise RuntimeError("LoRA target inspection found no matching modules")
    model.config.use_cache = False
    if args.quantization_mode == "qlora-nf4":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=LORA_TARGETS,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )
    if args.quantization_mode == "bf16":
        model.to("cuda")
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0:
        raise RuntimeError("LoRA attached no trainable parameters")

    device_batch = {key: value.to("cuda") for key, value in batch.items()}
    labels = device_batch.pop("labels")
    sample_weight = device_batch.pop("sample_weight")
    outputs = model(**device_batch)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    loss = normalized_completion_cross_entropy(
        logits,
        labels,
        sample_weight=sample_weight,
        first_supervised_token_weight=args.first_supervised_token_weight,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("the LoRA probe loss is not finite")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = sum(int(torch.isfinite(gradient).all().item()) for gradient in gradients)
    nonzero_gradients = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients)
    if not gradients or finite_gradients != len(gradients) or nonzero_gradients <= 0:
        raise RuntimeError("LoRA gradient verification failed")
    torch.cuda.synchronize()

    report = {
        "schema_version": 1,
        "status": "passed",
        "capacity_gate_only": True,
        "quality_result": False,
        "model_class": model.base_model.model.__class__.__name__,
        "quantization_mode": args.quantization_mode,
        "quantization": {
            "load_in_4bit": args.quantization_mode == "qlora-nf4",
            "quant_type": "nf4" if args.quantization_mode == "qlora-nf4" else None,
            "double_quantization": args.quantization_mode == "qlora-nf4",
            "compute_dtype": "bfloat16",
        },
        "batch_size": args.batch_size,
        "selected_sequence_tokens": [len(feature["input_ids"]) for feature in selected],
        "selected_completion_tokens": [
            sum(label != -100 for label in feature["labels"][1:]) for feature in selected
        ],
        "dataset_rows": len(dataset),
        "dataset_max_tokens": dataset.stats["tokens_max"],
        "max_length": args.max_length,
        "loss": round(float(loss.detach().cpu()), 6),
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            **inventory,
            "trainable_parameters": trainable,
            "total_parameters_with_adapter": total,
            "trainable_percent": round(100 * trainable / total, 6),
        },
        "finite_gradient_tensors": finite_gradients,
        "nonzero_gradient_tensors": nonzero_gradients,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "model": _model_fingerprint(args.model),
        "input": input_provenance,
        "code_sha256": code_fingerprints(REPO_ROOT),
        "privacy": {
            "row_text_logged": False,
            "report_to": "none",
            "output_root_ignored": _output_path_is_ignored(output_path),
        },
    }
    if fingerprint_file(args.train) != input_provenance["train_file"]:
        raise RuntimeError(
            "training input changed during the probe; refusing to write stale provenance"
        )
    # Revalidate after the potentially long GPU pass so a newly tracked path is not replaced.
    output_path = _resolve_safe_output_path(output_path)
    _atomic_write(output_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
