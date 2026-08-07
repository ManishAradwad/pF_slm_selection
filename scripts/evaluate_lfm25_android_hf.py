#!/usr/bin/env python3
"""Evaluate a model with the PocketFinancer app input and output behavior.

For a non-thinking model such as LFM2.5-350M this uses the same direct-answer
shape as the app. Thinking models use the app-shaped two-pass path while
respecting the model chat template's existing ``<think>`` opening. Transformers
still cannot reproduce the custom Android JNI, GGUF quantization, optional GBNF
sampler, or phone performance, so final deployment validation belongs to the
GGUF/device evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.android_contract import (  # noqa: E402
    ANDROID_DECODE_DEFAULTS,
    android_extraction_messages,
    context_compatibility,
    contract_provenance,
    decode_defaults,
    pocketfinancer_normalize_prediction,
    pocketfinancer_parse_prediction,
    selection_prefilter_sms,
    should_apply_prefilter,
    summarize_prefilter,
)
from lfm25.metrics import score_records  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402
from lfm25.provenance import (  # noqa: E402
    code_fingerprints,
    fingerprint_file,
    fingerprint_named_files,
)
from lfm25.training_provenance import (  # noqa: E402
    adapter_artifact_evidence,
    training_run_manifest_evidence,
)

_MODEL_LOCK_DOCUMENTATION_FILES = {"LICENSE", "README.md"}
_HF_RUNTIME_FIXED_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_HF_RUNTIME_WEIGHT_RE = re.compile(
    r"(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:safetensors|bin)"
    r"(?:\.index\.json)?"
)


def _safe_file_fingerprint(path: Path, *, filename: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.name != filename:
        raise ValueError(f"expected {filename}, got {resolved.name}")
    payload = resolved.read_bytes()
    return {
        "filename": filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _model_lock_runtime_files(model_lock: Path) -> tuple[dict[str, str], dict[str, Any]]:
    resolved = model_lock.resolve(strict=True)
    if resolved.name != "model.lock.json":
        raise ValueError(f"expected model.lock.json, got {resolved.name}")
    payload = resolved.read_bytes()
    lock_evidence = {
        "filename": "model.lock.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        lock = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("model.lock.json is not valid UTF-8 JSON") from error
    if not isinstance(lock, dict) or not isinstance(lock.get("model"), dict):
        raise ValueError("model.lock.json has no model object")
    raw_files = lock["model"].get("files")
    if not isinstance(raw_files, dict):
        raise ValueError("model.lock.json has no model.files object")

    locked: dict[str, str] = {}
    for name, digest in raw_files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("model.lock.json has an invalid model.files entry")
        if name not in _MODEL_LOCK_DOCUMENTATION_FILES:
            locked[name] = digest
    if (
        "config.json" not in locked
        or not ({"tokenizer.json", "tokenizer.model"} & set(locked))
        or not any(
            _HF_RUNTIME_WEIGHT_RE.fullmatch(name) is not None and not name.endswith(".index.json")
            for name in locked
        )
    ):
        raise ValueError("model.lock.json does not bind a complete HF runtime")
    return dict(sorted(locked.items())), lock_evidence


def _local_model_root(model_spec: str) -> tuple[Path, str | None]:
    model_path = Path(model_spec)
    if model_path.is_dir():
        return model_path.resolve(strict=True), None
    if model_path.is_file():
        raise ValueError("--model must be a local HF directory or cached identifier")

    from transformers.utils import cached_file

    cached_config = cached_file(model_spec, "config.json", local_files_only=True)
    if cached_config is None:
        raise RuntimeError(f"cannot resolve local model snapshot for provenance: {model_spec}")
    return Path(cached_config).parent.resolve(strict=True), model_spec


def _locked_hf_model_evidence(
    model_spec: str, model_lock: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    locked, lock_evidence = _model_lock_runtime_files(model_lock)
    root, identifier = _local_model_root(model_spec)
    present_runtime_files = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in _HF_RUNTIME_FIXED_FILES
            or _HF_RUNTIME_WEIGHT_RE.fullmatch(path.name) is not None
        )
    }
    unexpected = present_runtime_files - set(locked)
    if unexpected:
        raise ValueError(
            "local model has runtime files absent from model.lock.json: "
            + ", ".join(sorted(unexpected))
        )

    files: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in locked.items():
        candidate = root / name
        if not candidate.is_file():
            raise ValueError(f"local model is missing locked runtime file: {name}")
        item = fingerprint_file(candidate)
        if item["sha256"] != expected_sha256:
            raise ValueError(f"local model file differs from model.lock.json: {name}")
        files[name] = {"bytes": item["bytes"], "sha256": item["sha256"]}
    evidence: dict[str, Any] = {"path": str(root), "files": files}
    if identifier is not None:
        evidence["identifier"] = identifier
    return evidence, lock_evidence


def _evaluation_input_evidence(
    *,
    model: str,
    adapter: Path | None,
    dataset: Path,
    model_lock: Path,
) -> dict[str, Any]:
    model_evidence, lock_evidence = _locked_hf_model_evidence(model, model_lock)
    return {
        "model": model_evidence,
        "adapter": adapter_artifact_evidence(adapter) if adapter is not None else None,
        "training_run": training_run_manifest_evidence(adapter),
        "dataset": fingerprint_file(dataset),
        "model_lock": lock_evidence,
    }


def _assert_evaluation_inputs_unchanged(
    expected: Mapping[str, Any],
    *,
    model: str,
    adapter: Path | None,
    dataset: Path,
    model_lock: Path,
) -> None:
    try:
        observed = _evaluation_input_evidence(
            model=model,
            adapter=adapter,
            dataset=dataset,
            model_lock=model_lock,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("evaluation inputs changed after the pre-inference snapshot") from error
    if observed != expected:
        raise RuntimeError("evaluation inputs changed after the pre-inference snapshot")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _gold(row: dict[str, Any]) -> Any:
    if "expected" in row:
        return row["expected"]
    if "label" in row:
        return row["label"]
    if isinstance(row.get("messages"), list) and row["messages"]:
        return row["messages"][-1].get("content")
    raise ValueError("evaluation row has no target")


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(proportion * (len(ordered) - 1))))
    return round(ordered[index], 3)


_THINKING_KEYS = {
    "has_thinking_mode",
    "hasThinkingMode",
    "thinking",
    "always_reasons_before_answer",
}


def _thinking_flags(value: Any) -> list[bool]:
    flags: list[bool] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _THINKING_KEYS and isinstance(child, bool):
                flags.append(child)
            elif key == "thinking_mode" and isinstance(child, str):
                normalized = child.strip().lower()
                if normalized in {"on", "true", "thinking"}:
                    flags.append(True)
                elif normalized in {"off", "false", "direct"}:
                    flags.append(False)
            else:
                flags.extend(_thinking_flags(child))
    elif isinstance(value, list):
        for child in value:
            flags.extend(_thinking_flags(child))
    return flags


def _resolve_thinking_mode(
    requested: str,
    model_config: Mapping[str, Any] | None,
    model: str | Path,
) -> tuple[bool, str]:
    if requested == "on":
        return True, "cli"
    if requested == "off":
        return False, "cli"
    if requested != "auto":
        raise ValueError(f"unknown thinking mode: {requested!r}")

    flags = _thinking_flags(model_config) if model_config is not None else []
    if flags:
        if len(set(flags)) != 1:
            raise ValueError("model config contains conflicting thinking-mode flags")
        return flags[0], "model_config"

    identity = f"{model} {json.dumps(model_config or {}, sort_keys=True)}".lower()
    compact = identity.replace("_", "").replace("-", "").replace(".", "")
    if "lfm25350m" in compact:
        return False, "lfm2.5-350m_non_thinking"
    return False, "auto_safe_default_off"


def _prepare_thinking_prompt(prompt: str) -> tuple[str, bool]:
    """Open a thinking block only when the template did not already do so."""

    if re.search(r"<think>\s*$", prompt):
        return prompt, False
    separator = "" if not prompt or prompt.endswith("\n") else "\n"
    return f"{prompt}{separator}<think>\n", True


def _split_at_token_sequence(token_ids: list[int], stop_ids: list[int]) -> tuple[list[int], bool]:
    """Return tokens before the first exact stop sequence."""

    if not stop_ids:
        raise ValueError("stop token sequence cannot be empty")
    final_start = len(token_ids) - len(stop_ids)
    for start in range(max(0, final_start + 1)):
        if token_ids[start : start + len(stop_ids)] == stop_ids:
            return token_ids[:start], True
    return token_ids, False


def _thinking_context_budget(
    prompt_tokens: int,
    n_ctx: int,
    answer_tokens: int,
    closing_tokens: int,
    requested_thinking_tokens: int,
) -> dict[str, Any]:
    """Reserve the final answer, then cap reasoning to remaining context."""

    required_without_reasoning = prompt_tokens + closing_tokens + answer_tokens
    if required_without_reasoning > n_ctx:
        raise ValueError(
            f"prompt ({prompt_tokens}) + thinking close ({closing_tokens}) + "
            f"answer ({answer_tokens}) exceeds n_ctx={n_ctx}"
        )
    available = n_ctx - required_without_reasoning
    effective = min(requested_thinking_tokens, available)
    if effective <= 0:
        raise ValueError(
            f"prompt ({prompt_tokens}) + answer framing ({closing_tokens}) + "
            f"answer ({answer_tokens}) leaves no thinking capacity in n_ctx={n_ctx}"
        )
    return {
        "n_ctx": n_ctx,
        "prompt_tokens": prompt_tokens,
        "answer_max_tokens": answer_tokens,
        "thinking_close_tokens": closing_tokens,
        "requested_thinking_max_tokens": requested_thinking_tokens,
        "available_thinking_tokens": available,
        "effective_thinking_max_tokens": effective,
        "thinking_capped_by_context": effective < requested_thinking_tokens,
    }


def _hf_directory_evidence(directory: Path) -> dict[str, Any]:
    """Fingerprint every local HF weight shard/index and tokenizer metadata."""

    resolved = directory.resolve(strict=True)
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "adapter_config.json",
    }
    patterns = (
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "pytorch_model*.bin.index.json",
    )
    for pattern in patterns:
        names.update(path.name for path in resolved.glob(pattern) if path.is_file())
    return fingerprint_named_files(resolved, sorted(names))


def _hf_model_evidence(model_spec: str) -> dict[str, Any]:
    """Resolve identifiers to their local snapshot so weights are fingerprinted."""

    model_path = Path(model_spec)
    if model_path.is_dir():
        return _hf_directory_evidence(model_path)
    if model_path.is_file():
        return fingerprint_file(model_path)

    from transformers.utils import cached_file

    cached_config = cached_file(
        model_spec,
        "config.json",
        local_files_only=True,
    )
    if cached_config is None:
        raise RuntimeError(f"cannot resolve local model snapshot for provenance: {model_spec}")
    evidence = _hf_directory_evidence(Path(cached_config).parent)
    evidence["identifier"] = model_spec
    return evidence


def _contract_metadata(contract: str) -> dict[str, Any]:
    if contract in {"pocketfinancer", "android-prompt-proxy"}:
        return {
            "name": "pocketfinancer_hf_training_evaluation",
            "runtime_parity": False,
            "limitation": (
                "Transformers does not reproduce Android JNI, GGUF quantization, "
                "or optional llama.cpp GBNF sampling."
            ),
            "android_current_prompt_contract": contract_provenance(),
            "decode_defaults": decode_defaults(contract),
        }
    return {
        "name": "legacy_short_prompt",
        "runtime_parity": False,
        "message_builder": "lfm25.prompts.extraction_messages",
        "decode_defaults": decode_defaults("legacy"),
    }


def _execution_contract_evidence(contract: str) -> dict[str, Any]:
    return {
        "profile": _contract_metadata(contract),
        "evaluator": fingerprint_file(Path(__file__)),
        "code_sha256": code_fingerprints(REPO_ROOT),
    }


def _assert_execution_contract_unchanged(expected: Mapping[str, Any], *, contract: str) -> None:
    if _execution_contract_evidence(contract) != expected:
        raise RuntimeError("evaluator code or contract changed during inference")


def _context_summary(prompt_lengths: list[int]) -> dict[str, Any]:
    n_ctx = int(ANDROID_DECODE_DEFAULTS["n_ctx"])
    answer_tokens = int(ANDROID_DECODE_DEFAULTS["answer_max_tokens"])
    compatible = sum(
        context_compatibility(
            length,
            n_ctx=n_ctx,
            generation_tokens=answer_tokens,
        )["compatible"]
        for length in prompt_lengths
    )
    return {
        "android_current_n_ctx": n_ctx,
        "android_current_answer_budget": answer_tokens,
        "measured_rows": len(prompt_lengths),
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
        "compatible_rows": compatible,
        "incompatible_rows": len(prompt_lengths) - compatible,
        "runtime_parity": False,
    }


def _runtime_context_summary(
    prompt_lengths: list[int],
    n_ctx: int,
    answer_tokens: int,
    thinking_enabled: bool,
    closing_tokens: int,
    effective_budgets: list[int],
) -> dict[str, Any]:
    generation_reservation = answer_tokens + (closing_tokens if thinking_enabled else 0)
    compatible = sum(
        context_compatibility(
            length,
            n_ctx=n_ctx,
            generation_tokens=generation_reservation,
        )["compatible"]
        for length in prompt_lengths
    )
    return {
        "android_current_n_ctx": int(ANDROID_DECODE_DEFAULTS["n_ctx"]),
        "configured_n_ctx": n_ctx,
        "uses_android_current_3072_context": n_ctx == 3072,
        "answer_budget_tokens": answer_tokens,
        "android_current_answer_budget": answer_tokens,
        "thinking_close_tokens": closing_tokens if thinking_enabled else 0,
        "measured_rows": len(prompt_lengths),
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else None,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
        "prompt_plus_answer_compatible_rows": compatible,
        "prompt_plus_answer_incompatible_rows": len(prompt_lengths) - compatible,
        "compatible_rows": compatible,
        "incompatible_rows": len(prompt_lengths) - compatible,
        "effective_thinking_budget_min": (min(effective_budgets) if effective_budgets else None),
        "effective_thinking_budget_max": (max(effective_budgets) if effective_budgets else None),
        "runtime_parity": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "HF evaluation using PocketFinancer's exact prompt, prefilter, and "
            "direct-answer profile. Final runtime parity requires GGUF/device testing."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        choices=("pocketfinancer", "android-prompt-proxy", "legacy"),
        default="pocketfinancer",
    )
    parser.add_argument(
        "--apply-prefilter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("PocketFinancer SMS prefilter; enabled by default for the app profile"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--model-config",
        type=Path,
        help="optional JSON model config containing an explicit thinking flag",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto reads an explicit model config and otherwise treats LFM2.5-350M as non-thinking"
        ),
    )
    parser.add_argument(
        "--thinking-tokens",
        type=int,
        help="phase-one maximum before the per-row n_ctx cap",
    )
    args = parser.parse_args()

    model_config = None
    if args.model_config is not None:
        model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
        if not isinstance(model_config, dict):
            parser.error("--model-config must contain a JSON object")
    thinking_enabled, thinking_mode_source = _resolve_thinking_mode(
        args.thinking_mode,
        model_config,
        args.model,
    )

    defaults = decode_defaults(args.contract)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else defaults.get("answer_max_tokens", defaults.get("max_tokens", 256))
    )
    repeat_penalty = (
        args.repeat_penalty
        if args.repeat_penalty is not None
        else defaults.get("repeat_penalty", 1.0)
    )
    n_ctx = args.n_ctx if args.n_ctx is not None else defaults["n_ctx"]
    apply_prefilter = should_apply_prefilter(args.contract, args.apply_prefilter)
    pocketfinancer_profile = args.contract in {
        "pocketfinancer",
        "android-prompt-proxy",
    }
    if not pocketfinancer_profile:
        thinking_enabled = False
        thinking_mode_source = "legacy_contract"
    if thinking_enabled:
        thinking_tokens = (
            args.thinking_tokens
            if args.thinking_tokens is not None
            else int(defaults["thinking_max_tokens"])
        )
    else:
        thinking_tokens = 0

    if args.thinking_tokens is not None and args.thinking_tokens < 0:
        parser.error("--thinking-tokens cannot be negative")
    if thinking_enabled and thinking_tokens <= 0:
        parser.error("--thinking-tokens must be positive when thinking mode is on")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if repeat_penalty <= 0:
        parser.error("--repeat-penalty must be positive")
    if n_ctx <= max_new_tokens:
        parser.error("--n-ctx must be greater than --max-new-tokens")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative")

    input_evidence = _evaluation_input_evidence(
        model=args.model,
        adapter=args.adapter,
        dataset=args.dataset,
        model_lock=args.model_lock,
    )
    execution_contract_evidence = _execution_contract_evidence(args.contract)
    model_config_evidence = (
        fingerprint_file(args.model_config) if args.model_config is not None else None
    )
    rows = _read_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    _assert_evaluation_inputs_unchanged(
        input_evidence,
        model=args.model,
        adapter=args.adapter,
        dataset=args.dataset,
        model_lock=args.model_lock,
    )

    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.to("cuda").eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class StopOnTokenSequence(StoppingCriteria):
        def __init__(self, stop_ids: list[int]):
            self.stop_ids = stop_ids

        def __call__(self, input_ids, scores, **kwargs):
            del scores, kwargs
            if input_ids.shape[1] < len(self.stop_ids):
                return input_ids.new_zeros(input_ids.shape[0], dtype=torch.bool)
            suffix = input_ids[:, -len(self.stop_ids) :]
            stop = input_ids.new_tensor(self.stop_ids)
            return (suffix == stop).all(dim=1)

    records_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        disposition = (
            selection_prefilter_sms(str(row.get("sender", "")), str(row.get("sms", "")))
            if apply_prefilter
            else None
        )
        passed = disposition.accepted if disposition is not None else True
        common = {
            "id": row.get("id", row.get("public_id", index)),
            "gold": _gold(row),
            "selection_prefilter_applied": apply_prefilter,
            "prefilter_applied": apply_prefilter,
            "prefilter_passed": passed,
            "prefilter_rejection_stage": (
                disposition.rejection_stage if disposition is not None else None
            ),
            "prefilter_rejection_reason": (
                disposition.rejection_reason if disposition is not None else None
            ),
            "model_invoked": passed,
            **{key: row[key] for key in ("class", "template_family") if key in row},
        }
        if passed:
            pending.append((index, row))
            records_by_index[index] = common
        else:
            records_by_index[index] = {
                **common,
                "prediction": "null",
                "app_prediction": "null",
                "prompt_tokens": 0,
                "output_tokens": 0,
                "latency_ms_batch_amortized": 0.0,
            }

    latencies_ms: list[float] = []
    output_lengths: list[int] = []
    prompt_lengths: list[int] = []
    thinking_latencies_ms: list[float] = []
    answer_latencies_ms: list[float] = []
    thinking_lengths: list[int] = []
    effective_thinking_budgets: list[int] = []
    context_capped_rows = 0
    template_think_appended_rows = 0

    if thinking_enabled:
        thinking_stop_ids = tokenizer.encode("</think>", add_special_tokens=False)
        thinking_close_ids = tokenizer.encode("</think>\n", add_special_tokens=False)
        if not thinking_stop_ids or not thinking_close_ids:
            raise ValueError("tokenizer cannot encode the thinking close delimiter")

        for index, row in pending:
            messages = android_extraction_messages(
                str(row.get("sender", "")), str(row.get("sms", ""))
            )
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(rendered, str):
                raise TypeError("chat template did not render a string prompt")
            rendered, template_think_appended = _prepare_thinking_prompt(rendered)
            template_think_appended_rows += int(template_think_appended)
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            prompt_length = int(encoded["attention_mask"].sum().item())
            prompt_width = int(encoded["input_ids"].shape[1])
            budget = _thinking_context_budget(
                prompt_length,
                n_ctx,
                max_new_tokens,
                len(thinking_close_ids),
                thinking_tokens,
            )
            effective_budget = int(budget["effective_thinking_max_tokens"])
            effective_thinking_budgets.append(effective_budget)
            context_capped_rows += int(budget["thinking_capped_by_context"])

            torch.cuda.synchronize()
            thinking_started = time.perf_counter()
            with torch.inference_mode():
                phase_one = model.generate(
                    **encoded,
                    max_new_tokens=effective_budget,
                    do_sample=False,
                    repetition_penalty=repeat_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [StopOnTokenSequence(thinking_stop_ids)]
                    ),
                )
            torch.cuda.synchronize()
            thinking_latency_ms = (time.perf_counter() - thinking_started) * 1000
            phase_one_ids = phase_one[0, prompt_width:].tolist()
            reasoning_ids, thinking_stop_found = _split_at_token_sequence(
                phase_one_ids, thinking_stop_ids
            )
            trailing_special = {tokenizer.eos_token_id, tokenizer.pad_token_id}
            while reasoning_ids and reasoning_ids[-1] in trailing_special:
                reasoning_ids.pop()
            raw_reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()

            continuation_ids = reasoning_ids + thinking_close_ids
            continuation = encoded["input_ids"].new_tensor([continuation_ids])
            answer_input_ids = torch.cat([encoded["input_ids"], continuation], dim=1)
            answer_attention_mask = torch.ones_like(answer_input_ids)
            answer_prompt_length = int(answer_input_ids.shape[1])
            if answer_prompt_length + max_new_tokens > n_ctx:
                raise ValueError(
                    f"answer prompt ({answer_prompt_length}) + answer "
                    f"({max_new_tokens}) exceeds n_ctx={n_ctx}"
                )

            torch.cuda.synchronize()
            answer_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=answer_input_ids,
                    attention_mask=answer_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=repeat_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            answer_latency_ms = (time.perf_counter() - answer_started) * 1000
            answer_ids = generated[0, answer_prompt_length:]
            prediction = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            thinking_length = len(reasoning_ids)
            thinking_generation_length = len(phase_one_ids)
            answer_length = int(answer_ids.shape[0])
            elapsed_ms = thinking_latency_ms + answer_latency_ms
            latencies_ms.append(elapsed_ms)
            thinking_latencies_ms.append(thinking_latency_ms)
            answer_latencies_ms.append(answer_latency_ms)
            thinking_lengths.append(thinking_length)
            output_lengths.append(thinking_generation_length + answer_length)
            prompt_lengths.append(prompt_length)
            records_by_index[index].update(
                {
                    "prediction": prediction,
                    "app_prediction": pocketfinancer_normalize_prediction(prediction),
                    "raw_reasoning": raw_reasoning,
                    "prompt_tokens": prompt_length,
                    "answer_prompt_tokens": answer_prompt_length,
                    "thinking_tokens": thinking_length,
                    "thinking_generation_tokens": thinking_generation_length,
                    "answer_tokens": answer_length,
                    "output_tokens": thinking_generation_length + answer_length,
                    "thinking_latency_ms": round(thinking_latency_ms, 3),
                    "answer_latency_ms": round(answer_latency_ms, 3),
                    "latency_ms_batch_amortized": round(elapsed_ms, 3),
                    "thinking_stop_found": thinking_stop_found,
                    "template_think_appended": template_think_appended,
                    "thinking_context": budget,
                }
            )

        pending = []

    for start in range(0, len(pending), args.batch_size):
        batch_items = pending[start : start + args.batch_size]
        batch = [row for _, row in batch_items]
        if args.contract in {"pocketfinancer", "android-prompt-proxy"}:
            chats = [
                android_extraction_messages(str(row.get("sender", "")), str(row.get("sms", "")))
                for row in batch
            ]
        else:
            chats = [
                extraction_messages(str(row.get("sender", "")), str(row.get("sms", "")))
                for row in batch
            ]
        encoded = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        longest_prompt = int(encoded["attention_mask"].sum(dim=1).max().item())
        if longest_prompt + max_new_tokens > n_ctx:
            raise ValueError(
                f"proxy prompt ({longest_prompt}) + completion ({max_new_tokens}) "
                f"exceeds proxy n_ctx={n_ctx}; this guard does not change context size"
            )

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repeat_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        completions = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for offset, ((index, _row), prediction) in enumerate(zip(batch_items, decoded)):
            completion_ids = completions[offset]
            completion_length = int((completion_ids != tokenizer.pad_token_id).sum().item())
            prompt_length = int(encoded["attention_mask"][offset].sum().item())
            per_item_latency = elapsed_ms / len(batch)
            latencies_ms.append(per_item_latency)
            output_lengths.append(completion_length)
            prompt_lengths.append(prompt_length)
            records_by_index[index].update(
                {
                    "prediction": prediction.strip(),
                    "app_prediction": pocketfinancer_normalize_prediction(prediction.strip())
                    if pocketfinancer_profile
                    else prediction.strip(),
                    "prompt_tokens": prompt_length,
                    "output_tokens": completion_length,
                    "latency_ms_batch_amortized": round(per_item_latency, 3),
                }
            )

    output_records = [records_by_index[index] for index in range(len(rows))]
    model_records = [record for record in output_records if record["model_invoked"]]
    strict_whole_pipeline = score_records(output_records, slice_keys=("template_family", "class"))
    strict_conditional_model = score_records(model_records, slice_keys=("template_family", "class"))
    if pocketfinancer_profile:
        whole_pipeline = score_records(
            output_records,
            prediction_parser=pocketfinancer_parse_prediction,
            slice_keys=("template_family", "class"),
        )
        conditional_model = score_records(
            model_records,
            prediction_parser=pocketfinancer_parse_prediction,
            slice_keys=("template_family", "class"),
        )
    else:
        whole_pipeline = strict_whole_pipeline
        conditional_model = strict_conditional_model
    scored = dict(whole_pipeline)
    scored["whole_pipeline"] = whole_pipeline
    scored["conditional_model"] = conditional_model
    scored["app_interpreted"] = {
        "whole_pipeline": whole_pipeline,
        "conditional_model": conditional_model,
    }
    scored["strict_model_output"] = {
        "whole_pipeline": strict_whole_pipeline,
        "conditional_model": strict_conditional_model,
    }
    scored["selection_prefilter"] = summarize_prefilter(output_records, enabled=apply_prefilter)
    scored["prefilter"] = scored["selection_prefilter"]
    scored["android_current_context_compatibility"] = _runtime_context_summary(
        prompt_lengths,
        n_ctx,
        max_new_tokens,
        thinking_enabled,
        len(thinking_close_ids) if thinking_enabled else 0,
        effective_thinking_budgets,
    )
    scored["runtime"] = {
        "profile": args.contract,
        "android_runtime_parity": False,
        "prefilter_applied": apply_prefilter,
        "rows": len(output_records),
        "model_invocations": len(model_records),
        "batch_size": args.batch_size,
        "n_ctx": n_ctx,
        "max_new_tokens": max_new_tokens,
        "latency_ms_p50_batch_amortized": _percentile(latencies_ms, 0.5),
        "latency_ms_p95_batch_amortized": _percentile(latencies_ms, 0.95),
        "prompt_tokens_mean": round(statistics.fmean(prompt_lengths), 2)
        if prompt_lengths
        else None,
        "output_tokens_mean": round(statistics.fmean(output_lengths), 2)
        if output_lengths
        else None,
        "output_tokens_max": max(output_lengths) if output_lengths else None,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }

    scored["runtime"].update(
        {
            "thinking_mode": thinking_enabled,
            "thinking_mode_source": thinking_mode_source,
            "generation_path": ("sequential_two_pass" if thinking_enabled else "batched_direct"),
            "effective_batch_size": 1 if thinking_enabled else args.batch_size,
            "thinking_max_tokens_requested": (thinking_tokens if thinking_enabled else None),
            "thinking_context_capped_rows": context_capped_rows,
            "thinking_stop_found_rows": sum(
                int(record.get("thinking_stop_found", False)) for record in model_records
            ),
            "template_think_appended_rows": template_think_appended_rows,
            "thinking_tokens_mean": (
                round(statistics.fmean(thinking_lengths), 2) if thinking_lengths else None
            ),
            "thinking_tokens_max": (max(thinking_lengths) if thinking_lengths else None),
            "thinking_latency_ms_p50": _percentile(thinking_latencies_ms, 0.5),
            "thinking_latency_ms_p95": _percentile(thinking_latencies_ms, 0.95),
            "answer_latency_ms_p50": _percentile(answer_latencies_ms, 0.5),
            "answer_latency_ms_p95": _percentile(answer_latencies_ms, 0.95),
        }
    )

    dataset_evidence = {
        **input_evidence["dataset"],
        "row_count": len(rows),
        "row_limit": args.limit,
    }
    scored["provenance"] = {
        "profile": execution_contract_evidence["profile"],
        "evaluator": execution_contract_evidence["evaluator"],
        "model": input_evidence["model"],
        "model_lock": input_evidence["model_lock"],
        "adapter": input_evidence["adapter"],
        "training_run": input_evidence["training_run"],
        "dataset": dataset_evidence,
        "code_sha256": execution_contract_evidence["code_sha256"],
        "decode": {
            "engine": "transformers_prompt_training_proxy",
            "android_runtime_parity": False,
            "grammar_constrained": False,
            "do_sample": False,
            "repetition_penalty": repeat_penalty,
            "max_new_tokens": max_new_tokens,
            "n_ctx": n_ctx,
            "seed": args.seed,
        },
        "selection_prefilter": {
            "applied": apply_prefilter,
            "part_of_android_current": args.contract == "pocketfinancer",
            "rejected_prediction": "null",
        },
        "output_interpretation": {
            "primary_metrics": "pocketfinancer_kotlin_parser"
            if pocketfinancer_profile
            else "strict_research_parser",
            "strict_raw_metrics_also_reported": True,
        },
    }

    scored["provenance"]["model_config"] = model_config_evidence
    scored["provenance"]["decode"].update(
        {
            "transformers_version": transformers.__version__,
            "two_pass": thinking_enabled,
            "thinking_mode_source": thinking_mode_source,
            "thinking_max_tokens_requested": (thinking_tokens if thinking_enabled else None),
            "thinking_budget_policy": (
                "reserve_close_and_answer_then_cap_per_row" if thinking_enabled else None
            ),
            "thinking_stop": ["</think>"] if thinking_enabled else None,
            "answer_stop": None,
            "template_think_open_policy": (
                "reuse_unclosed_template_tag_else_append_once" if thinking_enabled else None
            ),
            "template_think_appended_rows": template_think_appended_rows,
            "thinking_context_capped_rows": context_capped_rows,
            "raw_reasoning_persistence": ("samples_jsonl_only" if thinking_enabled else None),
        }
    )

    _assert_evaluation_inputs_unchanged(
        input_evidence,
        model=args.model,
        adapter=args.adapter,
        dataset=args.dataset,
        model_lock=args.model_lock,
    )
    _assert_execution_contract_unchanged(
        execution_contract_evidence,
        contract=args.contract,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(scored, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(scored, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
