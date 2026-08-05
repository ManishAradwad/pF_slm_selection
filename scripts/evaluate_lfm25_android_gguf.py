#!/usr/bin/env python3
"""Evaluate any GGUF using the current PocketFinancer Android contract.

The primary profile runs the app's prefilter, uses the GGUF's built-in chat
template, and selects direct or two-pass generation from model configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.android_contract import (  # noqa: E402
    POCKETFINANCER_CONTRACT_ALIASES,
    android_extraction_messages,
    contract_provenance,
    decode_defaults,
    pocketfinancer_normalize_prediction,
    pocketfinancer_parse_prediction,
    pocketfinancer_prefilter_sms,
    should_apply_prefilter,
    summarize_prefilter,
)
from lfm25.metrics import score_records  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402
from lfm25.provenance import code_fingerprints, fingerprint_file  # noqa: E402


class GpuMemorySampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.baseline_mib = 0
        self.peak_total_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            used = int(completed.stdout.splitlines()[0].strip())
            self.peak_total_mib = max(self.peak_total_mib, used)
        except (IndexError, OSError, ValueError, subprocess.SubprocessError):
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._sample()
        self.baseline_mib = self.peak_total_mib
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=3)
        return False


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
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


def _contract_metadata(contract: str) -> dict[str, Any]:
    if contract in POCKETFINANCER_CONTRACT_ALIASES:
        return {
            **contract_provenance(),
            "evaluator_profile": "pocketfinancer_gguf",
        }
    return {
        "name": "legacy_short_prompt",
        "message_builder": "lfm25.prompts.extraction_messages",
        "decode_defaults": decode_defaults("legacy"),
    }


_THINKING_KEYS = {
    "has_thinking_mode",
    "hasThinkingMode",
    "thinking",
    "always_reasons_before_answer",
}


def _thinking_flags(value: Any) -> list[bool]:
    flags: list[bool] = []
    if isinstance(value, dict):
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
    *,
    model_config: dict[str, Any] | None,
    gguf: Path,
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

    identity = f"{gguf.name} {json.dumps(model_config or {}, sort_keys=True)}".lower()
    compact = identity.replace("_", "").replace("-", "").replace(".", "")
    if "lfm25350m" in compact:
        return False, "lfm2.5-350m_non_thinking"
    return False, "auto_safe_default_off"


def _fallback_chat_prompt(
    messages: list[dict[str, str]], *, enable_thinking: bool
) -> str:
    """Mirror PromptBuilder's Qwen3 fallback when GGUF metadata has no template."""

    system = messages[0]["content"]
    if enable_thinking:
        system += (
            "\n\nPlease think through this step by step inside <think> tags "
            "before giving your final answer."
        )
    user = messages[-1]["content"]
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _render_chat_prompt(
    model: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> tuple[str, str]:
    """Render Android's primary GGUF template, with its manual fallback."""

    template = getattr(model, "metadata", {}).get("tokenizer.chat_template")
    if isinstance(template, str) and template:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        def special(token_id: int) -> str:
            if token_id < 0:
                return ""
            return model.detokenize([token_id], special=True).decode(
                "utf-8", errors="replace"
            )

        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=special(model.token_eos()),
            bos_token=special(model.token_bos()),
            add_generation_prompt=True,
        )
        return formatter(messages=messages).prompt, "gguf_builtin"
    return (
        _fallback_chat_prompt(messages, enable_thinking=enable_thinking),
        "qwen3_manual_fallback",
    )


def _remove_template_bos(model: Any, prompt: str) -> tuple[str, bool]:
    """Remove a textual template BOS so create_completion adds exactly one."""

    token_id = model.token_bos()
    if token_id < 0:
        return prompt, False
    bos_text = model.detokenize([token_id], special=True).decode(
        "utf-8", errors="replace"
    )
    if bos_text and prompt.startswith(bos_text):
        return prompt[len(bos_text) :], True
    return prompt, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PocketFinancer GGUF evaluation: Android prefilter, model-aware "
            "direct/two-pass decoding, optional GBNF, and n_ctx=3072."
        )
    )
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument(
        "--tokenizer",
        help="deprecated compatibility option; GGUF tokenizer/template are authoritative",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="optional JSON model config containing an explicit thinking flag",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--grammar", default="DATA/sms_extraction.gbnf", type=Path)
    grammar_group = parser.add_mutually_exclusive_group()
    grammar_group.add_argument("--with-grammar", dest="grammar_enabled", action="store_true")
    grammar_group.add_argument("--no-grammar", dest="grammar_enabled", action="store_false")
    parser.set_defaults(grammar_enabled=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        choices=(
            "pocketfinancer",
            "android-current",
            "android",
            "android-runtime",
            "legacy",
        ),
        default="pocketfinancer",
    )
    parser.add_argument(
        "--apply-prefilter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "PocketFinancer SMS prefilter (enabled by default for the current profile)"
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto reads an explicit model config and otherwise treats "
            "LFM2.5-350M as non-thinking"
        ),
    )
    parser.add_argument(
        "--bos-mode",
        choices=("android-current", "single"),
        default="android-current",
        help=(
            "android-current preserves the app's template-BOS plus add-special "
            "behavior; single is an explicit one-BOS ablation"
        ),
    )
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--thinking-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    defaults = decode_defaults(args.contract)
    pocketfinancer_profile = args.contract in POCKETFINANCER_CONTRACT_ALIASES
    model_config = None
    if args.model_config is not None:
        model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
        if not isinstance(model_config, dict):
            parser.error("--model-config must contain a JSON object")
    thinking_enabled, thinking_mode_source = _resolve_thinking_mode(
        args.thinking_mode,
        model_config=model_config,
        gguf=args.gguf,
    )
    if not pocketfinancer_profile:
        thinking_enabled = False
        thinking_mode_source = "legacy_contract"
    n_ctx = args.n_ctx if args.n_ctx is not None else defaults["n_ctx"]
    n_gpu_layers = (
        args.n_gpu_layers
        if args.n_gpu_layers is not None
        else defaults.get("n_gpu_layers", -1)
    )
    if pocketfinancer_profile and thinking_enabled:
        thinking_tokens = (
            args.thinking_tokens
            if args.thinking_tokens is not None
            else defaults["thinking_max_tokens"]
        )
        default_answer_tokens = defaults["answer_max_tokens"]
    elif pocketfinancer_profile:
        thinking_tokens = 0
        default_answer_tokens = defaults["answer_max_tokens"]
    else:
        thinking_tokens = 0
        default_answer_tokens = defaults["max_tokens"]
    max_tokens = args.max_tokens if args.max_tokens is not None else default_answer_tokens
    repeat_penalty = (
        args.repeat_penalty
        if args.repeat_penalty is not None
        else defaults.get("repeat_penalty", 1.0)
    )
    grammar_enabled = (
        args.grammar_enabled if args.grammar_enabled is not None else defaults["grammar"]
    )
    apply_prefilter = should_apply_prefilter(args.contract, args.apply_prefilter)
    n_threads = (
        args.n_threads
        if args.n_threads is not None
        else min(os.cpu_count() or 1, defaults.get("max_cpu_threads", 4))
    )
    if n_ctx <= max_tokens:
        parser.error("--n-ctx must be greater than --max-tokens")
    if thinking_tokens < 0:
        parser.error("--thinking-tokens cannot be negative")
    if thinking_enabled and thinking_tokens <= 0:
        parser.error("--thinking-tokens must be positive when thinking mode is on")
    if max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if repeat_penalty <= 0:
        parser.error("--repeat-penalty must be positive")
    if n_threads <= 0:
        parser.error("--n-threads must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    from llama_cpp import Llama, LlamaGrammar
    if not args.gguf.is_file():
        raise FileNotFoundError(args.gguf)
    rows = _read_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    grammar = LlamaGrammar.from_file(str(args.grammar)) if grammar_enabled else None

    records_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        disposition = (
            pocketfinancer_prefilter_sms(
                str(row.get("sender", "")), str(row.get("sms", ""))
            )
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
                "latency_ms": 0.0,
            }

    latencies_ms: list[float] = []
    prompt_tokens: list[int] = []
    output_tokens: list[int] = []
    total_elapsed = 0.0
    total_prompt_tokens = 0
    total_output_tokens = 0
    template_bos_removed_rows = 0

    with GpuMemorySampler() as gpu_sampler:
        load_started = time.perf_counter()
        model = Llama(
            model_path=str(args.gguf.resolve()),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_batch=defaults.get("n_batch", 512),
            n_ubatch=defaults.get("n_ubatch", 256),
            flash_attn=defaults.get("flash_attention", False),
            use_mmap=defaults.get("use_mmap", True),
            seed=args.seed,
            verbose=False,
        )
        load_seconds = time.perf_counter() - load_started

        for model_index, (index, row) in enumerate(pending):
            if pocketfinancer_profile:
                messages = android_extraction_messages(
                    str(row.get("sender", "")), str(row.get("sms", ""))
                )
            else:
                messages = extraction_messages(
                    str(row.get("sender", "")), str(row.get("sms", ""))
                )
            prompt, chat_template_source = _render_chat_prompt(
                model,
                messages,
                enable_thinking=thinking_enabled,
            )
            template_bos_removed = False
            if args.bos_mode == "single":
                prompt, template_bos_removed = _remove_template_bos(model, prompt)
                template_bos_removed_rows += int(template_bos_removed)
            started = time.perf_counter()
            thinking_count = 0
            if thinking_enabled:
                think_prompt = prompt + "<think>\n"
                think_response = model.create_completion(
                    prompt=think_prompt,
                    temperature=0.0,
                    max_tokens=thinking_tokens,
                    stop=["</think>"],
                    echo=False,
                    repeat_penalty=repeat_penalty,
                    seed=args.seed,
                )
                think_text = str(think_response["choices"][0]["text"])
                thinking_count = int(
                    think_response.get("usage", {}).get("completion_tokens", 0)
                )
                answer_prompt = think_prompt + think_text + "</think>\n"
                response = model.create_completion(
                    prompt=answer_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    echo=False,
                    repeat_penalty=repeat_penalty,
                    grammar=grammar,
                    seed=args.seed,
                )
            else:
                response = model.create_completion(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    echo=False,
                    repeat_penalty=repeat_penalty,
                    grammar=grammar,
                    seed=args.seed,
                )
            elapsed = time.perf_counter() - started
            usage = response.get("usage", {})
            prompt_count = int(usage.get("prompt_tokens", 0))
            answer_count = int(usage.get("completion_tokens", 0))
            output_count = thinking_count + answer_count
            prediction = str(response["choices"][0]["text"]).strip()
            if model_index >= args.warmup:
                latencies_ms.append(elapsed * 1000)
                prompt_tokens.append(prompt_count)
                output_tokens.append(output_count)
                total_elapsed += elapsed
                total_prompt_tokens += prompt_count
                total_output_tokens += output_count
            records_by_index[index].update(
                {
                    "prediction": prediction,
                    "app_prediction": pocketfinancer_normalize_prediction(prediction)
                    if pocketfinancer_profile
                    else prediction,
                    "prompt_tokens": prompt_count,
                    "thinking_tokens": thinking_count,
                    "answer_tokens": answer_count,
                    "output_tokens": output_count,
                    "latency_ms": round(elapsed * 1000, 3),
                    "chat_template_source": chat_template_source,
                    "bos_mode": args.bos_mode,
                    "template_bos_removed": template_bos_removed,
                }
            )

    records = [records_by_index[index] for index in range(len(rows))]
    model_records = [record for record in records if record["model_invoked"]]
    strict_whole_pipeline = score_records(
        records, slice_keys=("template_family", "class")
    )
    strict_conditional_model = score_records(
        model_records, slice_keys=("template_family", "class")
    )
    if pocketfinancer_profile:
        whole_pipeline = score_records(
            records,
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
    metrics = dict(whole_pipeline)
    metrics["whole_pipeline"] = whole_pipeline
    metrics["conditional_model"] = conditional_model
    metrics["app_interpreted"] = {
        "whole_pipeline": whole_pipeline,
        "conditional_model": conditional_model,
    }
    metrics["strict_model_output"] = {
        "whole_pipeline": strict_whole_pipeline,
        "conditional_model": strict_conditional_model,
    }
    metrics["selection_prefilter"] = summarize_prefilter(
        records, enabled=apply_prefilter
    )
    metrics["prefilter"] = metrics["selection_prefilter"]
    measured = max(0, len(model_records) - args.warmup)
    metrics["runtime"] = {
        "profile": args.contract,
        "android_runtime_parity": False,
        "pocketfinancer_default_configuration": (
            pocketfinancer_profile
            and n_ctx == 3072
            and max_tokens == 256
            and not grammar_enabled
            and apply_prefilter
            and args.bos_mode == "android-current"
            and n_gpu_layers == 0
            and n_threads <= 4
        ),
        "selection_prefilter_applied": apply_prefilter,
        "rows": len(records),
        "model_invocations": len(model_records),
        "model_size_bytes": args.gguf.stat().st_size,
        "cold_process_load_seconds": round(load_seconds, 4),
        "n_ctx": n_ctx,
        "thinking_mode": thinking_enabled,
        "thinking_mode_source": thinking_mode_source,
        "bos_mode": args.bos_mode,
        "template_bos_removed_rows": template_bos_removed_rows,
        "thinking_max_tokens": thinking_tokens
        if thinking_enabled
        else None,
        "answer_max_tokens": max_tokens,
        "warmup_model_calls_excluded_from_timing": min(args.warmup, len(model_records)),
        "timed_model_calls": measured,
        "end_to_end_latency_ms_p50": _percentile(latencies_ms, 0.5),
        "end_to_end_latency_ms_p95": _percentile(latencies_ms, 0.95),
        "end_to_end_latency_ms_mean": round(statistics.fmean(latencies_ms), 3)
        if latencies_ms
        else None,
        "prompt_tokens_mean": round(statistics.fmean(prompt_tokens), 2)
        if prompt_tokens
        else None,
        "output_tokens_mean": round(statistics.fmean(output_tokens), 2)
        if output_tokens
        else None,
        "output_tokens_max": max(output_tokens) if output_tokens else None,
        "prompt_tokens_per_end_to_end_second": round(total_prompt_tokens / total_elapsed, 2)
        if total_elapsed
        else None,
        "output_tokens_per_end_to_end_second": round(total_output_tokens / total_elapsed, 2)
        if total_elapsed
        else None,
        "peak_process_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "gpu_memory_baseline_mib": gpu_sampler.baseline_mib,
        "gpu_memory_peak_total_mib": gpu_sampler.peak_total_mib,
        "gpu_memory_peak_delta_mib": max(
            0, gpu_sampler.peak_total_mib - gpu_sampler.baseline_mib
        ),
    }
    metrics["provenance"] = {
        "profile": _contract_metadata(args.contract),
        "evaluator": fingerprint_file(Path(__file__)),
        "model": fingerprint_file(args.gguf),
        "dataset": fingerprint_file(args.dataset),
        "tokenizer": "gguf_builtin",
        "deprecated_tokenizer_argument": args.tokenizer,
        "model_config": fingerprint_file(args.model_config)
        if args.model_config is not None
        else None,
        "grammar": fingerprint_file(args.grammar) if grammar_enabled else None,
        "code_sha256": code_fingerprints(REPO_ROOT),
        "decode": {
            "engine": "llama_cpp_python",
            "two_pass": thinking_enabled,
            "thinking_mode_source": thinking_mode_source,
            "thinking_grammar_constrained": False
            if thinking_enabled
            else None,
            "answer_grammar_constrained": grammar_enabled,
            "temperature": 0.0,
            "repeat_penalty": repeat_penalty,
            "thinking_max_tokens": thinking_tokens
            if thinking_enabled
            else None,
            "answer_max_tokens": max_tokens,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads,
            "n_threads_batch": n_threads,
            "n_batch": defaults.get("n_batch"),
            "n_ubatch": defaults.get("n_ubatch"),
            "flash_attention": defaults.get("flash_attention"),
            "use_mmap": defaults.get("use_mmap"),
            "seed": args.seed,
            "bos_mode": args.bos_mode,
            "template_bos_removed_rows": template_bos_removed_rows,
            "thinking_stop": ["</think>"]
            if thinking_enabled
            else None,
            "answer_stop": None,
        },
        "selection_prefilter": {
            "applied": apply_prefilter,
            "part_of_pocketfinancer": pocketfinancer_profile,
            "rejected_prediction": "null",
        },
        "output_interpretation": {
            "primary_metrics": "pocketfinancer_kotlin_parser"
            if pocketfinancer_profile
            else "strict_research_parser",
            "strict_raw_metrics_also_reported": True,
        },
        "parity_gaps": [
            "llama-cpp-python stop strings exclude </think>; Android JNI returns it",
            "llama-cpp-python and Android JNI have different KV-cache handling",
        ],
        "row_limit": args.limit,
        "row_count": len(rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
