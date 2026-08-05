#!/usr/bin/env python3
"""Short-prompt grammar-constrained GGUF evaluation and latency measurement."""

from __future__ import annotations

import argparse
import json
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

from lfm25.metrics import score_records  # noqa: E402
from lfm25.provenance import code_fingerprints, fingerprint_file  # noqa: E402
from lfm25.prompts import extraction_messages  # noqa: E402


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokenizer", default="TRAINING_ARTIFACTS/base/LFM2.5-350M")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--grammar", default="DATA/sms_extraction.gbnf", type=Path)
    parser.add_argument("--no-grammar", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from llama_cpp import Llama, LlamaGrammar
    from transformers import AutoTokenizer

    if not args.gguf.is_file():
        raise FileNotFoundError(args.gguf)
    rows = _read_rows(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    grammar = None if args.no_grammar else LlamaGrammar.from_file(str(args.grammar))

    with GpuMemorySampler() as gpu_sampler:
        load_started = time.perf_counter()
        model = Llama(
            model_path=str(args.gguf.resolve()),
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
            verbose=False,
        )
        load_seconds = time.perf_counter() - load_started

        records: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        prompt_tokens: list[int] = []
        output_tokens: list[int] = []
        total_elapsed = 0.0
        total_prompt_tokens = 0
        total_output_tokens = 0

        for index, row in enumerate(rows):
            messages = extraction_messages(str(row.get("sender", "")), str(row.get("sms", "")))
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            bos = tokenizer.bos_token
            if bos and prompt.startswith(bos):
                prompt = prompt[len(bos) :]
            started = time.perf_counter()
            generation_kwargs = dict(
                prompt=prompt,
                temperature=0.0,
                max_tokens=args.max_tokens,
                stop=["<|im_end|>"],
                echo=False,
                repeat_penalty=1.05,
            )
            if grammar is not None:
                generation_kwargs["grammar"] = grammar
            response = model.create_completion(**generation_kwargs)
            elapsed = time.perf_counter() - started
            usage = response.get("usage", {})
            prompt_count = int(usage.get("prompt_tokens", 0))
            output_count = int(usage.get("completion_tokens", 0))
            prediction = str(response["choices"][0]["text"]).strip()
            if index >= args.warmup:
                latencies_ms.append(elapsed * 1000)
                prompt_tokens.append(prompt_count)
                output_tokens.append(output_count)
                total_elapsed += elapsed
                total_prompt_tokens += prompt_count
                total_output_tokens += output_count
            records.append(
                {
                    "id": row.get("id", row.get("public_id", index)),
                    "gold": _gold(row),
                    "prediction": prediction,
                    "prompt_tokens": prompt_count,
                    "output_tokens": output_count,
                    "latency_ms": round(elapsed * 1000, 3),
                    **{
                        key: row[key]
                        for key in ("class", "template_family")
                        if key in row
                    },
                }
            )

    metrics = score_records(records, slice_keys=("template_family", "class"))
    measured = max(0, len(records) - args.warmup)
    metrics["runtime"] = {
        "model_size_bytes": args.gguf.stat().st_size,
        "cold_process_load_seconds": round(load_seconds, 4),
        "n_ctx": args.n_ctx,
        "max_tokens": args.max_tokens,
        "warmup_rows_excluded_from_timing": min(args.warmup, len(records)),
        "timed_rows": measured,
        "end_to_end_latency_ms_p50": _percentile(latencies_ms, 0.5),
        "end_to_end_latency_ms_p95": _percentile(latencies_ms, 0.95),
        "end_to_end_latency_ms_mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else None,
        "prompt_tokens_mean": round(statistics.fmean(prompt_tokens), 2) if prompt_tokens else None,
        "output_tokens_mean": round(statistics.fmean(output_tokens), 2) if output_tokens else None,
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
        "model": fingerprint_file(args.gguf),
        "dataset": fingerprint_file(args.dataset),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "grammar": None if args.no_grammar else fingerprint_file(args.grammar),
        "code_sha256": code_fingerprints(REPO_ROOT),
        "decode": {
            "engine": "llama_cpp_python",
            "grammar_constrained": not args.no_grammar,
            "temperature": 0.0,
            "repeat_penalty": 1.05,
            "max_tokens": args.max_tokens,
            "n_ctx": args.n_ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "seed": args.seed,
            "stop": ["<|im_end|>"],
        },
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
