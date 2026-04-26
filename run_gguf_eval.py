"""
Entry script: run the sms_extraction task with a GGUF model via llama-cpp-python,
optionally GBNF-grammar-constrained. Produces lm-eval-shaped outputs under
RESULTS/llamacpp/<model_slug>/ so they compare directly to HF baselines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "DATA"

# Make `import llamacpp_model` work (it lives in DATA/ alongside utils.py, which
# lm-eval will also need via --include_path).
sys.path.insert(0, str(DATA_DIR))

import llamacpp_model  # noqa: F401 — side-effect: @register_model("llamacpp")

import lm_eval
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager


def _model_slug(hf_id: str) -> str:
    return hf_id.replace("/", "__")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="HF repo id of the source model (for tokenizer + chat template, e.g. google/gemma-4-E2B-it).",
    )
    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to the GGUF weights file (e.g. MODELS/gemma-4-E2B-it-Q4_K_M.gguf).",
    )
    parser.add_argument(
        "--grammar",
        default=None,
        help="Path to a GBNF grammar file. Omit for unconstrained generation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap eval samples (smoke tests).")
    parser.add_argument("--n-ctx", type=int, default=0,
                        help="Context window size. 0 (default) = auto-detect from GGUF metadata.")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help="Thinking-mode generation. 'auto' detects from chat template; "
             "'on' forces two-phase (think → JSON); 'off' disables.",
    )
    parser.add_argument(
        "--thinking-max-tokens",
        type=int,
        default=4096,
        help="Token budget for the thinking phase (phase 1).",
    )
    parser.add_argument(
        "--thinking-repeat-penalty",
        type=float,
        default=1.1,
        help="repeat_penalty for the thinking phase. 1.0 = off; 1.1 breaks "
             "small-model verbatim loops without perturbing reasoning.",
    )
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "RESULTS" / "llamacpp"),
        help="Base directory for lm-eval output.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose llama.cpp logs.")
    args = parser.parse_args()

    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        parser.error(f"GGUF not found: {gguf_path}")
    if args.grammar:
        grammar_path = Path(args.grammar).resolve()
        if not grammar_path.exists():
            parser.error(f"Grammar not found: {grammar_path}")
    else:
        grammar_path = None

    model_args: dict[str, str | int | bool] = {
        "path": str(gguf_path),
        "tokenizer": args.model,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "thinking_max_tokens": args.thinking_max_tokens,
        "thinking_repeat_penalty": args.thinking_repeat_penalty,
        "verbose": args.verbose,
    }
    if grammar_path is not None:
        model_args["grammar_file"] = str(grammar_path)

    task_manager = TaskManager(include_path=str(DATA_DIR))

    tracker = EvaluationTracker(output_path=args.output_path)

    print(f"[run_gguf_eval] model={args.model} gguf={gguf_path.name} "
          f"grammar={'yes' if grammar_path else 'no'} thinking={args.thinking} "
          f"limit={args.limit}")

    results = lm_eval.simple_evaluate(
        model="llamacpp",
        model_args=model_args,
        tasks=["sms_extraction"],
        limit=args.limit,
        log_samples=True,
        apply_chat_template=True,
        task_manager=task_manager,
        evaluation_tracker=tracker,
    )

    if results is None:
        print("[run_gguf_eval] simple_evaluate returned None", file=sys.stderr)
        return 1

    # Persist — EvaluationTracker writes results_<ts>.json and samples_<task>_<ts>.jsonl
    # under output_path/<model_slug>/.
    tracker.general_config_tracker.model_source = "llamacpp"
    tracker.general_config_tracker.model_name = args.model
    tracker.general_config_tracker.model_name_sanitized = _model_slug(args.model)
    samples = results.pop("samples", None)
    tracker.save_results_aggregated(results=results, samples=samples)
    if samples is not None:
        for task_name, task_samples in samples.items():
            tracker.save_results_samples(task_name=task_name, samples=task_samples)

    # Top-level metric summary for quick scanning.
    # Keys in results include the filter suffix (",extract_json"), matching
    # the HF-baseline output format.
    task_results = results.get("results", {}).get("sms_extraction", {})
    headline = [
        "full_match_accuracy",
        "ghost_transaction_rate",
        "missed_transaction_rate",
        "json_validity",
    ]
    print("[run_gguf_eval] key metrics:")
    for name in headline:
        for full_key, v in task_results.items():
            if full_key.startswith(name + ",") and "_stderr" not in full_key:
                print(f"  {name:30s} = {v:.4f}")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
