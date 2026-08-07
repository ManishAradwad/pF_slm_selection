"""Configuration and command planning for the PocketFinancer model pipeline.

The public CLI deliberately orchestrates the existing, tested building blocks
instead of duplicating training or evaluation logic.  This module contains no
GPU imports, so plans can be validated in CI and on machines without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PIPELINE_SCHEMA_VERSION = 1
EXECUTION_STAGES = (
    "build-data",
    "evaluate-base-hf",
    "train",
    "evaluate-hf",
    "merge",
    "convert",
    "evaluate-gguf",
)
CANDIDATE_PROTOCOL_V1 = "candidate_protocol_v1"
CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION = "pocketfinancer-candidate-protocol-v1-sft-v1"
CANDIDATE_PROTOCOL_V1_SEEDS = (17, 29, 43)
CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES = (
    "build-candidate-data",
    "evaluate-direct-base-hf",
    "evaluate-selector-base-hf",
    "train-direct",
    "evaluate-direct-hf",
    "train-selector",
    "evaluate-selector-hf",
    "merge-direct",
    "convert-direct",
    "evaluate-direct-gguf",
    "merge-selector",
    "convert-selector",
    "compare-hf-seed-matrix",
)


class PipelineConfigError(ValueError):
    """Raised when a pipeline declaration is incomplete or inconsistent."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineConfigError(f"{name} must be numeric")
    return value


def load_pipeline_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(f"could not read pipeline config {path}: {error}") from error
    config = dict(_mapping(value, "pipeline config"))
    validate_pipeline_config(config)
    return config


def _validate_direct_pipeline_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineConfigError(f"schema_version must be {PIPELINE_SCHEMA_VERSION}")
    _text(config.get("name"), "name")
    _text(config.get("app_profile"), "app_profile")

    model = _mapping(config.get("model"), "model")
    _text(model.get("id"), "model.id")
    _text(model.get("local_path"), "model.local_path")
    _text(model.get("lock"), "model.lock")
    if not isinstance(model.get("thinking_mode"), bool):
        raise PipelineConfigError("model.thinking_mode must be true or false")

    data = _mapping(config.get("data"), "data")
    for key in ("manifest", "output_dir", "train", "dev"):
        _text(data.get(key), f"data.{key}")
    builder = _mapping(data.get("builder"), "data.builder")
    for key in (
        "dev_fraction",
        "seed",
        "minimum_silver_confidence",
        "max_per_template",
        "max_per_category",
        "max_null_to_transaction_ratio",
    ):
        _number(builder.get(key), f"data.builder.{key}")

    training = _mapping(config.get("training"), "training")
    if training.get("method") != "lora":
        raise PipelineConfigError("training.method currently supports only 'lora'")
    for key in ("output_dir", "loss_mode"):
        _text(training.get(key), f"training.{key}")
    for key in (
        "rank",
        "alpha",
        "dropout",
        "learning_rate",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "max_length",
        "first_supervised_token_weight",
        "warmup_ratio",
        "weight_decay",
        "early_stopping_patience",
        "seed",
    ):
        _number(training.get(key), f"training.{key}")
    if training.get("prompt_profile") != "pocketfinancer":
        raise PipelineConfigError(
            "training.prompt_profile must be 'pocketfinancer' on the supported path"
        )

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    for key in (
        "dataset",
        "hf_base_output_dir",
        "hf_output_dir",
        "gguf_output_dir",
    ):
        _text(evaluation.get(key), f"evaluation.{key}")
    _number(evaluation.get("n_ctx"), "evaluation.n_ctx")
    if str(data["train"]) == str(evaluation["dataset"]):
        raise PipelineConfigError("the regression dataset must never be used as training data")
    export = _mapping(config.get("export"), "export")
    for key in ("merged_dir", "gguf_prefix", "deployable_gguf"):
        _text(export.get(key), f"export.{key}")


def is_candidate_protocol(config: Mapping[str, Any]) -> bool:
    """Return whether a declaration is the isolated Candidate Protocol V1 track."""

    return config.get("protocol") == CANDIDATE_PROTOCOL_V1


def execution_stages(config: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES
        if is_candidate_protocol(config)
        else EXECUTION_STAGES
    )


def pipeline_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    if not is_candidate_protocol(config):
        training = _mapping(config.get("training"), "training")
        seed = training.get("seed")
        return (int(seed),) if isinstance(seed, int) and not isinstance(seed, bool) else ()
    training = _mapping(config.get("training"), "training")
    seeds = training.get("seeds")
    if not isinstance(seeds, list):
        return ()
    return tuple(
        int(seed) for seed in seeds if isinstance(seed, int) and not isinstance(seed, bool)
    )


def _seed_template(value: Any, name: str) -> str:
    template = _text(value, name)
    if "{seed}" not in template:
        raise PipelineConfigError(f"{name} must contain the {{seed}} placeholder")
    try:
        rendered = template.format(seed=CANDIDATE_PROTOCOL_V1_SEEDS[0])
    except (IndexError, KeyError, ValueError) as error:
        raise PipelineConfigError(f"{name} is not a valid seed template") from error
    if not rendered.strip():
        raise PipelineConfigError(f"{name} renders to an empty path")
    return template


def _validate_candidate_pipeline_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise PipelineConfigError(f"schema_version must be {PIPELINE_SCHEMA_VERSION}")
    if config.get("protocol") != CANDIDATE_PROTOCOL_V1:
        raise PipelineConfigError(f"protocol must be {CANDIDATE_PROTOCOL_V1!r}")
    _text(config.get("name"), "name")
    if config.get("app_profile") != "configs/contracts/pocketfinancer-candidate-v1.json":
        raise PipelineConfigError("Candidate V1 must use its versioned research profile")
    if config.get("baseline_app_profile") != (
        "configs/contracts/pocketfinancer-android-current.json"
    ):
        raise PipelineConfigError(
            "Candidate V1 direct control must use the current Android profile"
        )

    model = _mapping(config.get("model"), "model")
    if model.get("id") != "LiquidAI/LFM2.5-350M":
        raise PipelineConfigError("Candidate V1 is locked to LiquidAI/LFM2.5-350M")
    if model.get("local_path") != "TRAINING_ARTIFACTS/base/LFM2.5-350M":
        raise PipelineConfigError("Candidate V1 must use the locked local LFM2.5-350M path")
    if model.get("lock") != "configs/lfm25/model.lock.json":
        raise PipelineConfigError("Candidate V1 must use the immutable LFM2.5-350M model lock")
    for key in ("local_path", "lock"):
        _text(model.get(key), f"model.{key}")
    if model.get("thinking_mode") is not False:
        raise PipelineConfigError("Candidate V1 requires thinking_mode=false")

    data = _mapping(config.get("data"), "data")
    expected_data = {
        "input_dir": "PRIVATE_DATA/lfm25/pocketfinancer-android-a9b7df4-direct-v1",
        "output_dir": "PRIVATE_DATA/lfm25/pocketfinancer-candidate-protocol-v1",
        "train": (
            "PRIVATE_DATA/lfm25/pocketfinancer-candidate-protocol-v1/"
            "candidate_protocol_v1_train.jsonl"
        ),
        "dev": (
            "PRIVATE_DATA/lfm25/pocketfinancer-candidate-protocol-v1/"
            "candidate_protocol_v1_dev.jsonl"
        ),
        "report": (
            "PRIVATE_DATA/lfm25/pocketfinancer-candidate-protocol-v1/"
            "candidate_protocol_v1_report.json"
        ),
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise PipelineConfigError(
                f"data.{key} must use the fresh Candidate V1 path {expected!r}"
            )
    expected_rows = _mapping(data.get("expected_rows"), "data.expected_rows")
    if expected_rows != {"train": 152, "dev": 29}:
        raise PipelineConfigError("Candidate V1 expected rows must be train=152 and dev=29")
    builder = _mapping(data.get("builder"), "data.builder")
    if builder.get("script") != "scripts/build_lfm25_candidate_protocol_v1.py":
        raise PipelineConfigError("Candidate V1 must use its dedicated data builder")

    training = _mapping(config.get("training"), "training")
    if training.get("method") != "lora":
        raise PipelineConfigError("training.method currently supports only 'lora'")
    if training.get("objective") != "completion_only":
        raise PipelineConfigError("Candidate V1 requires completion-only LoRA")
    if tuple(training.get("seeds", ())) != CANDIDATE_PROTOCOL_V1_SEEDS:
        raise PipelineConfigError("Candidate V1 seeds must be exactly 17, 29, and 43")
    if training.get("loss_mode") != "per_example_completion_mean":
        raise PipelineConfigError("Candidate V1 requires per-example completion-only loss")
    for key in (
        "rank",
        "alpha",
        "dropout",
        "learning_rate",
        "first_supervised_token_weight",
        "warmup_ratio",
        "weight_decay",
        "early_stopping_patience",
    ):
        _number(training.get(key), f"training.{key}")
    arms = _mapping(training.get("arms"), "training.arms")
    expected_arm_contracts = {
        "direct": (
            "scripts/train_lfm25_lora.py",
            "pocketfinancer",
        ),
        "selector": (
            "scripts/train_lfm25_candidate_protocol_v1.py",
            CANDIDATE_PROTOCOL_V1,
        ),
    }
    for arm_name, (script, prompt_profile) in expected_arm_contracts.items():
        arm = _mapping(arms.get(arm_name), f"training.arms.{arm_name}")
        if arm.get("script") != script or arm.get("prompt_profile") != prompt_profile:
            raise PipelineConfigError(
                f"training.arms.{arm_name} must use its dedicated V1 contract"
            )
        _seed_template(
            arm.get("output_dir_template"),
            f"training.arms.{arm_name}.output_dir_template",
        )
        for key in (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation",
            "max_length",
        ):
            _number(arm.get(key), f"training.arms.{arm_name}.{key}")

    evaluation = _mapping(config.get("evaluation"), "evaluation")
    for key in ("dataset", "status"):
        _text(evaluation.get(key), f"evaluation.{key}")
    if evaluation.get("dataset") != "DATA/extraction_ds.jsonl":
        raise PipelineConfigError("Candidate V1 evaluation must use the locked diagnostic dataset")
    if evaluation.get("status") != "locked_reused_regression_not_fresh_test":
        raise PipelineConfigError("Candidate V1 diagnostic dataset role changed")
    if str(data["train"]) == str(evaluation["dataset"]):
        raise PipelineConfigError("the regression dataset must never be used as training data")
    if evaluation.get("deterministic") is not True:
        raise PipelineConfigError("Candidate V1 evaluation must be deterministic")
    if evaluation.get("thinking_mode") is not False:
        raise PipelineConfigError("Candidate V1 evaluation requires thinking off")
    evaluation_arms = _mapping(evaluation.get("arms"), "evaluation.arms")
    direct_eval = _mapping(evaluation_arms.get("direct"), "evaluation.arms.direct")
    selector_eval = _mapping(evaluation_arms.get("selector"), "evaluation.arms.selector")
    if direct_eval.get("hf_script") != "scripts/evaluate_lfm25_android_hf.py":
        raise PipelineConfigError("direct control must use the Android HF evaluator")
    if direct_eval.get("gguf_script") != "scripts/evaluate_lfm25_android_gguf.py":
        raise PipelineConfigError("direct control must use the Android GGUF evaluator")
    if selector_eval.get("hf_script") != ("scripts/evaluate_lfm25_candidate_protocol_v1_hf.py"):
        raise PipelineConfigError("selector must use the Candidate V1 HF evaluator")
    if selector_eval.get("max_new_tokens") != 64:
        raise PipelineConfigError("Candidate V1 selector answers are capped at 64 tokens")
    for arm_name, arm in (("direct", direct_eval), ("selector", selector_eval)):
        for key in ("n_ctx", "max_new_tokens", "batch_size", "repeat_penalty"):
            _number(arm.get(key), f"evaluation.arms.{arm_name}.{key}")
        for key in ("hf_base_output_dir_template", "hf_output_dir_template"):
            _seed_template(arm.get(key), f"evaluation.arms.{arm_name}.{key}")
    _seed_template(
        direct_eval.get("gguf_output_dir_template"),
        "evaluation.arms.direct.gguf_output_dir_template",
    )

    export = _mapping(config.get("export"), "export")
    export_arms = _mapping(export.get("arms"), "export.arms")
    for arm_name in ("direct", "selector"):
        arm = _mapping(export_arms.get(arm_name), f"export.arms.{arm_name}")
        for key in ("merged_dir_template", "gguf_prefix_template", "deployable_gguf_template"):
            _seed_template(arm.get(key), f"export.arms.{arm_name}.{key}")

    comparison = _mapping(config.get("comparison"), "comparison")
    expected_comparison = {
        "script": "scripts/compare_lfm25_candidate_protocol_v1.py",
        "output": ("RESULTS/pocketfinancer-candidate-v1/controlled-hf-seed-matrix.json"),
        "aggregate_only": True,
        "required_metric_files": 6,
        "force_overwrite": False,
    }
    if comparison != expected_comparison:
        raise PipelineConfigError(
            "Candidate V1 comparison must remain aggregate-only and overwrite-guarded"
        )

    gates = _mapping(config.get("gates"), "gates")
    decision = _mapping(gates.get("seed_matrix_decision"), "gates.seed_matrix_decision")
    if decision.get("status") != "unmet":
        raise PipelineConfigError("the three-seed Candidate V1 decision gate is not yet met")
    _text(decision.get("reason"), "gates.seed_matrix_decision.reason")
    if (
        decision.get("diagnostic_dataset_rows") != 203
        or decision.get("dataset_role") != "locked_reused_diagnostic_only"
    ):
        raise PipelineConfigError("the reused 203 rows must remain diagnostic only")
    required = _mapping(decision.get("required"), "gates.seed_matrix_decision.required")
    if tuple(required.get("seeds", ())) != CANDIDATE_PROTOCOL_V1_SEEDS:
        raise PipelineConfigError("the decision gate must cover seeds 17, 29, and 43")
    if required.get("selector_transaction_exact_vs_direct") != ("strictly_greater_on_every_seed"):
        raise PipelineConfigError("selector transaction exact must beat direct on every seed")
    if required.get("selector_ghost_transactions_vs_direct") != ("not_greater_on_any_seed"):
        raise PipelineConfigError("selector ghosts must not increase on any seed")
    if required.get("strict_schema_acceptance") != 1.0:
        raise PipelineConfigError("strict-schema acceptance gate must be 100%")
    if required.get("selected_values_source_grounded") != 1.0:
        raise PipelineConfigError("source-grounded selected values gate must be 100%")
    oracle = _mapping(
        required.get("oracle_coverage_floor"),
        "gates.seed_matrix_decision.required.oracle_coverage_floor",
    )
    expected_oracle = {
        "amount": {"covered": 114, "total": 114},
        "account": {"covered": 114, "total": 114},
        "counterparty": {"covered": 113, "total": 114},
        "joint": {"covered": 113, "total": 114},
    }
    if oracle != expected_oracle:
        raise PipelineConfigError("Candidate V1 oracle coverage floors changed")
    product_gate = _mapping(
        gates.get("fresh_human_gold_product_gate"),
        "gates.fresh_human_gold_product_gate",
    )
    if (
        product_gate.get("status") != "unmet"
        or product_gate.get("required_rows") != 1436
        or product_gate.get("dataset_role") != "fresh_human_gold_template_sender_held_out"
        or product_gate.get("promotion_blocked") is not True
    ):
        raise PipelineConfigError("the fresh 1,436-row human-gold product gate is unmet")
    _text(product_gate.get("reason"), "gates.fresh_human_gold_product_gate.reason")
    for name in (
        "selector_gguf_evaluation",
        "android_runtime_parity",
        "ios_runtime_parity",
        "device_measurement",
        "ios_device_measurement",
    ):
        gate = _mapping(gates.get(name), f"gates.{name}")
        if gate.get("status") != "unmet":
            raise PipelineConfigError(f"gates.{name}.status must remain 'unmet'")
        _text(gate.get("reason"), f"gates.{name}.reason")


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    if is_candidate_protocol(config):
        _validate_candidate_pipeline_config(config)
    else:
        _validate_direct_pipeline_config(config)


def _option(command: list[str], flag: str, value: Any) -> None:
    command.extend((flag, str(value)))


def _python(config: Mapping[str, Any]) -> str:
    runtime = config.get("runtime", {})
    if isinstance(runtime, Mapping):
        value = runtime.get("python")
        if isinstance(value, str) and value:
            return value
    return ".venv/bin/python"


def _candidate_seed(config: Mapping[str, Any], seed: int | None) -> int:
    allowed = pipeline_seeds(config)
    selected = allowed[0] if seed is None and allowed else seed
    if selected not in allowed:
        choices = ", ".join(str(value) for value in allowed)
        raise PipelineConfigError(f"Candidate V1 seed must be one of: {choices}")
    return int(selected)


def _render_seed_path(template: Any, name: str, seed: int) -> str:
    return _seed_template(template, name).format(seed=seed)


def _candidate_training_command(
    *,
    python: str,
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    training: Mapping[str, Any],
    arm_name: str,
    seed: int,
) -> list[str]:
    arm = _mapping(
        _mapping(training["arms"], "training.arms")[arm_name], f"training.arms.{arm_name}"
    )
    command = [python, str(arm["script"])]
    values: list[tuple[str, Any]] = [
        ("--model", model["local_path"]),
        ("--train", data["train"]),
        ("--eval", data["dev"]),
        ("--dataset-report", data["report"]),
        ("--model-lock", model["lock"]),
        (
            "--output-dir",
            _render_seed_path(
                arm["output_dir_template"],
                f"training.arms.{arm_name}.output_dir_template",
                seed,
            ),
        ),
    ]
    if arm_name == "direct":
        values.append(("--prompt-profile", arm["prompt_profile"]))
    values.extend(
        (
            ("--rank", training["rank"]),
            ("--alpha", training["alpha"]),
            ("--dropout", training["dropout"]),
            ("--learning-rate", training["learning_rate"]),
            ("--epochs", arm["epochs"]),
            ("--batch-size", arm["batch_size"]),
            ("--eval-batch-size", arm["eval_batch_size"]),
            ("--gradient-accumulation", arm["gradient_accumulation"]),
            ("--max-length", arm["max_length"]),
            ("--loss-mode", training["loss_mode"]),
            (
                "--first-supervised-token-weight",
                training["first_supervised_token_weight"],
            ),
            ("--warmup-ratio", training["warmup_ratio"]),
            ("--weight-decay", training["weight_decay"]),
            ("--early-stopping-patience", training["early_stopping_patience"]),
            ("--seed", seed),
        )
    )
    for flag, value in values:
        _option(command, flag, value)
    return command


def _candidate_hf_command(
    *,
    python: str,
    model: Mapping[str, Any],
    data: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    arm_name: str,
    seed: int,
    base: bool,
) -> list[str]:
    training_arm = _mapping(
        _mapping(training["arms"], "training.arms")[arm_name],
        f"training.arms.{arm_name}",
    )
    evaluation_arm = _mapping(
        _mapping(evaluation["arms"], "evaluation.arms")[arm_name],
        f"evaluation.arms.{arm_name}",
    )
    command = [python, str(evaluation_arm["hf_script"])]
    _option(command, "--model", model["local_path"])
    _option(command, "--model-lock", model["lock"])
    if not base:
        adapter_root = _render_seed_path(
            training_arm["output_dir_template"],
            f"training.arms.{arm_name}.output_dir_template",
            seed,
        )
        _option(command, "--adapter", f"{adapter_root}/adapter")
    output_key = "hf_base_output_dir_template" if base else "hf_output_dir_template"
    for flag, value in (
        ("--dataset", evaluation["dataset"]),
        (
            "--output-dir",
            _render_seed_path(
                evaluation_arm[output_key],
                f"evaluation.arms.{arm_name}.{output_key}",
                seed,
            ),
        ),
        ("--batch-size", evaluation_arm["batch_size"]),
        ("--max-new-tokens", evaluation_arm["max_new_tokens"]),
        ("--n-ctx", evaluation_arm["n_ctx"]),
        ("--repeat-penalty", evaluation_arm["repeat_penalty"]),
        ("--seed", seed),
    ):
        _option(command, flag, value)
    if arm_name == "direct":
        _option(command, "--contract", "pocketfinancer")
        _option(command, "--thinking-mode", "off")
    return command


def _build_candidate_stage_commands(
    config: Mapping[str, Any], *, force_data: bool, seed: int | None
) -> dict[str, list[str]]:
    selected_seed = _candidate_seed(config, seed)
    python = _python(config)
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export_arms = _mapping(_mapping(config["export"], "export")["arms"], "export.arms")

    build_data = [python, str(_mapping(data["builder"], "data.builder")["script"])]
    for flag, value in (("--input-dir", data["input_dir"]), ("--output-dir", data["output_dir"])):
        _option(build_data, flag, value)
    if force_data:
        build_data.append("--force")

    commands: dict[str, list[str]] = {"build-candidate-data": build_data}
    for arm_name in ("direct", "selector"):
        commands[f"evaluate-{arm_name}-base-hf"] = _candidate_hf_command(
            python=python,
            model=model,
            data=data,
            training=training,
            evaluation=evaluation,
            arm_name=arm_name,
            seed=selected_seed,
            base=True,
        )
        commands[f"train-{arm_name}"] = _candidate_training_command(
            python=python,
            model=model,
            data=data,
            training=training,
            arm_name=arm_name,
            seed=selected_seed,
        )
        commands[f"evaluate-{arm_name}-hf"] = _candidate_hf_command(
            python=python,
            model=model,
            data=data,
            training=training,
            evaluation=evaluation,
            arm_name=arm_name,
            seed=selected_seed,
            base=False,
        )
        training_arm = _mapping(
            _mapping(training["arms"], "training.arms")[arm_name], f"training.arms.{arm_name}"
        )
        export_arm = _mapping(export_arms[arm_name], f"export.arms.{arm_name}")
        adapter_root = _render_seed_path(
            training_arm["output_dir_template"],
            f"training.arms.{arm_name}.output_dir_template",
            selected_seed,
        )
        merged_dir = _render_seed_path(
            export_arm["merged_dir_template"],
            f"export.arms.{arm_name}.merged_dir_template",
            selected_seed,
        )
        merge = [python, "scripts/merge_lfm25_lora.py"]
        for flag, value in (
            ("--base-model", model["local_path"]),
            ("--adapter", f"{adapter_root}/adapter"),
            ("--output-dir", merged_dir),
        ):
            _option(merge, flag, value)
        commands[f"merge-{arm_name}"] = merge
        commands[f"convert-{arm_name}"] = [
            "bash",
            "scripts/convert_lfm25_gguf.sh",
            merged_dir,
            _render_seed_path(
                export_arm["gguf_prefix_template"],
                f"export.arms.{arm_name}.gguf_prefix_template",
                selected_seed,
            ),
        ]

    direct_eval = _mapping(
        _mapping(evaluation["arms"], "evaluation.arms")["direct"], "evaluation.arms.direct"
    )
    direct_export = _mapping(export_arms["direct"], "export.arms.direct")
    evaluate_direct_gguf = [python, str(direct_eval["gguf_script"])]
    for flag, value in (
        (
            "--gguf",
            _render_seed_path(
                direct_export["deployable_gguf_template"],
                "export.arms.direct.deployable_gguf_template",
                selected_seed,
            ),
        ),
        ("--tokenizer", model["local_path"]),
        ("--dataset", evaluation["dataset"]),
        (
            "--output-dir",
            _render_seed_path(
                direct_eval["gguf_output_dir_template"],
                "evaluation.arms.direct.gguf_output_dir_template",
                selected_seed,
            ),
        ),
        ("--contract", "pocketfinancer"),
        ("--thinking-mode", "off"),
        ("--n-ctx", direct_eval["n_ctx"]),
        ("--max-tokens", direct_eval["max_new_tokens"]),
        ("--repeat-penalty", direct_eval["repeat_penalty"]),
        ("--seed", selected_seed),
    ):
        _option(evaluate_direct_gguf, flag, value)
    evaluate_direct_gguf.append("--no-grammar")
    commands["evaluate-direct-gguf"] = evaluate_direct_gguf
    comparison = _mapping(config["comparison"], "comparison")
    selector_eval = _mapping(
        _mapping(evaluation["arms"], "evaluation.arms")["selector"],
        "evaluation.arms.selector",
    )
    compare = [python, str(comparison["script"])]
    for flag, value in (
        ("--direct-template", f"{direct_eval['hf_output_dir_template']}/metrics.json"),
        (
            "--selector-template",
            f"{selector_eval['hf_output_dir_template']}/metrics.json",
        ),
        ("--output", comparison["output"]),
        (
            "--config",
            "configs/pipelines/pocketfinancer-lfm2.5-350m-candidate-v1.json",
        ),
    ):
        _option(compare, flag, value)
    commands["compare-hf-seed-matrix"] = compare
    return {stage: commands[stage] for stage in CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES}


def build_stage_commands(
    config: Mapping[str, Any], *, force_data: bool = False, seed: int | None = None
) -> dict[str, list[str]]:
    """Return argv arrays for every reproducible pipeline stage."""

    validate_pipeline_config(config)
    if is_candidate_protocol(config):
        return _build_candidate_stage_commands(config, force_data=force_data, seed=seed)
    python = _python(config)
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    builder = _mapping(data["builder"], "data.builder")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export = _mapping(config["export"], "export")

    build_data = [python, "scripts/build_lfm25_private_sft_v2.py"]
    for flag, key in (
        ("--manifest", "manifest"),
        ("--output-dir", "output_dir"),
    ):
        _option(build_data, flag, data[key])
    for flag, key in (
        ("--dev-fraction", "dev_fraction"),
        ("--seed", "seed"),
        ("--minimum-silver-confidence", "minimum_silver_confidence"),
        ("--max-per-template", "max_per_template"),
        ("--max-per-category", "max_per_category"),
        ("--max-null-to-transaction-ratio", "max_null_to_transaction_ratio"),
    ):
        _option(build_data, flag, builder[key])
    if force_data:
        build_data.append("--force")

    train = [python, "scripts/train_lfm25_lora.py"]
    for flag, value in (
        ("--model", model["local_path"]),
        ("--model-lock", model["lock"]),
        ("--train", data["train"]),
        ("--eval", data["dev"]),
        ("--output-dir", training["output_dir"]),
        ("--prompt-profile", "pocketfinancer"),
        ("--rank", training["rank"]),
        ("--alpha", training["alpha"]),
        ("--dropout", training["dropout"]),
        ("--learning-rate", training["learning_rate"]),
        ("--epochs", training["epochs"]),
        ("--batch-size", training["batch_size"]),
        ("--eval-batch-size", training["eval_batch_size"]),
        ("--gradient-accumulation", training["gradient_accumulation"]),
        ("--max-length", training["max_length"]),
        ("--loss-mode", training["loss_mode"]),
        (
            "--first-supervised-token-weight",
            training["first_supervised_token_weight"],
        ),
        ("--warmup-ratio", training["warmup_ratio"]),
        ("--weight-decay", training["weight_decay"]),
        ("--early-stopping-patience", training["early_stopping_patience"]),
        ("--seed", training["seed"]),
    ):
        _option(train, flag, value)

    thinking_mode = "on" if model["thinking_mode"] else "off"
    evaluate_base_hf = [python, "scripts/evaluate_lfm25_android_hf.py"]
    for flag, value in (
        ("--model", model["local_path"]),
        ("--model-lock", model["lock"]),
        ("--dataset", evaluation["dataset"]),
        ("--output-dir", evaluation["hf_base_output_dir"]),
        ("--contract", "pocketfinancer"),
        ("--thinking-mode", thinking_mode),
        ("--n-ctx", evaluation["n_ctx"]),
        ("--max-new-tokens", 256),
        ("--seed", training["seed"]),
    ):
        _option(evaluate_base_hf, flag, value)

    adapter = f"{training['output_dir']}/adapter"
    evaluate_hf = [python, "scripts/evaluate_lfm25_android_hf.py"]
    for flag, value in (
        ("--model", model["local_path"]),
        ("--model-lock", model["lock"]),
        ("--adapter", adapter),
        ("--dataset", evaluation["dataset"]),
        ("--output-dir", evaluation["hf_output_dir"]),
        ("--contract", "pocketfinancer"),
        ("--thinking-mode", thinking_mode),
        ("--n-ctx", evaluation["n_ctx"]),
        ("--max-new-tokens", 256),
        ("--seed", training["seed"]),
    ):
        _option(evaluate_hf, flag, value)

    merge = [python, "scripts/merge_lfm25_lora.py"]
    for flag, value in (
        ("--base-model", model["local_path"]),
        ("--adapter", adapter),
        ("--output-dir", export["merged_dir"]),
    ):
        _option(merge, flag, value)

    convert = [
        "bash",
        "scripts/convert_lfm25_gguf.sh",
        str(export["merged_dir"]),
        str(export["gguf_prefix"]),
    ]

    evaluate_gguf = [python, "scripts/evaluate_lfm25_android_gguf.py"]
    for flag, value in (
        ("--gguf", export["deployable_gguf"]),
        ("--tokenizer", model["local_path"]),
        ("--dataset", evaluation["dataset"]),
        ("--output-dir", evaluation["gguf_output_dir"]),
        ("--contract", "pocketfinancer"),
        ("--thinking-mode", thinking_mode),
        ("--seed", training["seed"]),
    ):
        _option(evaluate_gguf, flag, value)
    if bool(evaluation.get("grammar", False)):
        _option(
            evaluate_gguf, "--grammar", evaluation.get("grammar_path", "DATA/sms_extraction.gbnf")
        )
        evaluate_gguf.append("--with-grammar")
    else:
        evaluate_gguf.append("--no-grammar")

    return {
        "build-data": build_data,
        "evaluate-base-hf": evaluate_base_hf,
        "train": train,
        "evaluate-hf": evaluate_hf,
        "merge": merge,
        "convert": convert,
        "evaluate-gguf": evaluate_gguf,
    }


@dataclass(frozen=True)
class InputRequirement:
    label: str
    path: Path


def _candidate_stage_requirements(
    config: Mapping[str, Any], stage: str, repo_root: Path, seed: int | None
) -> Sequence[InputRequirement]:
    selected_seed = _candidate_seed(config, seed)
    commands = build_stage_commands(config, seed=selected_seed)
    if stage not in commands:
        raise PipelineConfigError(f"unknown stage: {stage}")
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    training_arms = _mapping(_mapping(config["training"], "training")["arms"], "training.arms")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export_arms = _mapping(_mapping(config["export"], "export")["arms"], "export.arms")
    common = [
        InputRequirement("candidate_profile", repo_root / str(config["app_profile"])),
        InputRequirement("baseline_app_profile", repo_root / str(config["baseline_app_profile"])),
        InputRequirement("model_lock", repo_root / str(model["lock"])),
    ]
    if stage == "build-candidate-data":
        source = repo_root / str(data["input_dir"])
        return [
            *common,
            InputRequirement("source_train", source / "private_sft_v2_train.jsonl"),
            InputRequirement("source_dev", source / "private_sft_v2_dev.jsonl"),
        ]
    if stage == "compare-hf-seed-matrix":
        requirements = [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement(
                "diagnostic_dataset",
                repo_root / str(evaluation["dataset"]),
            ),
            InputRequirement("data_report", repo_root / str(data["report"])),
            InputRequirement(
                "golden_vectors",
                repo_root / "DATA" / "candidate_protocol_v1_golden.json",
            ),
        ]
        evaluation_arms = _mapping(evaluation["arms"], "evaluation.arms")
        for arm_name in ("direct", "selector"):
            arm = _mapping(evaluation_arms[arm_name], f"evaluation.arms.{arm_name}")
            template = _seed_template(
                arm["hf_output_dir_template"],
                f"evaluation.arms.{arm_name}.hf_output_dir_template",
            )
            for matrix_seed in CANDIDATE_PROTOCOL_V1_SEEDS:
                requirements.append(
                    InputRequirement(
                        f"{arm_name}_s{matrix_seed}_metrics",
                        repo_root / template.format(seed=matrix_seed) / "metrics.json",
                    )
                )
        return requirements
    candidate_data = [
        InputRequirement("train", repo_root / str(data["train"])),
        InputRequirement("dev", repo_root / str(data["dev"])),
        InputRequirement("data_report", repo_root / str(data["report"])),
    ]
    model_requirement = InputRequirement("model", repo_root / str(model["local_path"]))
    dataset_requirement = InputRequirement("dataset", repo_root / str(evaluation["dataset"]))
    if stage.startswith("evaluate-") and stage.endswith("-base-hf"):
        return [*common, model_requirement, dataset_requirement]
    if stage.startswith("train-"):
        return [*common, model_requirement, *candidate_data]
    arm_name = "selector" if "selector" in stage else "direct"
    training_arm = _mapping(training_arms[arm_name], f"training.arms.{arm_name}")
    adapter_root = _render_seed_path(
        training_arm["output_dir_template"],
        f"training.arms.{arm_name}.output_dir_template",
        selected_seed,
    )
    adapter_requirement = InputRequirement("adapter", repo_root / adapter_root / "adapter")
    if stage.endswith("-hf"):
        return [
            *common,
            model_requirement,
            adapter_requirement,
            dataset_requirement,
            InputRequirement("data_report", repo_root / str(data["report"])),
        ]
    if stage.startswith("merge-"):
        return [*common, model_requirement, adapter_requirement]
    export_arm = _mapping(export_arms[arm_name], f"export.arms.{arm_name}")
    if stage.startswith("convert-"):
        merged = _render_seed_path(
            export_arm["merged_dir_template"],
            f"export.arms.{arm_name}.merged_dir_template",
            selected_seed,
        )
        return [*common, InputRequirement("merged_model", repo_root / merged)]
    deployable = _render_seed_path(
        export_arm["deployable_gguf_template"],
        f"export.arms.{arm_name}.deployable_gguf_template",
        selected_seed,
    )
    return [
        *common,
        InputRequirement("gguf", repo_root / deployable),
        dataset_requirement,
    ]


def stage_requirements(
    config: Mapping[str, Any], stage: str, repo_root: Path, *, seed: int | None = None
) -> Sequence[InputRequirement]:
    """Return files/directories that must exist before a stage starts."""

    if is_candidate_protocol(config):
        return _candidate_stage_requirements(config, stage, repo_root, seed)
    commands = build_stage_commands(config)
    if stage not in commands:
        raise PipelineConfigError(f"unknown stage: {stage}")
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    export = _mapping(config["export"], "export")

    common = [
        InputRequirement("app_profile", repo_root / str(config["app_profile"])),
        InputRequirement("model_lock", repo_root / str(model["lock"])),
    ]
    if stage == "build-data":
        return [*common, InputRequirement("manifest", repo_root / str(data["manifest"]))]
    if stage == "train":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("train", repo_root / str(data["train"])),
            InputRequirement("dev", repo_root / str(data["dev"])),
        ]
    if stage == "evaluate-base-hf":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("dataset", repo_root / str(evaluation["dataset"])),
        ]
    if stage == "evaluate-hf":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("adapter", repo_root / str(training["output_dir"]) / "adapter"),
            InputRequirement("dataset", repo_root / str(evaluation["dataset"])),
        ]
    if stage == "merge":
        return [
            *common,
            InputRequirement("model", repo_root / str(model["local_path"])),
            InputRequirement("adapter", repo_root / str(training["output_dir"]) / "adapter"),
        ]
    if stage == "convert":
        return [*common, InputRequirement("merged_model", repo_root / str(export["merged_dir"]))]
    return [
        *common,
        InputRequirement("gguf", repo_root / str(export["deployable_gguf"])),
        InputRequirement("dataset", repo_root / str(evaluation["dataset"])),
    ]


def missing_requirements(
    config: Mapping[str, Any], stage: str, repo_root: Path, *, seed: int | None = None
) -> list[InputRequirement]:
    return [
        item
        for item in stage_requirements(config, stage, repo_root, seed=seed)
        if not item.path.exists()
    ]


def stage_output_paths(
    config: Mapping[str, Any], stage: str, repo_root: Path, *, seed: int | None = None
) -> Sequence[Path]:
    """Return material outputs guarded against accidental overwrite."""

    validate_pipeline_config(config)
    if not is_candidate_protocol(config):
        if stage == "train":
            training = _mapping(config["training"], "training")
            return [repo_root / str(training["output_dir"])]
        return []
    selected_seed = _candidate_seed(config, seed)
    data = _mapping(config["data"], "data")
    training_arms = _mapping(_mapping(config["training"], "training")["arms"], "training.arms")
    evaluation_arms = _mapping(
        _mapping(config["evaluation"], "evaluation")["arms"], "evaluation.arms"
    )
    export_arms = _mapping(_mapping(config["export"], "export")["arms"], "export.arms")
    if stage == "build-candidate-data":
        return [repo_root / str(data["output_dir"])]
    if stage == "compare-hf-seed-matrix":
        comparison = _mapping(config["comparison"], "comparison")
        return [repo_root / str(comparison["output"])]
    arm_name = "selector" if "selector" in stage else "direct"
    if stage.startswith("train-"):
        arm = _mapping(training_arms[arm_name], f"training.arms.{arm_name}")
        value = _render_seed_path(
            arm["output_dir_template"],
            f"training.arms.{arm_name}.output_dir_template",
            selected_seed,
        )
        return [repo_root / value]
    if stage.endswith("-hf"):
        arm = _mapping(evaluation_arms[arm_name], f"evaluation.arms.{arm_name}")
        key = "hf_base_output_dir_template" if "base" in stage else "hf_output_dir_template"
        value = _render_seed_path(arm[key], f"evaluation.arms.{arm_name}.{key}", selected_seed)
        return [repo_root / value]
    if stage.startswith("merge-"):
        arm = _mapping(export_arms[arm_name], f"export.arms.{arm_name}")
        value = _render_seed_path(
            arm["merged_dir_template"], f"export.arms.{arm_name}.merged_dir_template", selected_seed
        )
        return [repo_root / value]
    if stage.startswith("convert-"):
        arm = _mapping(export_arms[arm_name], f"export.arms.{arm_name}")
        prefix = _render_seed_path(
            arm["gguf_prefix_template"],
            f"export.arms.{arm_name}.gguf_prefix_template",
            selected_seed,
        )
        return [
            repo_root / f"{prefix}-BF16.gguf",
            repo_root / f"{prefix}-Q8_0.gguf",
            repo_root / f"{prefix}-Q4_K_M.gguf",
            repo_root / f"{prefix}-conversion.json",
        ]
    arm = _mapping(evaluation_arms["direct"], "evaluation.arms.direct")
    value = _render_seed_path(
        arm["gguf_output_dir_template"],
        "evaluation.arms.direct.gguf_output_dir_template",
        selected_seed,
    )
    return [repo_root / value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_candidate_profile(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Verify the checked-in research profile and its pinned local resources."""

    validate_pipeline_config(config)
    if not is_candidate_protocol(config):
        raise PipelineConfigError("candidate profile verification requires Candidate V1")
    profile_path = repo_root / str(config["app_profile"])
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(
            f"could not read candidate profile {profile_path}: {error}"
        ) from error
    if profile.get("schema_version") != 1 or profile.get("profile_version") != 1:
        raise PipelineConfigError("Candidate V1 profile schema/profile version mismatch")
    if profile.get("profile") != CANDIDATE_PROTOCOL_V1:
        raise PipelineConfigError("Candidate V1 profile name mismatch")
    if profile.get("status") != "experimental_not_android_deployed":
        raise PipelineConfigError("Candidate V1 profile must remain explicitly experimental")
    runtime = _mapping(profile.get("model_runtime_defaults"), "candidate profile runtime")
    expected_runtime = {
        "model": "LiquidAI/LFM2.5-350M",
        "thinking": False,
        "temperature": 0.0,
        "answer_max_tokens": 64,
        "passes": 1,
        "grammar": False,
        "repeat_penalty": 1.0,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise PipelineConfigError(f"Candidate V1 profile runtime mismatch for {key}")
    parity = _mapping(profile.get("platform_parity"), "candidate profile parity")
    if parity.get("host_reference_implemented") is not True:
        raise PipelineConfigError("Candidate V1 profile must keep host_reference_implemented=true")
    for key in (
        "android_implemented",
        "ios_implemented",
        "wire_compatible_with_pocketfinancer_android_current",
        "gguf_runtime_validated",
        "device_validated",
    ):
        if parity.get(key) is not False:
            raise PipelineConfigError(f"Candidate V1 profile must keep {key}=false")
    prefilter = _mapping(profile.get("prefilter"), "candidate profile prefilter")
    baseline_path = repo_root / str(config["baseline_app_profile"])
    if prefilter.get("profile_path") != str(config["baseline_app_profile"]):
        raise PipelineConfigError("Candidate V1 prefilter does not bind the baseline profile")
    if not baseline_path.is_file() or prefilter.get("profile_sha256") != _sha256(baseline_path):
        raise PipelineConfigError("Candidate V1 prefilter profile hash mismatch")
    resources = _mapping(profile.get("resources"), "candidate profile resources")
    golden = _mapping(resources.get("golden_vectors"), "candidate profile golden vectors")
    golden_path = repo_root / _text(golden.get("path"), "candidate golden path")
    golden_hash = _text(golden.get("sha256"), "candidate golden hash")
    if not golden_path.is_file() or golden_hash != _sha256(golden_path):
        raise PipelineConfigError("Candidate V1 golden-vector hash mismatch")
    return {
        "declaration_verified": True,
        "profile": CANDIDATE_PROTOCOL_V1,
        "profile_version": 1,
        "profile_sha256": _sha256(profile_path),
        "golden_vectors_sha256": golden_hash,
        "android_runtime_parity": False,
        "device_validated": False,
    }


def verify_candidate_data(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Fail closed unless Candidate V1 data matches its aggregate report."""

    validate_pipeline_config(config)
    if not is_candidate_protocol(config):
        raise PipelineConfigError("candidate data verification requires Candidate V1")
    data = _mapping(config["data"], "data")
    report_path = repo_root / str(data["report"])
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(
            f"could not read candidate data report {report_path}: {error}"
        ) from error
    if report.get("valid") is not True or report.get("candidate_protocol") != CANDIDATE_PROTOCOL_V1:
        raise PipelineConfigError("Candidate V1 data report is not valid for this protocol")
    if report.get("builder_version") != CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION:
        raise PipelineConfigError("Candidate V1 data builder version mismatch")
    implementations = _mapping(
        report.get("candidate_implementations"),
        "candidate data implementations",
    )
    expected_implementations = {
        "extractor": "lfm25/candidates.py",
        "protocol": "lfm25/candidate_protocol.py",
    }
    verified_implementations: dict[str, dict[str, str]] = {}
    for name, relative_path in expected_implementations.items():
        declared = _mapping(
            implementations.get(name),
            f"candidate data {name} implementation",
        )
        if declared.get("path") != relative_path:
            raise PipelineConfigError(f"Candidate V1 {name} implementation path mismatch")
        implementation_path = repo_root / relative_path
        if not implementation_path.is_file():
            raise PipelineConfigError(f"Candidate V1 {name} implementation is missing")
        implementation_hash = _sha256(implementation_path)
        if declared.get("sha256") != implementation_hash:
            raise PipelineConfigError(f"Candidate V1 {name} implementation hash mismatch")
        verified_implementations[name] = {
            "path": relative_path,
            "sha256": implementation_hash,
        }
    invariants = _mapping(report.get("invariants"), "candidate data invariants")
    required_true = (
        "only_original_train_rows",
        "candidate_oracle_covered",
        "same_rows_for_direct_and_selector",
        "android_prefilter_accepted",
    )
    missing_true = [key for key in required_true if invariants.get(key) is not True]
    if missing_true:
        raise PipelineConfigError(
            "Candidate V1 data invariants are incomplete: " + ", ".join(missing_true)
        )
    if invariants.get("historical_candidate_data_reused") is not False:
        raise PipelineConfigError("historical candidate data must not be reused")
    for key in (
        "sender_overlap_count",
        "template_overlap_count",
        "record_overlap_count",
        "sealed_test_rows_materialized",
    ):
        if invariants.get(key) != 0:
            raise PipelineConfigError(f"Candidate V1 data invariant failed: {key}")
    expected_rows = _mapping(data["expected_rows"], "data.expected_rows")
    inputs = _mapping(report.get("inputs"), "candidate data inputs")
    source_root = repo_root / str(data["input_dir"])
    source_hashes: dict[str, str] = {}
    source_contract = {
        "train": ("private_sft_v2_train.jsonl", 154),
        "dev": ("private_sft_v2_dev.jsonl", 29),
    }
    for split, (filename, expected_count) in source_contract.items():
        declared = _mapping(inputs.get(split), f"candidate data input {split}")
        source_path = source_root / filename
        if declared.get("filename") != filename or declared.get("rows") != expected_count:
            raise PipelineConfigError(f"Candidate V1 {split} source declaration mismatch")
        if not source_path.is_file() or source_path.stat().st_size <= 0:
            raise PipelineConfigError(f"Candidate V1 {split} source is missing or empty")
        source_hash = _sha256(source_path)
        if declared.get("sha256") != source_hash:
            raise PipelineConfigError(f"Candidate V1 {split} source hash mismatch")
        source_count = sum(
            1 for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if source_count != expected_count:
            raise PipelineConfigError(f"Candidate V1 {split} source row count mismatch")
        source_hashes[split] = source_hash
    artifacts = _mapping(report.get("artifacts"), "candidate data artifacts")
    splits = _mapping(report.get("splits"), "candidate data splits")
    verified_hashes: dict[str, str] = {}
    for split in ("train", "dev"):
        artifact = _mapping(artifacts.get(split), f"candidate data artifact {split}")
        path = repo_root / str(data[split])
        expected_count = int(expected_rows[split])
        if artifact.get("filename") != path.name or artifact.get("rows") != expected_count:
            raise PipelineConfigError(f"Candidate V1 {split} artifact declaration mismatch")
        if not path.is_file() or path.stat().st_size <= 0:
            raise PipelineConfigError(f"Candidate V1 {split} artifact is missing or empty")
        actual_hash = _sha256(path)
        if artifact.get("sha256") != actual_hash:
            raise PipelineConfigError(f"Candidate V1 {split} artifact hash mismatch")
        row_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if row_count != expected_count:
            raise PipelineConfigError(f"Candidate V1 {split} row count mismatch")
        split_report = _mapping(splits.get(split), f"candidate data split {split}")
        source_count = source_contract[split][1]
        excluded_count = source_count - expected_count
        if (
            split_report.get("input_rows") != source_count
            or split_report.get("excluded_rows") != excluded_count
        ):
            raise PipelineConfigError(f"Candidate V1 {split} source/output accounting mismatch")
        if split_report.get("output_rows") != expected_count:
            raise PipelineConfigError(f"Candidate V1 {split} split report mismatch")
        verified_hashes[split] = actual_hash
    train_report = _mapping(splits["train"], "candidate data train split")
    dev_report = _mapping(splits["dev"], "candidate data dev split")
    if train_report.get("exclusion_reasons") != {"candidate_missing_counterparty": 2}:
        raise PipelineConfigError("Candidate V1 train must have exactly two coverage exclusions")
    if dev_report.get("exclusion_reasons") != {}:
        raise PipelineConfigError("Candidate V1 dev must have no coverage exclusions")
    return {
        "verified": True,
        "protocol": CANDIDATE_PROTOCOL_V1,
        "builder_version": CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION,
        "candidate_implementations": verified_implementations,
        "rows": {"train": 152, "dev": 29},
        "sha256": verified_hashes,
        "source_sha256": source_hashes,
        "report": str(report_path.relative_to(repo_root)),
    }


def verify_locked_model(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """Verify the local HF base against the immutable checked-in model lock."""

    validate_pipeline_config(config)
    model = _mapping(config["model"], "model")
    lock_path = repo_root / str(model["lock"])
    model_root = repo_root / str(model["local_path"])
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineConfigError(f"could not read model lock {lock_path}: {error}") from error
    locked_model = _mapping(_mapping(lock, "model lock").get("model"), "model lock.model")
    if locked_model.get("repo") != model.get("id"):
        raise PipelineConfigError("pipeline model.id does not match model lock repo")
    files = _mapping(locked_model.get("files"), "model lock.model.files")
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in sorted(files.items()):
        relative_name = _text(relative, "locked model filename")
        expected_hash = _text(expected, f"model lock hash for {relative_name}")
        path = model_root / relative_name
        if not path.is_file():
            missing.append(relative_name)
        elif _sha256(path) != expected_hash:
            mismatched.append(relative_name)
    if missing or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if mismatched:
            details.append("hash_mismatch=" + ",".join(mismatched))
        raise PipelineConfigError("local base model does not match lock: " + "; ".join(details))
    return {
        "model_id": model["id"],
        "revision": locked_model.get("revision"),
        "verified_files": len(files),
        "lock": str(lock_path.relative_to(repo_root)),
    }


def _comparison_fingerprint(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineConfigError(f"Candidate V1 comparison anchor is missing: {label}")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def diagnostic_prefilter_evidence(dataset_path: Path) -> dict[str, Any]:
    """Bind aggregate prefilter behavior to one exact diagnostic byte snapshot."""

    from lfm25.android_contract import (
        pocketfinancer_prefilter_sms,
        summarize_prefilter,
    )

    path = Path(dataset_path)
    if not path.is_file():
        raise PipelineConfigError("Candidate V1 diagnostic dataset is missing")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise PipelineConfigError(
            f"could not read Candidate V1 diagnostic dataset: {error}"
        ) from error

    derived_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PipelineConfigError(
                f"Candidate V1 diagnostic row {line_number} is not valid JSON"
            ) from error
        if not isinstance(row, Mapping):
            raise PipelineConfigError(f"Candidate V1 diagnostic row {line_number} is not an object")
        if "expected" in row:
            gold = row["expected"]
        elif "label" in row:
            gold = row["label"]
        else:
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not isinstance(messages[-1], Mapping)
                or "content" not in messages[-1]
            ):
                raise PipelineConfigError(
                    f"Candidate V1 diagnostic row {line_number} has no target"
                )
            gold = messages[-1]["content"]
        disposition = pocketfinancer_prefilter_sms(
            str(row.get("sender", "")),
            str(row.get("sms", "")),
        )
        derived_rows.append(
            {
                "gold": gold,
                "prefilter_passed": disposition.accepted,
                "prefilter_rejection_stage": disposition.rejection_stage,
            }
        )

    summary = summarize_prefilter(derived_rows, enabled=True)
    return {
        "diagnostic_dataset": {
            "filename": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(derived_rows),
            "row_limit": None,
        },
        "diagnostic_prefilter": summary,
    }


def candidate_comparison_anchors(
    config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Build privacy-safe live anchors for the controlled HF comparison."""

    validate_pipeline_config(config)
    if not is_candidate_protocol(config):
        raise PipelineConfigError("comparison anchors require Candidate Protocol V1")

    data_evidence = verify_candidate_data(config, repo_root)
    model_evidence = verify_locked_model(config, repo_root)
    profile_evidence = verify_candidate_profile(config, repo_root)

    from lfm25.android_contract import contract_provenance as prefilter_provenance
    from lfm25.android_profile_sync import (
        AndroidProfileError,
        verify_profile_declaration,
    )
    from lfm25.candidate_protocol import contract_provenance as protocol_provenance
    from lfm25.provenance import code_fingerprints
    from lfm25.training_provenance import trainer_code_fingerprints

    data = _mapping(config["data"], "data")
    evaluation = _mapping(config["evaluation"], "evaluation")
    model = _mapping(config["model"], "model")
    profile_path = repo_root / str(config["app_profile"])
    baseline_path = repo_root / str(config["baseline_app_profile"])
    try:
        verify_profile_declaration(baseline_path)
    except AndroidProfileError as error:
        raise PipelineConfigError(
            f"Candidate V1 baseline profile anchor is invalid: {error}"
        ) from error

    profile = _mapping(
        json.loads(profile_path.read_text(encoding="utf-8")),
        "candidate profile",
    )
    resources = _mapping(profile.get("resources"), "candidate profile resources")
    golden = _mapping(resources.get("golden_vectors"), "candidate golden vectors")
    golden_path = repo_root / _text(golden.get("path"), "candidate golden path")

    diagnostic_path = repo_root / str(evaluation["dataset"])
    diagnostic_evidence = diagnostic_prefilter_evidence(diagnostic_path)
    diagnostic = diagnostic_evidence["diagnostic_dataset"]
    if diagnostic["rows"] != 203:
        raise PipelineConfigError("Candidate V1 diagnostic dataset must contain 203 rows")

    report_path = repo_root / str(data["report"])
    lock_path = repo_root / str(model["lock"])
    try:
        lock = _mapping(
            json.loads(lock_path.read_text(encoding="utf-8")),
            "model lock",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineConfigError(f"could not read Candidate V1 model lock: {error}") from error
    locked_model = _mapping(lock.get("model"), "model lock.model")
    locked_files = _mapping(locked_model.get("files"), "model lock.model.files")
    runtime_files = {
        _text(name, "model lock filename"): _text(
            digest,
            f"model lock digest for {name}",
        )
        for name, digest in sorted(locked_files.items())
        if name not in {"LICENSE", "README.md"}
    }
    if not runtime_files:
        raise PipelineConfigError("Candidate V1 model lock has no runtime files")

    evaluation_arms = _mapping(evaluation["arms"], "evaluation.arms")
    direct_eval = _mapping(evaluation_arms["direct"], "evaluation.arms.direct")
    selector_eval = _mapping(evaluation_arms["selector"], "evaluation.arms.selector")
    evaluator_code = {
        "direct": _sha256(repo_root / str(direct_eval["hf_script"])),
        "selector": _sha256(repo_root / str(selector_eval["hf_script"])),
        "selector_generation_engine": _sha256(
            repo_root / "scripts" / "evaluate_lfm25_candidate_hf.py"
        ),
        "comparator_module": _sha256(repo_root / "lfm25" / "candidate_protocol_compare.py"),
        "comparator_cli": _sha256(repo_root / "scripts" / "compare_lfm25_candidate_protocol_v1.py"),
    }
    return {
        "schema_version": 1,
        "diagnostic_dataset": diagnostic,
        "diagnostic_prefilter": diagnostic_evidence["diagnostic_prefilter"],
        "candidate_data": {
            "report": _comparison_fingerprint(report_path, "candidate data report"),
            "train_sha256": data_evidence["sha256"]["train"],
            "dev_sha256": data_evidence["sha256"]["dev"],
            "rows": data_evidence["rows"],
        },
        "model": {
            "lock": _comparison_fingerprint(lock_path, "model lock"),
            "id": model_evidence["model_id"],
            "revision": model_evidence["revision"],
            "files": runtime_files,
        },
        "profiles": {
            "candidate": _comparison_fingerprint(profile_path, "candidate profile"),
            "baseline": _comparison_fingerprint(baseline_path, "baseline profile"),
            "golden_vectors": _comparison_fingerprint(golden_path, "golden vectors"),
        },
        "prefilter_contract": prefilter_provenance(),
        "candidate_protocol": protocol_provenance(),
        "shared_code_sha256": code_fingerprints(repo_root),
        "evaluator_code_sha256": evaluator_code,
        "trainer_code_sha256": {
            arm: trainer_code_fingerprints(repo_root, arm) for arm in ("direct", "selector")
        },
        "platform_gates": {
            "hf_host_reference_only": True,
            "android_implemented": False,
            "ios_implemented": False,
            "android_runtime_parity": False,
            "ios_runtime_parity": False,
            "gguf_runtime_validated": False,
            "android_device_validated": False,
            "ios_device_validated": False,
            "product_promotion_allowed": False,
        },
        "profile_sha256": profile_evidence["profile_sha256"],
    }
