from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from lfm25.pipeline import (
    EXECUTION_STAGES,
    PipelineConfigError,
    build_stage_commands,
    load_pipeline_config,
    stage_requirements,
    verify_locked_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pipelines/pocketfinancer-lfm2.5-350m.json"
CONFIG_26B = ROOT / "configs/pipelines/pocketfinancer-lfm2.5-2.6b-base.json"
CONFIGS = (
    CONFIG,
    CONFIG_26B,
)
SCRIPT = ROOT / "scripts/run_pocketfinancer_pipeline.py"


def test_checked_in_pipeline_has_one_android_aligned_happy_path() -> None:
    config = load_pipeline_config(CONFIG)
    commands = build_stage_commands(config)

    assert tuple(commands) == EXECUTION_STAGES
    train = commands["train"]
    assert train[train.index("--prompt-profile") + 1] == "pocketfinancer"
    assert train[train.index("--max-length") + 1] == "2304"
    assert "legacy" not in train
    assert "candidate_selector" not in train

    hf = commands["evaluate-hf"]
    assert hf[hf.index("--contract") + 1] == "pocketfinancer"
    assert hf[hf.index("--n-ctx") + 1] == "3072"
    gguf = commands["evaluate-gguf"]
    assert gguf[gguf.index("--thinking-mode") + 1] == "off"
    assert "--no-grammar" in gguf


def test_pipeline_rejects_a_non_app_training_profile() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["training"]["prompt_profile"] = "legacy"

    with pytest.raises(PipelineConfigError, match="pocketfinancer"):
        build_stage_commands(config)


def test_pipeline_rejects_using_the_regression_set_for_training() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["data"]["train"] = config["evaluation"]["dataset"]

    with pytest.raises(PipelineConfigError, match="never be used as training"):
        build_stage_commands(config)


def test_plan_is_gpu_free_and_prints_all_stages() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--config", str(CONFIG)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)

    assert tuple(value) == EXECUTION_STAGES
    assert value["train"][0] == ".venv/bin/python"


def test_model_lock_verification_fails_closed_on_hash_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("original", encoding="utf-8")
    lock = {
        "model": {
            "repo": config["model"]["id"],
            "revision": "test",
            "files": {"config.json": "0" * 64},
        }
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    config["model"]["local_path"] = str(model_root.relative_to(tmp_path))
    config["model"]["lock"] = str(lock_path.relative_to(tmp_path))
    config["app_profile"] = "profile.json"

    with pytest.raises(PipelineConfigError, match="hash_mismatch"):
        verify_locked_model(config, tmp_path)


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_execution_order_captures_untouched_control_before_training() -> None:
    assert EXECUTION_STAGES == (
        "build-data",
        "evaluate-base-hf",
        "train",
        "evaluate-hf",
        "merge",
        "convert",
        "evaluate-gguf",
    )


@pytest.mark.parametrize("config_path", CONFIGS, ids=("350m", "2.6b-base"))
def test_checked_in_pipelines_plan_paired_hf_commands(config_path: Path) -> None:
    config = load_pipeline_config(config_path)
    commands = build_stage_commands(config)
    expected_thinking = "on" if config["model"]["thinking_mode"] else "off"

    base_hf = commands["evaluate-base-hf"]
    tuned_hf = commands["evaluate-hf"]
    assert base_hf[:2] == tuned_hf[:2]
    assert "--adapter" not in base_hf
    assert _command_value(base_hf, "--model") == config["model"]["local_path"]
    assert _command_value(base_hf, "--output-dir") == (
        config["evaluation"]["hf_base_output_dir"]
    )
    assert _command_value(base_hf, "--thinking-mode") == expected_thinking
    assert _command_value(tuned_hf, "--adapter") == (
        f"{config['training']['output_dir']}/adapter"
    )
    assert _command_value(tuned_hf, "--output-dir") == (
        config["evaluation"]["hf_output_dir"]
    )
    assert _command_value(tuned_hf, "--thinking-mode") == expected_thinking

    base_inputs = {
        requirement.label
        for requirement in stage_requirements(config, "evaluate-base-hf", ROOT)
    }
    tuned_inputs = {
        requirement.label
        for requirement in stage_requirements(config, "evaluate-hf", ROOT)
    }
    assert base_inputs == {"app_profile", "model_lock", "model", "dataset"}
    assert tuned_inputs == {*base_inputs, "adapter"}

    gguf = commands["evaluate-gguf"]
    assert _command_value(gguf, "--thinking-mode") == expected_thinking
    assert "--no-grammar" in gguf


def test_2_6b_base_pipeline_is_the_controlled_direct_lora_comparison() -> None:
    small = load_pipeline_config(CONFIG)
    candidate = load_pipeline_config(CONFIG_26B)

    assert candidate["model"] == {
        "id": "LiquidAI/LFM2.5-2.6B-Base",
        "local_path": "TRAINING_ARTIFACTS/base/LFM2.5-2.6B-Base",
        "lock": "configs/lfm25/model-2.6b-base.lock.json",
        "thinking_mode": False,
    }
    assert candidate["data"] == small["data"]
    assert {
        key: candidate["training"][key]
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
            "loss_mode",
            "first_supervised_token_weight",
            "seed",
        )
    } == {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 0.0001,
        "epochs": 4,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation": 32,
        "max_length": 2304,
        "loss_mode": "per_example_completion_mean",
        "first_supervised_token_weight": 3.0,
        "seed": 17,
    }
    assert (
        candidate["training"]["batch_size"]
        * candidate["training"]["gradient_accumulation"]
        == 32
    )
    assert candidate["evaluation"]["status"] == (
        "locked_reused_regression_not_fresh_test"
    )
    assert candidate["evaluation"]["grammar"] is False

    def output_paths(config: dict) -> set[str]:
        return {
            config["training"]["output_dir"],
            config["evaluation"]["hf_base_output_dir"],
            config["evaluation"]["hf_output_dir"],
            config["evaluation"]["gguf_output_dir"],
            *config["export"].values(),
        }

    assert output_paths(small).isdisjoint(output_paths(candidate))


@pytest.mark.parametrize("config_path", CONFIGS, ids=("350m", "2.6b-base"))
def test_checked_in_model_locks_match_and_verify(config_path: Path, tmp_path: Path) -> None:
    config = load_pipeline_config(config_path)
    checked_in_lock = json.loads(
        (ROOT / config["model"]["lock"]).read_text(encoding="utf-8")
    )
    assert checked_in_lock["model"]["repo"] == config["model"]["id"]
    assert checked_in_lock["model"]["revision"]
    assert checked_in_lock["model"]["files"]

    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_bytes(b"abc")
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "model": {
                    "repo": config["model"]["id"],
                    "revision": "test-revision",
                    "files": {
                        "config.json": (
                            "ba7816bf8f01cfea414140de5dae2223"
                            "b00361a396177a9cb410ff61f20015ad"
                        )
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    config["model"]["local_path"] = "model"
    config["model"]["lock"] = "lock.json"

    report = verify_locked_model(config, tmp_path)

    assert report == {
        "model_id": config["model"]["id"],
        "revision": "test-revision",
        "verified_files": 1,
        "lock": "lock.json",
    }


@pytest.mark.parametrize("config_path", CONFIGS, ids=("350m", "2.6b-base"))
def test_checked_in_plans_are_model_free(config_path: Path, tmp_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["local_path"] = "TRAINING_ARTIFACTS/base/not-downloaded"
    config["model"]["lock"] = "configs/lfm25/not-downloaded.lock.json"
    model_free_config = tmp_path / config_path.name
    model_free_config.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--config", str(model_free_config)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)

    assert tuple(value) == EXECUTION_STAGES
    assert _command_value(value["evaluate-base-hf"], "--model") == (
        "TRAINING_ARTIFACTS/base/not-downloaded"
    )
