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
    verify_locked_model,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pipelines/pocketfinancer-lfm2.5-350m.json"
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
