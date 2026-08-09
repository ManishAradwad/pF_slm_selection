from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from lfm25.pipeline import (
    CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES,
    CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION,
    CANDIDATE_PROTOCOL_V1_SEEDS,
    EXECUTION_STAGES,
    PipelineConfigError,
    build_stage_commands,
    diagnostic_prefilter_evidence,
    execution_stages,
    load_pipeline_config,
    pipeline_seeds,
    stage_output_paths,
    stage_requirements,
    verify_candidate_data,
    verify_candidate_profile,
)
from lfm25.android_profile_sync import AndroidProfileError
from scripts import run_pocketfinancer_pipeline as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pipelines" / "pocketfinancer-lfm2.5-350m-candidate-v1.json"
LEGACY_CONFIG = ROOT / "configs" / "pipelines" / "pocketfinancer-lfm2.5-350m.json"
SCRIPT = ROOT / "scripts" / "run_pocketfinancer_pipeline.py"
INVALIDATED_PATHS = (
    "PRIVATE_DATA/lfm25/private_sft_v2",
    "PRIVATE_DATA/lfm25/private_sft_v3",
    "PRIVATE_DATA/lfm25/candidate_sft_v4",
    "PRIVATE_DATA/lfm25/candidate_curriculum_v2",
)


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_candidate_v1_plan_is_a_distinct_controlled_two_arm_protocol() -> None:
    config = load_pipeline_config(CONFIG)

    assert pipeline_seeds(config) == CANDIDATE_PROTOCOL_V1_SEEDS == (17, 29, 43)
    assert execution_stages(config) == CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES
    assert "evaluate-selector-gguf" not in execution_stages(config)
    assert config["data"]["expected_rows"] == {"train": 152, "dev": 29}
    assert config["training"]["objective"] == "completion_only"
    assert config["evaluation"]["deterministic"] is True
    assert config["evaluation"]["thinking_mode"] is False
    decision = config["gates"]["seed_matrix_decision"]
    assert decision["status"] == "unmet"
    assert decision["diagnostic_dataset_rows"] == 203
    assert decision["dataset_role"] == "locked_reused_diagnostic_only"
    required = decision["required"]
    assert required["seeds"] == [17, 29, 43]
    assert required["selector_transaction_exact_vs_direct"] == ("strictly_greater_on_every_seed")
    assert required["selector_ghost_transactions_vs_direct"] == ("not_greater_on_any_seed")
    assert required["strict_schema_acceptance"] == 1.0
    assert required["selected_values_source_grounded"] == 1.0
    assert required["oracle_coverage_floor"] == {
        "amount": {"covered": 114, "total": 114},
        "account": {"covered": 114, "total": 114},
        "counterparty": {"covered": 113, "total": 114},
        "joint": {"covered": 113, "total": 114},
    }
    assert config["gates"]["fresh_human_gold_product_gate"]["required_rows"] == 1436
    assert config["comparison"] == {
        "script": "scripts/compare_lfm25_candidate_protocol_v1.py",
        "output": "RESULTS/pocketfinancer-candidate-v1/controlled-hf-seed-matrix.json",
        "aggregate_only": True,
        "required_metric_files": 6,
        "force_overwrite": False,
    }

    for seed in CANDIDATE_PROTOCOL_V1_SEEDS:
        commands = build_stage_commands(config, seed=seed)
        assert tuple(commands) == CANDIDATE_PROTOCOL_V1_EXECUTION_STAGES
        direct = commands["train-direct"]
        selector = commands["train-selector"]
        assert direct[1] == "scripts/train_lfm25_lora.py"
        assert _value(direct, "--prompt-profile") == "pocketfinancer"
        assert selector[1] == "scripts/train_lfm25_candidate_protocol_v1.py"
        assert "--prompt-profile" not in selector
        assert "candidate_selector" not in " ".join(selector)
        assert _value(direct, "--train") == _value(selector, "--train")
        assert _value(direct, "--eval") == _value(selector, "--eval")
        assert _value(direct, "--loss-mode") == "per_example_completion_mean"
        assert _value(selector, "--loss-mode") == "per_example_completion_mean"
        assert _value(direct, "--seed") == str(seed)
        assert _value(selector, "--seed") == str(seed)
        assert _value(direct, "--dataset-report") == config["data"]["report"]
        assert _value(selector, "--dataset-report") == config["data"]["report"]
        assert _value(direct, "--model-lock") == config["model"]["lock"]
        assert _value(selector, "--model-lock") == config["model"]["lock"]

        selector_hf = commands["evaluate-selector-hf"]
        assert selector_hf[1] == "scripts/evaluate_lfm25_candidate_protocol_v1_hf.py"
        assert _value(selector_hf, "--max-new-tokens") == "64"
        assert _value(selector_hf, "--repeat-penalty") == "1.0"
        assert "--hybrid-safety" not in selector_hf
        assert "--adapter" not in commands["evaluate-selector-base-hf"]
        for stage in (
            "evaluate-direct-base-hf",
            "evaluate-selector-base-hf",
            "evaluate-direct-hf",
            "evaluate-selector-hf",
        ):
            assert _value(commands[stage], "--model-lock") == config["model"]["lock"]

        compare = commands["compare-hf-seed-matrix"]
        assert compare[1] == "scripts/compare_lfm25_candidate_protocol_v1.py"
        assert _value(compare, "--direct-template") == (
            "RESULTS/pocketfinancer-candidate-v1/direct-r16-hf-s{seed}/metrics.json"
        )
        assert _value(compare, "--selector-template") == (
            "RESULTS/pocketfinancer-candidate-v1/selector-r16-hf-s{seed}/metrics.json"
        )
        assert _value(compare, "--output") == config["comparison"]["output"]
        assert _value(compare, "--config") == (
            "configs/pipelines/pocketfinancer-lfm2.5-350m-candidate-v1.json"
        )
        assert "--force" not in compare


def test_candidate_v1_paths_are_fresh_seed_isolated_and_not_historical() -> None:
    config = load_pipeline_config(CONFIG)
    serialized = json.dumps(config, sort_keys=True)
    assert all(path not in serialized for path in INVALIDATED_PATHS)

    output_sets = []
    for seed in CANDIDATE_PROTOCOL_V1_SEEDS:
        commands = build_stage_commands(config, seed=seed)
        outputs = {
            _value(commands["train-direct"], "--output-dir"),
            _value(commands["train-selector"], "--output-dir"),
            _value(commands["evaluate-direct-hf"], "--output-dir"),
            _value(commands["evaluate-selector-hf"], "--output-dir"),
            _value(commands["merge-direct"], "--output-dir"),
            _value(commands["merge-selector"], "--output-dir"),
        }
        assert all(f"s{seed}" in output for output in outputs)
        output_sets.append(outputs)
    assert output_sets[0].isdisjoint(output_sets[1])
    assert output_sets[0].isdisjoint(output_sets[2])
    assert output_sets[1].isdisjoint(output_sets[2])


def test_locked_diagnostic_prefilter_aggregate_is_live_and_private_safe() -> None:
    evidence = diagnostic_prefilter_evidence(ROOT / "DATA" / "extraction_ds.jsonl")
    dataset = evidence["diagnostic_dataset"]
    prefilter = evidence["diagnostic_prefilter"]

    assert dataset["rows"] == prefilter["n"] == 203
    assert prefilter["model_invocations"] == 115
    assert prefilter["rejected"] == 88
    assert prefilter["model_invocations"] + prefilter["rejected"] == prefilter["n"]
    assert sum(prefilter["rejections_by_stage"].values()) == prefilter["rejected"]
    assert set(evidence) == {"diagnostic_dataset", "diagnostic_prefilter"}
    assert "sender" not in json.dumps(evidence).casefold()
    assert "sms" not in json.dumps(evidence).casefold()


def test_candidate_v1_rejects_uncontrolled_seed_and_wrong_data_root() -> None:
    config = load_pipeline_config(CONFIG)
    with pytest.raises(PipelineConfigError, match="seed must be one of"):
        build_stage_commands(config, seed=54)

    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["data"]["input_dir"] = "PRIVATE_DATA/lfm25/candidate_sft_v4"
    with pytest.raises(PipelineConfigError, match="fresh Candidate V1 path"):
        build_stage_commands(changed, seed=17)


@pytest.mark.parametrize(
    "gate_name",
    ("ios_runtime_parity", "ios_device_measurement"),
)
def test_candidate_v1_rejects_met_ios_gate(gate_name: str) -> None:
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["gates"][gate_name]["status"] = "met"

    with pytest.raises(PipelineConfigError, match=gate_name):
        build_stage_commands(changed, seed=17)


def test_candidate_profile_rejects_ios_implementation_claim(tmp_path: Path) -> None:
    config = load_pipeline_config(CONFIG)
    for relative in (
        config["baseline_app_profile"],
        "DATA/candidate_protocol_v1_golden.json",
        config["app_profile"],
    ):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    profile_path = tmp_path / config["app_profile"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["platform_parity"]["ios_implemented"] = True
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineConfigError, match="ios_implemented=false"):
        verify_candidate_profile(config, tmp_path)


def test_candidate_plan_without_seed_prints_all_controlled_seed_plans() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--config", str(CONFIG)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)

    assert value["protocol"] == "candidate_protocol_v1"
    assert value["controlled_seeds"] == [17, 29, 43]
    assert tuple(value["plans"]) == ("17", "29", "43")
    assert value["plans"]["29"]["train-selector"][-1] == "29"
    assert all(gate["status"] == "unmet" for gate in value["unmet_gates"].values())


def _write_candidate_fixture(repo: Path, config: dict) -> None:
    data = config["data"]
    artifacts = {}
    splits = {}
    inputs = {}
    source_root = repo / data["input_dir"]
    source_root.mkdir(parents=True, exist_ok=True)
    for split, filename, count in (
        ("train", "private_sft_v2_train.jsonl", 154),
        ("dev", "private_sft_v2_dev.jsonl", 29),
    ):
        source_path = source_root / filename
        source_content = "".join(
            json.dumps({"source_fixture": index}) + "\n" for index in range(count)
        )
        source_path.write_text(source_content, encoding="utf-8")
        inputs[split] = {
            "filename": filename,
            "rows": count,
            "sha256": hashlib.sha256(source_content.encode()).hexdigest(),
        }
    for split, count in (("train", 152), ("dev", 29)):
        path = repo / data[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps({"fixture": index}) + "\n" for index in range(count))
        path.write_text(content, encoding="utf-8")
        artifacts[split] = {
            "filename": path.name,
            "rows": count,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        splits[split] = {
            "input_rows": 154 if split == "train" else 29,
            "output_rows": count,
            "excluded_rows": 2 if split == "train" else 0,
            "exclusion_reasons": (
                {"candidate_missing_counterparty": 2} if split == "train" else {}
            ),
        }
    implementations = {}
    for name, relative_path in {
        "extractor": "lfm25/candidates.py",
        "protocol": "lfm25/candidate_protocol.py",
    }.items():
        source_path = ROOT / relative_path
        implementation_path = repo / relative_path
        implementation_path.parent.mkdir(parents=True, exist_ok=True)
        implementation_content = source_path.read_bytes()
        implementation_path.write_bytes(implementation_content)
        implementations[name] = {
            "path": relative_path,
            "sha256": hashlib.sha256(implementation_content).hexdigest(),
        }

    report = {
        "builder_version": CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION,
        "valid": True,
        "candidate_protocol": "candidate_protocol_v1",
        "candidate_implementations": implementations,
        "artifacts": artifacts,
        "splits": splits,
        "inputs": inputs,
        "invariants": {
            "only_original_train_rows": True,
            "candidate_oracle_covered": True,
            "same_rows_for_direct_and_selector": True,
            "android_prefilter_accepted": True,
            "historical_candidate_data_reused": False,
            "sender_overlap_count": 0,
            "template_overlap_count": 0,
            "record_overlap_count": 0,
            "sealed_test_rows_materialized": 0,
        },
    }
    report_path = repo / data["report"]
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_candidate_data_reuse_requires_exact_report_hashes_and_counts(tmp_path: Path) -> None:
    config = load_pipeline_config(CONFIG)
    _write_candidate_fixture(tmp_path, config)

    verified = verify_candidate_data(config, tmp_path)
    assert verified["rows"] == {"train": 152, "dev": 29}
    assert verified["builder_version"] == CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION
    assert set(verified["candidate_implementations"]) == {
        "extractor",
        "protocol",
    }

    train_path = tmp_path / config["data"]["train"]
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(PipelineConfigError, match="hash mismatch"):
        verify_candidate_data(config, tmp_path)


def test_candidate_data_rejects_unaccepted_android_prefilter_rows(tmp_path: Path) -> None:
    config = load_pipeline_config(CONFIG)
    _write_candidate_fixture(tmp_path, config)
    report_path = tmp_path / config["data"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["invariants"]["android_prefilter_accepted"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(PipelineConfigError, match="android_prefilter_accepted"):
        verify_candidate_data(config, tmp_path)


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("builder", "builder version mismatch"),
        ("extractor", "extractor implementation hash mismatch"),
        ("protocol", "protocol implementation hash mismatch"),
    ],
)
def test_candidate_data_rejects_stale_builder_or_implementation(
    tmp_path: Path,
    component: str,
    message: str,
) -> None:
    config = load_pipeline_config(CONFIG)
    _write_candidate_fixture(tmp_path, config)
    report_path = tmp_path / config["data"]["report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if component == "builder":
        report["builder_version"] = "stale-builder"
    else:
        report["candidate_implementations"][component]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(PipelineConfigError, match=message):
        verify_candidate_data(config, tmp_path)


def test_candidate_stage_outputs_are_guarded_per_seed() -> None:
    config = load_pipeline_config(CONFIG)
    seed_17 = set(stage_output_paths(config, "convert-selector", ROOT, seed=17))
    seed_29 = set(stage_output_paths(config, "convert-selector", ROOT, seed=29))

    assert len(seed_17) == 4
    assert seed_17.isdisjoint(seed_29)
    assert any(path.name.endswith("-Q4_K_M.gguf") for path in seed_17)


def test_candidate_profile_verifies_declaration_and_explicit_android_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config(CONFIG)
    android_repo = tmp_path / "android-checkout"
    candidate_calls = []
    baseline_calls = []

    def fake_candidate_profile(value: dict, repo_root: Path) -> dict[str, object]:
        candidate_calls.append((value, repo_root))
        return {
            "declaration_verified": True,
            "profile": "candidate_protocol_v1",
        }

    def fake_android_profile(
        profile_path: Path,
        *,
        android_repo: Path | None = None,
        require_head: bool = True,
    ) -> dict[str, object]:
        baseline_calls.append((profile_path, android_repo, require_head))
        return {
            "declaration_verified": True,
            "repository_verified": True,
            "profile": "pocketfinancer",
        }

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "verify_candidate_profile", fake_candidate_profile)
    monkeypatch.setattr(
        runner,
        "verify_current_android_profile",
        fake_android_profile,
    )

    report = runner._app_profile_report(config, android_repo)

    assert candidate_calls == [(config, tmp_path)]
    assert baseline_calls == [
        (
            tmp_path / config["baseline_app_profile"],
            android_repo,
            True,
        )
    ]
    assert report["profile"] == "candidate_protocol_v1"
    assert report["baseline_android_profile"] == {
        "declaration_verified": True,
        "repository_verified": True,
        "profile": "pocketfinancer",
    }


def test_candidate_profile_fails_closed_on_android_checkout_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config(CONFIG)
    candidate_verified = False

    def fake_candidate_profile(_config: dict, _repo: Path) -> dict[str, object]:
        nonlocal candidate_verified
        candidate_verified = True
        return {"declaration_verified": True}

    def reject_android_profile(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AndroidProfileError("Android HEAD differs from pinned profile revision")

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "verify_candidate_profile", fake_candidate_profile)
    monkeypatch.setattr(runner, "verify_current_android_profile", reject_android_profile)

    with pytest.raises(AndroidProfileError, match="HEAD differs"):
        runner._app_profile_report(config, tmp_path / "stale-android")
    assert candidate_verified is True


def test_candidate_check_requires_profile_model_lock_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_pipeline_config(CONFIG)
    commands = build_stage_commands(config, seed=17)
    model_root = tmp_path / config["model"]["local_path"]
    model_root.mkdir(parents=True)
    report_path = tmp_path / config["data"]["report"]
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_app_profile_report",
        lambda _config, _android_repo: {"declaration_verified": True},
    )
    monkeypatch.setattr(
        runner,
        "verify_locked_model",
        lambda _config, _root: {"verified_files": 1},
    )
    monkeypatch.setattr(
        runner,
        "verify_candidate_data",
        lambda _config, _root: {"verified": False},
    )

    assert runner._check(config, commands, None, 17) == 1
    capsys.readouterr()

    monkeypatch.setattr(
        runner,
        "verify_candidate_data",
        lambda _config, _root: {"verified": True},
    )
    assert runner._check(config, commands, None, 17) == 0


def test_compare_stage_requires_all_six_metrics_and_guards_one_report() -> None:
    config = load_pipeline_config(CONFIG)
    requirements = stage_requirements(config, "compare-hf-seed-matrix", ROOT, seed=17)
    metric_labels = {
        requirement.label for requirement in requirements if requirement.label.endswith("_metrics")
    }

    assert metric_labels == {
        "direct_s17_metrics",
        "direct_s29_metrics",
        "direct_s43_metrics",
        "selector_s17_metrics",
        "selector_s29_metrics",
        "selector_s43_metrics",
    }
    assert {"model", "diagnostic_dataset", "data_report", "golden_vectors"} <= {
        requirement.label for requirement in requirements
    }
    assert list(stage_output_paths(config, "compare-hf-seed-matrix", ROOT, seed=17)) == [
        ROOT / config["comparison"]["output"]
    ]


def test_single_seed_all_stops_before_shared_matrix_comparison() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "all",
            "--dry-run",
            "--seed",
            "17",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)

    assert "convert-selector" in value
    assert "compare-hf-seed-matrix" not in value


def test_legacy_all_dry_run_expands_every_legacy_stage() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "all",
            "--dry-run",
            "--config",
            str(LEGACY_CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert tuple(json.loads(result.stdout)) == EXECUTION_STAGES


def test_legacy_training_and_hf_evaluation_pass_configured_model_lock() -> None:
    config = load_pipeline_config(LEGACY_CONFIG)
    commands = build_stage_commands(config)

    for stage in ("train", "evaluate-base-hf", "evaluate-hf"):
        assert _value(commands[stage], "--model-lock") == config["model"]["lock"]


def test_candidate_all_reuses_only_hash_verified_candidate_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_pipeline_config(CONFIG)
    data_output = tmp_path / "candidate-data"
    data_output.mkdir()
    (data_output / "report.json").write_text("{}", encoding="utf-8")
    verified_report = {
        "verified": True,
        "builder_version": CANDIDATE_PROTOCOL_V1_DATA_BUILDER_VERSION,
    }

    def output_paths(
        _config: dict,
        stage: str,
        _repo_root: Path,
        *,
        seed: int | None = None,
    ) -> list[Path]:
        assert seed == 17
        return [data_output] if stage == "build-candidate-data" else []

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "stage_output_paths", output_paths)
    monkeypatch.setattr(
        runner,
        "verify_candidate_data",
        lambda _config, _root: verified_report,
    )

    reusable = runner._candidate_all_preflight(
        config,
        ("build-candidate-data", "train-direct"),
        17,
        force_data=False,
    )
    assert reusable == {"build-candidate-data": verified_report}

    def reject_data(_config: dict, _root: Path) -> dict[str, object]:
        raise PipelineConfigError("candidate data provenance mismatch")

    monkeypatch.setattr(runner, "verify_candidate_data", reject_data)
    with pytest.raises(PipelineConfigError, match="provenance mismatch"):
        runner._candidate_all_preflight(
            config,
            ("build-candidate-data",),
            17,
            force_data=False,
        )
    assert (
        runner._candidate_all_preflight(
            config,
            ("build-candidate-data",),
            17,
            force_data=True,
        )
        == {}
    )


def test_candidate_all_rejects_later_output_before_running_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_pipeline_config(CONFIG)
    later_output = tmp_path / "selector-conversion.gguf"
    later_output.write_text("occupied", encoding="utf-8")

    def output_paths(
        _config: dict,
        stage: str,
        _repo_root: Path,
        *,
        seed: int | None = None,
    ) -> list[Path]:
        assert seed == 17
        return [later_output] if stage == "convert-selector" else []

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_pipeline_config", lambda _path: config)
    monkeypatch.setattr(
        runner,
        "_app_profile_report",
        lambda _config, _repo: {"declaration_verified": True},
    )
    monkeypatch.setattr(runner, "stage_output_paths", output_paths)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("preflight allowed stage execution"),
    )

    with pytest.raises(SystemExit) as raised:
        runner.main(["all", "--seed", "17", "--config", str(CONFIG)])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "no stages were run" in error
    assert "convert-selector=" in error


def test_comparison_collision_requires_a_fresh_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_pipeline_config(CONFIG)
    comparison_output = tmp_path / "comparison.json"
    comparison_output.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_pipeline_config", lambda _path: config)
    monkeypatch.setattr(
        runner,
        "_app_profile_report",
        lambda _config, _repo: {"declaration_verified": True},
    )
    monkeypatch.setattr(runner, "missing_requirements", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        runner,
        "verify_candidate_data",
        lambda _config, _root: {"verified": True},
    )
    monkeypatch.setattr(
        runner,
        "verify_locked_model",
        lambda _config, _root: {"verified_files": 1},
    )
    monkeypatch.setattr(
        runner,
        "stage_output_paths",
        lambda *_args, **_kwargs: [comparison_output],
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("collision allowed comparison overwrite"),
    )

    with pytest.raises(SystemExit) as raised:
        runner.main(["compare-hf-seed-matrix", "--config", str(CONFIG)])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "was not overwritten" in error
    assert "fresh comparison.output path" in error


def test_shared_matrix_comparison_dry_run_does_not_require_seed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare-hf-seed-matrix",
            "--dry-run",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)

    assert tuple(value) == ("compare-hf-seed-matrix",)


def test_legacy_check_preserves_profile_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_pipeline_config(
        ROOT / "configs" / "pipelines" / "pocketfinancer-lfm2.5-350m.json"
    )
    commands = build_stage_commands(config)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_app_profile_report", lambda _config, _repo: {"declaration_verified": False}
    )
    assert runner._check(config, commands, None, None) == 1
    capsys.readouterr()
