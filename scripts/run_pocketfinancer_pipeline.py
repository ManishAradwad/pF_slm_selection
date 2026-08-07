#!/usr/bin/env python3
"""One app-first entry point for PocketFinancer data, tuning, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lfm25.pipeline import (  # noqa: E402
    PipelineConfigError,
    build_stage_commands,
    execution_stages,
    is_candidate_protocol,
    load_pipeline_config,
    missing_requirements,
    pipeline_seeds,
    stage_output_paths,
    stage_requirements,
    verify_candidate_data,
    verify_candidate_profile,
    verify_locked_model,
)
from lfm25.android_profile_sync import (  # noqa: E402
    AndroidProfileError,
    verify_current_android_profile,
)


DEFAULT_CONFIG = Path("configs/pipelines/pocketfinancer-lfm2.5-350m.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Android-aligned PocketFinancer data/fine-tuning/evaluation "
            "pipeline. The Android app profile is the only supported happy path."
        )
    )
    parser.add_argument(
        "stage",
        help="pipeline stage to inspect or execute",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print argv without checking generated inputs or executing commands",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="allow the private data builder to replace only its fixed outputs",
    )
    parser.add_argument(
        "--android-repo",
        type=Path,
        help=(
            "optional PocketFinancer Android checkout; known WSL locations are "
            "auto-detected when omitted"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Candidate V1 controlled seed (17, 29, or 43)",
    )
    return parser


def _display(commands: dict[str, list[str]], stages: tuple[str, ...]) -> None:
    print(json.dumps({stage: commands[stage] for stage in stages}, indent=2))


def _app_profile_report(config: dict, android_repo: Path | None) -> dict[str, object]:
    profile_path = REPO_ROOT / str(config["app_profile"])
    if is_candidate_protocol(config):
        candidate_report = verify_candidate_profile(config, REPO_ROOT)
        baseline_path = REPO_ROOT / str(config["baseline_app_profile"])
        baseline_report = verify_current_android_profile(
            baseline_path,
            android_repo=android_repo,
        )
        if not bool(baseline_report.get("declaration_verified")):
            raise PipelineConfigError("Candidate V1 baseline Android profile verification failed")
        if android_repo is not None and baseline_report.get("repository_verified") is not True:
            raise PipelineConfigError("Candidate V1 explicit Android checkout was not verified")
        return {
            **candidate_report,
            "baseline_android_profile": baseline_report,
        }
    return verify_current_android_profile(
        profile_path,
        android_repo=android_repo,
    )


def _output_is_occupied(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def _occupied_stage_outputs(
    config: dict,
    stage: str,
    seed: int | None,
) -> list[Path]:
    return [
        path
        for path in stage_output_paths(config, stage, REPO_ROOT, seed=seed)
        if _output_is_occupied(path)
    ]


def _candidate_all_preflight(
    config: dict,
    stages: tuple[str, ...],
    seed: int,
    *,
    force_data: bool,
) -> dict[str, dict[str, object]]:
    """Reject collisions before execution and reuse only strongly verified data."""

    reusable: dict[str, dict[str, object]] = {}
    collisions: list[tuple[str, Path]] = []
    for stage in stages:
        occupied = _occupied_stage_outputs(config, stage, seed)
        if not occupied:
            continue
        if stage == "build-candidate-data":
            if force_data:
                continue
            reusable[stage] = verify_candidate_data(config, REPO_ROOT)
            continue
        collisions.extend((stage, path) for path in occupied)
    if collisions:
        details = ", ".join(f"{stage}={path.relative_to(REPO_ROOT)}" for stage, path in collisions)
        raise PipelineConfigError(
            "Candidate V1 all preflight found nonempty later-stage output(s); "
            "no stages were run. Use a new seed/run identity: "
            f"{details}"
        )
    return reusable


def _candidate_output_collision_message(stage: str, paths: list[Path]) -> str:
    details = ", ".join(str(path.relative_to(REPO_ROOT)) for path in paths)
    if stage == "compare-hf-seed-matrix":
        return (
            "comparison output already exists and was not overwritten; use a new "
            f"pipeline config with a fresh comparison.output path: {details}"
        )
    return f"{stage} output is not empty; use a new run identity: {details}"


def _check(
    config: dict,
    commands: dict[str, list[str]],
    android_repo: Path | None,
    seed: int | None,
) -> int:
    report = {}
    for stage in execution_stages(config):
        requirements = stage_requirements(config, stage, REPO_ROOT, seed=seed)
        report[stage] = {
            "ready": all(item.path.exists() for item in requirements),
            "inputs": {
                item.label: {
                    "path": str(item.path.relative_to(REPO_ROOT)),
                    "exists": item.path.exists(),
                }
                for item in requirements
            },
            "argv": commands[stage],
        }
    model_path = REPO_ROOT / str(config["model"]["local_path"])
    if model_path.is_dir():
        try:
            report["model_lock"] = {"verified": True, **verify_locked_model(config, REPO_ROOT)}
        except PipelineConfigError as error:
            report["model_lock"] = {"verified": False, "error": str(error)}
    else:
        report["model_lock"] = {"verified": False, "error": "local model is absent"}
    if is_candidate_protocol(config):
        report["controlled_seeds"] = list(pipeline_seeds(config))
        data_report = REPO_ROOT / str(config["data"]["report"])
        if data_report.is_file():
            try:
                report["candidate_data"] = verify_candidate_data(config, REPO_ROOT)
            except PipelineConfigError as error:
                report["candidate_data"] = {"verified": False, "error": str(error)}
        else:
            report["candidate_data"] = {"verified": False, "error": "not built"}
    try:
        report["app_profile"] = _app_profile_report(config, android_repo)
    except (AndroidProfileError, PipelineConfigError) as error:
        report["app_profile"] = {"declaration_verified": False, "error": str(error)}
    print(json.dumps(report, indent=2, sort_keys=True))
    profile_verified = bool(report["app_profile"].get("declaration_verified"))
    if is_candidate_protocol(config):
        ready = (
            profile_verified
            and bool(report["model_lock"].get("verified"))
            and bool(report["candidate_data"].get("verified"))
        )
        return 0 if ready else 1
    return 0 if profile_verified else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = load_pipeline_config(config_path)
    except PipelineConfigError as error:
        parser.error(str(error))

    available_stages = execution_stages(config)
    if args.stage not in {"check", "plan", "all", *available_stages}:
        parser.error(
            "unknown stage for this config; choose one of: "
            + ", ".join(("check", "plan", "all", *available_stages))
        )
    candidate = is_candidate_protocol(config)
    controlled_seeds = pipeline_seeds(config)
    if candidate and args.seed is not None and args.seed not in controlled_seeds:
        parser.error("Candidate V1 --seed must be one of 17, 29, or 43")
    if candidate and args.stage == "plan" and args.seed is None:
        plans = {
            str(seed): build_stage_commands(config, force_data=args.force_data, seed=seed)
            for seed in controlled_seeds
        }
        print(
            json.dumps(
                {
                    "protocol": config["protocol"],
                    "controlled_seeds": list(controlled_seeds),
                    "plans": plans,
                    "unmet_gates": config["gates"],
                },
                indent=2,
            )
        )
        return 0
    if (
        candidate
        and args.seed is None
        and args.stage
        not in {
            "check",
            "build-candidate-data",
            "compare-hf-seed-matrix",
        }
    ):
        parser.error("Candidate V1 execution requires --seed 17, 29, or 43")
    selected_seed = args.seed if candidate else None
    if candidate and selected_seed is None:
        selected_seed = controlled_seeds[0]
    try:
        commands = build_stage_commands(config, force_data=args.force_data, seed=selected_seed)
    except PipelineConfigError as error:
        parser.error(str(error))

    if candidate and args.stage == "all":
        stages = tuple(stage for stage in available_stages if stage != "compare-hf-seed-matrix")
    elif args.stage in {"plan", "all"}:
        stages = available_stages
    else:
        stages = (args.stage,)
    if args.stage == "plan" or args.dry_run:
        _display(commands, tuple(stage for stage in stages if stage in commands))
        return 0
    if args.stage == "check":
        return _check(config, commands, args.android_repo, selected_seed)

    try:
        app_profile = _app_profile_report(config, args.android_repo)
    except (AndroidProfileError, PipelineConfigError) as error:
        parser.error(str(error))
    print(json.dumps({"app_profile": app_profile}, sort_keys=True))

    reusable_stages: dict[str, dict[str, object]] = {}
    if candidate and args.stage == "all":
        try:
            reusable_stages = _candidate_all_preflight(
                config,
                stages,
                int(selected_seed),
                force_data=args.force_data,
            )
        except PipelineConfigError as error:
            parser.error(str(error))

    for stage in stages:
        missing = missing_requirements(config, stage, REPO_ROOT, seed=selected_seed)
        if missing:
            details = ", ".join(
                f"{item.label}={item.path.relative_to(REPO_ROOT)}" for item in missing
            )
            parser.error(f"{stage} is missing required input(s): {details}")
        if candidate and stage == "build-candidate-data" and not args.force_data:
            output_root = REPO_ROOT / str(config["data"]["output_dir"])
            if stage in reusable_stages or _output_is_occupied(output_root):
                try:
                    verified_data = verify_candidate_data(config, REPO_ROOT)
                except PipelineConfigError as error:
                    parser.error(
                        "Candidate V1 data output exists but is not reusable: "
                        f"{error}; pass --force-data only for an intentional rebuild"
                    )
                print(
                    json.dumps(
                        {"stage": stage, "skipped": True, "candidate_data": verified_data},
                        sort_keys=True,
                    )
                )
                continue
        if candidate and stage not in {
            "build-candidate-data",
            "evaluate-direct-base-hf",
            "evaluate-selector-base-hf",
        }:
            try:
                verify_candidate_data(config, REPO_ROOT)
            except PipelineConfigError as error:
                parser.error(str(error))
        if stage in {"evaluate-base-hf", "train", "evaluate-hf", "merge"}:
            try:
                verify_locked_model(config, REPO_ROOT)
            except PipelineConfigError as error:
                parser.error(str(error))
        if candidate and stage not in {
            "build-candidate-data",
            "convert-direct",
            "convert-selector",
        }:
            try:
                verify_locked_model(config, REPO_ROOT)
            except PipelineConfigError as error:
                parser.error(str(error))
        if stage == "train":
            output_dir = REPO_ROOT / str(config["training"]["output_dir"])
            if output_dir.exists() and any(output_dir.iterdir()):
                parser.error(
                    "training output directory is not empty; use a new run identity "
                    "or the low-level trainer's explicit resume option"
                )
        if candidate and not (stage == "build-candidate-data" and args.force_data):
            occupied = _occupied_stage_outputs(config, stage, selected_seed)
            if occupied:
                parser.error(_candidate_output_collision_message(stage, occupied))
        print(json.dumps({"stage": stage, "argv": commands[stage]}, sort_keys=True))
        completed = subprocess.run(
            commands[stage], cwd=REPO_ROOT, check=stage != "compare-hf-seed-matrix"
        )
        if stage == "compare-hf-seed-matrix" and completed.returncode:
            return completed.returncode
    if candidate and args.stage == "all":
        print(
            json.dumps(
                {
                    "next_stage": "compare-hf-seed-matrix",
                    "requires_completed_seeds": list(controlled_seeds),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
