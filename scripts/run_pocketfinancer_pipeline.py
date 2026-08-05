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
    EXECUTION_STAGES,
    PipelineConfigError,
    build_stage_commands,
    load_pipeline_config,
    missing_requirements,
    stage_requirements,
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
        choices=("check", "plan", "all", *EXECUTION_STAGES),
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
    return parser


def _display(commands: dict[str, list[str]], stages: tuple[str, ...]) -> None:
    print(json.dumps({stage: commands[stage] for stage in stages}, indent=2))


def _app_profile_report(
    config: dict, android_repo: Path | None
) -> dict[str, object]:
    profile_path = REPO_ROOT / str(config["app_profile"])
    return verify_current_android_profile(
        profile_path,
        android_repo=android_repo,
    )


def _check(
    config: dict,
    commands: dict[str, list[str]],
    android_repo: Path | None,
) -> int:
    report = {}
    for stage in EXECUTION_STAGES:
        requirements = stage_requirements(config, stage, REPO_ROOT)
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
    try:
        report["app_profile"] = _app_profile_report(config, android_repo)
    except AndroidProfileError as error:
        report["app_profile"] = {"declaration_verified": False, "error": str(error)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["app_profile"].get("declaration_verified") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = load_pipeline_config(config_path)
        commands = build_stage_commands(config, force_data=args.force_data)
    except PipelineConfigError as error:
        parser.error(str(error))

    stages = EXECUTION_STAGES if args.stage in {"plan", "all"} else (args.stage,)
    if args.stage == "plan" or args.dry_run:
        _display(commands, tuple(stage for stage in stages if stage in commands))
        return 0
    if args.stage == "check":
        return _check(config, commands, args.android_repo)

    try:
        app_profile = _app_profile_report(config, args.android_repo)
    except AndroidProfileError as error:
        parser.error(str(error))
    print(json.dumps({"app_profile": app_profile}, sort_keys=True))

    for stage in stages:
        missing = missing_requirements(config, stage, REPO_ROOT)
        if missing:
            details = ", ".join(
                f"{item.label}={item.path.relative_to(REPO_ROOT)}" for item in missing
            )
            parser.error(f"{stage} is missing required input(s): {details}")
        if stage in {"evaluate-base-hf", "train", "evaluate-hf", "merge"}:
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
        print(json.dumps({"stage": stage, "argv": commands[stage]}, sort_keys=True))
        subprocess.run(commands[stage], cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
