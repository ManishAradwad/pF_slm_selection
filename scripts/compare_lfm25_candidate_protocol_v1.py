#!/usr/bin/env python3
"""Compare the three-seed direct and Candidate Protocol V1 HF metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lfm25.candidate_protocol_compare import (  # noqa: E402
    ComparisonEvidenceError,
    SEEDS,
    compare_metric_files,
    write_report,
)
from lfm25.pipeline import (  # noqa: E402
    PipelineConfigError,
    candidate_comparison_anchors,
    load_pipeline_config,
)


DEFAULT_DIRECT_TEMPLATE = "RESULTS/pocketfinancer-candidate-v1/direct-r16-hf-s{seed}/metrics.json"
DEFAULT_SELECTOR_TEMPLATE = (
    "RESULTS/pocketfinancer-candidate-v1/selector-r16-hf-s{seed}/metrics.json"
)
DEFAULT_CONFIG = "configs/pipelines/pocketfinancer-lfm2.5-350m-candidate-v1.json"


def _arm_paths(
    args: argparse.Namespace, arm: str, parser: argparse.ArgumentParser
) -> dict[int, Path]:
    explicit = {seed: getattr(args, f"{arm}_s{seed}") for seed in SEEDS}
    supplied = [value is not None for value in explicit.values()]
    template = getattr(args, f"{arm}_template")
    if any(supplied) and not all(supplied):
        parser.error(f"all three --{arm}-sSEED paths are required together")
    if all(supplied) and template is not None:
        parser.error(f"--{arm}-template cannot be combined with explicit paths")
    if all(supplied):
        return {seed: Path(path) for seed, path in explicit.items() if path is not None}
    selected = template or (
        DEFAULT_DIRECT_TEMPLATE if arm == "direct" else DEFAULT_SELECTOR_TEMPLATE
    )
    if "{seed}" not in selected:
        parser.error(f"--{arm}-template must contain {{seed}}")
    return {seed: Path(selected.format(seed=seed)) for seed in SEEDS}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build an aggregate-only Candidate Protocol V1 controlled HF comparison.")
    )
    for arm in ("direct", "selector"):
        parser.add_argument(f"--{arm}-template")
        for seed in SEEDS:
            parser.add_argument(
                f"--{arm}-s{seed}",
                f"--{arm}-{seed}",
                type=Path,
                dest=f"{arm}_s{seed}",
            )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    direct_paths = _arm_paths(args, "direct", parser)
    selector_paths = _arm_paths(args, "selector", parser)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        config = load_pipeline_config(config_path)
        anchors = candidate_comparison_anchors(config, REPO_ROOT)
        report = compare_metric_files(
            direct_paths,
            selector_paths,
            trusted_anchors=anchors,
        )
        write_report(report, args.output, force=args.force)
    except (ComparisonEvidenceError, FileExistsError, PipelineConfigError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["controlled_hf_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
