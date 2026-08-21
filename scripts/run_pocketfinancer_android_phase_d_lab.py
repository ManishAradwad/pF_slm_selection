#!/usr/bin/env python3
"""Validate and optionally run the synthetic-only Android Phase D GGUF lab."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lfm25.android_protocol_lab import (  # noqa: E402
    ANDROID_COMMIT,
    BASELINE_ID,
    BASELINE_SHA256,
    LAB_ID,
    LAB_VERSION,
    SYNTHETIC_FIXTURE_PATH,
    AndroidProtocolLabError,
    load_lab_manifest,
    run_host_gguf_profile,
    run_parser_conformance,
    sha256_file,
    validate_result_set,
    verify_local_artifact,
)
from lfm25.workbench_v2 import write_aggregate_report  # noqa: E402


RESULTS_ROOT = REPOSITORY_ROOT / "RESULTS"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _profile_gap(
    profile: Mapping[str, Any],
    manifest: Mapping[str, Any],
    conformance: Mapping[str, Any],
    gap_code: str,
) -> dict[str, Any]:
    prompt = _read_json(REPOSITORY_ROOT / profile["prompt_profile"]["path"])
    return {
        "experiment_profile_id": profile["experiment_profile_id"],
        "runtime_variant_id": profile["runtime_variant_id"],
        "model_family": profile["model_family"],
        "protocol_id": profile["protocol_id"],
        "selected_profile_id": None,
        "provenance": {
            "android_commit": ANDROID_COMMIT,
            "baseline_id": BASELINE_ID,
            "baseline_manifest_sha256": BASELINE_SHA256,
            "model_revision": profile["model"]["model_revision"],
            "artifact_sha256": profile["model"]["artifact_sha256"],
            "chat_template_sha256": profile["model"]["chat_template_sha256"],
            "prompt_profile_id": prompt["prompt_profile_id"],
            "prompt_sha256": profile["prompt_profile"]["sha256"],
            "parser_sha256": manifest["resources"]["parser"]["sha256"],
            "runtime_profile_sha256": manifest["resources"]["runtime_profile"]["sha256"],
            "evaluator_sha256": manifest["resources"]["evaluator"]["sha256"],
            "fixture_sha256": manifest["resources"]["synthetic_fixture"]["sha256"],
        },
        "evidence": {
            "source_static": {"status": "measured", "conformance": dict(conformance)},
            "host_hf": {
                "status": "not_applicable",
                "gap_codes": ["hf_artifact_not_in_phase_d_scope"],
            },
            "host_gguf": {"status": "not_measured", "gap_codes": [gap_code]},
            "android_device": {
                "status": "not_measured",
                "gap_codes": ["android_protocol_harness_unavailable"],
            },
        },
        "gate_status": {
            "decision": "no_selection",
            "passed": False,
            "assessment": None,
            "additional_gap_codes": [
                "insufficient_synthetic_sample",
                "no_reproducible_quality_baseline",
                "protected_scoring_not_authorized",
            ],
        },
    }


def _verify_runtime_capabilities(runtime: Mapping[str, Any]) -> None:
    if runtime["accelerator"] != "nvidia_cuda":
        return
    from llama_cpp import llama_cpp

    if runtime["n_gpu_layers"] != -1 or not runtime["gpu_offload_required"]:
        raise AndroidProtocolLabError(
            "CUDA runtime must require all-layer GPU offload"
        )
    if not llama_cpp.llama_supports_gpu_offload():
        raise AndroidProtocolLabError("llama.cpp CUDA GPU offload is unavailable")


def _load_llama(profile: Mapping[str, Any], runtime: Mapping[str, Any]):
    from llama_cpp import Llama

    return Llama(
        model_path=str(REPOSITORY_ROOT / profile["model"]["artifact_path"]),
        n_gpu_layers=runtime["n_gpu_layers"],
        n_ctx=runtime["n_ctx"],
        n_batch=runtime["n_batch"],
        n_ubatch=runtime["n_ubatch"],
        n_threads=runtime["n_threads"],
        n_threads_batch=runtime["n_threads_batch"],
        flash_attn=runtime["flash_attention"],
        use_mmap=runtime["use_mmap"],
        seed=runtime["seed"],
        verbose=False,
    )


def _chat_template_sha256(llama: Any) -> str:
    template = llama.metadata.get("tokenizer.chat_template", "")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _result_set(
    *,
    manifest: Mapping[str, Any],
    implementation_commit: str,
    profiles: list[dict[str, Any]],
    device_smoke: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lab_id": LAB_ID,
        "lab_version": LAB_VERSION,
        "result_set_id": "pocketfinancer-android-phase-d-synthetic-cuda-v1",
        "status": "completed_evaluation_only_no_selection",
        "implementation_commit": implementation_commit,
        "bindings": {
            "lab_manifest_sha256": manifest["manifest_sha256"],
            "android_commit": ANDROID_COMMIT,
            "baseline_id": BASELINE_ID,
            "baseline_manifest_sha256": BASELINE_SHA256,
        },
        "privacy": {
            "classification": "aggregate_only_invented_synthetic",
            "contains_private_data": False,
            "contains_row_level_predictions": False,
            "raw_model_output_retained": False,
        },
        "device_runtime_smoke": dict(device_smoke),
        "profiles": profiles,
        "selection": {
            "decision": "no_selection",
            "selected_profile_id": None,
            "direct_v2_selected": False,
            "candidate_v2_selected": False,
            "model_selected": False,
            "runtime_variant_selected": False,
            "production_defaults_changed": False,
        },
        "phase_e_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--run-host-gguf", action="store_true")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--implementation-commit")
    parser.add_argument("--device-smoke-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        manifest = load_lab_manifest()
        conformance = run_parser_conformance()
        profiles = manifest["profiles"]
        requested = set(args.profile)
        if requested:
            profiles = [item for item in profiles if item["experiment_profile_id"] in requested]
            if {item["experiment_profile_id"] for item in profiles} != requested:
                raise AndroidProtocolLabError("an unknown --profile was requested")

        artifact_status = {}
        if args.verify_artifacts or args.run_host_gguf:
            for profile in profiles:
                variant = profile["runtime_variant_id"]
                if variant not in artifact_status:
                    artifact_status[variant] = verify_local_artifact(profile)

        if not args.run_host_gguf:
            summary = {
                "lab_id": manifest["lab_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "profiles_validated": len(manifest["profiles"]),
                "parser_conformance": conformance["protocols"],
                "artifacts_verified": sum(
                    item.get("status") == "verified" for item in artifact_status.values()
                ),
                "selection": "none",
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if requested and len(profiles) != len(manifest["profiles"]):
            raise AndroidProtocolLabError(
                "partial host runs may be diagnosed but cannot emit a final result set"
            )
        if args.output is None:
            raise AndroidProtocolLabError("--output is required with --run-host-gguf")
        implementation_commit = args.implementation_commit or _git_head()
        if len(implementation_commit) != 40:
            raise AndroidProtocolLabError("implementation commit must be a full Git object ID")
        runtime_profile_path = (
            REPOSITORY_ROOT / manifest["resources"]["runtime_profile"]["path"]
        )
        runtime_profile = _read_json(runtime_profile_path)
        runtime = runtime_profile["host_gguf_runtime"]
        _verify_runtime_capabilities(runtime)
        annotation_package = _read_json(SYNTHETIC_FIXTURE_PATH)
        conformance_by_protocol = conformance["protocols"]

        aggregate_profiles = []
        by_variant: dict[str, list[Mapping[str, Any]]] = {}
        for profile in profiles:
            by_variant.setdefault(profile["runtime_variant_id"], []).append(profile)
        for variant, variant_profiles in by_variant.items():
            status = artifact_status[variant]
            if status["status"] != "verified":
                for profile in variant_profiles:
                    aggregate_profiles.append(
                        _profile_gap(
                            profile,
                            manifest,
                            conformance_by_protocol[profile["protocol_id"]],
                            status["gap_code"],
                        )
                    )
                continue
            try:
                llama = _load_llama(variant_profiles[0], runtime)
                if _chat_template_sha256(llama) != variant_profiles[0]["model"][
                    "chat_template_sha256"
                ]:
                    raise AndroidProtocolLabError("embedded GGUF chat template hash mismatch")
                for profile in variant_profiles:
                    aggregate_profiles.append(
                        run_host_gguf_profile(
                            llama,
                            profile=profile,
                            manifest=manifest,
                            annotation_package=annotation_package,
                            runtime_profile=runtime_profile,
                            conformance=conformance_by_protocol[profile["protocol_id"]],
                        )
                    )
            except Exception:
                existing = {item["experiment_profile_id"] for item in aggregate_profiles}
                for profile in variant_profiles:
                    if profile["experiment_profile_id"] not in existing:
                        aggregate_profiles.append(
                            _profile_gap(
                                profile,
                                manifest,
                                conformance_by_protocol[profile["protocol_id"]],
                                "gguf_host_runtime_error",
                            )
                        )
            finally:
                if "llama" in locals():
                    del llama
                gc.collect()

        device_smoke = (
            _read_json(args.device_smoke_json)
            if args.device_smoke_json is not None
            else {
                "status": "not_measured",
                "evidence_class": "android_device",
                "protocol_comparison_claim": False,
                "gap_codes": ["android_protocol_harness_unavailable"],
            }
        )
        result = _result_set(
            manifest=manifest,
            implementation_commit=implementation_commit,
            profiles=sorted(
                aggregate_profiles, key=lambda item: item["experiment_profile_id"]
            ),
            device_smoke=device_smoke,
        )
        validate_result_set(result, manifest=manifest)
        write_aggregate_report(args.output, result)
        print(
            json.dumps(
                {
                    "profiles_recorded": len(aggregate_profiles),
                    "host_gguf_measured": sum(
                        item["evidence"]["host_gguf"]["status"] == "measured"
                        for item in aggregate_profiles
                    ),
                    "android_device_protocol_profiles_measured": sum(
                        item["evidence"]["android_device"]["status"] == "measured"
                        for item in aggregate_profiles
                    ),
                    "selection": "none",
                    "output_sha256": sha256_file(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (AndroidProtocolLabError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Phase D lab failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
