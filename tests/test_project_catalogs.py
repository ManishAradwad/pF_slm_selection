from __future__ import annotations

import hashlib
import json
from pathlib import Path

from DATA.utils import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]


def _read_json(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_android_contract_catalog_matches_checked_in_assets() -> None:
    contract = _read_json("configs/contracts/pocketfinancer-android-current.json")
    prompt = contract["prompt"]
    examples = json.dumps(FEW_SHOT_EXAMPLES, ensure_ascii=False, indent=2).encode("utf-8")
    system_hash = _sha256_bytes(SYSTEM_PROMPT.encode("utf-8"))
    examples_hash = _sha256_bytes(examples)
    grammar_hash = _sha256_bytes((ROOT / "DATA/sms_extraction.gbnf").read_bytes())

    assert contract["profile"] == "pocketfinancer"
    assert contract["profile_version"] == 3
    assert contract["status"] == "current_app_profile"
    assert contract["android_source"]["revision"] == (
        "a9b7df44be2183daac3a05cadbfd40b8f309cd4b"
    )
    assert contract["android_source"]["hash_basis"] == (
        "sha256_of_git_blob_content_at_revision"
    )
    assert prompt["few_shot_examples"] == 7
    assert prompt["assets"]["system_prompt.txt"] == system_hash
    assert prompt["assets"]["few_shot_examples.json"] == examples_hash
    assert prompt["assets"]["sms_extraction.gbnf"] == grammar_hash
    source_files = contract["android_source"]["files"]
    assert source_files["inference/src/main/assets/system_prompt.txt"] == system_hash
    assert source_files["inference/src/main/assets/few_shot_examples.json"] == examples_hash
    assert source_files["inference/src/main/assets/sms_extraction.gbnf"] == grammar_hash

    assert contract["preprocessing"]["enabled"] is True
    assert len(contract["preprocessing"]["ordered_stages"]) == 6
    assert contract["runtime"]["n_ctx"] == 3072
    assert contract["runtime"]["n_gpu_layers"] == 0
    assert contract["runtime"]["max_cpu_threads"] == 4
    assert contract["generation"]["thinking_mode"] == "model_config"
    assert contract["generation"]["non_thinking"] == {
        "passes": 1,
        "answer_max_tokens": 256,
        "stop": None,
    }
    assert contract["generation"]["grammar"] == {
        "optional": True,
        "default_enabled": False,
    }
    assert contract["models"]["LiquidAI/LFM2.5-350M"] == {
        "has_thinking_mode": False,
        "generation_path": "non_thinking",
    }


def test_clean_experiment_catalog_keeps_data_and_runtime_status_separate() -> None:
    catalog = _read_json("configs/experiments/lfm2.5-350m-clean-research.json")
    by_protocol = {item["protocol"]: item for item in catalog["experiments"]}
    direct = by_protocol["direct_four_field_generation"]
    candidate = by_protocol["grounded_candidate_selector"]

    assert catalog["evaluation_boundary"]["status"] == (
        "locked_reused_regression_not_fresh_test"
    )
    assert catalog["contract_warning"]["android_runtime_parity"] is False
    assert direct["train"]["rows"] == 160
    assert direct["dev"]["rows"] == 25
    assert candidate["train"]["rows"] == 158 + 280 == 438
    assert candidate["android_wire_compatible"] is False
    assert candidate["result"]["all_emitted_fields_source_grounded"] is True


def test_2_6b_local_diagnostic_record_matches_locks() -> None:
    model = _read_json("configs/models/lfm2.5-2.6b.json")
    post_lock = _read_json(model["locks"]["post_trained"])
    base_lock = _read_json(model["locks"]["base"])
    files = model["gguf"]["files"]

    assert model["status"] == "evaluated_local_diagnostic_only"
    assert model["locks"] == {
        "post_trained": "configs/lfm25/model-2.6b-post.lock.json",
        "base": "configs/lfm25/model-2.6b-base.lock.json",
    }
    assert model["artifacts"] == {"weights": "local_only", "gitignored": True}
    assert model["support"] == {"android": False, "deployment": False}
    assert (
        model["post_trained"]["revision"]
        == post_lock["model"]["revision"]
        == "dca1825886789bd40b94368f53b1d9ada4c94598"
    )
    assert (
        model["base"]["revision"]
        == base_lock["model"]["revision"]
        == "78f33a52fbe65f7665963f482179dcc3e75f0d9e"
    )
    assert model["post_trained"]["parameters"] == post_lock["model"]["parameters"] == 2_697_198_592
    assert model["base"]["parameters"] == base_lock["model"]["parameters"] == 2_697_198_592
    assert files["LFM2.5-2.6B-Q4_0.gguf"] < files["LFM2.5-2.6B-Q8_0.gguf"]
    assert model["license"]["commercial_revenue_threshold_usd"] == 10_000_000


def test_2_6b_experiment_record_is_aggregate_and_not_production() -> None:
    record = _read_json("configs/experiments/lfm2.5-2.6b-base-r16-s17.json")

    assert record["schema_version"] == 1
    assert record["status"] == "evaluated_local_diagnostic_only"
    assert record["recorded_at"] == "2026-08-05"
    assert record["run_id"] == "lfm2.5-2.6b-base-direct-r16-s17"

    expected_revisions = {
        "post_trained": "dca1825886789bd40b94368f53b1d9ada4c94598",
        "base": "78f33a52fbe65f7665963f482179dcc3e75f0d9e",
    }
    for model_name, expected_revision in expected_revisions.items():
        model = record["models"][model_name]
        lock = _read_json(model["lock"])
        assert model["revision"] == lock["model"]["revision"] == expected_revision

    boundary = record["evaluation_boundary"]
    assert boundary["status"] == "locked_reused_regression_not_fresh_test"
    assert (boundary["rows"], boundary["transactions"], boundary["nulls"]) == (
        203,
        114,
        89,
    )
    assert boundary["sha256"] == (
        "fec483b11cf458212b6a636f508632649790beacc91050efdd52abb2b590d44e"
    )
    assert boundary["used_for_training"] is False
    assert boundary["human_gold"] is False

    blind_review = record["blinded_human_review"]
    assert blind_review == {
        "metadata": "PRIVATE_DATA/lfm25/blinded_test_review_metadata.json",
        "package_version": "lfm25-blinded-test-review-v1",
        "rows": 1436,
        "pending_rows": 1436,
        "completed_rows": 0,
        "status": "package_frozen_pending_human_review",
        "human_gold": False,
        "used_for_training": False,
        "used_for_results": False,
    }

    training_data = record["training_data"]
    assert training_data["human_gold"] is False
    assert (training_data["train"]["rows"], training_data["train"]["sha256"]) == (
        154,
        "66a5ffe1f0722a594838f6ee405cefa1f3adb8290dec92db222021c6a6e12f56",
    )
    assert (training_data["dev"]["rows"], training_data["dev"]["sha256"]) == (
        29,
        "126b9337e49ecbbbc437dbf68b113f0a51d00ad28fe5bb1b6590427f742f9513",
    )

    training = record["training"]
    assert training["lora"]["rank"] == 16
    assert training["lora"]["trainable_parameters"] == 24_461_312
    assert training["lora"]["target_module_coverage"]["matched_module_count"] == 166
    assert training["optimization"]["optimizer"] == "adamw_torch"
    assert training["optimization"]["effective_batch_size"] == 32
    assert training["checkpoint_selection"]["best_step"] == 5
    assert training["checkpoint_selection"]["final_global_step"] == 15
    assert training["checkpoint_selection"]["restored_best_checkpoint"] is True
    assert training["timing"]["training_wall_time_seconds"] == 1227.775
    assert training["peak_vram_mib"] == 8560.9
    assert record["capacity_probe"]["peak_vram_mib"] == 7351.9

    artifacts = record["artifacts"]
    assert (artifacts["adapter"]["bytes"], artifacts["adapter"]["sha256"]) == (
        97_889_536,
        "7e1f2d0f47ab901942b4e307b06e02d45f09393ff5d94a8e8aac45f149ce7b18",
    )
    expected_gguf = {
        "bf16": (
            5_403_156_736,
            "ad830af7b17bb2f90eecc70fedf7f939291dda5d267ce7cc58fee1d84d1cf7f5",
        ),
        "q8_0": (
            2_874_777_856,
            "fccdc839f5e3213a769b280abbd2d8ca9df375807105b9fb0683db673f1c6c74",
        ),
        "q4_k_m": (
            1_674_453_248,
            "9849f6377eab1a292665714b91819ef17ed2cf52cff20717713fa646e0f7f77c",
        ),
    }
    for name, expected in expected_gguf.items():
        artifact = artifacts["gguf"][name]
        assert (artifact["bytes"], artifact["sha256"]) == expected

    post_lock = _read_json(record["models"]["post_trained"]["lock"])
    official_post_q4 = artifacts["official_post_q4_k_m_reference"]
    assert official_post_q4["bytes"] == 1_674_454_848
    assert official_post_q4["sha256"] == post_lock["official_gguf"]["files"][
        "LFM2.5-2.6B-Q4_K_M.gguf"
    ]

    by_result = {item["id"]: item for item in record["results"]}
    expected_app_results = {
        "post_trained_hf": (174, 86, 0.982301, 203),
        "post_trained_official_q4_single_bos": (176, 88, 0.973214, 203),
        "base_untuned_hf": (186, 98, 0.995633, 203),
        "base_lora_hf": (184, 96, 0.991228, 203),
        "base_lora_bf16_single_bos": (184, 96, 0.991228, 203),
        "base_lora_q8_single_bos": (184, 96, 0.991228, 203),
        "base_lora_q4_android_current": (166, 78, 0.949772, 203),
        "base_lora_q4_single_bos": (173, 85, 0.959276, 203),
    }
    assert set(by_result) == set(expected_app_results)
    for result_id, expected in expected_app_results.items():
        result = by_result[result_id]
        app = result["app_interpreted"]
        assert (
            app["whole_exact"],
            app["transaction_exact"],
            app["transaction_f1"],
            app["valid_outputs"],
        ) == expected
        assert app["schema_validity"] == 1.0
        assert result["android_runtime_parity"] is False

    assert by_result["post_trained_hf"]["strict_model_output"]["valid_outputs"] == 202
    assert (
        by_result["post_trained_official_q4_single_bos"]["strict_model_output"][
            "valid_outputs"
        ]
        == 198
    )

    by_pair = {item["id"]: item for item in record["paired_comparisons"]}
    assert by_pair["base_untuned_hf_vs_base_lora_hf"]["mcnemar_exact_p"] == 0.7265625
    assert (
        by_pair["q4_android_current_vs_q4_single_bos"]["mcnemar_exact_p"]
        == 0.0390625
    )
    assert by_pair["base_lora_hf_vs_q4_single_bos"]["mcnemar_exact_p"] == 0.00738525

    assert record["parity"] == {
        "runtime": False,
        "android": False,
        "device": False,
        "deployment": False,
    }
    assert record["support"] == {
        "android": False,
        "deployment": False,
        "production": False,
    }
    assert record["decision"]["promote_lora"] is False
    assert record["decision"]["ship"] is False

    serialized = json.dumps(record)
    assert "samples.jsonl" not in serialized
    assert "internal_map" not in serialized
    assert '"human_gold": true' not in serialized

def test_repository_guidance_is_harness_neutral() -> None:
    required_files = (
        "README.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        ".github/pull_request_template.md",
        "docs/history/GENERAL_GGUF_BENCHMARK_2026-04-25.md",
    )
    for relative in required_files:
        assert (ROOT / relative).is_file(), relative

    assert not (ROOT / "CLAUDE.md").exists()
    assert not (ROOT / ".claude/scheduled_tasks.lock").exists()
    assert not (ROOT / ".devcontainer/post-create.sh").exists()

    guidance_files = (
        "README.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "docs/README.md",
        "scripts/README.md",
        "scripts/evaluate.sh",
        "scripts/fetch_models.sh",
    )
    guidance = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in guidance_files
    ).lower()
    assert "claude.md" not in guidance
    assert "@anthropic-ai" not in guidance

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/run_pocketfinancer_pipeline.py" in readme
    assert "AGENTS.md" in readme
