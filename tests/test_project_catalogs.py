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


def test_2_6b_candidate_record_is_pinned_but_not_downloaded() -> None:
    model = _read_json("configs/models/lfm2.5-2.6b.json")
    files = model["gguf"]["files"]

    assert model["status"] == "candidate_not_downloaded"
    assert len(model["post_trained"]["revision"]) == 40
    assert files["LFM2.5-2.6B-Q4_0.gguf"] < files["LFM2.5-2.6B-Q8_0.gguf"]
    assert model["license"]["commercial_revenue_threshold_usd"] == 10_000_000
