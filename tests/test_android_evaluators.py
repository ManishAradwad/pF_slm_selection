from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
HF_EVALUATOR = ROOT / "scripts" / "evaluate_lfm25_android_hf.py"
GGUF_EVALUATOR = ROOT / "scripts" / "evaluate_lfm25_android_gguf.py"


def _help(script: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_script(script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"test_{script.stem}", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hf_auto_thinking_mode_matches_model_config_and_safe_defaults() -> None:
    evaluator = _load_script(HF_EVALUATOR)

    assert evaluator._resolve_thinking_mode(
        "auto",
        model_config={"post_trained": {"always_reasons_before_answer": True}},
        model="LiquidAI/LFM2.5-2.6B",
    ) == (True, "model_config")
    assert evaluator._resolve_thinking_mode(
        "auto",
        model_config=None,
        model="TRAINING_ARTIFACTS/base/LFM2.5-350M",
    ) == (False, "lfm2.5-350m_non_thinking")
    assert evaluator._resolve_thinking_mode("auto", model_config=None, model="unknown-model") == (
        False,
        "auto_safe_default_off",
    )
    assert evaluator._resolve_thinking_mode(
        "on", model_config={"hasThinkingMode": False}, model="anything"
    ) == (True, "cli")

    with pytest.raises(ValueError, match="conflicting"):
        evaluator._resolve_thinking_mode(
            "auto",
            model_config={
                "hasThinkingMode": True,
                "nested": {"has_thinking_mode": False},
            },
            model="anything",
        )


def test_hf_thinking_prompt_reuses_the_template_opening() -> None:
    evaluator = _load_script(HF_EVALUATOR)
    rendered = "<|im_start|>assistant\n<think>"

    assert evaluator._prepare_thinking_prompt(rendered) == (rendered, False)
    assert evaluator._prepare_thinking_prompt(rendered + "\n") == (
        rendered + "\n",
        False,
    )
    appended, changed = evaluator._prepare_thinking_prompt("<|im_start|>assistant\n")
    assert changed is True
    assert appended.endswith("<think>\n")
    assert appended.count("<think>") == 1
    assert evaluator._split_at_token_sequence([10, 11, 12, 13], [12, 13]) == (
        [10, 11],
        True,
    )
    assert evaluator._split_at_token_sequence([10, 11], [12]) == (
        [10, 11],
        False,
    )


def test_hf_thinking_context_caps_reasoning_after_reserving_answer() -> None:
    evaluator = _load_script(HF_EVALUATOR)

    full = evaluator._thinking_context_budget(1780, 3072, 256, 2, 1024)
    assert full["effective_thinking_max_tokens"] == 1024
    assert full["thinking_capped_by_context"] is False

    capped = evaluator._thinking_context_budget(1900, 3072, 256, 2, 1024)
    assert capped["available_thinking_tokens"] == 914
    assert capped["effective_thinking_max_tokens"] == 914
    assert capped["thinking_capped_by_context"] is True

    with pytest.raises(ValueError, match="exceeds n_ctx=3072"):
        evaluator._thinking_context_budget(2815, 3072, 256, 2, 1024)
    with pytest.raises(ValueError, match="leaves no thinking capacity"):
        evaluator._thinking_context_budget(2814, 3072, 256, 2, 1024)


def test_hf_fingerprints_all_safetensor_shards_and_index(tmp_path: Path) -> None:
    evaluator = _load_script(HF_EVALUATOR)
    names = {
        "config.json": "{}",
        "model-00001-of-00002.safetensors": "first-shard",
        "model-00002-of-00002.safetensors": "second-shard",
        "model.safetensors.index.json": "{}",
        "tokenizer.json": "{}",
    }
    for name, content in names.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    evidence = evaluator._hf_model_evidence(str(tmp_path))

    assert set(evidence["files"]) == set(names)
    assert all("sha256" in item for item in evidence["files"].values())


def test_hf_evaluator_uses_current_prompt_prefilter_and_direct_answer_profile() -> None:
    output = _help(HF_EVALUATOR)

    assert "--contract {pocketfinancer,android-prompt-proxy,legacy}" in output
    assert "PocketFinancer's exact prompt, prefilter, and direct-" in output
    assert "Final runtime parity requires GGUF/device testing" in output
    assert "--apply-prefilter" in output
    assert "--model-config" in output
    assert "--thinking-mode {auto,on,off}" in output
    assert "--thinking-tokens" in output
    source = HF_EVALUATOR.read_text(encoding="utf-8")
    assert 'default="pocketfinancer"' in source
    assert '"android_runtime_parity": False' in source
    assert '"engine": "transformers_prompt_training_proxy"' in source
    assert "LlamaGrammar" not in source
    assert "raw_reasoning" in source
    assert "StopOnTokenSequence" in source


def test_gguf_evaluator_exposes_model_aware_generation_and_optional_grammar() -> None:
    output = _help(GGUF_EVALUATOR)

    assert "--contract {pocketfinancer,android-current,android,android-runtime,legacy}" in output
    assert "--model-config" in output
    assert "--thinking-mode {auto,on,off}" in output
    assert "--bos-mode {android-current,single}" in output
    assert "--thinking-tokens" in output
    assert "--with-grammar" in output
    assert "--no-grammar" in output
    assert "treats LFM2.5-350M as non-thinking" in output
    source = GGUF_EVALUATOR.read_text(encoding="utf-8")
    assert 'default="pocketfinancer"' in source
    assert 'default="android-current"' in source
    assert 'stop=["</think>"]' in source
    assert "grammar=grammar" in source
    assert '"answer_stop": None' in source
    assert "Android current always requires GBNF" not in source
    assert '"android_runtime_parity": False' in source
    assert '"raw_reasoning"' in source
    assert '"thinking_max_tokens_effective_min"' in source
    assert '"thinking_context_capped_rows"' in source
    assert '"thinking_finish_reason_counts"' in source
    assert '"thinking_latency_ms_p50"' in source
    assert '"template_think_appended_rows"' in source
    assert '"loader_generation_tags_normalized"' in source


def test_gguf_auto_thinking_mode_uses_model_config_and_safe_lfm_default() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)
    lfm = Path("LFM2.5-350M-Q4_0.gguf")

    assert evaluator._resolve_thinking_mode(
        "auto", model_config={"hasThinkingMode": False}, gguf=lfm
    ) == (False, "model_config")
    assert evaluator._resolve_thinking_mode(
        "auto", model_config={"model": {"has_thinking_mode": True}}, gguf=lfm
    ) == (True, "model_config")
    assert evaluator._resolve_thinking_mode("auto", model_config=None, gguf=lfm) == (
        False,
        "lfm2.5-350m_non_thinking",
    )
    assert evaluator._resolve_thinking_mode(
        "auto", model_config=None, gguf=Path("unknown-model.gguf")
    ) == (False, "auto_safe_default_off")

    with pytest.raises(ValueError, match="conflicting"):
        evaluator._resolve_thinking_mode(
            "auto",
            model_config={"hasThinkingMode": True, "has_thinking_mode": False},
            gguf=lfm,
        )


def test_gguf_normalizes_transformers_generation_blocks_for_local_jinja() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)
    template = "A{%- generation -%}B{% endgeneration %}C"

    compatible, count = evaluator._normalize_chat_template_generation_tags(
        template
    )

    assert count == 2
    assert compatible == "A{#- generation -#}B{# endgeneration #}C"
    assert "{% generation" not in compatible
    assert "{% endgeneration" not in compatible


def test_gguf_thinking_prompt_reuses_the_template_opening() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)
    rendered = "<|im_start|>assistant\n<think>"

    assert evaluator._prepare_thinking_prompt(rendered) == (rendered, False)
    assert evaluator._prepare_thinking_prompt(rendered + "\n") == (
        rendered + "\n",
        False,
    )
    appended, changed = evaluator._prepare_thinking_prompt(
        "<|im_start|>assistant\n"
    )
    assert changed is True
    assert appended.endswith("<think>\n")
    assert appended.count("<think>") == 1


def test_gguf_thinking_context_reserves_close_and_answer_before_capping() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)

    full = evaluator._thinking_context_budget(1780, 3072, 256, 2, 1024)
    assert full["effective_thinking_max_tokens"] == 1024
    assert full["thinking_capped_by_context"] is False

    capped = evaluator._thinking_context_budget(1900, 3072, 256, 2, 1024)
    assert capped["available_thinking_tokens"] == 914
    assert capped["effective_thinking_max_tokens"] == 914
    assert capped["thinking_capped_by_context"] is True

    with pytest.raises(ValueError, match="exceeds n_ctx=3072"):
        evaluator._thinking_context_budget(2815, 3072, 256, 2, 1024)
    with pytest.raises(ValueError, match="leaves no thinking capacity"):
        evaluator._thinking_context_budget(2814, 3072, 256, 2, 1024)


def test_gguf_context_preflight_mirrors_completion_framing_and_fails_closed() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)

    class FakeBackend:
        add_eos = False

        def token_cls(self) -> int:
            return -1

        def token_sep(self) -> int:
            return -1

        def add_bos_token(self) -> bool:
            return True

        def add_eos_token(self) -> bool:
            return self.add_eos

    class FakeModel:
        _model = FakeBackend()

        def tokenize(self, _text, *, add_bos: bool, special: bool) -> list[int]:
            assert special is True
            return [1, 10, 11, 12] if add_bos else [10, 11, 12]

        def token_bos(self) -> int:
            return 1

        def token_eos(self) -> int:
            return 2

    class OpaqueModel:
        def tokenize(self, _text, *, add_bos: bool, special: bool) -> list[int]:
            return [1] if add_bos else []

    with pytest.raises(RuntimeError, match="cannot preflight"):
        evaluator._completion_prompt_token_count(OpaqueModel(), "prompt")

    model = FakeModel()
    assert evaluator._completion_prompt_token_count(model, "prompt") == 4
    model._model.add_eos = True
    assert evaluator._completion_prompt_token_count(model, "prompt") == 5

    evaluator._require_completion_capacity(
        2816, 3072, 256, phase="direct"
    )
    with pytest.raises(ValueError, match="direct prompt .* exceeds n_ctx=3072"):
        evaluator._require_completion_capacity(
            2817, 3072, 256, phase="direct"
        )


def test_single_bos_ablation_removes_only_a_leading_template_bos() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)

    class FakeModel:
        def token_bos(self) -> int:
            return 1

        def detokenize(self, _tokens, special=True) -> bytes:
            assert special is True
            return b"<bos>"

    model = FakeModel()
    assert evaluator._remove_template_bos(model, "<bos>prompt") == ("prompt", True)
    assert evaluator._remove_template_bos(model, "prompt") == ("prompt", False)


@pytest.mark.parametrize("script", [HF_EVALUATOR, GGUF_EVALUATOR])
def test_sample_records_never_persist_sender_sms_or_content_prompt_hash(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "prompt_sha256" not in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "prediction" in keys:
            assert keys.isdisjoint({"sender", "sms"})


@pytest.mark.parametrize("script", [HF_EVALUATOR, GGUF_EVALUATOR])
def test_evaluator_reports_conditional_and_whole_pipeline_views(script: Path) -> None:
    source = script.read_text(encoding="utf-8")

    assert '"whole_pipeline"' in source
    assert '"conditional_model"' in source
    assert '"selection_prefilter"' in source
    assert '"rejected_prediction": "null"' in source
