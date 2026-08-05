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


def test_hf_evaluator_uses_current_prompt_prefilter_and_direct_answer_profile() -> None:
    output = _help(HF_EVALUATOR)

    assert "--contract {pocketfinancer,android-prompt-proxy,legacy}" in output
    assert "PocketFinancer's exact prompt, prefilter, and direct-" in output
    assert "Final runtime parity requires GGUF/device testing" in output
    assert "--apply-prefilter" in output
    source = HF_EVALUATOR.read_text(encoding="utf-8")
    assert 'default="pocketfinancer"' in source
    assert '"android_runtime_parity": False' in source
    assert '"engine": "transformers_prompt_training_proxy"' in source
    assert "LlamaGrammar" not in source


def test_gguf_evaluator_exposes_model_aware_generation_and_optional_grammar() -> None:
    output = _help(GGUF_EVALUATOR)

    assert (
        "--contract {pocketfinancer,android-current,android,android-runtime,legacy}"
        in output
    )
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


def test_gguf_auto_thinking_mode_uses_model_config_and_safe_lfm_default() -> None:
    evaluator = _load_script(GGUF_EVALUATOR)
    lfm = Path("LFM2.5-350M-Q4_0.gguf")

    assert evaluator._resolve_thinking_mode(
        "auto", model_config={"hasThinkingMode": False}, gguf=lfm
    ) == (False, "model_config")
    assert evaluator._resolve_thinking_mode(
        "auto", model_config={"model": {"has_thinking_mode": True}}, gguf=lfm
    ) == (True, "model_config")
    assert evaluator._resolve_thinking_mode(
        "auto", model_config=None, gguf=lfm
    ) == (False, "lfm2.5-350m_non_thinking")
    assert evaluator._resolve_thinking_mode(
        "auto", model_config=None, gguf=Path("unknown-model.gguf")
    ) == (False, "auto_safe_default_off")

    with pytest.raises(ValueError, match="conflicting"):
        evaluator._resolve_thinking_mode(
            "auto",
            model_config={"hasThinkingMode": True, "has_thinking_mode": False},
            gguf=lfm,
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
