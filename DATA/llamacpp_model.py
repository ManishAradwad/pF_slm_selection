"""
Out-of-tree lm-eval model adapter for `llama-cpp-python`.

Registers as `@register_model("llamacpp")`. Same engine as `llama.rn` (the
production on-device runtime), so dev-time metrics honestly predict on-device
behavior. Supports GBNF grammar-constrained decoding via `--model_args grammar_file=...`.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def _maybe_str(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in ("", "none", "null") else s


@register_model("llamacpp")
class LlamaCppLM(LM):
    """lm-eval adapter over `llama-cpp-python` with optional GBNF grammar."""

    def __init__(
        self,
        path: str,
        tokenizer: str,
        grammar_file: str | None = None,
        n_ctx: int = 131072,
        n_gpu_layers: int = -1,
        max_tokens: int = 512,
        verbose: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()

        # Defer heavy imports so `import llamacpp_model` stays fast for registration-only use.
        from llama_cpp import Llama, LlamaGrammar
        from transformers import AutoTokenizer

        path = str(path)
        tokenizer_id = str(tokenizer)
        grammar_file = _maybe_str(grammar_file)
        n_ctx = int(n_ctx)
        n_gpu_layers = int(n_gpu_layers)
        max_tokens = int(max_tokens)
        verbose = _to_bool(verbose)
        seed = int(seed)

        if not Path(path).exists():
            raise FileNotFoundError(f"GGUF not found: {path}")

        self.llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            seed=seed,
        )
        self.hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.grammar = (
            LlamaGrammar.from_file(grammar_file) if grammar_file else None
        )
        self._tokenizer_name = tokenizer_id
        self.max_tokens = max_tokens

    # ── Required abstracts ──────────────────────────────────────────────────

    def generate_until(self, requests, disable_tqdm: bool = False):
        outputs: list[str] = []
        pbar = tqdm(
            requests,
            total=len(requests),
            disable=disable_tqdm,
            desc="llamacpp",
        )
        for instance in pbar:
            context, gen_kwargs = instance.args
            if not isinstance(gen_kwargs, dict):
                gen_kwargs = {}

            until = gen_kwargs.get("until") or None
            temperature = float(gen_kwargs.get("temperature", 0.0) or 0.0)
            max_tokens = int(
                gen_kwargs.get("max_gen_toks", self.max_tokens)
                or self.max_tokens
            )

            # HF chat-template output starts with "<bos>" (recorded in samples log
            # for parity with the hf-backend baseline). llama-cpp tokenizes strings
            # with add_bos=True by default, which would duplicate it and hurt
            # quality. Strip here so the model sees exactly one BOS.
            prompt = context
            if prompt.startswith("<bos>"):
                prompt = prompt[len("<bos>"):]

            call_kwargs = dict(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=until,
                echo=False,
            )
            if self.grammar is not None:
                call_kwargs["grammar"] = self.grammar

            out = self.llm.create_completion(**call_kwargs)
            text = out["choices"][0]["text"]
            outputs.append(text)

        return outputs

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood not supported by the llamacpp adapter "
            "(only generate_until tasks are supported)."
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not supported by the llamacpp adapter."
        )

    # ── Chat template + tokenizer metadata ──────────────────────────────────

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        return self.hf_tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name
