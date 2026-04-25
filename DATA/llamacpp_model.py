"""
Out-of-tree lm-eval model adapter for `llama-cpp-python`.

Registers as `@register_model("llamacpp")`. Same engine as `llama.rn` (the
production on-device runtime), so dev-time metrics honestly predict on-device
behavior. Supports GBNF grammar-constrained decoding via `--model_args grammar_file=...`.
"""

from __future__ import annotations

import struct
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


# GGUFValueType enum (from gguf spec)
_GGUF_TYPE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1}  # uint8..bool
_GGUF_STRING = 8
_GGUF_ARRAY  = 9


def _read_gguf_n_ctx_train(path: str) -> int | None:
    """
    Parse GGUF metadata to find <arch>.context_length without loading the model.
    Returns None if the key is absent or the file can't be parsed.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version, = struct.unpack("<I", f.read(4))
            if version not in (2, 3):
                return None
            _n_tensors, = struct.unpack("<Q", f.read(8))
            n_kv,       = struct.unpack("<Q", f.read(8))

            def read_str() -> str:
                length, = struct.unpack("<Q", f.read(8))
                return f.read(length).decode("utf-8", errors="replace")

            def skip_value(vtype: int) -> None:
                if vtype in _GGUF_TYPE_SIZE:
                    f.read(_GGUF_TYPE_SIZE[vtype])
                elif vtype == _GGUF_STRING:
                    length, = struct.unpack("<Q", f.read(8))
                    f.read(length)
                elif vtype == _GGUF_ARRAY:
                    elem_type, = struct.unpack("<I", f.read(4))
                    count,     = struct.unpack("<Q", f.read(8))
                    for _ in range(count):
                        skip_value(elem_type)

            arch = None
            context_length = None

            for _ in range(n_kv):
                key = read_str()
                vtype, = struct.unpack("<I", f.read(4))
                if key == "general.architecture":
                    arch = read_str()
                elif key.endswith(".context_length") and vtype == 4:  # uint32
                    context_length, = struct.unpack("<I", f.read(4))
                else:
                    skip_value(vtype)
                if arch and context_length is not None:
                    return context_length

        return context_length
    except Exception:
        return None


@register_model("llamacpp")
class LlamaCppLM(LM):
    """lm-eval adapter over `llama-cpp-python` with optional GBNF grammar."""

    def __init__(
        self,
        path: str,
        tokenizer: str,
        grammar_file: str | None = None,
        n_ctx: int = 0,
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

        # n_ctx=0 → use the model's trained context from GGUF metadata, capped at the
        # practical VRAM budget for an RTX 4070 (12 GB). llama.cpp pre-allocates the
        # full KV cache up-front, so a 128k or 262k native context would OOM even
        # though SMS prompts only use ~1-2k tokens. 32k fits every model in the slate
        # with headroom; override with --n-ctx N if you need more.
        _VRAM_CTX_CAP = 32768
        if n_ctx == 0:
            native = _read_gguf_n_ctx_train(path)
            if native:
                n_ctx = min(native, _VRAM_CTX_CAP)
                if n_ctx < native:
                    print(f"[llamacpp] n_ctx: native={native}, capped to {n_ctx} (VRAM budget)")
                else:
                    print(f"[llamacpp] n_ctx: native={native}")
            else:
                n_ctx = _VRAM_CTX_CAP
                print(f"[llamacpp] n_ctx: metadata not found, using {n_ctx}")

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

            # HF chat-template prepends the model's BOS token as text. llama-cpp
            # adds its own BOS during tokenization, so strip the text copy here
            # to avoid a duplicate. Use bos_token dynamically so this works for
            # any model (Gemma: "<bos>", LFM2.5: "<|startoftext|>", etc.).
            prompt = context
            bos = self.hf_tokenizer.bos_token
            if bos and prompt.startswith(bos):
                prompt = prompt[len(bos):]

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
