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
        thinking: str = "auto",
        thinking_max_tokens: int = 4096,
        thinking_repeat_penalty: float = 1.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # forward-compat absorber for unrecognised model_args keys

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
        thinking = str(thinking).strip().lower()
        thinking_max_tokens = int(thinking_max_tokens)
        thinking_repeat_penalty = float(thinking_repeat_penalty)

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

        # Thinking-mode detection. A model is considered a thinking model if its
        # chat template references `</think>` as a literal close tag — Qwen3 and
        # Qwen3.5 hit this; Gemma/arcee/LFM2.5 do not (LFM2.5 mentions thinking
        # only as a no-op past-message stripper). When on, we run two-phase
        # generation: phase 1 reasons freely up to `</think>`, phase 2 emits
        # grammar-constrained JSON. Without phase 1, the grammar suppresses any
        # reasoning tokens and Qwen3 collapses into copying the few-shot answers.
        chat_tmpl = (self.hf_tokenizer.chat_template or "")
        auto_think = ("</think>" in chat_tmpl) and self._template_lets_model_emit_think()
        if thinking == "auto":
            self.thinking = auto_think
        elif thinking in ("on", "true", "1", "yes", "y"):
            self.thinking = True
        else:
            self.thinking = False
        self.thinking_max_tokens = thinking_max_tokens
        self.thinking_repeat_penalty = thinking_repeat_penalty
        if self.thinking:
            print(f"[llamacpp] thinking: ON (budget={thinking_max_tokens} tokens, "
                  f"repeat_penalty={thinking_repeat_penalty}, "
                  f"phase 1 ungrammared, stop=</think>)")
        elif thinking == "auto" and "</think>" in chat_tmpl:
            print(f"[llamacpp] thinking: OFF (template mentions </think> but "
                  f"doesn't expect the model to emit one — likely a past-turn "
                  f"stripper rule, not real thinking)")

    # ── Required abstracts ──────────────────────────────────────────────────

    def _template_lets_model_emit_think(self) -> bool:
        """Detect a true thinking model.

        A model is thinking-aware iff its chat template actually responds to
        `enable_thinking` — i.e. `apply_chat_template(..., enable_thinking=True)`
        and `enable_thinking=False` produce *different* renderings. Qwen3 and
        Qwen3.5 satisfy this; LFM2.5 and Gemma do not (their templates mention
        `</think>` only as a no-op past-message stripper, with no thinking flag
        in the conditional).

        Additionally, the True-rendering must NOT pre-inject a closing
        `</think>` tag — otherwise the model is being told "thinking is over,
        just answer", which is the False semantics anyway.
        """
        msgs = [{"role": "user", "content": "X"}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            r_true = self.hf_tokenizer.apply_chat_template(
                msgs, enable_thinking=True, **kwargs
            )
            r_false = self.hf_tokenizer.apply_chat_template(
                msgs, enable_thinking=False, **kwargs
            )
        except TypeError:
            return False
        # Template ignored the flag → not a real thinking model.
        if r_true == r_false:
            return False
        # Even with the flag respected, if the True render still contains
        # `</think>`, the template is closing the block for us — model is
        # not expected to emit thinking.
        return "</think>" not in r_true

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

            if self.thinking:
                text = self._generate_with_thinking(
                    prompt=prompt,
                    until=until,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
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

    def _generate_with_thinking(
        self,
        prompt: str,
        until,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Two-phase generation for thinking models.

        Phase 1: no grammar, stop at `</think>`. The model produces a free-form
        reasoning trace. We always force the prompt to start with `<think>\\n`
        so we can rely on the close tag as a stop boundary even for templates
        that don't auto-open the block.

        Phase 2: prompt += phase1 + `</think>\\n`. Grammar applied. The model
        emits the JSON answer.

        We return the full string `<think>{trace}</think>\\n{json}` so the
        downstream filter (which strips `<think>...</think>`) sees the same
        shape regardless of phase boundary.
        """
        # Phase 1 — open the think block ourselves and let the model fill it.
        # Mild `repeat_penalty` breaks the verbatim-loop failure mode small
        # thinking models hit at greedy temperature (Qwen3.5-0.8B repeats
        # phrases until it eats the budget). 1.1 is the conservative default
        # — high enough to break loops, low enough not to perturb reasoning.
        phase1_prompt = prompt + "<think>\n"
        phase1_out = self.llm.create_completion(
            prompt=phase1_prompt,
            temperature=temperature,
            max_tokens=self.thinking_max_tokens,
            stop=["</think>"],
            echo=False,
            repeat_penalty=self.thinking_repeat_penalty,
        )
        thinking_text = phase1_out["choices"][0]["text"]

        # Phase 2 — close the block and constrain to JSON via grammar.
        phase2_prompt = (
            phase1_prompt + thinking_text + "</think>\n"
        )
        call_kwargs = dict(
            prompt=phase2_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=until,
            echo=False,
        )
        if self.grammar is not None:
            call_kwargs["grammar"] = self.grammar
        phase2_out = self.llm.create_completion(**call_kwargs)
        json_text = phase2_out["choices"][0]["text"]

        return f"<think>\n{thinking_text}</think>\n{json_text}"

    def loglikelihood(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "loglikelihood not supported by the llamacpp adapter "
            "(only generate_until tasks are supported)."
        )

    def loglikelihood_rolling(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "loglikelihood_rolling not supported by the llamacpp adapter."
        )

    # ── Chat template + tokenizer metadata ──────────────────────────────────

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # Forward `enable_thinking` only if the template understands it. We
        # explicitly pass False when self.thinking is False so Qwen3's default
        # (which is True) doesn't leak in. When self.thinking is True we let
        # the template render in thinking-on mode, then phase 1 of
        # `_generate_with_thinking` opens the `<think>` block on top of that.
        try:
            return self.hf_tokenizer.apply_chat_template(
                chat_history,
                enable_thinking=self.thinking,
                **kwargs,
            )
        except TypeError:
            return self.hf_tokenizer.apply_chat_template(chat_history, **kwargs)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name
