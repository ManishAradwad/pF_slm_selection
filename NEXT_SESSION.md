# 2026-04-25 — CUDA-backed llama.cpp container bootstrap

## Why this doc exists (this section)
The prior session (2026-04-21, archived below) implemented the out-of-tree `@register_model("llamacpp")` lm-eval adapter and shipped grammar-constrained Gemma-4-E2B-it evaluation. That work landed and is committed. The current limitation is throughput: `llama-cpp-python` was running CPU-only because the original `python:3.11-bullseye`-based dev container shipped glibc 2.31 — too old for prebuilt CUDA wheels (which need glibc ≥ 2.32 and GLIBCXX_3.4.29/30) and we couldn't easily build from source either (no CUDA toolkit in the image).

## Why this matters (purpose / background)
A full 203-sample eval takes ~10 min on CPU — tolerable for one model. But the project has four SLMs to benchmark (Gemma-4-E2B-it, Qwen3.5-0.8B, Qwen3-0.6B, LFM2.5-1.2B-Instruct), each in at least two configurations (no-grammar vs with-grammar), plus prompt-ablation runs ("does the generalized prompt survive under grammar?" — see "Prior-session context" below). On CPU that's hours of serial wall-clock per iteration cycle. With GPU offload via `n_gpu_layers=-1` we expect 5-10× speedup, bringing a full sweep to ~10-15 min — short enough to iterate on the prompt, the grammar, and few-shot examples without being gated by the eval loop.

On 2026-04-24 the dev container was rebuilt on `nvidia/cuda:12.1.1-devel-ubuntu22.04`. That base image gives us:
- glibc 2.35 (Ubuntu 22.04) — prebuilt CUDA wheels work out of the box.
- nvcc 12.1 + CUDA development headers in the image — `llama-cpp-python` can be built from source with `-DGGML_CUDA=on` directly during `docker build`, no host toolchain needed.
- Existing GPU passthrough (`runArgs: ["--gpus", "all"]`) continues to work — `nvidia-smi` sees the RTX 4070 (12 GB) inside the container.

But the rebuild left bootstrapping incomplete — the new image is missing pip deps that previously lived in the per-developer venv, and the on-disk `pf_docker/` venv is stale (points to a Python interpreter that no longer exists in the new image). This session captures everything in the Dockerfile + `devcontainer.json` so the next rebuild Just Works without manual fixup.

## Diagnosis (state at start of this session)
1. **`llama-cpp-python` not installed in the rebuilt container.** The Dockerfile sets `CMAKE_ARGS="-DGGML_CUDA=on"` and runs `pip install "lm_eval[gguf]"`, but the `[gguf]` extras only pull `requests` — upstream's gguf adapter is an HTTP client to `llama-server` (see "lm-eval's `gguf` adapter doesn't help" in the prior section). The CUDA build env is set up but nothing ever consumes it. `pip show llama-cpp-python` → not found.
2. **`transformers`, `torch`, `accelerate` missing.** `DATA/llamacpp_model.py:52` imports `AutoTokenizer` for chat-template rendering; nothing in the image satisfies the import.
3. **Stale `pf_docker/` venv.** Created in the prior bullseye container. `pyvenv.cfg` references `/usr/local/bin/python3.11`, which doesn't exist in the new Ubuntu 22.04 image (only `/usr/bin/python3.11`). `source pf_docker/bin/activate` silently no-ops — `python` keeps resolving to system Python and `sys.path` doesn't include the venv's `site-packages`. That's why `import llama_cpp` raised `ModuleNotFoundError` even after activation. The activation looked successful but wasn't.

GPU + CUDA toolkit themselves are healthy: `nvidia-smi` shows the 4070, `nvcc --version` reports 12.1.105, and the adapter at `DATA/llamacpp_model.py:69` already passes `n_gpu_layers=-1` (offload all layers). Everything downstream of "deps installed" is ready.

## Plan

### `requirements.txt` (new, version-pinned)
Single source of truth for pip deps. Referenced from the Dockerfile.
- `torch` (CUDA 12.1 wheel from PyTorch's index)
- `transformers`, `accelerate`, `huggingface_hub`
- `llama-cpp-python` (built from source with `-DGGML_CUDA=on`)
- lm-evaluation-harness pinned to a specific commit (currently `git+https://...` grabs HEAD — silent version drift between rebuilds is a real footgun)

### Dockerfile (`.devcontainer/Dockerfile`)
- Keep the `nvidia/cuda:12.1.1-devel-ubuntu22.04` base.
- Keep the Python 3.11 install via deadsnakes PPA.
- After `update-alternatives`, `pip install -r requirements.txt` — but `llama-cpp-python` line uses `--no-binary=llama-cpp-python` to force from-source build with `CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1`. Without `--no-binary`, pip can grab a prebuilt CPU wheel and silently miss CUDA — same trap that bit the prior container.

### `devcontainer.json`
- `containerEnv: {"HF_TOKEN": "${localEnv:HF_TOKEN}"}` — forward host HF token (set once in `~/.bashrc` on WSL).
- `HF_HOME=/workspaces/pF_slm_selection/.hf_cache` — persistent tokenizer/model cache survives rebuilds (saves redownload of gated Gemma tokenizer every rebuild).
- `postCreateCommand` extended: install Claude Code (existing) → create `pf_docker/` venv with `--system-site-packages` (thin shim, inherits heavy deps from system Python — no double-install on rebuild) → run `scripts/verify_gpu.py` so a broken build fails loudly at container creation rather than every eval silently falling back to CPU.

### Helper scripts (new, `scripts/`)
- `scripts/verify_gpu.py` — load the smallest available GGUF with `n_gpu_layers=-1, verbose=True`; assert `ggml_cuda_init: found 1 CUDA devices` appears in stderr. Exits non-zero if missing.
- `scripts/fetch_models.sh` — `huggingface-cli download` lines for each of the four SLMs into `MODELS/`. Reproducible model directory without baking 8+ GB of weights into the image.

### Cleanup
- Delete `pf/` (legacy WSL2-bare venv with hardcoded paths — never used inside the container).
- Delete the stale `pf_docker/` directory.
- Recreate `pf_docker/` via the new `postCreateCommand` path.

### CLAUDE.md updates
- Replace "**`llama-cpp-python` is CPU-only in the Docker container.**" paragraph with the new CUDA-enabled reality (and the trapdoor: if a future rebuild silently regresses to CPU, `--no-binary=llama-cpp-python` is the line to check).
- Add `HF_TOKEN` requirement note + where to set it.
- Note `scripts/` directory and what each script does.

## Concrete first moves (in order)

1. **Host-side prereq (user action):** ensure `HF_TOKEN` is exported in WSL `~/.bashrc`. Without it, gated Gemma tokenizer download fails — independent of GPU.
2. Write `requirements.txt`.
3. Rewrite `.devcontainer/Dockerfile` against the requirements file.
4. Update `.devcontainer/devcontainer.json` (containerEnv, HF_HOME, extended postCreateCommand).
5. Add `scripts/verify_gpu.py` and `scripts/fetch_models.sh`.
6. Update `.gitignore` for `.hf_cache/` and `scripts/__pycache__/`.
7. **In the running container** (no rebuild yet): pip-install the same deps into system Python, create the new `pf_docker/` venv with `--system-site-packages`, run `verify_gpu.py`. Confirm `ggml_cuda_init` shows in stderr and `nvidia-smi` reports VRAM allocated during a smoke load.
8. Delete old `pf/` and stale `pf_docker/` — **only after** step 7 verifies the replacement works.
9. Update `CLAUDE.md`.
10. **One container rebuild** (user-driven) to validate the chain reproduces from a clean image.
11. **Smoke eval, GPU, 10 samples**, confirm metrics within noise of the prior CPU-grammar run. If they diverge, kernel-level numerical differences are a known quirk of CUDA vs CPU llama.cpp; investigate before scaling up.

## Non-goals this session
- Running the full SLM eval sweep on GPU. That's the next session — this one ends when `verify_gpu.py` passes and a 10-sample smoke run completes with `nvidia-smi` showing GPU utilization.
- Multi-GPU support (we have one 4070; not worth the complexity).
- Quantization sweep beyond Q4_K_M. Q4_K_M is what `llama.rn` ships on-device — metric honesty beats ceiling metrics. See CLAUDE.md "Evaluation approach".
- Replacing the Dockerfile base with a slimmer image (`-runtime` instead of `-devel`). `-devel` is needed because we build `llama-cpp-python` from source; the alternative is a separate builder stage, premature complexity for a single-developer project.

---

# 2026-04-21 — Switch dev eval from `hf` to `llama.cpp`, add GBNF grammar (archived)

## Why this doc exists
Handoff from the 2026-04-21 session. We diagnosed the exact shape of the `llama.cpp` / lm-eval gap and settled on the architecture for fixing it (out-of-tree `@register_model("llamacpp")` adapter using `llama-cpp-python`). Nothing has been coded yet. This session implements that.

Prior plan docs, for lineage:
- `/home/vscode/.claude/plans/snappy-beaming-mountain.md` — prompt rewrite (2026-04-15)
- `/home/vscode/.claude/plans/buzzing-watching-ullman.md` — Priority 1 attempt (2026-04-17, regressed)

## The core problem

### 1. We're evaluating with the wrong backend
Every current SLM eval runs through lm-eval's `hf` backend (HuggingFace `transformers`). That is **not** the runtime the app will use. `pocket-financer` ships the SLM on Android via **`llama.rn`** — a React Native binding over **`llama.cpp`**. `transformers.generate(...)` and `llama.cpp`'s sampler differ in tokenization edge cases, quantization (HF = fp16/bf16 on GPU; `llama.rn` = Q4_K_M on-device), and sampling plumbing. Dev metrics obtained via `hf` do not honestly predict how the model behaves on a user's phone.

The fix is to move dev eval to **`llama-cpp-python`** — the Python binding over the *same* `llama.cpp` engine that `llama.rn` wraps. Anything that works in `llama-cpp-python` maps 1:1 to what the app will do on-device.

### 2. We need GBNF grammar-constrained decoding
From the 2026-04-17 Gemma-4-E2B-it regression analysis, **71 of the 82 ghost cases were shape-compliance failures**, not classification failures:
- 23 cases: model emitted `{"amount": null, "merchant": null, "date": null, "type": null, "account": null}` instead of the literal `null`.
- 48 cases: model emitted partial JSON with most fields null.
- Only 11: genuine hallucinations.

A prompt rewrite can only nudge token probabilities. A **grammar** makes invalid output shapes *literally unreachable* at generation time — the sampler masks out every token that would lead to a non-parseable output. `llama.cpp` supports this natively (GBNF via `grammar` arg); `llama.rn` exposes it on-device; so the same GBNF we validate at dev time is deployable without modification.

### 3. lm-eval's `gguf` adapter doesn't help
`lm-evaluation-harness/lm_eval/models/gguf.py:52-62`:
```python
request = {
    "prompt": prompt,
    "logprobs": self.logprobs,
    "temperature": self.temperature,
}
if continuation: ...
if stop is not None: request["stop"] = stop
response = requests.post(f"{self.base_url}/v1/completions", json=request)
```
- HTTP-only: POSTs to a separate `llama-server` process. Not in-process.
- Body carries `prompt`, `logprobs`, `temperature`, optional `stop`/`max_tokens`/`echo`. **No `grammar` field.**
- `llama-server` *does* accept `grammar` per-request, but lm-eval never sends one.
- Upstream: last commit on `main` is 2026-04-08 (checked 2026-04-21). No open work visible on this. Not a path we can wait for.

## The approach

**Write an out-of-tree lm-eval model adapter named `llamacpp`**, registered via the public `@register_model` decorator, that runs `llama-cpp-python` in-process and passes a GBNF grammar directly to `Llama.create_completion(...)`.

### Why this, not the alternatives
Three options were considered end-to-end:

| | (A) Fork `gguf.py` | (B) Standalone harness | (C) Plug-in `@register_model` adapter |
|---|---|---|---|
| Comparable to existing HF baselines | yes | drift risk | **yes** |
| In-process (no server) | no | yes | **yes** |
| Modifies upstream | yes | no | **no** |
| Reuses lm-eval pipeline | full | reimplements | **full** |

(C) is the only option where every run — HF-baseline, GGUF-no-grammar, GGUF-with-grammar, across every SLM we benchmark later — goes through the **identical** lm-eval pipeline: same `doc_to_text`, same `extract_json_filter`, same 11 metric functions, same output-JSON schema. The only thing that varies is which adapter lm-eval dispatches to. For cross-model selection that's the gold standard.

`CLAUDE.md` was updated 2026-04-21 to reflect this convention: prefer out-of-tree extensions via `--include_path` (tasks) and `@register_model` (models); in-tree edits to `lm-evaluation-harness/` are allowed but a conscious choice, not the default.

### Registry plumbing (verified against upstream source this session)
- `lm_eval/api/registry.py:465-488` — `register_model(*names)` is a plain decorator, no "must live under `lm_eval/models/`" constraint.
- `lm_eval/models/__init__.py:25-57` — pre-populates lazy placeholders for built-in models; any alias not in `MODEL_MAPPING` is fair game for out-of-tree registration.
- `lm_eval/api/model.py:25-128` — `LM` abstract class requires `loglikelihood`, `loglikelihood_rolling`, `generate_until`. Our `sms_extraction.yaml` has `output_type: generate_until`, so only `generate_until` needs a real implementation; the other two can raise `NotImplementedError`.
- `apply_chat_template` + `tokenizer_name` must also be implemented because we pass `--apply_chat_template`.

### Adapter shape (~100 lines)

```
DATA/llamacpp_model.py
  from lm_eval.api.model import LM
  from lm_eval.api.registry import register_model

  @register_model("llamacpp")
  class LlamaCppLM(LM):
      def __init__(self, path, tokenizer, grammar_file=None,
                   n_ctx=4096, n_gpu_layers=-1, max_tokens=512):
          # llama_cpp.Llama(model_path=path, n_gpu_layers=..., n_ctx=...)
          # AutoTokenizer.from_pretrained(tokenizer)  # HF tokenizer for chat template
          # LlamaGrammar.from_file(grammar_file) if provided else None

      def generate_until(self, requests, disable_tqdm=False):
          # for each Instance(args=(ctx, gen_kwargs)):
          #     self.llm.create_completion(
          #         prompt=ctx, grammar=self.grammar,
          #         temperature=0, max_tokens=self.max_tokens,
          #         stop=gen_kwargs["until"])
          # returns list[str]

      def apply_chat_template(self, chat_history, add_generation_prompt=True):
          return self.hf_tokenizer.apply_chat_template(
              chat_history, tokenize=False,
              add_generation_prompt=add_generation_prompt)

      @property
      def tokenizer_name(self):
          return self._tokenizer_name

      # loglikelihood / loglikelihood_rolling → NotImplementedError
```

```
DATA/run_gguf_eval.py          (~30 lines)
  import DATA.llamacpp_model        # side effect: @register_model("llamacpp")
  import lm_eval
  lm_eval.simple_evaluate(
      model="llamacpp",
      model_args="path=MODELS/gemma-4-E2B-it-Q4_K_M.gguf,"
                 "tokenizer=google/gemma-4-E2B-it,"
                 "grammar_file=DATA/sms_extraction.gbnf",
      tasks=["sms_extraction"],
      task_manager=lm_eval.tasks.TaskManager(include_path="DATA"),
      log_samples=True,
      apply_chat_template=True,
      output_path="./RESULTS/llamacpp",
  )
```

### Design choices already settled
- **Chat template via HF `AutoTokenizer.apply_chat_template`, not GGUF-embedded.** Exact parity with existing HF runs; zero divergence risk. Cost: one extra model_arg (`tokenizer=<hf_repo_id>`).
- **Quantization: Q4_K_M.** Matches what `llama.rn` actually ships on-device. Honest metric beats ceiling metric.
- **GGUF source: verify at run time.** bartowski or lmstudio-community mirror for Gemma-4-E2B-it; pick the most-downloaded reputable repo.
- **Field order pinned in grammar**: `amount, merchant, date, type, account`. Verified 100 % consistent across all 114 txn-gold rows and all 3 few-shot examples.
- **Nullability per field** (dataset-verified, reconciled with prompt wording):
  | field | nullable? | evidence |
  |---|---|---|
  | `amount` | no | 0/114 null in gold |
  | `merchant` | **yes** | 18/114 (15.8 %) null; prompt explicitly allows |
  | `date` | **yes** | 0/114 in gold, but prompt says "no date → null" and "no year → null" — match prompt, not dataset, or we force the model to invent dates |
  | `type` | no, strict enum | prompt pins verbs to `debit`/`credit` |
  | `account` | no | 0/114 null |
- **Output-JSON shape**: controlled by lm-eval, identical to existing HF runs. Only the output directory differs (`RESULTS/llamacpp/` vs `RESULTS/new_pipeline/`).

## Draft GBNF grammar

```gbnf
root        ::= "null" | transaction
transaction ::= "{" ws
                "\"amount\""   ws ":" ws amount        ws "," ws
                "\"merchant\"" ws ":" ws maybe-string  ws "," ws
                "\"date\""     ws ":" ws maybe-string  ws "," ws
                "\"type\""     ws ":" ws type-enum     ws "," ws
                "\"account\""  ws ":" ws string        ws
                "}"
amount       ::= [0-9]+ ("." [0-9]+)?
type-enum    ::= "\"debit\"" | "\"credit\""
maybe-string ::= "null" | string
string       ::= "\"" char* "\""
char         ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws           ::= [ \t\n\r]*
```

GBNF has no `{n}` quantifier — the unicode escape must be unrolled to 4 explicit hex chars. (An earlier draft of this doc had this wrong.)

## Prior-session context that's still load-bearing

- **Uncommitted prompt change in `DATA/utils.py`** (lines 28–31 region): the single sender-ID hint was replaced with a 5-line pattern-based description. Under HF-no-grammar this regressed Gemma (`full_match_accuracy` 0.704 → 0.458, `ghost_transaction_rate` 0.187 → 0.404). **Keep the change; do not revert.** The bet is that 71 of the 82 new ghosts are shape-compliance failures that grammar structurally eliminates. If the grammar run still comes in meaningfully below 0.704, *then* revert and isolate whether grammar alone closes the gap.
- **Two HF baselines to compare against**, both Gemma-4-E2B-it:
  - `RESULTS/new_pipeline/google__gemma-4-E2B-it/results_2026-04-15T17-36-26.264874.json` — old prompt, HF, `full_match_accuracy` **0.704**
  - `RESULTS/new_pipeline/google__gemma-4-E2B-it/results_2026-04-17T08-41-23.435467.json` — current prompt, HF, `full_match_accuracy` **0.458**
- **Eval distribution caveat**: 203 samples, 89 null-gold (44 %). Production's `new_pipeline.py` pre-filter (11,659 → 1,032 survivors) removes most non-transactions upstream of the SLM, so the app's real residual null rate is probably 5–15 %, not 44 %. Current ghost metrics overstate real-world hurt — keep in mind when interpreting deltas.

## Concrete first moves

1. **Verify `llama-cpp-python` CUDA build.** `llama_cpp` 0.3.19 is already installed in `pf_docker`, but likely CPU-only (default pip wheel). Load a small GGUF with `n_gpu_layers=-1` and watch `nvidia-smi`. If no VRAM usage → `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade`.
2. **Pull Gemma-4-E2B-it Q4_K_M GGUF** from HF (bartowski / lmstudio-community; pick at run time). Store under `MODELS/` (gitignored).
3. **Write `DATA/llamacpp_model.py`** with the `LlamaCppLM` adapter above.
4. **Write `DATA/run_gguf_eval.py`** wrapper (~30 lines).
5. **Write `DATA/sms_extraction.gbnf`** using the grammar drafted above.
6. **Smoke test, no grammar, 10 samples.** Confirm outputs are in the same ballpark as the HF current-prompt 0.458 baseline. If wildly off, chat-template or tokenization is the first suspect — fix before adding grammar.
7. **Full 203-sample runs on Gemma-4-E2B-it, in this order:**
   - (a) llama.cpp-no-grammar — isolates the "path effect" (HF → llama.cpp).
   - (b) llama.cpp-with-grammar — isolates the "grammar effect" on top of the path.
8. **Decide whether the generalized prompt survives under grammar.** Compare run (b) against both HF baselines. If (b) ≥ 0.704, the generalized prompt is vindicated and we ship it. If (b) < 0.704 meaningfully, revert the prompt, re-run, and isolate.

## Non-goals this session
- Running Qwen3.5-0.8B / Qwen3-0.6B / LFM2.5-1.2B-Instruct. They come after the Gemma flow is proven.
- Simplifying `SYSTEM_PROMPT` (removing defensive "output either JSON or null" prose now that grammar enforces it). Worth doing once grammar lands, but not this session.
- Held-out test split, external friends/family data, public Indian SMS corpora, synthetic stress-test SMS.
- Any code edits to `lm-evaluation-harness/`.

## Key file paths
- `DATA/utils.py` — `SYSTEM_PROMPT` (13–53), `FEW_SHOT_EXAMPLES` (55–88), `doc_to_text`, `extract_json_filter`, 11 metric functions, sanity tests. **No edits needed** — the adapter reuses this unchanged via lm-eval.
- `DATA/sms_extraction.yaml` — task config. **No edits needed.**
- `DATA/extraction_ds.jsonl` — 203 samples. Unchanged.
- `DATA/llamacpp_model.py` — **new**, the adapter.
- `DATA/run_gguf_eval.py` — **new**, entry script.
- `DATA/sms_extraction.gbnf` — **new**, the grammar.
- `MODELS/` — **new (gitignored)**, GGUF files.
- `RESULTS/new_pipeline/google__gemma-4-E2B-it/` — existing HF baselines to beat.
- `RESULTS/llamacpp/google__gemma-4-E2B-it/` — new output target.
- `CLAUDE.md` — convention updated 2026-04-21 for out-of-tree extensions.

## Useful reference
- `llama.rn` = React Native binding over `llama.cpp`. Exposes `grammar` in its completion API.
- `llama-cpp-python` = Python binding over `llama.cpp`. Same engine. `LlamaGrammar.from_file(...)` + `Llama.create_completion(..., grammar=...)`.
- GBNF spec: `github.com/ggerganov/llama.cpp/blob/master/grammars/README.md`. Reference grammars in `llama.cpp/grammars/` (e.g. `json.gbnf`).
- `@register_model` lives in `lm_eval.api.registry`. Apply to an `LM` subclass; import the module before calling `lm_eval.simple_evaluate` so the decorator runs.

## Conventions
- Dev container venv: `source pf_docker/bin/activate`
- HF lm-eval command template (unchanged, for future HF comparison runs):
  ```
  lm_eval --model hf --model_args pretrained=<MODEL_ID> --tasks sms_extraction --include_path DATA/ --device cuda:0 --log_samples --output_path ./RESULTS/new_pipeline --apply_chat_template
  ```
- llama.cpp path is invoked via `python DATA/run_gguf_eval.py` (not via `lm_eval` CLI — the out-of-tree adapter has to be imported before the model registry is queried).
- Do NOT commit data files (csv / json / db / xlsx / jsonl in DATA).
- Qwen3-0.6B: append `/no_think` to skip CoT (applies when we get to Qwen later).
