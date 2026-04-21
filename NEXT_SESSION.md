# Next Session — Constrained Decoding via GBNF Grammar

## Why this doc exists
This is a handoff from the 2026-04-17 session. We started on *Priority 1* from the previous handoff (replace two hardcoded sender lists in `SYSTEM_PROMPT` with pattern-based descriptions) and it **regressed hard** against Gemma-4-E2B-it. Diagnosing the regression surfaced a deeper issue: the SLM is doing two jobs in one forward pass (binary classification AND 5-field extraction), and the output contract (`JSON object OR literal "null"`) is fragile on a 2B model. Instead of iterating further on prompt wording, the plan now is to **fix the output contract structurally via grammar-constrained decoding in llama.cpp**, which is exactly what the production runtime (`llama.rn`) supports natively.

Previous plans for reference:
- `/home/vscode/.claude/plans/snappy-beaming-mountain.md` — prompt rewrite (2026-04-15)
- `/home/vscode/.claude/plans/buzzing-watching-ullman.md` — Priority 1 attempt (2026-04-17, regressed)

## Current state of the working tree (READ FIRST)

**Uncommitted changes in `DATA/utils.py`:** the `SYSTEM_PROMPT` hint line (line 28) was replaced with a 5-line pattern-based description during the 2026-04-17 attempt. Under the HF-transformers / no-grammar path, this caused a measurable Gemma regression:

- `full_match_accuracy`: 0.704 → 0.458 (−0.246)
- `rejection_accuracy`: 0.793 → 0.596 (−0.197)
- `ghost_transaction_rate`: 0.187 → 0.404 (**+0.217 — worse**)

**Decision: keep the current (more general) prompt, don't revert.** Reasoning: of the 82 ghost cases in the new run, **23 are "all-null dict" + 48 are "partial-null dict" = 71 are shape-compliance failures**, not classification failures. The model knew it was a non-transaction; it just emitted `{"amount": null, ...}` instead of the literal `null`. Grammar-constrained decoding makes that output shape literally unreachable, so those 71 cases should recover for free. Only 11 were genuine hallucinations.

The bet: keep the generalized prompt + add grammar → recover or beat the 0.704 baseline. If the grammar run still underperforms the 0.704 number meaningfully, *then* revert the prompt and isolate whether grammar alone fixes it.

So there are now **two HF baselines to compare against** in the grammar run:
- `results_2026-04-15T17-36-26.264874.json` (old prompt, HF, no grammar) — `full_match_accuracy` 0.704
- `results_2026-04-17T08-41-23.435467.json` (new prompt, HF, no grammar) — `full_match_accuracy` 0.458

Expected under grammar: the new-prompt run should move a lot closer to 0.704. If it does, the generalization win stands.

Other state:
- `lm-evaluation-harness/` cloned (upstream; do not modify).
- Qwen3.5-0.8B, Qwen3-0.6B, LFM2.5-1.2B-Instruct have never been re-run against the current prompt and definitely haven't been run under grammar.
- Nothing committed.

## The core insight driving this session

### Why the dual-role output contract is fragile

The eval has 89 null-gold samples out of 203 (44% null). With the pattern-based hint, Gemma's failure mode was not "can't reject promos" — it was **can't commit to a rejection output shape**:
- 23 cases: model emits `{"amount": null, "merchant": null, "date": null, "type": null, "account": null}` instead of the literal `null`. Classification reasoning is right; output shape is wrong; metric counts it as a ghost.
- 48 cases: model emits partial JSON (e.g. `amount: 75, type: "debit"`, rest null). Classification is wavering; model hedges with half-extracted shape.
- 11 cases: genuine hallucinations (some are few-shot value leakage — `merchant: "DEMO SHOP DAILY"`, `account: "XX0000"` copied verbatim from sentinel examples).

A prompt rewrite can only influence tokens through the attention mechanism. A **grammar rewrite** makes invalid output shapes *literally unreachable* at generation time.

### Production path supports this for free

`pocket-financer` deploys the model via **`llama.rn`**, which is a React Native wrapper around **`llama.cpp`**. GBNF grammar-constrained decoding is a first-class llama.cpp feature — you pass a `grammar` string alongside the prompt and the engine masks logits each step so only tokens that keep the output valid against the grammar can be sampled.

Critical consequence: the same GBNF grammar used on-device in `llama.rn` can be used at dev-eval time via `llama-cpp-python` (Python bindings of the same engine). **Dev metrics will then actually reflect production behaviour** — which is not true today (HF transformers path has different generation semantics from the llama.cpp path the app uses).

### Methodological note — eval distribution

The 203-sample eval has 44% null-gold, but the production flow runs `new_pipeline.py` as a pre-filter (11,659 → 1,032 survivors, mostly real transactions). The SLM in the app probably sees a ~5–15% residual null rate, not 44%. Today's ghost numbers overstate real-world hurt. Worth keeping in mind when interpreting any future metric deltas.

## Priorities for this session

### Priority 1 — Stand up a llama-cpp-python eval path alongside the HF one
Goal: ability to run the 203-sample eval through `llama-cpp-python` (same engine as `llama.rn`) so we can pass a grammar.

Concrete sub-steps:
1. `pip install llama-cpp-python` into the `pf_docker` venv. (Target CUDA build: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade`.)
2. Pull GGUF-quantized versions of the target models from HF (Q4_K_M or Q5_K_M — quantization matches what `llama.rn` ships on-device, so metrics are more honest). Candidate repos:
   - `google/gemma-4-E2B-it` → check for a GGUF mirror or re-quantize via `llama.cpp/convert_hf_to_gguf.py`
   - Same for Qwen3.5-0.8B, Qwen3-0.6B, LFM2.5-1.2B-Instruct
3. Write an lm-eval `--model` adapter (or a standalone Python eval harness that consumes `DATA/extraction_ds.jsonl` + `DATA/utils.py`'s `doc_to_text` / metrics). lm-evaluation-harness does have a `gguf` model type — worth trying first before rolling our own.
4. Sanity-check: run without a grammar first, confirm metrics are within noise of the HF baseline for Gemma. That rules out "HF vs llama.cpp path differences" as a confound.

### Priority 2 — Write the GBNF grammar
Draft (iterate in session):
```gbnf
root        ::= "null" | transaction
transaction ::= "{" ws
                "\"amount\":"   ws amount       "," ws
                "\"merchant\":" ws maybe-string "," ws
                "\"date\":"     ws maybe-string "," ws
                "\"type\":"     ws type-enum    "," ws
                "\"account\":"  ws maybe-string ws
                "}"
amount       ::= [0-9]+ ("." [0-9]+)?
type-enum    ::= "\"debit\"" | "\"credit\""
maybe-string ::= "null" | string
string       ::= "\"" char* "\""
char         ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F]{4}
ws           ::= [ \t\n]*
```

Design notes to sort out:
- Should `amount` also allow `null`? Currently the schema assumes amount is always populated when not rejecting. Check the eval dataset — are there any non-null cases with missing amount?
- Should we pin `type` to the enum even when the field is absent? Grammar says always present; the dataset might have cases where gold has no `type`. Verify.
- Field ordering in the grammar is strict. The model must emit in this order. Current few-shot examples use a specific order — align the grammar with it.
- Whitespace: allow any amount of whitespace between tokens so the model can emit pretty-printed or compact JSON.

### Priority 3 — Simplify the prompt
With the grammar enforcing output shape, a lot of `SYSTEM_PROMPT`'s defensive prose becomes redundant:
- Drop: "Output either a single JSON object or the literal word null. Nothing else — no prose, no markdown fences, no explanations." (grammar enforces it)
- Drop: the STEP 2 per-field format notes that exist purely to get the model to emit the right shape (e.g. parsing Rs.1,500 as 1500.0 — grammar forces numeric, but the normalization still matters; keep that)
- Keep: everything about *what* to extract (STEP 1 rejection list, verb→type map, merchant phrase patterns, date normalization rules, account label rules).

This is a second-order win: fewer tokens → faster inference on-device + less for the model to get wrong.

### Priority 4 — Re-run all four target models under grammar
Only after Priorities 1–3 are stable. Models to benchmark:
- `google/gemma-4-E2B-it` (our current leader, `full_match_accuracy` 0.704 HF-baseline)
- `Qwen/Qwen3-0.6B` (append `/no_think` to the prompt to skip CoT — see memory)
- `Qwen/Qwen3.5-0.8B`
- `LiquidAI/LFM2.5-1.2B-Instruct`

For each, log: HF-transformers-no-grammar result, llama.cpp-no-grammar result, llama.cpp-with-grammar result. This gives us a clean ablation: path effect vs grammar effect vs model effect.

### Priority 5 — Re-open the generalization question
The original concern that started all this (prompt overfits to one inbox) is still live. The *current* hint line is already a first pass at generalizing (pattern-based, not inbox-specific) — if it holds up under grammar, great. If not, further iteration is safer once format variance is off the board. No additional prompt-generalization work happens before grammar lands.

## Deferred (from the prior handoff — still on the roadmap, but not this session)

- **Held-out test split** (prior Priority 2). Still worth doing before any prompt iteration. Once grammar is in, tune/test split becomes useful for all the *semantic* prompt work we deferred.
- **External data from friends/family with different banks** (prior Priority 3).
- **Public Indian SMS corpora search** (prior Priority 4).
- **Synthetic stress-test SMS for untested types** — EMI auto-debit, SI debit, BBPS, international spend, ATM-with-location, NACH, reversals (prior Priority 5).

## Key file paths

- `DATA/utils.py` — lines 13-50 `SYSTEM_PROMPT`, lines 52-85 `FEW_SHOT_EXAMPLES`, lines 513-591 sanity tests, filters + metrics in between. **Needs revert before anything else**.
- `DATA/sms_extraction.yaml` — lm-eval task config; may need a sibling `.yaml` once we add the llama-cpp path (e.g. `sms_extraction_gguf.yaml`)
- `DATA/extraction_ds.jsonl` — 203 labeled samples (unchanged)
- `new_pipeline.py` — rule-based pre-filter; produces 1,032 transaction-shaped SMS from 11,659 commercial messages. Relevant because it shapes the production input distribution the SLM actually sees.
- `RESULTS/new_pipeline/google__gemma-4-E2B-it/results_2026-04-15T17-36-26.264874.json` — **the baseline to beat**. Every grammar run should compare against this.
- `RESULTS/new_pipeline/google__gemma-4-E2B-it/results_2026-04-17T08-41-23.435467.json` — the regressed run (Priority-1 attempt). Keep it; useful as a negative reference for failure-mode analysis.
- `CLAUDE.md` — project overview + conventions

## Useful context for the new session

- `llama.rn` = React Native binding for `llama.cpp`. Inherits all llama.cpp features. Exposes `grammar` param in its completion API.
- `llama-cpp-python` is the Python binding over the same engine. Also exposes `grammar`. Use this for dev-time eval.
- **GBNF spec**: `lm-evaluation-harness/` is NOT the right reference — look at `github.com/ggerganov/llama.cpp/blob/master/grammars/README.md` for GBNF syntax. Reference grammars (like `json.gbnf`) live in `llama.cpp/grammars/`.
- Structured-output libraries worth knowing exist but aren't relevant here because llama.cpp has native support: Outlines, XGrammar, LM Format Enforcer, Guidance. If `llama-cpp-python` grammar support ever proves insufficient we can fall back to these.

## Conventions to keep in mind

- Dev container venv: `source pf_docker/bin/activate`
- HF lm-eval command template (current):
  ```
  lm_eval --model hf --model_args pretrained=<MODEL_ID> --tasks sms_extraction --include_path DATA/ --device cuda:0 --log_samples --output_path ./RESULTS/new_pipeline --apply_chat_template
  ```
- llama.cpp lm-eval command template (to establish this session, draft):
  ```
  lm_eval --model gguf --model_args model=<PATH_TO_GGUF>,grammar_file=<PATH_TO_GBNF> --tasks sms_extraction --include_path DATA/ --log_samples --output_path ./RESULTS/llamacpp
  ```
  (verify arg names against lm-eval's actual gguf adapter)
- Do not modify anything under `lm-evaluation-harness/`
- Never commit data files (csv / json / db / xlsx / jsonl in DATA)
- When running Qwen3-0.6B, append `/no_think` to skip CoT

## Recommended first move in the new session

1. **Confirm sanity baseline.** `python DATA/utils.py` → 41/41 PASS. (The uncommitted prompt change doesn't affect sanity tests; this is a 30-second check, not a revert.)
2. **Install `llama-cpp-python` (CUDA build) and get a single GGUF** (Gemma-4-E2B-it Q4_K_M). Verify it loads and generates.
3. **No-grammar smoke test**: run ~10 samples through llama-cpp-python without any grammar, confirm outputs look sane and are in the same ballpark as the HF run on the same inputs. This de-risks the new eval path before we layer grammar on top.
4. Only then draft the grammar and wire it in.
5. Compare the grammar run against both HF baselines (0.704 old-prompt and 0.458 current-prompt). If grammar + current prompt is meaningfully below 0.704, revert the prompt and re-run to isolate where the deficit comes from.

Don't try to land all four models + grammar + simplified prompt in one session. The right beat for this session is: new eval path standing → Gemma under grammar → decision on whether the generalized prompt survives. Everything else is a follow-up.
