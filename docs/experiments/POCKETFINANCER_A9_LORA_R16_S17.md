# PocketFinancer Android-aligned LFM2.5-350M run

Run ID: `lfm2.5-350m-direct-r16-s17`

Date: 2026-08-05

Android source: `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`

## 2.6B follow-up

The executed
[LFM2.5-2.6B Base LoRA follow-up](POCKETFINANCER_LFM25_2_6B_R16_S17.md)
is dated 2026-08-05 and completes the prospective 2.6B step discussed below. Its
BF16 rank-16 probe passed at 7,351.9 MiB peak allocated VRAM, so QLoRA was not
used. On the same 154-train / 29-dev silver materialization, the adapter did not
reliably improve the untouched Base on the reused 203-row regression fixture.
Human-gold, Android-device, and deployment gates remain open; teacher labels
still require source grounding and human review. The 350M results in this report
remain the historical record of this run.

## Conclusion

The end-to-end training, merge, GGUF conversion, and Android-profile evaluation
pipeline works on the WSL2 gaming PC and used its NVIDIA GeForce RTX 4070 for
training. The LoRA materially improved LFM2.5-350M over the untouched base, but
the resulting direct-extraction model is not accurate enough to ship.

The most deployable artifact from this run is the 229,311,680-byte Q4_K_M GGUF.
Under the current host emulation of the Android path it achieved 120/203 whole-
pipeline exact matches and 31/114 exact transaction extractions. A single-BOS
diagnostic reached 125/203 and 36/114, but its paired improvement was not
statistically decisive. This is a lead for an Android tokenization A/B test, not
a model promotion.

## What was aligned to the app

The run pinned and hash-checked the PocketFinancer Android source, including its
six-stage SMS prefilter, exact prompt assets and seven examples, generic outer
system message, built-in GGUF chat template, 3,072-token context, direct 256-token
answer path for LFM2.5-350M, optional/default-off grammar, and Kotlin-compatible
output interpretation. The host GGUF evaluator still uses `llama-cpp-python`, not
the app's custom JNI build, so only a device run can establish runtime parity.

The six-stage filter rejected 88/203 regression rows before inference and passed
all 114 labeled transactions plus one labeled null. Consequently the model was
invoked 115 times. Whole-pipeline metrics count those 88 deterministic nulls;
transaction-only exact match is the more revealing model-quality number. The zero
ghost count is weak model evidence because only one negative reached generation;
0/1 has a 79.35% Wilson upper bound, so a hard-negative set that passes the filter
is still required.

## Data build

The builder examined 14,075 raw rows but made only the original train partition
eligible for derived training data. The original 1,379-row dev and 1,436-row test
partitions remained sealed and zero of their rows were materialized.

| Split | Rows | Transactions | Nulls | Role |
|---|---:|---:|---:|---|
| Train | 154 | 139 | 15 | Updates LoRA weights |
| Dev | 29 | 22 | 7 | Silver tuning/checkpoint selection only |

Of the 183 materialized rows, 75 used consensus-silver labels and 108 used
source-grounded silver labels. The derived train/dev split has no sender or
template overlap. This is a meaningful improvement in leakage control, but it is
not human-gold data. The 29-row dev set is too small for a reliable ghost-rate or
generalization claim.

The previously discussed 251 train and 54 dev rows belong to an invalidated older
materialization. See the experiment catalog for their exact lineage and the 122
counterparty-label contamination finding. They were not reused here.

## Training

This was completion-only supervised fine-tuning with LoRA rank 16, alpha 32, and
dropout 0.05. It trained 5,996,544 parameters out of 360,480,512, or 1.6635%.
Prompts were masked from loss; each completion was reduced to a per-example mean;
provenance sample weights were applied; and the first `null`-versus-JSON decision
token received weight 3.0.

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 |
| Peak training VRAM | 3,886.7 MiB |
| Runtime | 358.3 seconds |
| Effective batch | 32 (microbatch 2, accumulation 16) |
| Learning rate | 0.0001 |
| Requested epochs | 4 |
| Best checkpoint | Epoch 2 / step 10 |
| Dev loss by epoch | 0.6752, **0.1800**, 0.8466, 0.8849 |

The rising dev loss after epoch 2 is a clear small-data overfitting signal. The
trainer restored the epoch-2 adapter before export.

## App-interpreted regression results

The 203 rows below are a repeatedly consulted compatibility/regression set: 114
transactions and 89 nulls. They are not a fresh sealed test. Scores use the
PocketFinancer parser as the primary interpretation.

| Variant | Overall exact | Transaction exact | TP / miss | Ghosts | Valid after app parser |
|---|---:|---:|---:|---:|---:|
| Untuned base, HF proxy | 89/203 (43.84%) | 0/114 (0.00%) | 0 / 114 | 0 | 203/203 |
| LoRA adapter, HF proxy | 121/203 (59.61%) | 32/114 (28.07%) | 72 / 42 | 0 | 203/203 |
| Merged BF16 GGUF | 122/203 (60.10%) | 33/114 (28.95%) | 88 / 26 | 0 | 203/203 |
| Q8_0 GGUF | 120/203 (59.11%) | 31/114 (27.19%) | 88 / 26 | 0 | 203/203 |
| Q4_K_M, current Android tokenization | 120/203 (59.11%) | 31/114 (27.19%) | 85 / 29 | 0 | 203/203 |
| Q4_K_M, single-BOS diagnostic | 125/203 (61.58%) | 36/114 (31.58%) | 91 / 23 | 0 | 203/203 |
| Q4_K_M, grammar enabled | 120/203 (59.11%) | 31/114 (27.19%) | 85 / 29 | 0 | 203/203 |

Under strict raw-output parsing, the HF adapter had 202/203 valid rows, BF16 and
Q8 each had 201/203, and both Q4 variants had 203/203. The app parser safely
failed malformed outputs closed to `null`; therefore 100% app-interpreted validity
does not mean every raw generation was valid. On the 115 invoked rows, strict
contract validity was 0 for base, 114 for HF adapter, 113 for BF16, 113 for Q8,
and 115 for both Q4 tokenization modes. Saved JSON-validity and schema-validity
are two labels for the same contract-parser decision, not independent tests. The
untuned base's invoked outputs were not usable extractions; its 89 app-interpreted
exact matches are null classifications, mostly supplied by the prefilter.

The paired strict-output comparison from the HF base to the adapter had 33 wins
and zero regressions. The overall app-interpreted gain is 32 rows because the app
parser already converted one malformed base response into the correct null.

## Field-level detail

Accuracy below uses all 114 gold transaction rows, so a missed transaction makes
all four fields wrong.

| Variant | Amount | Account | Type | Counterparty |
|---|---:|---:|---:|---:|
| HF adapter | 62.28% | 63.16% | 56.14% | 30.70% |
| BF16 GGUF | 74.56% | 76.32% | 62.28% | 34.21% |
| Q8_0 GGUF | 74.56% | 76.32% | 64.91% | 30.70% |
| Q4_K_M current | 71.93% | 65.79% | 50.88% | 41.23% |
| Q4_K_M single BOS | 78.07% | 72.81% | 64.04% | 36.84% |

Counterparty extraction remains the clearest field bottleneck, while transaction
misses remain the largest whole-example failure class. Quantization changes which
fields fail rather than causing a simple uniform degradation.

Among rows where the model detected a transaction but missed exact extraction, the
HF adapter's 40 errors were dominated by counterparty-only (32) and
counterparty-plus-type (5). BF16's 55 were dominated by counterparty-only (35) and
counterparty-plus-type (13). Current Q4 had 54 such rows: counterparty-only 18,
type-only 12, counterparty-plus-type 11, counterparty-plus-account 6, and smaller
combinations 7. Single-BOS had 55: counterparty-only 30,
counterparty-plus-type 12, counterparty-plus-account 5, type-only 3, and other
combinations 5. This makes counterparty ambiguity the first data target and shows
that Q4's type/account regressions also need explicit coverage.

## Quantization and host runtime

| GGUF | File bytes | Host p50 | Host p95 | Peak process RSS |
|---|---:|---:|---:|---:|
| BF16 | 711,484,608 | 1,315 ms | 1,557 ms | 1,307.9 MiB |
| Q8_0 | 379,217,088 | 1,337 ms | 1,537 ms | 767.6 MiB |
| Q4_K_M current | 229,311,680 | 1,117 ms | 1,253 ms | 687.7 MiB |
| Q4_K_M single BOS | 229,311,680 | 1,143 ms | 1,318 ms | 686.9 MiB |

These are desktop host CPU measurements through `llama-cpp-python`, with four
threads and 115 model invocations. They are not Android phone latency, RAM,
thermal, or battery measurements. Q8 produced no exact-match gain over Q4 in this
run. Relative to BF16, Q8 had four BF16-only correct rows and two Q8-only rows
(paired p=0.6875); Q4 had nine BF16-only and seven Q4-only rows (p=0.8036).
The HF timing is batched GPU-amortized timing and is not comparable to this
sequential CPU timing. Adapted outputs were also much shorter than the unusable
base outputs: mean generated lengths were about 21 tokens for HF adapter and
24-26 for GGUF, versus about 43 for the base.

## BOS and grammar diagnostics

The GGUF chat template emits a BOS token, while the current fresh-completion path
also asks llama.cpp tokenization to add special tokens. Source inspection and the
host duplicate-BOS warning indicate that the current app path likely begins with
two BOS tokens. The default evaluator preserved that behavior. The single-BOS
ablation removed the template BOS and let the completion layer add one.

Single BOS had ten uniquely correct rows versus five uniquely correct rows for the
current path, with 188 ties and paired p=0.3018. The direction is encouraging but
not conclusive on 203 reused rows. Before changing Android, instrument the exact
input token IDs or run a controlled phone A/B.

Enabling the existing GBNF grammar changed zero prediction strings and zero
semantic predictions: all 203 pairs tied and paired p=1.0. Since Q4 was already
100% strict-valid, grammar added no quality benefit in this run.

## Local artifacts and reproducibility

Private data, adapters, per-row outputs, and GGUF files remain local and ignored.
The important hashes are:

- Train data: `66a5ffe1f0722a594838f6ee405cefa1f3adb8290dec92db222021c6a6e12f56`
- Dev data: `126b9337e49ecbbbc437dbf68b113f0a51d00ad28fe5bb1b6590427f742f9513`
- Adapter: `920925bedaf523967ae330d22be46c0f46e736fbaafdf2e23f505da705250bb1`
- Q4_K_M: `fb470e9ad5fb6330748e0d862cc949ab5124a1d5192a9e4120949511f64dca97`
- Q8_0: `0167dd65edadce928bc3e230b73dd1abcc547e97d561483cdb42e8fd7ea62c6d`
- BF16: `30d2d5541004cfeb02c4b1980d2d0745055769cce508eca26e9b8f6fa6022b47`

## Decision and next experiment ladder

1. Do not promote this adapter yet; 31-36/114 exact transactions is far below an
   app-quality target.
2. Build a fresh, human-reviewed, template-held-out gold test before making more
   model decisions.
3. Expand clean, source-grounded training coverage and run learning curves across
   multiple seeds. Focus first on missed transactions, counterparty ambiguity,
   debit/credit phrasing, and hard nulls that pass the deterministic filter.
4. Verify single versus duplicate BOS on the real Android JNI path.
5. Use LFM2.5-2.6B as a like-for-like quality ceiling and reviewed teacher; train
   the 350M student on direct `null`/JSON answers, not hidden reasoning.
6. If direct 350M plateaus, revisit deterministic candidate extraction plus a
   compact learned selector, then task-specific architectures only after the
   direct and candidate learning curves establish the bottleneck.

This run proves the new pipeline is operational and reproducible. It does not yet
prove that a 350M model can reach near-perfect production performance; the next
clean-data learning curve is the experiment that can answer whether scale or task
formulation is the limiting factor.
