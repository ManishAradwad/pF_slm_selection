# PocketFinancer Android-aligned LFM2.5-2.6B evaluation and rank-16 run

Run ID: `lfm2.5-2.6b-base-direct-r16-s17`

Date: 2026-08-05

Android source: `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`

Status: executed local diagnostic; not approved for Android support or deployment

## Conclusion

The untouched Base checkpoint was the best completed HF variant on the locked
203-row regression fixture. It beat the post-trained checkpoint by 186 versus 174
whole-pipeline exact rows and 98 versus 86 exact transaction rows. The paired
result favored Base on 15 rows and Post on three (`p = 0.00753784`). This is useful
local evidence that mandatory general-purpose reasoning did not help this direct
extraction task. The official untouched Post Q4_K_M single-BOS diagnostic scored
176 rather than 174 exact rows; with eight HF-only wins, ten Q4-only wins, and
`p = 0.81452942`, it did not establish a reliable change from Post HF.

Rank-16 LoRA did not reliably improve Base. The adapter scored 184 rather than 186
whole-pipeline exact rows and 96 rather than 98 exact transactions. In the paired
comparison, Base alone was correct on five rows, LoRA alone on three, and 195 tied
(`p = 0.7265625`). The point estimate is slightly worse and the paired result does
not establish a difference. One small silver-data run and one seed are not enough
to conclude that Base is generally untunable.

The tuned Q4_K_M Android-current host run fell to 166/203 whole-pipeline and 78/114
transaction exact while emitting a duplicate-leading-BOS warning on every invoked
row. Removing the template BOS recovered seven exact rows, to 173/203 and 85/114;
the paired improvement was significant (`p = 0.0390625`). Single-BOS Q4 still
trailed tuned HF by 11 net exact rows (`p = 0.00738525`). Single-BOS BF16 and Q8
preserved tuned-HF app-interpreted exact correctness, while Q8 exactly reproduced
BF16 predictions, localizing the remaining loss to Q4 quantization under this host
engine rather than merge or the BF16/Q8 engine path.

None of these results is production evidence: the 203 rows are reused diagnostics,
the 1,436-row blinded package has no human-gold adjudication, and no custom-JNI
Android device run was performed.

## Evidence boundary

All scores in this report use `DATA/extraction_ds.jsonl`, SHA-256
`fec483b11cf458212b6a636f508632649790beacc91050efdd52abb2b590d44e`.
This is the grandfathered, repeatedly consulted 203-row regression fixture, with
114 labeled transactions and 89 labeled nulls. It is not fresh, sealed, or valid
for a production claim, and it was not used for training.

The Android six-stage prefilter rejected 88/203 rows, all labeled nulls. It passed
all 114 transactions and one null, so each model was invoked only 115 times.
Whole-pipeline exact scores therefore include 88 deterministic correct rejections.
Transaction-only exact match is the more informative model-quality measure. The
single generated null became a false positive for every completed variant, so a
100% conditional ghost rate means 1/1, not a well-measured false-positive rate.

A reviewer-blind package has been frozen locally for all 1,436 rows in the held-out
test partition. Its aggregate state is 1,436 pending: no row has completed human
adjudication, and the package must not be described as human gold. It contributed
no score or training row here.

## Why Post used two passes and Base used direct generation

The post-trained lock pins `LiquidAI/LFM2.5-2.6B` at revision
`dca1825886789bd40b94368f53b1d9ada4c94598`. Its model configuration declares
mandatory reasoning. The current PocketFinancer profile routes thinking models
through a reasoning pass of up to 1,024 tokens, stopped by `</think>`, followed by
a direct answer pass of up to 256 tokens. The HF proxy therefore used the required
sequential two-pass path rather than forcing a non-faithful direct-only ablation.

The Base lock pins `LiquidAI/LFM2.5-2.6B-Base` at revision
`78f33a52fbe65f7665963f482179dcc3e75f0d9e`. Base has no thinking mode, so both its
untuned and adapted evaluations used the app's single direct 256-token path. This
comparison holds the extraction contract, prefilter, greedy decoding, 3,072-token
context, and app parser fixed while respecting each model's declared generation
path; it does not isolate model weights from the cost or behavior of reasoning.

All 115 invoked Post prompts fit the 3,072-token context only by reserving the
answer and closing-tag budget: effective reasoning capacity ranged from 933 to
1,002 tokens, and all 115 rows were context-capped. Only 35/115 reasoning passes
found the thinking stop. Post averaged 886.76 thinking tokens, with host-GPU p50
thinking latency 24.576 seconds and p50 answer latency 1.438 seconds. Its total
p50/p95 were 26.541/30.130 seconds. Base was evaluated in batches of eight and had
426.675/519.791 ms batch-amortized p50/p95; those timing modes are not directly
comparable and neither is Android phone latency.

## App-interpreted and strict results

The app-interpreted columns apply the PocketFinancer Kotlin-compatible parser.
Strict columns require the raw model output itself to satisfy the contract.

| Variant | Generation | App exact | Strict exact | App/strict transaction exact | App/strict F1 | App/strict valid |
|---|---|---:|---:|---:|---:|---:|
| Post, untouched HF | Two-pass reasoning | 174/203 (85.71%) | 174/203 (85.71%) | 86/114 / 86/114 (75.44%) | 0.9823 / 0.9868 | 203/203 / 202/203 |
| Post, official Q4_K_M | Two-pass, single BOS, GPU | 176/203 (86.70%) | 176/203 (86.70%) | 88/114 / 88/114 (77.19%) | 0.9732 / 0.9732 | 203/203 / 198/203 |
| Base, untouched HF | Direct | **186/203 (91.63%)** | **186/203 (91.63%)** | **98/114 / 98/114 (85.96%)** | **0.9956 / 0.9956** | 203/203 / 203/203 |
| Base rank-16 LoRA, HF | Direct | 184/203 (90.64%) | 184/203 (90.64%) | 96/114 / 96/114 (84.21%) | 0.9912 / 0.9956 | 203/203 / 203/203 |
| Base LoRA, BF16 GGUF | Direct, single BOS, GPU | 184/203 (90.64%) | 183/203 (90.15%) | 96/114 / 95/114 (84.21% / 83.33%) | 0.9912 / 0.9956 | 203/203 / 203/203 |
| Base LoRA, Q8_0 GGUF | Direct, single BOS, GPU | 184/203 (90.64%) | 183/203 (90.15%) | 96/114 / 95/114 (84.21% / 83.33%) | 0.9912 / 0.9956 | 203/203 / 203/203 |
| Base LoRA, Q4_K_M GGUF | Direct, Android-current BOS, CPU | 166/203 (81.77%) | 166/203 (81.77%) | 78/114 / 78/114 (68.42%) | 0.9498 / 0.9498 | 203/203 / 203/203 |
| Base LoRA, Q4_K_M GGUF | Direct, single BOS, CPU | 173/203 (85.22%) | 173/203 (85.22%) | 85/114 / 85/114 (74.56%) | 0.9593 / 0.9593 | 203/203 / 203/203 |

Post HF produced one strict-invalid output. Its strict transaction detection was
112 true positives / two misses versus 111 / three after app interpretation, even
though both exact counts stayed 174. Official Post Q4 produced five strict-invalid
outputs; the app parser recovered all five to contract-valid interpretations
without changing its exact score. All Base variants were strict-valid on all 203
rows,
although BF16/Q8 strict interpretation changed one exact row relative to app
interpretation. App-interpreted transaction detection was 111 true positives and
three misses for Post HF, 109 and five for Post Q4, 114 and zero for Base, 113 and
one for HF LoRA/BF16/Q8, 104 and ten for Android-current tuned Q4, and 106 and eight
for single-BOS tuned Q4. Every variant had one generated false positive.

App-interpreted field accuracy on all 114 transaction rows was:

| Variant | Amount | Account | Type | Counterparty |
|---|---:|---:|---:|---:|
| Post HF | 97.37% | 97.37% | 97.37% | 75.44% |
| Post official Q4_K_M | 95.61% | 95.61% | 95.61% | 77.19% |
| Base HF | **100.00%** | **100.00%** | **100.00%** | **85.96%** |
| Base LoRA HF | 98.25% | 96.49% | 99.12% | 84.21% |
| Base LoRA BF16/Q8, single BOS | 98.25% | 98.25% | 99.12% | 84.21% |
| Base LoRA Q4_K_M, Android-current BOS | 91.23% | 87.72% | 91.23% | 69.30% |
| Base LoRA Q4_K_M, single BOS | 92.98% | 89.47% | 92.98% | 75.44% |

Counterparty remained the weakest field for all variants. The adapter reduced the
Base point estimate in every field, although the paired whole-row comparison does
not establish a reliable overall change.

### Paired interpretation

| Comparison | First only correct | Second only correct | Ties | Exact McNemar p |
|---|---:|---:|---:|---:|
| Post HF first, Post Q4 second | 8 | 10 | 185 | 0.81452942 |
| Post HF first, Base second | 3 | **15** | 185 | 0.00753784 |
| Base first, LoRA HF second | 5 | 3 | 195 | 0.7265625 |
| LoRA HF first, BF16 single-BOS second | 0 | 0 | 203 | 1.0 |
| BF16 first, Q8_0 single-BOS second | 0 | 0 | 203 | 1.0 |
| Android-current Q4 first, single-BOS Q4 second | 1 | **8** | 194 | 0.0390625 |
| LoRA HF first, single-BOS Q4 second | **13** | 2 | 188 | 0.00738525 |

These are paired app-interpreted exact results over the same 203 rows. Post HF
and official Post Q4 did not differ reliably. The Post-to-Base comparison supports
the observed Base advantage on this fixture; Base-to-LoRA does not support a
reliable LoRA improvement. Tuned HF and BF16 had three semantic prediction
differences but no exact-correctness flips; BF16 and Q8 had no prediction-string
differences. Thus merge/BF16 and Q8 preserved app exact correctness. Removing the
duplicate BOS significantly improved Q4, but an 11-row net, significant Q4 loss
remained. With BOS held single and the engine path controlled by BF16/Q8, that
remaining quality loss is specific to Q4 quantization in these host diagnostics.

## BF16 capacity probe

The pre-training gate tested a real BF16 forward/backward pass on the longest
selected training example: 1,912 total sequence tokens, including 39 completion
tokens, with maximum length 2,304 and microbatch one. It passed in 9.972 seconds at
7,351.9 MiB peak VRAM on the RTX 4070. The probe loaded no 4-bit weights, found
332 finite LoRA gradient tensors and 166 nonzero gradient tensors, and reported a
finite loss of 0.257812.

The probe covered all configured hybrid modules: 22 `in_proj`, eight each of
`q_proj`, `k_proj`, and `v_proj`, 30 `out_proj`, and 30 each of `w1`, `w2`, and
`w3`, for 166 matched modules. It was explicitly a capacity gate, not a quality
result. Its success justified BF16 LoRA instead of introducing QLoRA as another
experimental variable.

## Training data and objective

The run reused the same local Android-profile silver materialization as the 350M
experiment; it did not materialize any held-out test row. These derived labels are
not human gold. The builder configuration used seed 25,052,027, a 15% dev fraction,
minimum silver confidence 0.86, at most eight rows per template, at most 512 per
category, and a maximum 1:1 null-to-transaction ratio.

| Split | Rows | SHA-256 | Token range / mean / p95 | Explicit weight sum |
|---|---:|---|---|---:|
| Train | 154 | `66a5ffe1f0722a594838f6ee405cefa1f3adb8290dec92db222021c6a6e12f56` | 1,816-1,912 / 1,870.08 / 1,903 | 122.34 |
| Dev | 29 | `126b9337e49ecbbbc437dbf68b113f0a51d00ad28fe5bb1b6590427f742f9513` | 1,827-1,891 / 1,864.07 / 1,889 | 24.66 |

No row exceeded the 2,304-token training limit. Mean completion length was 28.73
tokens for train and 25.14 for dev. Every row carried an explicit provenance
weight, ranging from 0.645 to 0.99; mean weights were 0.794416 and 0.850345.

Training was completion-only causal SFT: prompt tokens were masked with ignore
index -100, causal shifting was enabled, each completion was reduced to a token
mean, and examples were combined as a sample-weighted mean. The first supervised
`null`-versus-JSON decision token received weight 3.0. This objective prevents
long prompts from dominating the loss and emphasizes the app's primary rejection
decision.

## LoRA and optimization

Rank 16, alpha 32, and dropout 0.05 were applied to `in_proj`, `q_proj`, `k_proj`,
`v_proj`, `out_proj`, `w1`, `w2`, and `w3`. The adapter trained 24,461,312 of
2,721,659,904 parameters, or 0.898764%. Module coverage matched the BF16 probe's
166-module inventory exactly.

| Setting | Recorded value |
|---|---|
| GPU and software | NVIDIA GeForce RTX 4070; Python 3.11.14; Torch 2.6.0+cu124; Transformers 5.6.2; PEFT 0.20.0 |
| Precision | BF16 training, TF32 enabled |
| Batch | Microbatch 1; accumulation 32; effective batch 32; eval batch 1 |
| Length | 2,304 tokens; gradient checkpointing enabled, non-reentrant |
| Optimizer | `adamw_torch`; learning rate 0.0001; max gradient norm 1.0 |
| Schedule | Cosine; warmup ratio 0.05 / one warmup step; weight decay 0.01 |
| Control | Seed 17; full determinism; evaluate and save each epoch |
| Stopping | Four epochs requested; early-stopping patience two |

The best dev loss was 0.2533269525 at epoch 1 / step 5. With patience two, training
ended at epoch 3 / step 15 after 15 optimizer steps, and
`load_best_model_at_end` restored `checkpoint-5` before adapter export. Recorded
run train loss was 6.5609316508; final evaluation of the restored checkpoint was
0.2533269525.

Training took 1,227.775 seconds wall time (1,227.452 seconds in Trainer). The run
through adapter save took 1,258.523 seconds, and final evaluation took 20.044
seconds. Peak training VRAM was 8,560.9 MiB. The exported adapter weights were
97,889,536 bytes with SHA-256
`7e1f2d0f47ab901942b4e307b06e02d45f09393ff5d94a8e8aac45f149ce7b18`.

## Merge, GGUF conversion, and runtime diagnostics

Conversion used llama.cpp commit `fe2adf0e722f30f5295fdec8a0f1dc788f7498bc`
(tree `3d5eccf08dbfe6edab1939314964475a603ab2de`) and produced:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Tuned BF16 GGUF | 5,403,156,736 | `ad830af7b17bb2f90eecc70fedf7f939291dda5d267ce7cc58fee1d84d1cf7f5` |
| Tuned Q8_0 GGUF | 2,874,777,856 | `fccdc839f5e3213a769b280abbd2d8ca9df375807105b9fb0683db673f1c6c74` |
| Tuned Q4_K_M GGUF | 1,674,453,248 | `9849f6377eab1a292665714b91819ef17ed2cf52cff20717713fa646e0f7f77c` |
| Official Post Q4_K_M reference | 1,674,454,848 | `79fdf00351b46cf26f020aead28d01889886be87c55fa0eb907e6f9b00bfee14` |

GGUF magic verification passed for all three tuned artifacts. A one-token fixed-
synthetic load/generation smoke passed for tuned Q4_K_M with `llama-cli`; the
conversion script itself did not load-smoke BF16 or Q8_0.

The full regression diagnostics used the 3,072-token context and default-off
grammar. Tuned BF16 and Q8 and untouched Post Q4 were single-BOS GPU-offloaded
quality probes; tuned Q4 used the four-thread CPU path. Their host measurements
were:

| Artifact / BOS | App exact | Transaction exact | F1 | End-to-end p50 / p95 | Peak RSS | Peak GPU delta |
|---|---:|---:|---:|---:|---:|---:|
| Post Q4_K_M / single | 176/203 | 88/114 | 0.973214 | 5,624.481 / 5,978.984 ms | 671.7 MiB | 1,953 MiB |
| BF16 / single | 184/203 | 96/114 | 0.991228 | 649.621 / 737.360 ms | 1,176.2 MiB | 5,523 MiB |
| Q8_0 / single | 184/203 | 96/114 | 0.991228 | 431.504 / 486.011 ms | 709.7 MiB | 3,095 MiB |
| Q4_K_M / Android-current | 166/203 | 78/114 | 0.949772 | 3,021.512 / 3,553.862 ms | 2,280.5 MiB | CPU-only |
| Q4_K_M / single | 173/203 | 85/114 | 0.959276 | 3,018.742 / 3,546.215 ms | 2,280.4 MiB | CPU-only |

Official Post Q4 averaged 764.18 reasoning tokens. It stopped or reached EOG on
53/115 invoked rows and exhausted the context-derived thinking budget on 62/115;
thinking p50/p95 were 5,376.687/5,620.817 ms and peak GPU memory was 2,439 MiB total.

The tuned Android-current evaluator removed the template BOS on 0/115 invoked rows
and emitted duplicate-leading-BOS warnings on all 115. Single-BOS tuned Q4 removed
it on 115/115 rows and recovered seven exact rows with essentially unchanged
latency and RSS. It is a controlled improvement, but it is not the PocketFinancer
default and does not close the Q4 quantization gap. Post Q4 and tuned BF16/Q8
timings are GPU-offloaded while tuned Q4 timings are CPU; cross-row timing
comparisons are therefore not meaningful.

All measurements are desktop `llama-cpp-python`, not Android phone measurements.
They cannot establish latency, memory, thermal, battery, or custom-JNI parity.

## Reproducibility and privacy

The Base model contains 2,697,198,592 parameters. Its locked BF16 weight file is
5,394,427,448 bytes with SHA-256
`3331a7db7672c4d6feb91352ae17bc62ae8b6b3263b4c290886ac452402e0551`.
Prompt assets and the Android source were hash-checked through profile version 3;
the profile pins a 3,072-token context, seven few-shot examples, the six-stage
prefilter, greedy decoding, default-off grammar, and Kotlin-compatible parsing.

All private data, adapters, merged weights, GGUFs, and row-level predictions remain
local and ignored. Training used `report_to = none`, `push_to_hub = false`, and did
not log raw examples. This work does not authorize uploading the private corpus,
model proposals, annotations, predictions, adapters, merged weights, or GGUFs.

## Release and deployment gates

1. Complete qualified blinded human adjudication of the 1,436 held-out test rows,
   import only the reviewed manifest, and evaluate without tuning on the result.
2. Decide and audit the correct BOS behavior in PocketFinancer's pinned custom JNI
   path, then verify exact prompt token IDs and HF-to-GGUF behavior there. A host
   `llama-cpp-python` result is not Android runtime parity.
3. Measure the selected artifact on target phones: cold load, p50/p95 latency, peak
   RAM, sustained thermals, battery, and failure recovery. The approximately
   1.67 GB Q4 file is only part of runtime memory.
4. Establish a quality threshold on fresh human gold and repeat training across
   controlled seeds/data sizes before claiming a LoRA effect or choosing a model.
5. Review data rights and privacy, and obtain an explicit release/deployment
   decision. Local preparation does not authorize publication or app rollout.
6. Confirm the applicable entity and LFM Open License v1.0 terms before shipping.
   The lock records a USD 10 million annual-revenue threshold for commercial use;
   redistribution also requires the license and notices. This is not legal advice.

The current model configuration deliberately records Android support and deployment
as false. The untouched Base HF result is the quality leader on this diagnostic,
but no 2.6B artifact should be promoted until the human-gold, BOS, custom-JNI,
device, privacy, data-rights, and license gates are closed.
