# LFM2.5-350M PocketFinancer pipeline v2

> **Current app correction (2026-08-05):** PocketFinancer Android `a9b7df44`
> supersedes the `a6c8a11` runtime described in this experiment chronology. The
> app now applies its SMS filter, uses a 3,072 context, branches on model thinking
> capability, and defaults optional grammar off. New runs use
> `scripts/run_pocketfinancer_pipeline.py`; saved scores below retain their original
> experimental provenance.
> The first run under that current profile is now complete; its canonical record is
> `docs/experiments/POCKETFINANCER_A9_LORA_R16_S17.md`. That record supersedes this
> chronology for current status and Android-profile scores.

Date: 2026-08-05
Status: local research snapshot; Android-runtime interpretation superseded

> **Historical `a6c8a11` correction:** A read-only audit at that older revision
> proved that the evaluator then called "exact Android" matched prompt assets and
> chat roles, but not that revision's runtime. At `a6c8a11` the app had no
> six-stage prefilter, used `n_ctx=1024`, and forced a 1,024-token thinking pass
> followed by a grammar-constrained 256-token answer. Android `a9b7df44` later
> changed those choices to the same prefilter, 3,072-token context,
> model-dependent thinking, and default-off optional grammar now declared here.
> The saved scores were not regenerated after that change and remain host research
> evidence, not end-to-end app/device results. Read
> `docs/architecture/ANDROID_RUNTIME_AUDIT.md` and
> `docs/experiments/EXPERIMENT_CATALOG.md` as the canonical interpretation.

## Executive answer

The original LFM2.5-350M experiment did **not** train or evaluate with the exact
prompt and runtime contract used by PocketFinancer Android. It trained on a short
five-line system prompt with no demonstrations and a 512-token limit. Its locked
regression evaluation also used the legacy decoding path. The selected Q8 artifact's
58/203 (28.6%) result therefore does not answer how the model behaves in the app.

A prompt-aligned experimental evaluator now exists. Under its long prompt,
selection-repo prefilter, and single-pass desktop defaults, the untouched base
model gets 88/203 (43.3%) entirely from deterministic prefilter rejections and gets
0/114 transactions exactly right. The old synthetic
adapter improves this to 114/203 (56.2%) and 26/114 transaction exact, but it is still
not viable. A newly trained private-v2 LoRA finally uses the exact Android prompt
assets and role placement:
it reaches 124/203 (61.0837%) overall and 35/114 (30.7018%) transaction exact, with no
ghosts, only three misses, and fully valid output. A later audit found that this
adapter's private-v2 source contained 122 contaminated silver labels, so 124/203 is
historical prompt-contract evidence, not clean training evidence.

The corrected prompt-aligned LoRA trains on private v3 and reaches 140/203 (68.9655%)
overall and 51/114 (44.7368%) transaction exact. This is the current direct research
result, not a current-app result: 94 true positives, 20 misses, no ghosts, and
202/203 valid/schema-valid outputs. Compared with the contaminated adapter, cleaning
improves counterparty and joint extraction but lowers transaction recall.

The best defensible LFM result is the rebuilt grounded selector: strict and safe
decoding both reach 154/203 (75.8621%) overall and 66/114 (57.8947%) transaction
exact, with 102 true positives, 12 misses, one ghost, and 203/203 valid outputs. The
safe path made zero interventions, so this score does not depend on a post-hoc
override. The older 168/203 score remains an unsafe, contaminated diagnostic only;
the 151/203 currency-only counterfactual it motivated is superseded by the clean
rebuild. The candidate approach is not a drop-in replacement for the current app,
but it is the clearest evidence that 350M can be useful after deterministic span
extraction and reconstruction.

## Historical correction: prompt alignment versus runtime parity

The earlier report's phrase “production prompt” referred to the experiment's short
repository prompt. It did not refer to the prompt actually assembled by
PocketFinancer Android. The prompt assets and roles were later ported into
`lfm25/android_contract.py`, using `DATA/utils.py`. The first host runtime
assumptions did not match Android `a6c8a11`; Android `a9b7df44` later converged on
the same high-level direct-LFM profile.

| Contract element | Previous LFM experiment | Prompt-aligned saved experiment |
|---|---|---|
| Outer system message | Five-line extraction policy | Generic financial-SMS assistant message |
| User message | Current sender and SMS | Long extraction policy, seven demonstrations, then current sender and SMS |
| Few-shot examples | None | Seven |
| Prompt length on locked set | Fits the 512-token training limit | About 1,871–2,017 tokens before the answer; mean 1,905 |
| Context | 512 train; 2,048 evaluation | 3,072 (matches Android `a9b7df44`; `a6c8a11` used 1,024) |
| Generation cap | 96 | 256 |
| Repetition penalty | 1.05 | 1.0 (Android has no repeat sampler) |
| Grammar | Enabled in final legacy GGUF evaluation | Disabled (matches the default at `a9b7df44`; `a6c8a11` required it) |
| Prefilter | None | Six ordered stages (matches `a9b7df44`; absent at `a6c8a11`) |

The current app also uses model configuration to choose direct versus two-pass
generation. LFM2.5-350M is treated as non-thinking, so its app path is a direct
256-token answer. The remaining parity gap is the host runtime and saved-output
provenance, not these high-level settings.

Role placement is part of the contract. Android puts the long policy and all seven
demonstrations inside the **user** message and uses only the generic sentence as the
outer **system** message. Moving the long policy into the system role changes the
tokenized input and is not equivalent.

The mismatch is structural, not cosmetic. The app prompt alone is roughly four
times the old 512-token training ceiling, so the exact Android serialization could
not have been present in those training examples. The former private SFT path was a
third contract: it used the long policy as a system message but omitted the seven
demonstrations. It was not used for the selected artifact.

### Six-stage prefilter and locked-set behavior

The saved selection experiment applies these stages in order: reject a personal
numeric sender; require a currency-denominated amount; require a masked account/card;
require a completed-transaction verb; reject OTP/security messages; reject collect
or mandate requests. Android `a9b7df44` now uses the same ordered logic. On the
203-row locked set the saved host experiment:

- passes all 114/114 gold transactions;
- rejects 88/89 gold nulls, leaving 115 model invocations;
- rejects 43 rows at the currency stage, 40 at account/card, three at transaction
  verb, and two at OTP.

Whole-pipeline scores must therefore be read alongside conditional model scores.
For example, the untouched model's 88 exact rows come from the filter; its model
output is schema-valid on 0/115 invoked rows under the long Android prompt.

## What the original experiment actually trained

### Data

The completed training ladder used only a programmatic synthetic pool:

- 350 training rows from 14 template families;
- 100 development rows from four wholly held-out families;
- no private SMS-derived training row;
- programmatic labels rather than human gold or consensus silver.

The raw-data preparation track existed, but it did not feed the selected model. It
decontaminated and component-split 14,075 eligible incoming messages into 11,260
train, 1,379 dev, and 1,436 test rows. Three local proposal models produced 128
accepted consensus labels, which materialized as 101 train, 27 dev, and zero test
examples. The optional private pilot was skipped because the accepted dev set had
only three transactions, below its predeclared minimum of six.

### Training and evaluation

The old run used BF16 rank-16 LoRA with completion-only, token-averaged loss, an
effective batch size of 32, cosine scheduling, epoch evaluation, and a six-epoch
cap. Rank 16, learning rate 1e-4, seed 29 was selected on the synthetic dev set.
The final locked-set evaluation invoked the model on all 203 rows with the short
prompt, 2,048 context, a 96-token cap, repetition penalty 1.05, and a JSON grammar.

Its best Q8 result was 58/203 (28.6%) exact, including 8/114 transaction exact,
39/89 ghosts, and 26/114 misses. This was below the always-null baseline of 89/203
(43.8%). See `docs/LFM25_350M_EXPERIMENT_REPORT.md` for the complete historical run,
quantization, latency, and provenance record; interpret its “production prompt”
wording using the correction above.

## Private SFT v3: corrected raw-data rebuild

`lfm25/private_sft_v2.py` builds a new local-only dataset from the existing
decontaminated split manifest. Its important design choice is to seal the original
dev and test partitions **before** filtering, labeling, sampling, or derived
splitting. Only the original 11,260-row train partition is eligible.

The builder then:

1. applies the exact Android prefilter;
2. chooses labels in priority order: reviewed human gold, accepted local-model
   consensus silver, then high-confidence source-grounded heuristic silver;
3. requires transaction fields to be grounded in the current source message and
   excludes ambiguous or insufficiently grounded labels;
4. caps repeated templates and categories;
5. creates an inner train/dev split over connected sender/template components, so
   neither sender nor normalized template crosses the boundary; and
6. attaches provenance and a top-level `sample_weight` to every row.

The first materialization, called private v2, was later invalidated. A stricter
candidate-grounding audit found 87 old-train rows and 37 old-dev rows whose labeled
counterparty could not be selected from source-backed candidates. Two of the 87
train rows are legitimate candidate-extractor coverage misses; the other 85 train
rows plus all 37 dev rows are contaminated silver labels: **122 total**.

The corrected builder tightens counterparty grounding, including rejection of
currency-bearing counterparty labels, and rebuilds from the original raw split
manifest rather than deleting derived rows. Aggregate `no_acceptable_grounded_label`
exclusions rise from 134 to 256. Component membership and the inner split are
recomputed, so private v3 also reflects normal selection/splitting effects.

No human-gold row is available in the corrected set. Of 467 Android-filter passes,
256 lack an acceptable grounded label and 26 are removed by the template cap. The
clean result is:

| Derived split | Rows | Transactions | Nulls | Consensus silver | Grounded silver |
|---|---:|---:|---:|---:|---:|
| Train | 160 | 140 | 20 | 59 | 101 |
| Dev | 25 | 23 | 2 | 17 | 8 |
| Total | 185 | 163 | 22 | 76 | 109 |

There is zero sender overlap and zero normalized-template overlap between derived
train and dev. The original 1,379 dev rows and 1,436 test rows remain sealed, with
zero materialized. Grounded silver receives `0.75 * heuristic_confidence`, accepted
consensus receives the minimum matching-proposal confidence, and reviewed human gold
would receive 1.0.

The corrected private-v3 train/dev artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `private_sft_v3/private_sft_v2_train.jsonl` | `c48ea5d565bebc9858489ea1c2a0623f96dea276673a9f4b429321e14e1318f9` |
| `private_sft_v3/private_sft_v2_dev.jsonl` | `e61fa0ec3a945c8c25af961e885382d57f367ada5ddf2f815e4f6168200d282c` |

The derived dev split is a silver tuning set, not a gold benchmark. With only two
null rows, it is unusable for a credible ghost-rate or natural-prevalence claim.

## Grounded candidate selector

The direct four-field task asks a tiny model both to decide semantics and to copy
arbitrary amounts, accounts, and counterparties exactly. `lfm25/candidates.py`
separates those jobs:

1. deterministic regex code extracts source-backed candidate spans;
2. the model emits only compact IDs plus transaction/null and debit/credit decisions;
3. a strict parser rejects any unknown ID or wrong schema; and
4. deterministic reconstruction copies the chosen source values into the existing
   four-field PocketFinancer object.

Amounts use `A*` IDs and accounts use `C*`. Counterparty IDs encode the source cue,
such as at/to/by/from/VPA/linked-mobile/towards/UPI-or-NEFT/for/on, plus an explicit
no-counterparty choice. The ID semantics remove the positional ambiguity found in
the first selector pilot.

On the locked 114 transactions, the deterministic candidate oracle covers amount
114/114, account 114/114, and counterparty 113/114; joint coverage is 113/114
(99.12%). This is an extractor upper bound, not a model score. Candidate v4 applies
the same oracle requirement to corrected private v3. It excludes only two
counterparty-uncovered train transactions and no dev rows, producing 158 train and
25 dev rows. Their hashes are
`fb6e073e1a4df1ead1fa0fa6a0e098022486cf56e63f19d56e3706b3ef291a40`
and `1807c8faf13229946e153d4825bcac28d773b0abacc1dd09d9d6dbd23bbd9e19`.
Candidate v4 and the clean selector training manifest both bind the current
`lfm25/candidates.py` implementation SHA-256
`e4b2c9dc2b6eeb85e2ab494ac905a492bd7ef4692560313b6ec0cab1ed07b8c1`;
there is no candidate-implementation mismatch in the clean training/evaluation
chain.

### Low-weight semantic curriculum

`lfm25/candidate_curriculum.py` supplies sparse cue and hard-negative coverage
without copying private messages. Curriculum v2 generates 220 transactions,
including a 20-row no-counterparty ATM family that teaches the explicit `PN`
selection, plus 60 policy-valid hard negatives. Every one of the 280 rows passes the
Android prefilter and is candidate-oracle-covered. Each synthetic row has weight
0.2.

Mixing those 280 rows with 158 clean private selector rows gives 438 rows: 358
transactions and 80 nulls. The private weight sums to 124.485 and synthetic weight
to 56.0, so 69.0% of effective weight remains private. The curriculum hash is
`b0f6e6cf0094275e54324caa9487dba161586c8356ce0026b6c261131874fee7`; the mixed
train hash is `13e2e7eab1d708542a679e43e3134feca47e93d6df552d1f55608ff501fc87fc`.

### Objective and training setup

The v2 trainer in `scripts/train_lfm25_lora.py` supports `legacy`, `android`, and
`candidate_selector` prompt profiles. It refuses overlength rows instead of silently
truncating them. Android defaults to 3,072 tokens; legacy and selector profiles
default to 512.

V2 uses causal, completion-only cross-entropy normalized to a mean **per example**,
then combines examples with their provenance weights. The first supervised decision
token is weighted 3.0 in all runs reported here. This prevents long transaction
targets from overwhelming short null targets; without per-example normalization,
93.67% of supervised target tokens came from transactions and only 6.33% from nulls.

Every v2 run used BF16 rank-16 LoRA with 5,996,544 trainable parameters (1.6635%),
learning rate 1e-4, cosine scheduling, effective batch 32, seed 17, and patience-two
early stopping. Exact-Android runs use batch 2, gradient accumulation 16, and a
2,304-token ceiling. The contaminated run used 253/54 rows; the clean run uses the
160/25 private-v3 rows. Both completed and selected epoch four. Selector runs use
batch 8, accumulation 4, and a six-epoch cap.

| Training run | Train rows | Epochs run / selected | Eval loss | Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Exact Android, contaminated private v2 | 253 | 4 / 4 | 0.058073 | 610.9 s | 3,887 MiB |
| **Exact Android, clean private v3** | **160** | **4 / 4** | **0.0809661** | **406.3 s** | **3,886.7 MiB** |
| Positional-ID private pilot, contaminated | 251 | 6 / 4 | 0.024616 | 168.6 s | 3,076 MiB |
| Semantic-ID private, contaminated | 251 | 6 / 5 | 0.025285 | 185.0 s | 3,647 MiB |
| Semantic-ID + curriculum v1, contaminated | 521 | 5 / 3 | 0.024465 | 304.8 s | 3,647 MiB |
| **Clean semantic-ID + curriculum v2** | **438** | **5 / 3** | **0.0247496** | **254.0 s** | **3,645.2 MiB** |

Every row labeled contaminated predates the grounding correction and is historical
only. The clean exact-Android adapter is at
`TRAINING_ARTIFACTS/lfm25_v2/android_private_clean_r16_lr1e4_seed17/adapter`.
The clean selector adapter is at
`TRAINING_ARTIFACTS/lfm25_v2/candidate_semantic_curriculum_clean_r16_lr1e4_seed17/adapter`.
Both have 5,996,544 trainable parameters. Selector training stopped after epoch five
and restored epoch three, the lowest silver-dev loss.

### Hybrid history and final safe behavior

The old contaminated curriculum-v1 adapter scored 145/203 strictly and 168/203 when
the first broad `--hybrid-safety` rule made 29 counterparty overrides. Pairing
those saved rows showed 23 fixes and no locked-set breaks, but an oracle stress audit
over 596 candidate-covered labeled transactions found that the rule would corrupt
61 of 63 already-correct selections it altered. The 168/203 score is therefore an
unsafe benchmark diagnostic and must never be treated as the deployable result.

A currency-only narrowing produced a 151/203, 63/114 counterfactual on the old saved
outputs. That estimate was never the final clean pipeline and is now superseded by
the raw-data rebuild, candidate v4, curriculum v2, and clean retraining.

The final evaluator retains only the evidence-supported safe behavior. On the clean
adapter it makes **zero interventions**: strict and `--hybrid-safety` both score
154/203 overall and 66/114 transaction exact, with identical predictions and
metrics. The defensible number is therefore 154/203, independent of a hybrid rescue.

## Result ladder on the locked 203-row regression set

The set contains 114 transactions and 89 nulls. It has been used repeatedly during
development, so it is a compatibility benchmark rather than an unbiased test.
Android-aligned and candidate rows below include the same six-stage prefilter.

| System | Whole exact | Transaction exact | Ghosts | Misses | JSON/schema valid |
|---|---:|---:|---:|---:|---:|
| Always null | 89/203 (43.8%) | 0/114 | 0/89 | 114/114 | 203/203 |
| Historical short-prompt LFM Q8 | 58/203 (28.6%) | 8/114 (7.0%) | 39/89 | 26/114 | 203/203 |
| Exact Android contract, untouched LFM | 88/203 (43.3%) | 0/114 | 0/89 | 114/114 | 88/203 |
| Exact Android contract, old synthetic adapter | 114/203 (56.2%) | 26/114 (22.8%) | 1/89 | 57/114 | 196/203 |
| Exact Android, contaminated private-v2 adapter | 124/203 (61.0837%) | 35/114 (30.7018%) | 0/89 | 3/114 | 203/203 |
| **Exact Android, clean private-v3 adapter** | **140/203 (68.9655%)** | **51/114 (44.7368%)** | **0/89** | **20/114** | **202/203** |
| Positional selector, contaminated | 133/203 (65.5%) | 45/114 (39.5%) | 1/89 | 11/114 | 203/203 |
| Semantic selector, contaminated | 140/203 (69.0%) | 52/114 (45.6%) | 1/89 | 11/114 | 203/203 |
| Curriculum-v1 selector, contaminated | 145/203 (71.4%) | 57/114 (50.0%) | 1/89 | 12/114 | 202/203 |
| Currency-only old counterfactual, superseded | 151/203 (74.38%) | 63/114 (55.26%) | not persisted | not persisted | not persisted |
| Broad old hybrid, unsafe diagnostic | 168/203 (82.8%) | 80/114 (70.2%) | 1/89 | 12/114 | 202/203 |
| **Clean curriculum-v2 selector, strict** | **154/203 (75.8621%)** | **66/114 (57.8947%)** | **1/89** | **12/114** | **203/203** |
| **Clean curriculum-v2 selector, safe** | **154/203 (75.8621%)** | **66/114 (57.8947%)** | **1/89** | **12/114** | **203/203** |
| Historical Gemma Q4 reference | 175/203 (86.2%) | not recorded | 17/89 | 0/114 | not recorded |
| Historical Gemma Q8 reference | 176/203 (86.7%) | not recorded | 15/89 | 0/114 | not recorded |

The contaminated exact-Android private adapter's transaction precision/recall/F1
are 100% / 97.37% / 98.67%, from 111 true positives, three misses, and zero ghosts.
Its amount, account, type, and counterparty accuracies are
107/114 (93.86%), 108/114 (94.74%), 82/114 (71.93%), and 50/114 (43.86%).
It learned detection and source copying far better than the old adapter, but
counterparty and type errors hold joint transaction exactness to 30.70%. Because its
training set is contaminated, these remain historical observations; the following
paragraph is the clean exact-Android result.

The clean exact-Android adapter's transaction precision/recall/F1 are 100% /
82.4561% / 90.3846%, from 94 true positives, 20 misses, and zero ghosts. Amount,
account, type, and counterparty accuracy are 94/114 (82.4561%), 92/114 (80.7018%),
75/114 (65.7895%), and 65/114 (57.0175%); joint transaction exact is 51/114
(44.7368%). Relative to the contaminated run, cleaning gains 16 joint-exact
transactions and 15 counterparty-correct transactions but loses 17 detection true
positives. It produces 202/203 valid/schema-valid outputs. This 140/203 score is the
current direct result for the exact app prompt and output contract.

The clean selector's transaction precision/recall/F1 are 99.03% / 89.47% / 94.01%,
from 102 true positives, 12 misses, and one ghost. Amount, account, and type are each
102/114 (89.4737%); counterparty and joint transaction exact are both 66/114
(57.8947%). Strict parsing and the safe path are identical, with zero safe
interventions and 203/203 valid/schema-valid outputs. This 154/203 result is the
best defensible LFM score: it is 21 exact rows below historical Gemma Q4 and 22 below
Q8. Those references came from earlier repository evaluation work, so the table is a
useful quality bar, not a claim of identical mobile runtime conditions.

## What this says about the 350M feasibility question

The evidence currently supports a conditional “possibly,” not a production “yes”:

- The untouched checkpoint cannot handle direct long-prompt generation, and the old
  synthetic adapter transfers only partially to the exact app contract.
- The clean exact-prompt run reaches 68.9655% overall and 44.7368% transaction
  exact. Cleaning materially improves counterparty and joint extraction, although
  transaction recall falls from the contaminated diagnostic's 97.37% to 82.46%.
- Task decomposition changes the picture substantially: deterministic extraction and
  reconstruction let the model focus on a much smaller semantic decision space.
- The clean task-decomposed model reaches 75.8621% without any safe intervention;
  counterparty selection remains its dominant field error.
- The larger 82.8% number combines contaminated training data with an unsafe broad
  rule and is not evidence for deployment.

The next credible decision point is a blinded, human-gold, sender/template-held-out
test at natural prevalence. More tuning against the 203-row set cannot establish
production quality.

## Limitations and non-claims

- The 203 rows are locked only as a regression fixture; this v2 design and hybrid
  rule were developed after inspecting aggregate failures on it.
- The clean dev set is all silver and has only two null rows. Its low eval loss
  cannot validate semantics, ghost rate, or deployment readiness.
- No original test row was trained on or materialized, but the original test also
  has no completed human-gold adjudication, so there is no new primary test score.
- The clean 154/203 system is not wire-compatible with the current Android prompt.
  It needs candidate extraction, selector prompting, and reconstruction added and
  tested in the app.
- The broad hybrid counterparty override fails oracle stress testing despite its
  strong locked-set gain. The 168/203 row is diagnostic evidence, not a deployable
  configuration.
- The 151/203 currency-only score belongs to the superseded contaminated adapter and
  was only a counterfactual. The final clean strict/safe evaluations are persisted.
- The contaminated exact-Android result is retained only to explain the effect of
  cleaning. The clean direct result still has 20 transaction misses and one invalid
  output, so it does not meet the production quality bar.
- The prompt-aligned HF evaluator is a training/research proxy. A new GGUF evaluator
  now mirrors Android `a9b7df44`'s prefilter, model-dependent direct/two-pass choice,
  context, grammar default, and Kotlin parser semantics, but the saved scores in
  this chronology predate that profile. Android still uses custom CPU/JNI llama.cpp,
  so exact template/KV behavior, device memory, latency, thermals, battery, and
  quantization parity remain unverified until the experiments are rerun.
- The candidate oracle has one known locked transaction outside its candidate set;
  no selector can recover it until deterministic extraction improves.
- Silver labels can encode errors from heuristics or local proposal models. Confidence
  and grounding reduce risk but do not make them human gold.
- No model, adapter, data, or report is authorized for upload, publication, or
  production promotion.

## Privacy, sealing, and decontamination

- All raw-data processing, labeling, training, and inference stayed local and
  offline; W&B, hub pushes, hosted labeling, and raw-example logging were disabled.
- Raw values are not printed by the builders. Reports contain counts, configuration,
  hashes, and provenance only.
- The original preparation excluded 3,509 regression relatives before private
  splitting: 366 exact, 2,566 normalized-template, and 577 configured near relatives.
- Private v3 inherits that boundary and additionally seals original dev/test before
  any eligibility decision. Derived train/dev have zero sender and template overlap.
- The 280-row curriculum v2 is programmatic and not private-derived. Its persisted
  aggregate audit records zero exact and zero normalized-template overlaps against
  all 203 locked rows and all 14,075 private split-manifest rows. It contains 280
  unique exact messages and 112 distinct normalized templates. No private text or
  candidate-to-private mapping is retained in the audit.
- Private SFT, results, adapters, and training artifacts stay in ignored local trees;
  `scripts/check_repo_safety.py` remains the publication guard.

## Reproduction commands

Run from the prepared WSL workspace with all pinned model assets already local.
These commands do not download or upload anything. Builders that target canonical
paths replace only their fixed ignored outputs when `--force` is supplied; the
curriculum output directory must not already exist. Use a fresh output namespace for
reruns if the canonical training/results directories are present.

```bash
cd /home/tojinotzenin/pF_slm_selection
source scripts/activate_wsl.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true

# Reproduce the stricter audit of the invalidated old private-v2 materialization.
python scripts/build_lfm25_candidate_sft.py \
  --input-dir PRIVATE_DATA/lfm25/private_sft_v2 \
  --output-dir PRIVATE_DATA/lfm25/candidate_sft_v3 \
  --force

# Build corrected private v3 from the prepared, decontaminated split manifest.
python scripts/build_lfm25_private_sft_v2.py \
  --manifest PRIVATE_DATA/lfm25/split_manifest.jsonl \
  --output-dir PRIVATE_DATA/lfm25/private_sft_v3 \
  --dev-fraction 0.15 \
  --seed 25052027 \
  --minimum-silver-confidence 0.86 \
  --max-per-template 8 \
  --max-per-category 512 \
  --max-null-to-transaction-ratio 1.0 \
  --force

# Convert source-grounded labels into compact selector targets.
python scripts/build_lfm25_candidate_sft.py \
  --input-dir PRIVATE_DATA/lfm25/private_sft_v3 \
  --output-dir PRIVATE_DATA/lfm25/candidate_sft_v4 \
  --force

# Build the low-weight semantic curriculum and mixed training file.
python scripts/build_lfm25_candidate_curriculum.py \
  --private-train PRIVATE_DATA/lfm25/candidate_sft_v4/candidate_sft_train.jsonl \
  --output-dir PRIVATE_DATA/lfm25/candidate_curriculum_v2 \
  --rows-per-relation 20 \
  --rows-per-negative 10 \
  --sample-weight 0.2 \
  --seed 35025 \
  --decontamination-corpus DATA/extraction_ds.jsonl \
  --decontamination-corpus PRIVATE_DATA/lfm25/split_manifest.jsonl

# Audit the deterministic candidate upper bound on the locked set.
python scripts/audit_lfm25_candidate_coverage.py \
  --dataset DATA/extraction_ds.jsonl \
  --label-field expected \
  --output RESULTS/lfm25/v2_candidate_oracle_coverage.json
```

Reproduce the final clean selector training run:

```bash
python scripts/train_lfm25_lora.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --train PRIVATE_DATA/lfm25/candidate_curriculum_v2/candidate_mixed_train.jsonl \
  --eval PRIVATE_DATA/lfm25/candidate_sft_v4/candidate_sft_dev.jsonl \
  --output-dir TRAINING_ARTIFACTS/lfm25_v2/candidate_semantic_curriculum_clean_r16_lr1e4_seed17 \
  --prompt-profile candidate_selector \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 0.0001 --epochs 6 \
  --batch-size 8 --eval-batch-size 8 --gradient-accumulation 4 \
  --max-length 512 \
  --loss-mode per_example_completion_mean \
  --first-supervised-token-weight 3.0 \
  --warmup-ratio 0.1 --weight-decay 0.01 \
  --early-stopping-patience 2 --seed 17
```

Reproduce the saved prompt-aligned experimental baselines and final clean selector:

```bash
# Saved long-prompt experimental contract: untouched base and old adapter.
python scripts/evaluate_lfm25_android_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/repro_android_untouched_seed17 \
  --contract android-prompt-proxy --apply-prefilter --seed 17

python scripts/evaluate_lfm25_android_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_experiments_final/final_r16_lr0p0001_seed29/adapter \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/repro_android_old_adapter_seed17 \
  --contract android-prompt-proxy --apply-prefilter --seed 17

# Final clean adapter, strict and safe modes.
python scripts/evaluate_lfm25_candidate_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_v2/candidate_semantic_curriculum_clean_r16_lr1e4_seed17/adapter \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/v2_candidate_semantic_curriculum_clean_seed17_regression \
  --apply-prefilter --no-hybrid-safety --seed 17

python scripts/evaluate_lfm25_candidate_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_v2/candidate_semantic_curriculum_clean_r16_lr1e4_seed17/adapter \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/v2_candidate_semantic_curriculum_clean_safe_hybrid_seed17_regression \
  --apply-prefilter --hybrid-safety --seed 17
```

Reproduce the clean prompt-aligned direct training and saved proxy evaluation:

```bash
python scripts/train_lfm25_lora.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --train PRIVATE_DATA/lfm25/private_sft_v3/private_sft_v2_train.jsonl \
  --eval PRIVATE_DATA/lfm25/private_sft_v3/private_sft_v2_dev.jsonl \
  --output-dir TRAINING_ARTIFACTS/lfm25_v2/android_private_clean_r16_lr1e4_seed17 \
  --prompt-profile android \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 0.0001 --epochs 4 \
  --batch-size 2 --eval-batch-size 2 --gradient-accumulation 16 \
  --max-length 2304 \
  --loss-mode per_example_completion_mean \
  --first-supervised-token-weight 3.0 \
  --warmup-ratio 0.05 --weight-decay 0.01 \
  --early-stopping-patience 2 --seed 17

python scripts/evaluate_lfm25_android_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_v2/android_private_clean_r16_lr1e4_seed17/adapter \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/v2_android_private_clean_seed17_regression \
  --contract android-prompt-proxy --apply-prefilter --seed 17
```

The earlier contaminated prompt-aligned run is retained for prompt-contract diagnosis,
but its private-v2 labels are now known to be contaminated. Its adapter is
`TRAINING_ARTIFACTS/lfm25_v2/android_private_r16_lr1e4_seed17/adapter`. The
historical training and evaluation commands were:

```bash
python scripts/train_lfm25_lora.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --train PRIVATE_DATA/lfm25/private_sft_v2/private_sft_v2_train.jsonl \
  --eval PRIVATE_DATA/lfm25/private_sft_v2/private_sft_v2_dev.jsonl \
  --output-dir TRAINING_ARTIFACTS/lfm25_v2/android_private_r16_lr1e4_seed17 \
  --prompt-profile android \
  --rank 16 --alpha 32 --dropout 0.05 \
  --learning-rate 0.0001 --epochs 4 \
  --batch-size 2 --eval-batch-size 2 --gradient-accumulation 16 \
  --max-length 2304 \
  --loss-mode per_example_completion_mean \
  --first-supervised-token-weight 3.0 \
  --warmup-ratio 0.05 --weight-decay 0.01 \
  --early-stopping-patience 2 --seed 17

python scripts/evaluate_lfm25_android_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_v2/android_private_r16_lr1e4_seed17/adapter \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/v2_android_private_seed17_regression \
  --contract android-prompt-proxy --apply-prefilter --seed 17
```

Relevant focused tests are in `tests/test_android_contract.py`,
`tests/test_android_evaluators.py`, `tests/test_private_sft_v2.py`,
`tests/test_lfm25_training.py`, `tests/test_lfm25_candidates.py`,
`tests/test_candidate_sft.py`, and `tests/test_candidate_curriculum.py`.
