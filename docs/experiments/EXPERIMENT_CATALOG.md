# Experiment and dataset catalog

> **Historical research catalog.** The active SMS-processing architecture is now
> `src/pocketfinancer_sms` and the Grounded Candidate Selector contract. Terms such
> as “current” below describe status within the measured model-research lineage;
> they do not make Candidate Protocol V1, Semantic V2, or the old private splits
> active product/data architecture. See the
> [historical evidence index](../history/SMS_PROCESSING_EVIDENCE_INDEX.md).

Updated: 2026-08-08

Current Android profile: `a9b7df44` with active SMS prefilter, `n_ctx=3072`,
model-dependent thinking, and optional/default-off grammar. The `a6c8a11` parity
correction later in this chronology is historical. New experiments use
`configs/contracts/pocketfinancer-android-current.json`.

This is the canonical status table. It separates data quality, prompt alignment,
runtime parity, and benchmark freshness so that a high score cannot hide a broken
assumption.

## Evaluation boundary

`DATA/extraction_ds.jsonl` contains 203 labeled rows: 114 transactions and 89
nulls. It has been consulted throughout model and heuristic development. Keep it
locked as a regression/compatibility set; it is not a fresh production test and
must not be used for training.

No run below has yet been validated on a new, human-reviewed, template-held-out
test set or through a working target-phone Android inference path.

A frozen reviewer-blind package covers all 1,436 rows in the held-out test
partition, but all 1,436 remain pending human adjudication. It is not human gold
and contributed no training row or result below.

## Dataset lineage

| Dataset | Train | Dev/tuning | Status | Meaning |
|---|---:|---:|---|---|
| Historical synthetic direct SFT | 350 | 100 | Historical | Programmatic templates; selected old adapter |
| Private v2 direct | 253 | 54 | **Invalidated** | Silver data before strict counterparty grounding |
| Private v2 candidate subset | **251** | **54** | **Invalidated** | Candidate-covered subset of private v2 |
| Private v3 direct | **160** | **25** | Current clean silver | Rebuilt from sealed original-train partition |
| PocketFinancer a9 direct v1 | **154** | **29** | Current app-aligned silver | Filter-passing, sender/template-disjoint; used by r16-s17 |
| Candidate Protocol V1 paired arms | **152** | **29** | Current controlled silver | Fresh candidate-covered a9 materialization; shared by direct and selector arms |
| Candidate v4 private | **158** | **25** | Current clean silver | Private v3 minus two true candidate-coverage misses |
| Curriculum v2 | 280 | - | Current synthetic support | 220 transactions + 60 hard negatives, weight 0.2 |
| Candidate mixed training | **438** | **25** | Current clean experiment | 158 private candidate + 280 curriculum |

### What "251 train rows and 54 tuning rows" means

These were not 305 independent human-gold examples and the 54 rows were not a
final test set.

1. An older private-v2 builder materialized 253 train and 54 dev rows from only the
   original raw **train** partition. The original raw dev and test partitions stayed
   sealed.
2. The candidate builder could map 251/253 train rows to source-backed amount,
   account, and counterparty candidate IDs. Two valid transaction rows were omitted
   because the candidate extractor could not represent their counterparties.
3. All 54 old dev rows were candidate-covered. The resulting composition was 230
   transactions + 21 nulls in train and 53 transactions + 1 null in dev.
4. Training updated weights on the 251 rows. The 54 silver dev rows selected the
   epoch and hyperparameters; they contributed no gradients.
5. A stricter audit then found 85 contaminated train labels and all 37 affected dev
   labels, or 122 contaminated counterparty labels in the parent materialization.
   Together with the two genuine extractor misses, 87 old-train and 37 old-dev rows
   failed the candidate-grounding audit.

Therefore the 251/54 run is useful history about training mechanics, not evidence
about clean model quality. The correct response was to rebuild labels and component
splits from the original sealed raw split, producing 160/25 direct and 158/25
candidate data. We did not merely delete bad rows in place.

The older current 25-row dev set is still silver and has only two nulls. The new
app-aligned 29-row dev has 22 transactions and seven nulls. Both can choose a
checkpoint by loss, but neither can support a credible ghost-rate claim.

## Result catalog

| System | Exact on 203 | Transaction exact | Status and interpretation |
|---|---:|---:|---|
| LFM2.5-2.6B untouched Post, HF proxy | 174/203 | 86/114 | Two-pass local diagnostic; not Android runtime |
| LFM2.5-2.6B untouched Post, official Q4 single-BOS | 176/203 | 88/114 | Host-GPU diagnostic; no reliable change from Post HF (`p=0.815`) |
| **LFM2.5-2.6B untouched Base, HF proxy** | **186/203** | **98/114** | Best completed 2.6B result; reused-fixture ceiling evidence only |
| LFM2.5-2.6B Base rank-16 LoRA, HF proxy | 184/203 | 96/114 | No reliable gain over untouched Base (`p=0.727`) |
| LFM2.5-2.6B Base LoRA, BF16 single-BOS | 184/203 | 96/114 | Host-GPU merge reference; not phone runtime |
| LFM2.5-2.6B Base LoRA, Q8 single-BOS | 184/203 | 96/114 | Prediction-identical to BF16 on this host run |
| LFM2.5-2.6B Base LoRA, Q4 Android-current BOS | 166/203 | 78/114 | Host CPU diagnostic; duplicate-BOS warning on all invocations |
| LFM2.5-2.6B Base LoRA, Q4 single-BOS | 173/203 | 85/114 | Better than current BOS (`p=0.039`), still below HF (`p=0.007`) |
| Always null | 89/203 | 0/114 | Sanity baseline |
| Android a9 untouched base, HF proxy | 89/203 | 0/114 | Current baseline; app parser, not JNI runtime |
| **Android a9 r16-s17 adapter, HF proxy** | **121/203** | **32/114** | Current app-aligned training diagnostic |
| **Android a9 r16-s17 BF16 GGUF** | **122/203** | **33/114** | Current host GGUF reference; not phone runtime |
| Android a9 r16-s17 Q8 GGUF | 120/203 | 31/114 | No exact gain over Q4 in this run |
| **Android a9 r16-s17 Q4 GGUF** | **120/203** | **31/114** | 229 MB deployable candidate; not accurate enough |
| Android a9 r16-s17 Q4 single-BOS | 125/203 | 36/114 | Diagnostic only; paired p=0.302 |
| Historical short-prompt LFM Q8 | 58/203 | 8/114 | Historical; wrong prompt for app |
| Prompt-aligned old synthetic adapter | 114/203 | 26/114 | Historical; experimental prefilter/runtime |
| Prompt-aligned contaminated direct LoRA | 124/203 | 35/114 | Invalidated data; diagnostic only |
| **Prompt-aligned clean direct LoRA** | **140/203** | **51/114** | Current clean direct research result |
| **Clean grounded candidate selector** | **154/203** | **66/114** | Best defensible 350M research result; not Android wire-compatible |
| Candidate V1 direct HF, seeds 17/29/43 | 102 / 136 / 95 of 203 | 13 / 47 / 6 of 114 | Controlled baselines for the paired selector runs |
| **Candidate V1 selector HF, seeds 17/29/43** | **148 / 150 / 147 of 203** | **60 / 62 / 59 of 114** | Accuracy won every seed; declared gate failed because FP was 1 vs direct 0 every seed |
| Candidate V1 direct Q4 host, seeds 17/29/43 | 119 / 124 / 102 of 203 | 30 / 35 / 13 of 114 | Android-profile host diagnostics; duplicate-BOS mode, not phone runtime |
| Old broad hybrid | 168/203 | 80/114 | Unsafe diagnostic; corrupts correct selections under stress |
| Historical Gemma Q4 reference | 175/203 | Not recorded | Useful quality bar; nonidentical runtime |

The full current-run record is
[PocketFinancer Android-aligned LFM2.5-350M run](POCKETFINANCER_A9_LORA_R16_S17.md).
Its 203 rows are the reused regression set, not a fresh test. Grammar-on produced
exactly the same Q4 output strings as grammar-off. The single-BOS direction is
interesting but not statistically decisive and needs Android JNI verification.

The [completed 2.6B diagnostic](POCKETFINANCER_LFM25_2_6B_R16_S17.md) and its
[machine-readable record](../../configs/experiments/lfm2.5-2.6b-base-r16-s17.json)
catalog the newer aggregate evidence. The untouched Base's 186/203 is local
quality-ceiling evidence on the reused fixture only. Rank-16 LoRA reached 184/203
and did not establish a reliable gain over Base (`p=0.7265625`). Single-BOS BF16
and Q8 preserved the LoRA HF app-exact result and matched one another exactly;
Q4 fell to 166/203 under Android-current BOS handling and recovered to 173/203
with single BOS. Quantization and correct BOS handling remain unresolved for
shipping. No result establishes custom-JNI runtime, device, or deployment parity.

The direct clean result has 94 true positives, 20 transaction misses, no ghosts,
and 202/203 valid/schema-valid outputs under its saved experimental evaluator. The
candidate result has 102 true positives, 12 misses, one ghost, and 203/203 valid
outputs. All 103 emitted candidate transactions are reconstructed from source-backed
values; the main remaining failure is choosing the wrong counterparty span.

The newer [Candidate Protocol V1 controlled run](POCKETFINANCER_LFM25_350M_CANDIDATE_PROTOCOL_V1.md)
is a separate three-seed experiment on a fresh 152/29 paired materialization. It
uses compact candidate IDs, deterministic reconstruction, a versioned byte-level
wire contract, and a strict trusted-evidence comparator. Selector transaction
exact improved by 47, 15, and 53 rows, with 100% strict-schema acceptance and
source grounding, but the selector introduced one false transaction on every
seed while direct introduced none. Its preregistered HF gate therefore failed.
This does not supersede the older 154/203 candidate result or authorize product
promotion; both used the reused 203-row diagnostic rather than fresh human gold.

## Android parity chronology

Earlier reports called the 140/203 path "exact Android." That label is superseded.
The prompt text, seven demonstrations, generic outer system role, and message role
placement match the Android assets. The rest of the saved evaluation contract does
not match the current app:

| Concern | Saved selection experiment | Android at `a6c8a11` |
|---|---|---|
| Prefilter | Six-stage experimental filter | None |
| Context | 3,072 | 1,024 |
| Generation | Single pass | Forced 1,024-token think pass + answer pass |
| Grammar | Disabled by default | Always enabled for answer |
| Answer cap | 256 | 256 |
| Runtime | HF / llama-cpp-python desktop | CPU-only custom Kotlin/JNI llama.cpp `b9198` |
| Model | LFM2.5-350M | No LFM tier; reachable branches select Qwen |

At `a6c8a11`, the 1,862-2,017-token LFM2.5-350M prompt could not fit the
1,024-token context even before thinking or answering. That finding explains why
the original "exact Android" label was invalid at the time.

Android `a9b7df44` later added the same six-stage prefilter, raised context to 3,072,
made thinking model-dependent, and made grammar optional/default-off. The locked
LFM2.5-350M prompt plus its direct 256-token answer budget now fits. This resolves
the saved experiment's high-level configuration mismatch, but it does not
retroactively rerun old outputs through the current built-in GGUF template, Kotlin
parser, custom JNI runtime, or phone hardware.

Consequently, none of the saved LFM scores is yet a current-app/device result. New
runs must use `configs/contracts/pocketfinancer-android-current.json` and the unified
pipeline. The [Android runtime audit](../architecture/ANDROID_RUNTIME_AUDIT.md)
remains a historical `a6c8a11` snapshot.

Candidate Protocol V1 is an experimental research contract layered on this
baseline. Neither Android nor iOS currently implements its candidate enumeration,
tiny-ID selector parser, or deterministic reconstruction path. Its HF comparison
and converted GGUF files are not mobile-runtime parity evidence.

## Decision

- **2.6B untouched Base:** use only as controlled, reused-regression
  quality-ceiling evidence; it is not a production benchmark or deployment choice.
- **2.6B rank-16 LoRA:** do not promote; one silver-data seed showed no reliable
  improvement over untouched Base.
- **2.6B BF16/Q8:** identical single-BOS host predictions are useful conversion
  parity evidence, not Android runtime or device evidence.
- **2.6B Q4/BOS:** the Q4 loss and Android-current duplicate-BOS behavior are
  unresolved shipping blockers; do not ship.
- **350M direct artifact:** continue research; do not promote.
- **Latest direct Q4:** operational pipeline artifact, but only 31/114 exact
  transactions under current tokenization; do not ship.
- **350M candidate architecture:** most promising clean direction; requires Android
  candidate extraction, selector parsing, and deterministic reconstruction.
- **Candidate Protocol V1 controlled gate:** reject promotion despite large
  transaction-exact gains; the selector had one false transaction on every seed
  versus zero for direct. Diagnose that repeatable safety regression and rerun the
  full controlled matrix before reconsidering.
- **168/203 broad hybrid:** reject for deployment.
- **Private v2 and 251/54 candidate data:** invalidated; never reuse for selection.
- **Production claim:** blocked on fresh human-gold data and an aligned Android
  runtime/device benchmark.
