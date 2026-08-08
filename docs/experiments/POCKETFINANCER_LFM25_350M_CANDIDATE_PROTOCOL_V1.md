# PocketFinancer LFM2.5-350M Candidate Protocol V1 controlled run

Run ID: `lfm2.5-350m__candidate-protocol-v1-vs-direct__r16-s17-s29-s43__data-ff27b95d`

Date: 2026-08-08

Android source: `a9b7df44be2183daac3a05cadbfd40b8f309cd4b`

Status: executed controlled HF research diagnostic; rejected by the declared
false-positive safety gate; not approved for runtime support or deployment.

## Conclusion

Candidate Protocol V1 made the 350M model substantially better at the extraction
task on every controlled seed. Transaction-exact results improved from 13 to 60,
47 to 62, and 6 to 59 out of 114. Every accepted selector transaction was strict
schema-valid and reconstructed only from source-backed candidates.

The experiment still **failed** its preregistered gate. The direct arm produced no
false transaction on any seed, while the selector produced one on every seed.
The gate required selector false positives to be no greater than direct for every
seed. This single failed criterion is enough to reject promotion; the accuracy
gain does not override it.

The correct product decision is therefore to retain Candidate Protocol V1 as a
promising research architecture, investigate the per-seed null-to-transaction
failure mode, and require a new controlled run before promotion. No Android or
iOS app implements this protocol today.

## Evidence boundary

The evaluation used the locked `DATA/extraction_ds.jsonl` regression fixture:

- 203 rows: 114 transactions and 89 nulls;
- SHA-256 `fec483b11cf458212b6a636f508632649790beacc91050efdd52abb2b590d44e`;
- never used for training in this experiment;
- repeatedly consulted during development, so it is not a fresh test;
- not a substitute for a sender/template-held-out human-gold benchmark.

The Android prefilter passed all 114 transactions and one null to the model: 115
model invocations and 88 deterministic rejections. The frozen 1,436-row blind
package remains entirely unadjudicated and contributed no score here.

## Protocol under test

The direct arm retained the current Android four-field output. Candidate Protocol
V1 moved deterministic work to the host:

1. enumerate source-backed amount, account, and counterparty candidates;
2. give the model compact candidate IDs;
3. require `{"transaction":0}` for a non-transaction or the fixed five-member
   selector object for a transaction;
4. reject malformed, unknown, reordered, duplicate, or ungrounded output;
5. reconstruct the canonical transaction locally and use the SMS timestamp as the
   transaction date.

The protocol fixes canonical UTF-8 request bytes, candidate ordering and IDs,
exact decimal handling, portable integer bounds, JSON member order and escaping,
and fail-closed parsing. Sixteen invented synthetic golden vectors cover the wire
contract. They contain no private messages.

## Candidate data

Both arms used the same fresh app-aligned materialization and split:

| Split | Rows | SHA-256 |
|---|---:|---|
| Train | 152 | `ff27b95d93a807a943d845af68804e214441e10152cc97088ed5d6893dd50bfc` |
| Dev | 29 | `5290db7d463b339c1daf402aea875abc8686c50215e09e6a334d5a3f8e095bf9` |

The builder excluded two train records that could not be represented under the
strict candidate oracle. Train/dev record, sender, and template overlap are all
zero. These labels are private silver data, not independent human gold. The
direct target and selector target were materialized from the same record before
training; no historical candidate curriculum or invalidated 251/54 data was used.

## Controlled training design

The arms shared the locked `LiquidAI/LFM2.5-350M` revision
`36aa424c15e1bd69acab3380c0854b3d188e1036`, seeds 17/29/43, rank 16, alpha 32,
dropout 0.05, learning rate `1e-4`, six requested epochs, effective batch 32,
completion-only per-example loss, and first decision-token weight 3. The direct
arm used batch 2, accumulation 16, and max length 2,304; the selector used batch
8, accumulation 4, and max length 1,024.

Every saved adapter is bound to its model lock, data/report hashes, trainer code,
selected checkpoint, and adapter-file tree. Training restored the best checkpoint
before saving:

| Arm | Seed | Best step | Best epoch | Eval loss | Final step/epoch |
|---|---:|---:|---:|---:|---:|
| Direct | 17 | 10 | 2 | 0.204779 | 20 / 4 |
| Direct | 29 | 10 | 2 | 0.196840 | 20 / 4 |
| Direct | 43 | 10 | 2 | 0.205212 | 20 / 4 |
| Selector | 17 | 20 | 4 | 0.006598 | 30 / 6 |
| Selector | 29 | 30 | 6 | 0.015605 | 30 / 6 |
| Selector | 43 | 20 | 4 | 0.008114 | 30 / 6 |

HF decoding was greedy with thinking disabled and repetition penalty 1.0. Direct
used `n_ctx=3072` and `max_new_tokens=256`; selector used `n_ctx=1024` and
`max_new_tokens=64`.

## Controlled HF result

| Seed | Direct whole / transaction exact | Selector whole / transaction exact | Direct FP | Selector FP | Transaction delta |
|---:|---:|---:|---:|---:|---:|
| 17 | 102/203 / 13/114 | 148/203 / 60/114 | 0 | 1 | +47 |
| 29 | 136/203 / 47/114 | 150/203 / 62/114 | 0 | 1 | +15 |
| 43 | 95/203 / 6/114 | 147/203 / 59/114 | 0 | 1 | +53 |

Untuned direct behavior was stable at 89/203 whole-pipeline exact and 0/114
transaction exact. Untuned selector inference accepted no model output; the strict
parser rejected all 115 model invocations. Those baselines are diagnostics, not
additional seeds of trained evidence.

The deterministic oracle could represent 114/114 amounts, 114/114 accounts,
113/114 counterparties, and 113/114 complete transactions. All three trained
selector runs achieved 100% strict-schema acceptance and 100% source grounding
among accepted transactions.

### Declared decision gate

| Criterion, required on every seed | Result |
|---|---|
| Selector transaction exact strictly greater than direct | Pass |
| Selector false positives no greater than direct | **Fail** |
| Strict-schema acceptance 100% | Pass |
| Accepted transactions source-grounded 100% | Pass |
| Oracle coverage floors satisfied | Pass |

The trusted report records `evidence_validated: true`,
`criteria_satisfied: false`, and `passed: false`. Its scope is exactly **HF
research evidence only**. Product promotion is always false in this report type.

## Packaging and host GGUF diagnostics

All six adapters were merged. Each merged model was converted to BF16, Q8_0, and
Q4_K_M with locked llama.cpp commit
`fe2adf0e722f30f5295fdec8a0f1dc788f7498bc`, producing 18 hash-verified GGUFs.
Every Q4 received a load-and-one-token smoke. BF16 and Q8 have hash and GGUF-magic
evidence but no separate load smoke.

Only the direct Q4 artifacts have a supported Android-profile host evaluator:

| Seed | Whole / transaction exact | FP | p50 / p95 end-to-end | Peak RSS |
|---:|---:|---:|---:|---:|
| 17 | 119/203 / 30/114 | 0 | 1016.5 / 1146.1 ms | 688.1 MiB |
| 29 | 124/203 / 35/114 | 0 | 1053.3 / 1139.3 ms | 688.4 MiB |
| 43 | 102/203 / 13/114 | 0 | 817.7 / 1102.4 ms | 688.3 MiB |

These are WSL host measurements, not Android phone timings. Selector GGUF
evaluation is not implemented; its nine selector GGUF artifacts are packaging artifacts,
not runtime evidence.

All three direct Q4 runs used the profile's `android-current` BOS mode. During
execution, the host console emitted its duplicate-leading-BOS warning. The metrics
do not persist a warning count; they record zero template-BOS removals and 115
model invocations per seed. This declared template-BOS-plus-add-special diagnostic
behavior is not file corruption, but it prevents a claim of Android token-stream
parity or isolated quantization loss. A custom-JNI trace or controlled single-BOS
A/B is still required.

## Reproducibility anchors

The aggregate comparison report is local and ignored at
`RESULTS/pocketfinancer-candidate-v1/controlled-hf-seed-matrix.json`; its SHA-256
is `6496682002b18fd0cae47441b7f61ea3941ff7a249089d826ff77e73efbccd28`.

| Anchor | SHA-256 |
|---|---|
| Model lock | `21327a6411d3e9d828d33ce3e21abe15951e6ab6c3ac6b5fda779ac582a4345f` |
| Android baseline profile | `b818397851bb03fc3365710fe8ddfe82cadfe896525d931b93dbb8a6653c9d3f` |
| Candidate V1 profile | `0b9c81c2d0e95cf6313700cd5eb6155df7894697b2db245c3c1e490e21afddef` |
| Candidate V1 golden vectors | `f9adec44390b749c3add8e10163131491729ca3fb36ddd7f4c814857a27f26f1` |
| Candidate protocol identity | `0de86c5d3676d485cb865e2a345425971e02758762c74f0777e0a51a39da448c` |
| Candidate data report | `3ec4c6d155a09537b62f40259a3ddebaeec3e7c6976c44ab7c5b37a51256d538` |

Adapted HF metrics hashes:

| Arm | Seed 17 | Seed 29 | Seed 43 |
|---|---|---|---|
| Direct | `fdb42096f14efcc4302e6e44f7e806b59445025d1a4d6acc61d6cbbb82bc8620` | `7fc02709db8027a9fdae19fd9f436f924a55df7c0c1772529e9a0ae47eb05333` | `b4f8973867d43a96906aa4f9ef3f1fe5485584d5af7a240e2ff49e6d850726f1` |
| Selector | `a61ee5e21f239bb269e7399fa9b4dfc1a003a3f19915d92901aba7d3d16062de` | `cc0b566ddfdcccadf6df328d7fd40e7a826ab4002fd5bc9f63ff780ae30e7e38` | `3b75d111c6eb1f79c813a3a527281ff5010d6974dd4887c77cc36d76b1aa680f` |

Direct Q4 artifacts and metrics:

| Seed | Q4 artifact SHA-256 | Metrics SHA-256 |
|---:|---|---|
| 17 | `d981c9a66a1098a638cf830045ee0abe52bd5eded3301684b0a7e5695d1d789f` | `327958c38632c4d072e03ee6410189086667257dee83e65e3bdb6e6f85b9e949` |
| 29 | `e6da3e41be45f29842d7787d95794dba58daa2d3ce358e06b74e8f20c30852b9` | `3676a1a274b81293647763bfa1a219be25b9d6510dc0da7c879e2547912362dd` |
| 43 | `187152dba976876f42731d496b4609dd95b76e3d4a66fea7aca7e4352d1af84c` | `ec483e058df46e494aeea0bfd70fa8783c22dc8668d5a9284b8d7c2d72856d28` |

Implementation is anchored by commits `ee4bb3d` (Candidate Protocol V1) and
`2fb143b` (sparse-counter evidence validation). The generated model, adapter,
prediction, and aggregate result files remain in ignored local roots.

## Unmet gates and decision

- Fresh sender/template-held-out human-gold evaluation: unmet.
- Selector GGUF runtime evaluation: unmet.
- Android Candidate Protocol implementation and runtime parity: unmet.
- iOS Candidate Protocol implementation and runtime parity: unmet.
- Android and iOS device latency, memory, thermal, and battery evidence: unmet.
- Privacy, data-rights, model-license, and target-device release review: not
  authorized by this experiment.

Candidate Protocol V1 is retained for research because its accuracy improvement
is large and consistent. It is not promoted because its false-positive regression
is also consistent. No artifact is approved for upload, publication, release, or
deployment.
