# LFM2.5-350M PocketFinancer experiment report

Date: 2026-08-04
Status: local pre-release execution complete; public candidate unreleased; human review pending

> **Historical scope correction (2026-08-05):** this report's later phrase
> "production prompt" refers only to the experiment repository's short prompt. It
> is not the long seven-shot prompt/runtime contract used by PocketFinancer Android.
> The artifact documented here was neither trained nor evaluated with that exact app
> contract. See `docs/LFM25_350M_PIPELINE_V2.md` for the corrected contract audit and
> subsequent exact-prompt/private-data experiments. For the completed current
> Android-profile run, use
> `docs/experiments/POCKETFINANCER_A9_LORA_R16_S17.md`.

> **Repository cleanup note (2026-08-07):** the generated
> `error_analysis.txt` and `results_analysis.ipynb` artifacts described below
> have since been removed from the current tree. Their original commits remain
> available in Git history. The repository-safety exception now covers only the
> frozen `DATA/extraction_ds.jsonl` regression fixture.

## Decision

**Reject the current trained LFM2.5-350M artifact as PocketFinancer's default
extractor.**

The best evaluated quantized artifact, Q8_0, reached 58/203 (28.6%) four-field
exact match on the locked regression set. It is below the always-null baseline at
89/203 (43.8%) and far below the existing Gemma references: 175/203 (86.2%) for Q4
and 176/203 (86.7%) for Q8. Q8_0 recovered some transaction cases, but still
produced 39/89 ghosts and only 8/114 transaction-exact outputs. Its small size and
speed do not offset that quality gap.

This rejects the current default-model candidate, not the architecture forever. A
future attempt needs a sufficiently large, balanced, policy-valid private training
set and a newly adjudicated, sender/template-held-out primary test. No production
claim is made from the synthetic dev set or the repeatedly tuned 203-row regression
set. Android compatibility and the model-license implications also remain unverified.

## Boundaries observed

- All work ran natively in WSL2 at `/home/tojinotzenin/pF_slm_selection`; Docker was
  not used.
- The instruction checkpoint `LiquidAI/LFM2.5-350M` is pinned locally at revision
  `36aa424c15e1bd69acab3380c0854b3d188e1036`.
- Official GGUF baselines are pinned at revision
  `bb7ee58b243e4cede04187e323e760b04f8a0091`.
- No SMS or transformed SMS was sent to hosted inference, remote labeling, W&B, or
  another third party.
- During this work, no new data, adapter, merged weights, or GGUF was uploaded, pushed, published, or
  released.
- Newly generated raw/private data, public candidates, results, models, upstream
  sources, and training artifacts remain ignored by Git. The pre-existing tracked
  `DATA/extraction_ds.jsonl` regression corpus is an explicit exact-path exception.
- At the time of this run, pre-existing tracked `error_analysis.txt` and
  `results_analysis.ipynb` were left untouched. They contained legacy
  private/regression examples and model-output material outside the
  public-candidate package. The later cleanup note above records their removal
  from the current tree without rewriting this experiment-time state.
- The public candidate remains `unreleased`; all 120 rows remain
  `manual_review=pending`; neither a release nor a dataset-license decision exists.

Base-model, tokenizer, license-text, and official-GGUF hashes are recorded in
`configs/lfm25/model.lock.json`. The llama.cpp commit, source tree, source files, and
binaries are pinned in `configs/lfm25/llama_cpp.lock.json`. Trained-GGUF hashes are in
the conversion manifest and direct evaluator provenance.

## Runtime and model setup

- Native WSL2 Ubuntu 24.04, Python 3.11.14.
- NVIDIA RTX 4070, 12,282 MiB reported VRAM.
- PyTorch 2.6.0+cu124, Transformers 5.6.2, TRL 1.9.2, PEFT 0.20.0.
- Model architecture: 354,483,968 base parameters.
- Rank-16 adapter: 5,996,544 trainable parameters, 1.6635% of the PEFT model.
- Adapter targets: `in_proj`, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `w1`, `w2`,
  and `w3`, covering convolution, attention, projection, and MLP linears.
- A real BF16 completion-only backward pass produced finite, nonzero gradients in
  all 148 gradient tensors, with loss 3.692791 and peak allocated VRAM 1,954.3 MiB.
  The persisted verifier includes direct model and code hashes.

The production prompt is short and contains no few-shot example. Training uses the
official chat template and assistant-completion-only loss. Outputs are literal JSON
`null` or an object containing exactly `amount`, `counterparty`, `type`, and
`account`.

## Data boundary

### Private annotation and review track

The complete export contains 17,830 messages: 17,584 incoming rows were eligible for
preparation and 246 outgoing rows were excluded before decontamination. The pipeline
then excluded 3,509 incoming rows related to the frozen regression corpus:

- 366 exact relatives;
- 2,566 normalized-template relatives;
- 577 configurable near relatives.

The remaining 14,075 rows were assigned as atomic sender/template components:

| Split | Rows | Distinct senders | Assignment components |
|---|---:|---:|---:|
| Train | 11,260 | 1,549 | 937 |
| Dev | 1,379 | 565 | 416 |
| Test | 1,436 | 379 | 258 |

There is zero sender overlap, zero sender component crossing, and zero normalized
template-family crossing between splits. All timestamps parse, and component time is
used in deterministic assignment. Strict row-level chronological boundaries are not
claimed: under the chosen component-contiguous assignment, long-lived connected
sender/template components span both nominal boundaries. This result does not prove
that every possible component assignment must overlap. The observed overlap is
recorded explicitly in `PRIVATE_DATA/lfm25/validation_report.json`.

Heuristics proposed 358 transactions and 13,717 nulls, all marked silver rather than
gold. The review queue contains 12,639 pending train/dev rows and 1,436 required-human
test rows. Test rows were never shown to the local proposal models, and only reviewed
human gold may be materialized from test. The review queue itself is not a blinded
adjudication view: it retains heuristic metadata for prioritization and must not be
described as reviewer-blind.

The local proposal stage selected all 326 heuristic train/dev positives plus 58
deterministically stratified hard negatives (344 train, 40 dev). Three pinned local
GGUFs supplied grammar-constrained greedy proposals: Gemma-4 E2B Q4, Gemma-4 E2B Q8,
and Qwen3-1.7B Q8. Format confidence was 0.99 only for schema-valid deterministic
decoding; semantic acceptance still required three valid proposals, at least two
exactly agreeing models, and at least two independent model families. Test rows were
never proposed, and raw reasoning/output was not retained.

Final proposal and materialization outcome:

- proposals: **1,152**, schema-valid: **1,152/1,152 (100%)**;
- consensus statuses: **128 accepted, 33 insufficient agreement, 223 insufficient model-family diversity**;
- accepted consensus labels: **128 total (81 transaction, 47 null)**;
- materialized train/dev/test: **101/27/0**;
- materialized label balance: **train 78 transaction/23 null; dev 3 transaction/24 null; test 0/0**.

The optional supplemental mixed-data pilot was **skipped**. Although the proposal,
schema-validity, train-size, and dev-size gates passed, the accepted dev split had
only 3 transaction examples, below the predeclared minimum of 6; using it to select
a training result would be underpowered.

### Programmatic synthetic SFT track

The completed experiment ladder used a separate programmatic pool, without private
row derivation:

- 350 local-only train rows across 14 template families;
- 100 dev rows across four entirely held-out families;
- held-out families: bill due, failed payment, merchant refund, and utility bill;
- zero train/dev template-family overlap;
- label source: programmatic synthetic, not human gold and not consensus silver.

The train/dev SHA-256 values are
`28fad00deca4633a70724946815c83c2905a9ada710da4948a6c1e2ee7777369` and
`3458507091c5b875f20008db51bd3c4f5f3552c9839f940a741e4e66e0fb8771`.
All 450 rows were compared in memory against the complete 17,830-message export plus
the 203 regression rows: 18,033 documents and 13,775 unique normalized documents.
No row was rejected or rewritten; maximum accepted similarity was 0.517375 against a
0.72 threshold, and maximum four-word Jaccard was 0.05. No private text, private hash,
or candidate-to-private mapping was persisted.

## Baselines on the locked regression set

The 203 rows contain 114 transactions and 89 gold-null cases. They are a
compatibility/regression benchmark only because earlier model and prompt work tuned
against them. The generated local summaries fingerprint their model/sample inputs,
dataset, decode/filter settings, scorer, and relevant code as applicable.

| Artifact / prompt | Exact | Txn exact | Ghost | Miss | JSON valid |
|---|---:|---:|---:|---:|---:|
| Always null | 89/203 (43.8%) | 0.0% | 0.0% | 100.0% | 100.0% |
| Untouched official Q4, prior repository prompt | 10/203 (4.9%) | 5.3% | 95.5% | 0.9% | 99.5% |
| Untouched official Q8, prior repository prompt | 27/203 (13.3%) | 22.8% | 98.9% | 0.0% | 100.0% |
| Untouched HF BF16, short prompt | 8/203 (3.9%) | 5.3% | 88.8% | 8.8% | 91.6% |
| Untouched official Q8, short prompt | 13/203 (6.4%) | 6.1% | 93.3% | 7.0% | 100.0% |
| Untouched official Q4, short prompt | 52/203 (25.6%) | 2.6% | 44.9% | 56.1% | 100.0% |
| Gemma-4 E2B Q4 reference | 175/203 (86.2%) | — | 19.1% | 0.0% | — |
| Gemma-4 E2B Q8 reference | 176/203 (86.7%) | — | 16.9% | 0.0% | — |

Ghost rate is conditional on the 89 gold-null rows; miss rate is conditional on the
114 transactions. Null/null rows do not inflate transaction-field accuracy.

## LoRA comparison and seed variation

All runs used BF16 LoRA, completion-only loss, gradient checkpointing, effective
batch size 32, a maximum of six epochs, epoch evaluation/early stopping, cosine
scheduling, and deterministic seeds. The higher-learning-rate run stopped after four
epochs; six was a cap, not a claim that every run completed six.

| Rank | LR | Seed | Epochs | Dev exact | Dev txn exact | Ghost | Miss | Eval loss | Peak train VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1e-4 | 17 | 6 | 70.0% | 40.0% | 0.0% | 0.0% | 0.045850 | 3,116 MiB |
| 16 | 1e-4 | 17 | 6 | 73.0% | 46.0% | 0.0% | 0.0% | 0.044414 | 3,156 MiB |
| 16 | 2e-4 | 17 | 4 | 66.0% | 32.0% | 0.0% | 0.0% | 0.051417 | 3,156 MiB |
| 16 | 1e-4 | 29 | 6 | 73.0% | 46.0% | 0.0% | 0.0% | **0.043721** | 3,156 MiB |
| 16 | 1e-4 | 41 | 6 | 73.0% | 46.0% | 0.0% | 0.0% | 0.044729 | 3,156 MiB |

Rank 16 / 1e-4 won the rank/LR comparison. All three finalist seeds tied on
headline quality; seed 29 won the predefined lower-eval-loss tie-break. Full BF16
fine-tuning was not justified, and the measured memory headroom did not justify
QLoRA.

The synthetic dev result mainly demonstrates formatting and null/transaction
classification. For the selected HF adapter, account, counterparty, and type were
correct on all 50 transaction rows, but amount was correct on only 46%. The two
hard-negative families were stronger: bill-due and failed-payment each scored 25/25
with zero ghosts for BF16, Q8_0, and Q4_K_M.

## Merge, conversion, and parity

The seed-29 adapter was merged into the pinned BF16 base. Adapter and merged HF each
scored 73/100 on dev with no correctness flips. Three already-incorrect parsed
outputs differed, so this is quality parity rather than byte-identical decoding. An
exact merge rerun preserved the merge-manifest SHA-256
`40e2f03d6285bac803a928973a7bcf326458d7de255615522a0af80ad43d0b3c`.

The official `ggml-org/llama.cpp` checkout is pinned to commit
`fe2adf0e722f30f5295fdec8a0f1dc788f7498bc` and tree
`3d5eccf08dbfe6edab1939314964475a603ab2de`. A native CUDA 12.8 / sm_89 build
produced and hash-verified `llama-cli`, `llama-quantize`, and `llama-bench`. Pushes
from the ignored upstream checkout are disabled.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| BF16 | 711,484,512 | `a3c65f67f397b21006499b5ba63965890376868ff78e241d3c71d98ec2360b32` |
| Q8_0 | 379,216,992 | `fa7a7a5bae447fcf0411a9ceefca60f430e7e41285b1b6f823568bbe847e3798` |
| Q4_K_M | 229,311,584 | `bdaccf7d6c8ac299df671984c598ef86ef00ddedc9ad2628e879ab6376f9241c` |

All three files passed GGUF-magic integrity checks. The conversion script performed
its suppressed CUDA load/one-token smoke on Q4_K_M only; later full evaluations
independently loaded all three variants. The conversion manifest enforces the exact
toolchain and artifact hashes on reuse. An offline rerun verified and retained all
three existing GGUFs without overwriting them and repeated the Q4_K_M smoke check.

HF-to-BF16-GGUF parity used a common unconstrained greedy path on both sides:

- dev: merged HF 73/100, BF16 GGUF 74/100, one BF16-only correct;
- regression: merged HF 56/203, BF16 GGUF 57/203, one BF16-only correct.

The BF16-to-Q8/Q4 comparisons then used the common grammar-constrained evaluator.
Against BF16, Q8 lost two exact dev rows and gained one net regression row. Q4 had
six BF16-only versus three Q4-only dev rows, and 12 BF16-only versus five Q4-only
regression rows. The paired regression Q4 comparison was not statistically decisive
at this sample size (exact McNemar p=0.1435), but every variant remains far below the
quality bar.

## Quality, size, and latency frontier

All final rows used the same short prompt, tokenizer/chat template, four-field
grammar, 96-token cap, filter, and scorer. Confidence intervals are Wilson 95% CIs.

| Variant | Exact (95% CI) | Txn exact | Ghost (95% CI) | Miss (95% CI) | Precision / recall / F1 | JSON / schema |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 28.1% (22.3–34.6) | 7.0% | 44.9% (35.0–55.3) | 21.1% (14.6–29.4) | 69.2 / 78.9 / 73.8% | 100 / 100% |
| Q8_0 | **28.6% (22.8–35.1)** | 7.0% | **43.8% (34.0–54.2)** | 22.8% (16.1–31.3) | 69.3 / 77.2 / 73.0% | 100 / 100% |
| Q4_K_M | 24.6% (19.2–31.0) | 5.3% | 50.6% (40.4–60.7) | **8.8% (4.8–15.4)** | 69.8 / 91.2 / 79.1% | 100 / 100% |

| Variant | Account | Amount | Counterparty | Type |
|---|---:|---:|---:|---:|
| BF16 | 44.7% | 63.2% | 11.4% | 71.1% |
| Q8_0 | 49.1% | 60.5% | 11.4% | 69.3% |
| Q4_K_M | 53.5% | 78.9% | 15.8% | 64.0% |

Counterparty extraction is the clearest shared failure. Q4's higher recall and some
higher marginal field scores do not recover joint exactness and come with more
ghosts.

End-to-end timing is per SMS on the RTX 4070; one warmup row was excluded. True
prompt/decode throughput is a separate five-repetition `llama-bench` measurement at
214 prompt and 32 generated tokens. The benchmark JSON is schema v2 and stores
aggregate statistics only. It hashes the GGUFs, benchmark binary, lockfile, and
benchmark script alongside the exact argv, pinned source commit/tree, workload,
and runtime identifiers.

| Artifact | File MiB | Dev exact | Regression exact | Output tokens mean/max | p50 / p95 ms | Load s | GPU delta MiB | RSS MiB | PP tok/s | TG tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 678.5 | 74.0% | 57/203 (28.1%) | 21.47 / 42 | 270.2 / 434.7 | 0.601 | 1,017 | 1,556.3 | 30,503.239 ± 12,295.659 | 452.802 ± 8.877 |
| Q8_0 | 361.6 | 72.0% | **58/203 (28.6%)** | 20.80 / 42 | 262.4 / 396.5 | 0.336 | 715 | 1,285.4 | 44,231.006 ± 12,394.233 | 639.132 ± 10.074 |
| Q4_K_M | **218.7** | 71.0% | 50/203 (24.6%) | 23.53 / 44 | **261.5 / 342.1** | **0.307** | **573** | **1,281.0** | 42,401.367 ± 13,072.779 | 694.067 ± 34.286 |

Q8 is the best observed quality point; Q4 is the size, load-time, latency, and
decode-throughput point. Neither is production-viable. Cold-load values vary with OS
caching; separate dev processes observed 1.0601 s BF16, 0.6620 s Q8, and 0.4865 s Q4.

## Privacy and unreleased public candidate

The release candidate contains 120 fully programmatic rows across 18 classes and 18
template families. Its authoritative full-corpus audit compared every row against
the complete export plus regression set: 18,033 documents and 13,775 unique
normalized documents.

The ignored local package includes the complete candidate, a safe preview, the audit
report, dataset card, memorization manifest, and a non-conclusive license/data-rights
review note.

- 120 accepted, zero rejected, zero rewritten;
- zero normalized exact private matches;
- maximum accepted similarity 0.498024 at threshold 0.72;
- maximum four-word n-gram Jaccard 0.048780;
- zero blocked PII, quasi-ID, URL, email, VPA, phone, reference, account, date,
  location, or secret-like findings;
- all 120 rows remain pending manual review;
- release and license decisions remain `not_made`.

Aggregate-only model probes found:

- 64 private-prefix continuations sampled in memory;
- zero verbatim next-three-word matches;
- zero shared four-word n-gram matches and maximum four-word Jaccard 0;
- maximum withheld-suffix sequence ratio 0.512, mean 0.159966;
- lower-completion-loss membership AUC 0.653771;
- synthetic-train mean loss 0.021497 versus unseen-template-dev mean 0.028256;
- no private text, private hash, or generated completion persisted.

The membership AUC requires human interpretation. It compares repetitive synthetic
training templates with wholly unseen synthetic dev templates, so distribution shift
is confounded with membership. It is not an estimate of membership in the private SMS
archive.

The model carries the pinned LFM Open License v1.0 text. A qualified legal reviewer
must assess its conditions before any product or redistribution decision. No release
license has been selected for the dataset candidate, and this report makes no legal
conclusion.

## Android deployment limitation

No PocketFinancer Android/llama.rn repository or target device is present in this
workspace. App compatibility, exact chat-template/grammar integration, mobile RAM,
thermals, sustained speed, and battery impact are therefore unverified. Desktop CUDA
success must not be represented as Android validation.

## CI

`.github/workflows/ci.yml` is a deliberately small checks-only workflow:

- push, pull-request, and manual triggers;
- read-only repository permission, disabled credential persistence, concurrency
  cancellation, and a ten-minute job timeout;
- Python 3.11 with four exact lightweight CI dependency pins;
- a pathname-only tracked-file guard that rejects the enumerated database, raw-export,
  model/checkpoint/result, and patch-backup patterns plus every file under the six
  protected output trees; it is not a content scanner for legacy source/analysis
  files;
- repo-wide Ruff checks, ShellCheck for every tracked shell script, and the complete
  unit suite;
- no model download, GPU work, artifact upload, publishing, release, or deployment.

The final local mirror passed repository safety, both Ruff passes, ShellCheck, Bash
syntax, dependency consistency, `git diff --check`, and **94 tests**. The
tests also passed in an isolated environment containing only the CI runtime
dependencies.

## Exact reproduction commands

Run from the already prepared native WSL workspace described by the goal brief.
These offline commands intentionally do not acquire gated/licensed model or tokenizer
assets. The pinned base checkpoint, official LFM GGUFs, proposal GGUFs, and tokenizer
snapshots must already exist at the lock/config paths; proposal, toolchain, conversion,
and evaluation provenance checks bind the assets actually used. Exact private split
reproduction also requires the existing ignored `PRIVATE_DATA/lfm25/.hash_key`; it
must remain local and must never be published. Most `--force` data/audit writes are
atomic. Proposal `--force` intentionally discards each model checkpoint before a full
recompute; omit it to resume durably from existing checkpoints.

```bash
cd /home/tojinotzenin/pF_slm_selection
source scripts/activate_wsl.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true

# Private split, three-model local proposals, validation, and policy-gated materialization.
python scripts/prepare_lfm25_private_data.py prepare --source all_sms.json --force
python scripts/prepare_lfm25_private_data.py validate --source all_sms.json
python scripts/propose_lfm25_private_labels.py --force
python scripts/prepare_lfm25_private_data.py validate --source all_sms.json
python scripts/prepare_lfm25_private_data.py materialize --force

# Separate unreleased public candidate and complete-export privacy audit.
python scripts/build_lfm25_public_candidate.py --force
python scripts/audit_lfm25_public_candidate.py \
  --private-export all_sms.json \
  --private-jsonl DATA/extraction_ds.jsonl \
  --force

# Programmatic-only SFT pool with whole held-out template families.
python scripts/build_lfm25_synthetic_sft.py \
  --count 450 \
  --seed 25052027 \
  --holdout-family merchant_refund \
  --holdout-family utility_bill \
  --holdout-family failed_payment \
  --holdout-family bill_due \
  --private-export all_sms.json \
  --private-jsonl DATA/extraction_ds.jsonl \
  --force

# Real completion-only backward verification and provenance-bound always-null baseline.
python scripts/verify_lfm25_backward.py
python scripts/score_lfm25_results.py \
  --always-null DATA/extraction_ds.jsonl \
  --dataset DATA/extraction_ds.jsonl \
  --out RESULTS/lfm25/metrics/always_null.json

# Untouched short-prompt baselines on the locked regression set.
python scripts/evaluate_lfm25_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/short_prompt_hf_bf16 \
  --seed 17
python scripts/evaluate_lfm25_gguf.py \
  --gguf MODELS/LFM2.5-350M-official/LFM2.5-350M-Q8_0.gguf \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/short_prompt_q8 \
  --seed 17
python scripts/evaluate_lfm25_gguf.py \
  --gguf MODELS/LFM2.5-350M-official/LFM2.5-350M-Q4_K_M.gguf \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/short_prompt_q4 \
  --seed 17

# Preserve provenance for the earlier repository-prompt official-GGUF samples.
python scripts/score_lfm25_results.py \
  --samples RESULTS/lfm25/untouched_q4/TRAINING_ARTIFACTS__base__LFM2.5-350M/samples_sms_extraction_2026-08-03T20-18-01.839155.jsonl \
  --samples RESULTS/lfm25/untouched_q8/TRAINING_ARTIFACTS__base__LFM2.5-350M/samples_sms_extraction_2026-08-03T20-16-48.306326.jsonl \
  --dataset DATA/extraction_ds.jsonl \
  --out RESULTS/lfm25/metrics/official_gguf_baselines.json

# Fixed LoRA rank/LR comparison and finalist seeds.
python scripts/run_lfm25_experiments.py \
  --config configs/lfm25/experiments.json \
  --train PRIVATE_DATA/lfm25/synthetic_sft_train.jsonl \
  --dev PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --artifacts-root TRAINING_ARTIFACTS/lfm25_experiments_final \
  --results-root RESULTS/lfm25/experiments_final

# Merge, verify pinned llama.cpp offline, and convert/hash all GGUFs.
python scripts/merge_lfm25_lora.py \
  --adapter TRAINING_ARTIFACTS/lfm25_experiments_final/final_r16_lr0p0001_seed29/adapter \
  --output-dir TRAINING_ARTIFACTS/lfm25_merged_seed29
bash scripts/setup_lfm25_llama_cpp.sh --offline
bash scripts/convert_lfm25_gguf.sh \
  TRAINING_ARTIFACTS/lfm25_merged_seed29 \
  TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29

# Adapter and merged-HF evaluations.
python scripts/evaluate_lfm25_hf.py \
  --model TRAINING_ARTIFACTS/base/LFM2.5-350M \
  --adapter TRAINING_ARTIFACTS/lfm25_experiments_final/final_r16_lr0p0001_seed29/adapter \
  --dataset PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --output-dir RESULTS/lfm25/experiments_final/final_r16_lr0p0001_seed29/dev \
  --seed 29
python scripts/evaluate_lfm25_hf.py \
  --model TRAINING_ARTIFACTS/lfm25_merged_seed29 \
  --dataset PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --output-dir RESULTS/lfm25/merged_seed29/dev \
  --seed 29
python scripts/evaluate_lfm25_hf.py \
  --model TRAINING_ARTIFACTS/lfm25_merged_seed29 \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/merged_seed29/regression \
  --seed 29

# Common grammar-constrained dev and regression evaluation for each GGUF.
for variant in BF16 Q8_0 Q4_K_M; do
  python scripts/evaluate_lfm25_gguf.py \
    --gguf "TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-${variant}.gguf" \
    --dataset PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
    --output-dir "RESULTS/lfm25/gguf_final/${variant}/dev" \
    --seed 29
  python scripts/evaluate_lfm25_gguf.py \
    --gguf "TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-${variant}.gguf" \
    --dataset DATA/extraction_ds.jsonl \
    --output-dir "RESULTS/lfm25/gguf_final/${variant}/regression" \
    --seed 29
done

# Unconstrained BF16 control for the HF-to-GGUF comparison.
python scripts/evaluate_lfm25_gguf.py \
  --gguf TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-BF16.gguf \
  --dataset PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --output-dir RESULTS/lfm25/gguf_final/BF16_unconstrained/dev \
  --no-grammar --seed 29
python scripts/evaluate_lfm25_gguf.py \
  --gguf TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-BF16.gguf \
  --dataset DATA/extraction_ds.jsonl \
  --output-dir RESULTS/lfm25/gguf_final/BF16_unconstrained/regression \
  --no-grammar --seed 29

# Provenance-bound paired comparisons.
python scripts/compare_lfm25_predictions.py \
  --first RESULTS/lfm25/experiments_final/final_r16_lr0p0001_seed29/dev/samples.jsonl \
  --second RESULTS/lfm25/merged_seed29/dev/samples.jsonl \
  --first-name adapter_hf --second-name merged_hf \
  --output RESULTS/lfm25/merged_seed29/adapter_merge_parity.json
python scripts/compare_lfm25_predictions.py \
  --first RESULTS/lfm25/merged_seed29/dev/samples.jsonl \
  --second RESULTS/lfm25/gguf_final/BF16_unconstrained/dev/samples.jsonl \
  --first-name merged_hf_unconstrained --second-name bf16_gguf_unconstrained \
  --output RESULTS/lfm25/gguf_final/hf_bf16_dev_parity.json
python scripts/compare_lfm25_predictions.py \
  --first RESULTS/lfm25/merged_seed29/regression/samples.jsonl \
  --second RESULTS/lfm25/gguf_final/BF16_unconstrained/regression/samples.jsonl \
  --first-name merged_hf_unconstrained --second-name bf16_gguf_unconstrained \
  --output RESULTS/lfm25/gguf_final/hf_bf16_regression_parity.json
declare -A quant_slug=([Q8_0]=q8 [Q4_K_M]=q4)
for variant in Q8_0 Q4_K_M; do
  for split in dev regression; do
    slug="${quant_slug[$variant]}"
    python scripts/compare_lfm25_predictions.py \
      --first "RESULTS/lfm25/gguf_final/BF16/${split}/samples.jsonl" \
      --second "RESULTS/lfm25/gguf_final/${variant}/${split}/samples.jsonl" \
      --first-name bf16 --second-name "${variant,,}" \
      --output "RESULTS/lfm25/gguf_final/bf16_${slug}_${split}.json"
  done
done

# True PP/TG benchmark and aggregate-only model privacy probes.
python scripts/benchmark_lfm25_gguf.py \
  --gguf BF16=TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-BF16.gguf \
  --gguf Q8_0=TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-Q8_0.gguf \
  --gguf Q4_K_M=TRAINING_ARTIFACTS/lfm25_gguf/lfm25-seed29-Q4_K_M.gguf
python scripts/probe_lfm25_memorization.py \
  --model TRAINING_ARTIFACTS/lfm25_merged_seed29 \
  --train PRIVATE_DATA/lfm25/synthetic_sft_train.jsonl \
  --dev PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --private-export all_sms.json \
  --private-jsonl DATA/extraction_ds.jsonl
```

Local pre-commit verification:

```bash
cd /home/tojinotzenin/pF_slm_selection
set -euo pipefail
ci_root="$(mktemp -d)"
trap 'rm -rf "$ci_root"' EXIT
python3.11 -m venv "$ci_root/venv"
source "$ci_root/venv/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1
python scripts/check_repo_safety.py
python -m pip install --requirement requirements-ci.txt
python -m ruff check .
python -m ruff check --select E4,E7,E9,F lfm25 scripts tests
shell_list="$ci_root/shell-files"
git ls-files -z --cached --others --exclude-standard -- '*.sh' > "$shell_list"
mapfile -d '' -t shell_files < "$shell_list"
if ((${#shell_files[@]})); then
  shellcheck --external-sources \
    --exclude=SC1090,SC1091,SC2086,SC2162,SC2317 \
    -- "${shell_files[@]}"
  bash -n -- "${shell_files[@]}"
fi
python -m pytest
git diff --check
git diff --cached --check
```

## Required human checkpoints

### Production evidence

A qualified reviewer must adjudicate all 1,436 rows in the sender/template-held-out
primary test queue before any renewed production claim. Natural prevalence must be
preserved. The current `reject` decision does not need this review to remain valid;
the review is required only to reconsider the model.

### Prospective dataset release — stop here

No release is authorized. Before any publication or redistribution:

1. manually review every one of the 120 candidate rows;
2. resolve the membership-probe interpretation;
3. complete privacy, trademark, data-rights, and legal/license review;
4. review the legacy tracked regression/analysis artifacts before any repository
   publication;
5. select a dataset license explicitly, if appropriate;
6. inspect the complete local candidate package; and
7. obtain explicit user approval in a separate decision.

Until all seven occur, the candidate and every derived artifact remain local and
unreleased.
