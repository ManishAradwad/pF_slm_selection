# LFM2.5-350M PocketFinancer Goal Brief

## Objective

Determine whether `LiquidAI/LFM2.5-350M` can become PocketFinancer's small, fast default model for extracting structured financial transactions from SMS. Build the data pipeline, establish untouched baselines, fine-tune, evaluate, quantize, benchmark, and finish with an evidence-backed `promote`, `continue`, or `reject` recommendation.

In parallel, prepare a **local, unreleased, privacy-safe candidate open dataset** for this problem. Dataset preparation is authorized; publication is not.

Do not stop after producing a plan. Continue through all locally executable phases. Ask the user only when human adjudication, target-device access, a license decision, or another consequential choice is genuinely required.

## Non-negotiable boundaries

- Work natively in WSL2 at `/home/tojinotzenin/pF_slm_selection`; do not use Docker.
- Read `TRAINING_READINESS.md` and the current dataset, prompt, grammar, filter, scorer, and evaluation code before changing anything.
- Preserve all pre-existing and uncommitted work. Never reset or overwrite unrelated changes.
- Raw SMS, private derived datasets, mappings, adapters, and checkpoints are sensitive. Keep them local and ignored by Git.
- Do not send raw or transformed SMS to hosted APIs, cloud inference, W&B, remote labeling services, or other third parties.
- Do not print raw SMS or identifiers in routine logs and reports.
- **Do not upload, publish, release, or push dataset rows or weights to Hugging Face, GitHub, or any other external destination.** Do not select or assert a release license. Stop for explicit user approval after the complete candidate package has been inspected locally.

## Runtime

Activate `scripts/activate_wsl.sh`. The prepared environment uses Python 3.11, CUDA PyTorch, Transformers, TRL, PEFT, Accelerate, bitsandbytes, and CUDA-enabled llama.cpp on an RTX 4070 12 GB. Verify CUDA and a real backward pass before a long run; install or upgrade only what is actually missing.

## Model

Use the instruction-tuned [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M), not the older LFM2-350M and not the separate Base checkpoint initially.

Primary references:

- [Official model card](https://huggingface.co/LiquidAI/LFM2.5-350M)
- [Official GGUFs](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)
- [TRL fine-tuning guide](https://docs.liquid.ai/lfm/fine-tuning/trl)
- [Dataset-format guide](https://docs.liquid.ai/lfm/fine-tuning/datasets)
- [Release post](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)

Pin and record the model revision, file hashes, license text, tokenizer, chat template, architecture, and conversion/runtime revisions. Verify exact llama.cpp and eventual llama.rn support rather than assuming family-level compatibility.

## Data inventory and evaluation boundary

- `sms.db`: 17,830 raw messages; 17,584 are incoming.
- `all_sms.csv` and `all_sms.json`: exact exports of the same archive.
- There is currently no project `sms_db.parquet`; Parquet is optional at this scale.
- `DATA/extraction_ds.jsonl`: 203 labeled examples, comprising 114 transactions and 89 null cases.

The 203-row set has been repeatedly tuned against and is selection-biased. Lock it as a regression/compatibility benchmark. Do not train on its rows or on exact, near, or normalized-template relatives if historic results will be compared. A new template-held-out, human-reviewed test set must be the primary basis for production claims.

Split raw candidates by normalized template family, sender, and time **before** labeling. Do not rely on random row splits. Persist private split manifests, hashes, template-group IDs, label provenance, confidence, and review status.

## Extraction contract

Emit exactly `null` or JSON containing only:

```json
{
  "amount": "...",
  "counterparty": "...",
  "type": "...",
  "account": "..."
}
```

Follow the repository's current canonicalization and nullability rules. Remove legacy `date` and `merchant` targets from all new training examples. Format SFT examples with the model's official chat template and compute loss only on assistant/completion tokens. Prefer a short production prompt without few-shot examples after tuning.

## Private training-data track

Treat the raw archive as an annotation pool, not ground truth. Use local heuristics and the strongest available local models to propose labels. Retain `label_source` and `label_confidence`; accept automatic silver labels only under a defined consensus policy. Never describe silver labels as human gold.

Build a human-review queue containing disagreements, low-confidence examples, final test examples, and hard negatives such as OTPs, balance alerts, offers, bill reminders, payment requests, mandates, pending/failed/declined transactions, reversals, and messages containing amounts without completed transactions. Preserve natural prevalence in test data; balanced sampling or loss weighting may be used for training.

Never train directly from `RESULTS/llamacpp/**/samples_*.jsonl`; these contain evaluation leakage, repetitions, and model errors.

## Public-dataset candidate track

Build this as a separate pipeline and output tree from private training data. Never mutate the source archive.

Prefer newly synthesized or substantially rewritten messages over lightly redacted originals. A row is not release-safe merely because obvious PII was replaced. Remove or synthesize names, phone numbers, emails, account/card suffixes, VPAs, transaction/reference IDs, URLs, exact timestamps, locations, merchant identifiers, device identifiers, and other direct or quasi-identifiers. Resample amounts and dates while preserving extraction semantics and keep every replacement consistent with its target JSON.

Do not publish raw-text hashes, reversible mappings, source database IDs, or stable identifiers that connect a candidate row to the private archive. Use fresh public IDs. Guard against rare-template and linkage re-identification. Measure n-gram/near-duplicate similarity between candidate and private text, and rewrite or reject rows that remain too close.

Run layered PII/secret scanning, duplicate and leakage checks, label-consistency validation, and memorization probes. Because the candidate will be small, require manual review of every row proposed for release. Produce:

- a local-only release-candidate dataset;
- a redacted preview safe for the user to inspect;
- a dataset card covering schema, generation/transformation process, provenance, limitations, intended and prohibited uses, class/template coverage, and known biases;
- an audit report with scan results, similarity findings, rejected-row counts, and reviewer status;
- a license and data-rights review note that makes no final legal conclusion.

The candidate remains private until the user explicitly approves the exact reviewed artifact and separately authorizes publication.

## Experiment ladder

1. Establish always-null and untouched-model baselines.
2. Evaluate untouched HF BF16 plus official Q8_0 and Q4_K_M GGUFs on the current four-field schema.
3. Start BF16 LoRA SFT with completion-only loss, measured short sequence length, gradient checkpointing, deterministic seeds, and early stopping. A reasonable first adapter is rank 16, alpha 32, dropout 0.05.
4. Inspect LFM2 modules. Target appropriate attention, projection, and MLP linear layers; do not blindly restrict adapters to Q/K/V. Record the selected modules and trainable parameter count.
5. Run a small rank/learning-rate comparison, then multiple seeds for the finalist. Use full BF16 fine-tuning only if LoRA underfits. Use QLoRA only if measured memory pressure justifies it.
6. Merge the winner and verify adapter/merged-HF parity.
7. Convert with a compatible upstream llama.cpp revision. Produce an F16 or Q8 reference GGUF and Q4_K_M; add Q5_K_M only if useful. Establish HF-to-reference parity before attributing loss to Q4.
8. Re-evaluate every deployable artifact through the same grammar, prompt, filter, and scorer.

## Metrics and baselines

Report valid JSON, four-field exact match, transaction-only exact match, conditional ghost rate over gold-null cases, conditional miss rate over transactions, transaction precision/recall/F1, per-field accuracy on transaction cases, unseen-template and hard-negative results, leakage checks, paired per-example differences, confidence intervals where useful, and seed variation. Do not let null/null rows inflate field accuracy or hide a null-policy collapse.

Current regression references:

- Gemma-4 E2B Q4: 175/203 exact, 0/114 misses, 17/89 ghosts, about 3.11 GB.
- Gemma-4 E2B Q8: 176/203 exact, 0/114 misses, 15/89 ghosts, about 5.05 GB.
- Always-null: 89/203 exact with 114/114 misses.

Historical LFM2.5-1.2B Q4 results used an obsolete schema and collapsed almost entirely to `null`; rerun current-schema comparisons rather than treating that artifact as an architecture verdict.

## Throughput and deployment

Measure model size, cold-load time, peak VRAM/RAM, prompt-processing speed, decode speed, output length, and end-to-end p50/p95 latency per SMS on the RTX 4070. Use the short tuned prompt and a measured output-token cap. Vendor throughput is context, not evidence.

Test the same chat template, grammar, and GGUF on PocketFinancer's Android/llama.rn runtime when its repository and device are available. Otherwise document that app compatibility, mobile RAM, thermals, sustained speed, and battery impact remain unverified.

## Deliverables and completion rule

Produce reproducible data-building, training, conversion, evaluation, and benchmark scripts/configs; private ignored manifests; model artifacts; complete results; a redacted experiment report; and the unreleased public-dataset candidate package and audit.

Conclude with:

- the best quality/size/latency frontier;
- an honest `promote`, `continue`, or `reject` decision;
- remaining data or device limitations;
- exact reproduction commands;
- a clearly separated user-review checkpoint for any prospective dataset release.

No publication action is part of this goal.
