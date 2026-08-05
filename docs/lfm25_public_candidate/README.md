# LFM2.5 public-candidate workflow

This workflow creates a separate, local, **unreleased** candidate at
`PUBLIC_CANDIDATE/lfm25/`. It does not modify the existing evaluation dataset or use
private records as generation inputs. All messages, senders, account suffixes,
amounts, dates, references, counterparties, VPA handles, URLs, email addresses, phone
tokens, and locations come from deterministic fictional generators.

Every candidate row remains `manual_review=pending`. This workflow makes neither a
release decision nor a license decision.

## Build and audit

Run in WSL2 from the repository root:

```bash
python3 scripts/build_lfm25_public_candidate.py --force
python3 scripts/audit_lfm25_public_candidate.py \
  --private-export all_sms.json \
  --private-jsonl DATA/extraction_ds.jsonl \
  --force
```

The complete `all_sms.json` export is the authoritative archive input for this audit.
It is loaded read-only through `--private-export` using its `text` field, while
`DATA/extraction_ds.jsonl` supplies the regression examples through
`--private-jsonl` using its `sms` field. Use `--private-export-text-field` or
`--private-text-field` only for alternate sources with different field names.

SQLite input remains available through `--private-sqlite` for compatibility with
older local environments. It is not authoritative and must not replace the complete
export when reproducing this audit. Inputs are retained in memory only while
comparing texts. The script does not print private messages, copy private rows, make
candidate-to-private mappings, or persist private identifiers or hashes. It fails
closed before writing if `PUBLIC_CANDIDATE/lfm25/` is not ignored by Git.

## Candidate contract

`expected` is a JSON string with exactly four keys:

- `amount`
- `counterparty`
- `type`
- `account`

Transactional examples have populated values and a `debit` or `credit` type.
Hard-negative examples use the literal JSON value `null`. Public IDs are newly
generated UUID4-shaped identifiers and do not encode row contents or any
private record identity.

## Audit gates

The auditor applies:

- exact schema and four-field label validation;
- duplicate public-ID and normalized-message checks;
- normalized exact-match, word n-gram Jaccard, and near-duplicate sequence checks
  against local private texts;
- deterministic label-preserving rewrites followed by rejection if similarity remains
  at or above the configured threshold;
- aggregate-only scans for PII, quasi-identifiers, URLs, emails, VPA handles, phones,
  references, accounts, dates, locations, and secret-like strings;
- narrow allow rules for generator-owned reserved tokens such as `.example` URLs,
  `@synthupi` handles, `SYNREF-` references, future synthetic dates, and reserved
  account suffixes;
- class, template-family, sender, rejection, and reviewer-status coverage reports.

Allow rules apply only when the row has the exact generator-owned provenance marker.
Unexpected realistic-looking matches are rejected.

## Aggregate model memorization probe

The aggregate-only model memorization probe has been completed for the current local
candidate. Reproduce it with the same authoritative private inputs:

```bash
python3 scripts/probe_lfm25_memorization.py \
  --model TRAINING_ARTIFACTS/lfm25_merged_seed29 \
  --train PRIVATE_DATA/lfm25/synthetic_sft_train.jsonl \
  --dev PRIVATE_DATA/lfm25/synthetic_sft_dev.jsonl \
  --private-export all_sms.json \
  --private-jsonl DATA/extraction_ds.jsonl
```

The report retains aggregate counts and loss summaries only; it does not persist
private text, private hashes, or generated completions. Reviewers must interpret the
reported lower-loss membership AUC. It compares programmatic synthetic training rows
with unseen-template synthetic development rows, so it is sensitive to distribution
shift and is not an estimate of membership in the private archive.

## Ignored local artifacts

The build step writes:

- `candidate_unreviewed.jsonl`
- `generation_manifest.json`

The audit step writes only accepted/re-written synthetic rows and aggregate results:

- `candidate.jsonl`
- `safe_preview.jsonl`
- `audit_report.json`
- `memorization_probe_manifest.json`
- `dataset_card.md`
- `license_data_rights_review.md`

Rejected row texts are not written. The memorization manifest records the completed
data-similarity and aggregate-only model probes, including the membership AUC that
still requires human interpretation. Complete the review checklist before
considering any external use.
