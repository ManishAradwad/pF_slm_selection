# Workbench Requirements and Data Flow

Status: **active and implemented**
Entry point: `python scripts/run_sms_processing.py serve-workbench`

## Private layout

```text
PRIVATE_DATA/sms_processing/
  .source-id-key
  CURRENT.json
  runs/<immutable-run-id>/
    canonical_manifest.jsonl
    deterministic_analysis.jsonl
    weak_operational_segregation.jsonl
    grouping.jsonl
    pool_assignments.jsonl
    legacy_review_asset.jsonl
    annotation_queues/*.jsonl
    reports/*.json
    provenance.json
  workbench/
    workbench.sqlite3
    backups/*.sqlite3
    backups/*.manifest.json
    exports/<hash-bound-export-id>/
```

Directories are mode `0700`; files are `0600`. The entire root is Git-ignored and
guarded against force-add. Queue files contain source IDs and reasons only and are
views over the canonical manifest, never independent datasets.

## Local service boundary

The browser server binds only to `127.0.0.1`, emits no request log, uses a random
per-run token, rejects foreign origins, sets a self-only content security policy,
and serves checked-in HTML/CSS/JavaScript with no remote assets. Private UI
screenshots are prohibited; browser smoke tests use invented rows.

## Screens and workflow

The unified screen provides:

- aggregate progress, queue counts, field/core candidate coverage, and
  pool/class/family/rail coverage;
- local search, filters, sorting, paging, and message skimming;
- pool, sender, normalized-template, sender-template group, date/month, weak
  class/event-state/family/rail, disposition, selector action, and review-state
  navigation;
- exact source message selection for amount, direction, account, and counterparty;
- analyzer clauses, cues, reason codes, exact candidate IDs/evidence, and queue
  rationale;
- canonical decisions including ambiguous and multiple-event;
- absent/unknown optional fields, uncertainty, notes, family, and rail;
- drafts, submission, revision, adjudication, weak-segregation correction,
  Candidate Selector preview, backups, and exports.

Weak corrections are append-only records separate from source and human truth.
Validation errors identify the missing or inconsistent field and do not change a
label into `none`.

## Blind-first protected review

For `protected_test` and `later_time_holdout`:

1. The reviewer can see the original source message but not deterministic
   analysis, weak facets, candidate prefill, queue reasons, or prior labels.
2. Drafts can be saved without lifting blindness.
3. A complete initial canonical decision must be submitted.
4. The reviewer explicitly chooses reveal.
5. Only then are deterministic suggestions, group context, prior weak corrections,
   disagreement state, and adjudication context shown.

Filtering protected pools by hidden weak facets is rejected, and list rows have
those facets removed until the reviewer reveals them.

## Persistence model

SQLite runs in WAL mode with full synchronous durability, foreign keys, a busy
timeout, and immediate write transactions. Annotation revisions and weak
corrections are hash-chained and append-only. Each save uses an expected revision;
stale clients receive a conflict and must reload.

Backups use SQLite's consistent backup API, mode `0600`, an integrity check, and a
SHA-256 sidecar bound to schema version and corpus run. Recovery verifies the
expected hash, schema, corpus-run binding, revision chains, and SQLite integrity;
it then removes stale sidecars and atomically replaces the database. Submitted or
adjudicated exports revalidate revision chains, are deterministically sorted,
bound to the immutable corpus run and revision hashes, and addressed by content
hash.

Adjudication requires two submitted labels whose canonical content disagrees. The
resolution stores both source revision hashes and creates a new append-only
adjudicated revision rather than overwriting either review.

## Candidate-oracle feedback

Target preview revalidates the submitted rich label against the source and active
analysis. A valid single event displays only the compact ID target. Missing or
mismatched fields return one aggregate-safe projection reason, allowing the
reviewer to distinguish annotation error from analyzer candidate-coverage error.

The workbench currently holds the complete 17,830-row canonical run and has a
verified initial backup. No final train/dev/test SFT target exists.
