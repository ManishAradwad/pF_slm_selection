# Workbench Requirements and Data Flow

Status: **implemented v1 foundation; canonical-label/2 focused mode planned**
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

## Current implementation boundary

The current browser form writes `pocketfinancer.canonical-label/1`, exposes the
older five-way decision UI, and previews compact Candidate Selector targets.
Those behaviors are historical implementation facts, not the target annotation
contract for the SLM-primary direct extractor.

The secure store, canonical corpus import, pool and category navigation,
blind-first controls, source-span selection, append-only revisions,
adjudication, backup, restore, and encrypted export remain reusable. The focused
mode must add a versioned `canonical-label/2` path, preserve v1 revisions
read-only, and project approved v2 truth directly to the extractor without
requiring analyzer candidate IDs.

## Focused personal-corpus annotation mode

The existing unified workbench proves the storage, privacy, queue, revision, and
labeling foundations. The planned focused mode optimizes repeated annotation of
the owner's personal SMS corpus without creating a second database or label
format.

The focused mode must provide:

- a visible annotation-contract identity and an explicit v2 editing path; legacy
  v1 revisions remain readable but cannot be silently edited as v2;
- queue entry from pool, weak category, family, rail, sender/template group,
  review state, disagreement, candidate coverage, or imported-feedback views;
- one complete immutable SMS at a time, with clear position and remaining count;
- keyboard-first `posted`, `none`, and `abstain` actions;
- direct Amount, Direction, Account, and optional Counterparty span selection on
  the source, with accessible field colors and exactly one active selection;
- clearly marked analyzer/native-feedback suggestions in non-protected pools,
  with one action to accept a correct suggestion and ordinary controls to replace
  it;
- continuous draft preservation, save-and-next, skip, previous, and deterministic
  resume at the last unfinished item;
- compact validation messages that keep the current annotation intact;
- progress and coverage summaries without exposing protected-pool hidden facets;
  and
- append-only correction, adjudication, backup, restore, and explicit export
  through the existing secure workbench boundaries.

Pool assignment and weak segregation serve different purposes. Pools preserve
split/leakage boundaries and do not change when a human label changes. Weak
operational class, event state, family, and rail are browsing suggestions; they
remain separate from submitted canonical labels. Protected pools keep the
blind-first behavior below even when the fast workflow is used.

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

## Historical Candidate Selector preview

The implemented v1 target preview revalidates a submitted legacy label against
the active analysis and displays a compact Candidate Selector target. Keep it
available only for reproducing historical experiments. The focused v2 path
instead validates exact source-grounded fields and previews the direct-extractor
target independently of analyzer candidate coverage.

The workbench currently holds the complete 17,830-row canonical run and has a
verified initial backup. The focused annotation mode above remains planned. No
training-ready train/dev/test SFT target exists.

## Native Review remains separate

The local browser workbench remains the implemented corpus-labeling tool. The
native review screen is a separate product surface governed by its versioned
review and feedback contracts.

For every retained review operation, Android and iOS must show:

- the complete immutable source message and read-only receipt time;
- separate amount, direction, account, and counterparty highlights;
- analyzer candidates/cues explicitly marked as advisory suggestions;
- the validated extractor suggestion when available;
- stable messages resolved from reason-code-registry/2;
- account-resolution status without silently selecting a default account.

A correction uses native source selection to produce Unicode-scalar evidence.
When the message provides no selectable direction word, the reviewer may choose
debit or credit with the explicit direction control and that provenance is
recorded. A posted correction must select an existing account. Counterparty may
remain absent.

Confirm, correct, and reject actions carry expected and resulting review
revisions and are append-only/idempotent by action ID. Receipt time cannot be
edited. Feedback does not silently become canonical truth or authorize
persistence, export, telemetry, or training.
