# Local Annotation Workbench

This guide covers the local browser workbench for applying [Annotation Handbook
V1](ANNOTATION_HANDBOOK_V1.md). It is an operating procedure, not authority to
upload, publish, train, release, or deploy anything.

### Open the served URL, not the HTML file

The workbench is a local web application, not a standalone HTML document. Opening
`lfm25/annotation_assets/index.html` directly produces a `file://` URL. That loads
only the static markup: the browser cannot fetch the server-owned stylesheet,
JavaScript session, bootstrap data, or SMS row. The resulting unstyled page stays
on **Loading**, shows `0 / 0`, and has an empty Sender/SMS area. This is expected
for an incorrectly opened file and does not indicate that a private row was lost.

Activate the environment first, enter the checkout that owns the code, and launch
the server command:

```bash
cd /home/tojinotzenin/pF_slm_selection
source scripts/activate_wsl.sh
python scripts/run_lfm25_annotation_workbench.py blinded \
  --reviewer <stable-id> \
  --batch-size 50
```

Before merge, use a linked worktree only for the invented smoke fixture and unit
tests. Do not attach or copy canonical private data: the runtime privacy guard
intentionally rejects private roots that resolve outside the active checkout.
Run the real private workflow from the canonical checkout after the change is
merged.

The command prints one aggregate-only JSON line containing a URL such as
`http://127.0.0.1:8765/`. Open that exact `http://127.0.0.1:.../` URL in the local
browser. The SMS is delivered only by the authenticated local `/api/row` request
after the page establishes its server session; it is intentionally not embedded
in `index.html`.

### Safe UI smoke test with invented rows

Before opening any private package, exercise the real HTML, CSS, JavaScript,
cookie, API, and SQLite stack with three wholly invented rows:

```bash
python scripts/run_lfm25_annotation_workbench_smoke.py
```

That command keeps prefill off, matching the production default. To verify the
candidate-assisted form population using only the same invented rows, launch:

```bash
python scripts/run_lfm25_annotation_workbench_smoke.py \
  --source-prefill unambiguous
```

The smoke launcher binds only to `127.0.0.1` on an OS-selected unused port and
prints its exact URL. It never calls a `PRIVATE_DATA` loader or export path. Its
fixed reviewer, SMS messages, senders, and queue tags are invented in code, and
its database lives in a temporary directory that is removed after **Close**,
`Ctrl+C`, or normal shutdown. The page repeatedly labels every message
`SYNTHETIC DEMO ONLY - NO PRIVATE DATA`; do not treat the fixture as data,
annotation-quality, model-quality, or production evidence.

This command is also the manual fallback when browser automation is unavailable
in the current host environment. Open the printed URL and
verify that the SMS card and styled controls appear. Do not substitute a
`file://` tab, and never use private rows for a screenshot or UI bug report.

## 1. Privacy boundary before launch

Run from the canonical WSL repository after activating its prepared environment.
Keep the machine offline while reviewing when practical. The workbench is
intentionally local-only: use only the loopback URL printed by the local process;
never expose it on a LAN address, through a tunnel, proxy, remote browser, shared
screen, or port-forward.

Use a dedicated local browser profile with account sign-in, sync, telemetry,
history synchronization, cloud spell-check, password-manager capture, translation,
AI assistants, and nonessential extensions disabled. Do not paste private text
into search, developer-assistant, chat, email, issue, or cloud-document fields.
Treat browser cache, history, downloads, crash reports, clipboard contents,
screenshots, and accessibility/automation logs as private artifacts. Close other
profiles before review and do not reuse a review profile for ordinary browsing.

Raw SMS, senders, exact spans, reviewer notes, proposals, row mappings, locks,
backups, and per-row predictions must remain inside ignored private storage. Safe
terminal and report output is aggregate-only. Never print a row to diagnose a UI
problem; use its opaque local ID.

## 2. Choose exactly one isolated mode

Do not run both modes concurrently or reuse a browser tab/profile between them.
Mode, input, state, lock, backup, and provenance must stay isolated.

### Blinded test mode

```bash
python scripts/run_lfm25_annotation_workbench.py blinded \
  --reviewer <stable-id> \
  --batch-size 50
```

This mode reads only the frozen reviewer-blind test package. It must not reveal
silver labels, heuristic reasons, model proposals, confidence, template groups,
private hashes, source mappings, training state, or prior model predictions.
There is no proposal-reveal or active-queue path in this mode.

### Training mode

```bash
python scripts/run_lfm25_annotation_workbench.py training \
  --reviewer <stable-id> \
  --pool <explicit-private-jsonl> \
  --batch-size 50
```

Training mode requires an explicit private JSONL pool. It must fail closed if the
pool contains, derives from, overlaps, or cannot prove exclusion of frozen test
rows. Do not point it at the blinded package, a reviewed test manifest, or an
untracked convenience copy. Training annotations and active queues are never test
evidence.

Both commands accept optional `--repo-root` when deliberately operating on a
specific local checkout. Resolve and verify that root before review; do not use it
to place state or outputs outside the ignored private boundary.

The durable defaults are `PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3`
and `PRIVATE_DATA/lfm25/annotation_workbench/training_curation.sqlite3`. Keep and
reuse the matching database on every relaunch. If you pass `--db`, keep it below
`PRIVATE_DATA/lfm25` and repeat that exact argument for resume, export, and recovery.

### Default unaided mode and explicit source-assisted mode

Source prefill is **off by default**. Omitting `--source-prefill` preserves the
unaided review method and its existing database binding. This remains the default
for evidence intended to measure an unaided human annotation pass.

All existing unaided commands and paths remain unchanged. Do not add the assisted
flag to an in-progress unaided package or reuse an unaided artifact in an assisted
run.

An assisted blinded run starts by exporting its own policy-selected package before
the first workbench launch:

```bash
python scripts/review_lfm25_blinded_test.py export \
  --source-prefill unambiguous
```

The immutable source manifest remains the common input, but this command writes
the distinct assisted review projection, internal map, and metadata:
`blinded_test_candidate_assisted_review.jsonl`,
`blinded_test_candidate_assisted_review_internal_map.jsonl`, and
`blinded_test_candidate_assisted_review_metadata.json`. Do not use `--force` to
reinitialize an in-progress package. Validate the policy-selected package with:

```bash
python scripts/review_lfm25_blinded_test.py validate \
  --source-prefill unambiguous
```

Then launch the assisted workbench, which may ask the existing deterministic
source-candidate extractor to populate only fields it can ground unambiguously:

```bash
python scripts/run_lfm25_annotation_workbench.py blinded \
  --reviewer <stable-id> \
  --batch-size 50 \
  --source-prefill unambiguous
```

This selects
`PRIVATE_DATA/lfm25/annotation_workbench/blinded_test_candidate_assisted.sqlite3`.
The assisted server reads and updates only the assisted review projection; it does
not share the unaided review JSONL, map, metadata, projection, or database.

For training curation, use the same flag with the explicit pool and a separate
assisted database, for example
`PRIVATE_DATA/lfm25/annotation_workbench/training_curation_candidate_assisted.sqlite3`.
Never point assisted and unaided runs at the same database. The frozen binding
records the prefill policy as `human_verified_candidate_assisted`; a mismatch
fails closed instead of silently mixing methods.

The server exposes prefill as a separate read-only suggestion DTO: loading a row
does not change its annotation, decision, revision, or audit history. The UI may
copy compatible suggestions into the form only after the reviewer independently
chooses **Transaction**. That explicit edit follows the normal draft-autosave
path, but suggestions never complete a row on their own.

Prefill is an editing aid, not a label. Read the full SMS, make the transaction
decision independently, and verify every suggested component, exact selected
text, UTF-8 span, decimal, and direction before completion. Fields that are
ambiguous or unsupported remain for the reviewer. Candidate extraction can be
systematically wrong, so assisted annotations must be reported separately from
unaided annotations and must not be used as a causal human-quality comparison.
Candidate-assisted blinded labels are not unbiased gold for measuring that same
candidate extractor's coverage or selection accuracy; retain unaided gold for
that evaluation or disclose and analyze the assisted dependency separately. The
option does not authorize a hosted model, telemetry, proposal auto-acceptance, or
test/training crossover.

Resume every assisted session with the same `--source-prefill unambiguous`, mode,
reviewer, pool, repository root, and assisted database. Use assisted validation
when checking the projection after interruption. At the final gate, validate and
import with the same policy:

```bash
python scripts/review_lfm25_blinded_test.py validate \
  --source-prefill unambiguous

python scripts/review_lfm25_blinded_test.py import \
  --source-prefill unambiguous
```

Import selects the assisted DB and writes
`split_manifest_candidate_assisted_human_reviewed.jsonl` plus
`blinded_test_candidate_assisted_review_import_report.json`. Omitting the flag at
export, validation, launch/resume, or import is a provenance error, not a request
to convert the assisted run into an unaided one. Never copy annotations or
projections between the two policy lineages.

## 3. Identity, timestamps, and 50-row sessions

Use one stable, non-personal local reviewer ID for your work. Do not use an email
address, phone number, credential, or change IDs mid-review. Never share an ID
between people. The workbench records the reviewer ID, offset-aware save timestamp,
and timezone; do not type, edit, backdate, or copy those values.

Use `--batch-size 50` for normal sessions. A session is an attention boundary, not
a target to rush through, and the dashboard's **Batch** counter applies only to
rows completed by the current local process.

A normal 50-row batch is:

1. launch the intended mode with `--batch-size 50`;
2. work from **Next pending**, resolving difficult rows instead of skipping them;
3. stop when **Batch** reaches `50 / 50` (or when no eligible row remains); and
4. confirm **Saved**, close the review tab, then stop the process with `Ctrl+C`.

Take a real break before another batch. Do not start extra batches to avoid
uncertain or difficult rows, and never stop while the current form says Unsaved,
Saving, or Save failed.

State is resumable. Relaunch the same mode with the same reviewer, database, pool
(for training), repository root, and batch size. The next process opens a new
50-row attention session over the durable state and starts from pending work. Do
not delete the database, change the pool, create a replacement export, or use a
force option to manufacture a clean start.

## 4. Review and navigation

Read the full SMS before assigning a decision. For a transaction, select amount,
account, and optional counterparty directly in the source display, then verify the
normalized decimal, direction, exact selected text, and boundaries before saving.
For `not_transaction`, confirm that all extraction fields are empty. Record a
short categorical note and `uncertain` status when required by the handbook.

Use the in-app Previous/Next controls and save/status indicator rather than the
browser Back/Forward buttons. A navigation shortcut must never discard a dirty
form; resolve any unsaved-change warning before moving. Before using a single-key
shortcut, move focus out of notes or another text input. V1 bindings are:

| Keys | Action |
|---|---|
| `J` / `K` | Move between the previous/next rows |
| `N` | Move to the next pending row |
| `T` | Set `transaction` |
| `X` | Set `not_transaction` / null |
| `U` | Toggle uncertainty |
| `Ctrl+S` | Submit the current annotation |
| `Alt+A` | Assign the current source selection as amount |
| `Alt+C` | Assign the current source selection as account |
| `Alt+P` | Assign the current source selection as counterparty |
| `Alt+0` | Record explicit no-counterparty |

The amount selection must cover the full written currency expression; the
workbench stores its exact decimal separately. Account selection includes the cue
and mask, while counterparty selection excludes relation cues such as `to`,
`from`, `at`, and `by`.

Filters are review aids, not changes to batch membership or QC eligibility. V1
provides `pending`, `completed`, `uncertain`, `noted`, `transaction`, `null`, and
`QC` filters. Training mode also provides a generic `active_learning` priority
filter. It does not reveal why a pending row was prioritized. Clear filters before
declaring a session complete. Never use free-text browser search or an extension
to index SMS content.

## 5. Training mode: blind first, proposals second

Every training row starts blind. Before revealing a heuristic or local-model
proposal:

1. read the source and create a complete provisional annotation;
2. save the blind annotation, uncertainty, reviewer, and timestamp;
3. explicitly choose Reveal proposal only if comparison is useful; and
4. after comparison, save any revision with a reason while preserving the blind
   annotation and reveal audit.

Proposal reveal is irreversible for that review record. It must record reveal
time and proposal provenance, and it must never rewrite the blind-first payload.
Do not reveal first and then reconstruct what you think your blind answer would
have been. A proposal is a fallible local aid, never gold and never a reason to
ignore source spans or the handbook.

The training-only `active_learning` filter prioritizes rows with one or more of
these locally computed category tags: model disagreement, low-confidence output,
candidate-coverage miss, hard negative with an amount signal, OTP/security,
pending/failed/declined/hold, payment request/reminder, refund/reversal, multiple
entities, and rare sender/template. The filter name stays generic and its category
reasons stay hidden until that row has one complete blind label. After completion,
the UI may show those local category reasons. Proposal contents and confidence
remain separately hidden unless the reviewer explicitly chooses **Reveal
proposal**; category display is not a proposal reveal.

The priority view is deterministic from the frozen explicit pool and its recorded
file order. Freeze and record the input-pool digest, eligibility rule, queue
algorithm and version, tie-break rule, ordered opaque row IDs, proposal/model
revision, contract/handbook version, and creation time. It is built only from
eligible training/development rows and never from test rows. Queue ranking must
not change the frozen test or leak test labels, predictions, templates, or
membership into training selection.

### Export reviewed training curation

Stop the training workbench process first so the exporter can take the exclusive
database lock. Then reuse the exact reviewer, pool, and database binding:

```bash
python scripts/run_lfm25_annotation_workbench.py export-training \
  --reviewer <same-stable-id> \
  --pool <same-explicit-private-jsonl> \
  --db PRIVATE_DATA/lfm25/annotation_workbench/training_curation.sqlite3
```

The defaults write `PRIVATE_DATA/lfm25/training_curation_human_reviewed.jsonl` and
`PRIVATE_DATA/lfm25/training_curation_export_report.json`. The command refuses to
overwrite nonempty outputs. Its console report is aggregate-only; check its
completed, pending, and uncertain counts, and never treat pending or uncertain rows
as approved training labels.

For an assisted training workspace, repeat the policy explicitly during export:

```bash
python scripts/run_lfm25_annotation_workbench.py export-training \
  --reviewer <same-stable-id> \
  --pool <same-explicit-private-jsonl> \
  --source-prefill unambiguous
```

Without path overrides, this selects the candidate-assisted database and writes
`training_curation_candidate_assisted_human_reviewed.jsonl` plus
`training_curation_candidate_assisted_export_report.json` under
`PRIVATE_DATA/lfm25`. Never export an assisted database under the unaided policy
or combine the two outputs.

Each completed exported row retains `human_annotation_workbench`, including the
generic annotation and blind-first/final event provenance. A private component
evaluation builder can pass its `annotation` value directly to
`adapt_workbench_annotation(...)`, together with that row's private source fields,
to form evaluator input without changing decimals or UTF-8 spans. Keep the
resulting paired rows under `PRIVATE_DATA`; never print or commit their raw source,
annotation, spans, event hashes, or per-row results.

## 6. QC and adjudication policy

The required QC population is the union of:

- every row whose final primary decision is `transaction`;
- every row with nonempty notes;
- every row that was ever uncertain, even if the flag was later cleared; and
- a deterministic sample of at least 10% of final `not_transaction` rows, rounded
  up.

Sample `ceil(10% * all final nulls)` from the full null set using opaque stable IDs
and a frozen, versioned deterministic rule. Record the null-population digest,
policy version, sample size, selected-ID digest, and tie-break/seed material.
Never hand-pick easy nulls or resample for a preferred result. A sampled noted or
ever-uncertain null retains both QC reasons; overlap does not invalidate the
deterministic sample.

V1 performs QC in a distinct, delayed session by the same reviewer. It stores the
initial, QC, and any adjudication actions as separate events, each with its own
reviewer and timestamp; the QC event never overwrites the initial event. QC
rechecks the decision, every component, exact source spans, notes, uncertainty,
and proposal-visibility history. A disagreement or unresolved uncertainty goes to
adjudication; it is not resolved by silently editing the primary record. Final
gold keeps the event history, identities, timestamps, and reasons as an audit
trail. The actionable initial/QC/adjudication history is shown only after pending
QC blindness lifts; while a required QC row is pending, its prior annotation,
category reasons, proposal state, and history remain hidden.

### Perform the delayed pass

1. Complete every primary row and resolve every active uncertainty; the UI will
   reject QC start while any row is incomplete or unresolved.
2. End the primary process at a saved boundary, take the intended delay, then
   relaunch the identical mode, reviewer, database, batch size, and training pool.
3. Select **Start delayed QC pass**. This freezes the required queue; do not change
   inputs or try to resample it.
4. Re-annotate each queued row from the source and choose **Complete row**. Pending
   QC hides the prior annotation and history, and QC has no draft/autosave path.
5. An exact agreement passes. A disagreement is retained as a separate failed QC
   event; review the now-available history, choose the final source-grounded label,
   and submit that row again to record the explicit adjudication.

Once QC starts, the database persists that phase. Relaunching the same command and
database resumes the frozen QC queue automatically; there is no separate `--qc`
command and no permission to return to or rewrite the primary pass.

### Future independent-review record

The append-only event schema retains reviewer identity per initial, QC, and
adjudication event so it can support an independent reviewer later, but that
workflow is deferred and no standalone second-review command is documented. A
future independent QC event must be keyed by opaque row ID and review round and
store reviewer role/ID, an offset-aware timestamp and timezone, the full annotation
with spans, uncertainty and notes, handbook/QC-policy versions,
parent-annotation/provenance digest, proposal-visibility state, and
disagreement/adjudication status. The second reviewer must not see the primary
annotation or proposals before committing their own result, must use a different
reviewer ID, and must never overwrite the primary event. Until this capability is
implemented and validated, describe V1 accurately as delayed same-reviewer QC,
not independent double annotation.

## 7. Backups, locking, and recovery

Only one writable process may own a mode/state at a time. If a lock is reported,
first check whether another local workbench process is using that workspace. Never
run a second writer, edit a lock, or delete it. A different browser tab
does not create a safe second writer.

Workbench writes are transactional, and each session must retain rolling local
backups with validated source/state binding. Backups belong under the ignored
boundary, not in a synced desktop, shared drive, email, issue attachment, or Git.
Do not edit JSONL/state by hand, concatenate backups, or restore only part of a
state/provenance set.

For an interrupted write, stale-lock warning, or failed integrity check:

1. stop interacting with the UI and stop the local server;
2. confirm no process owns the database; never edit or remove its lock file;
3. choose one existing `.snapshot.bak` file from the database's exact
   `<database>.backups` directory without opening, copying, or modifying it;
4. run the recovery command with the database and that exact backup path; and
5. relaunch the identical mode and arguments, then verify aggregate
   completed/pending/QC counts before continuing.

For the default blinded-test database, replace `TIMESTAMP.UUID` below with the
exact basename segment of the selected local backup:

```bash
python scripts/run_lfm25_annotation_workbench.py recover \
  --db PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3 \
  --backup PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3.backups/blinded_test.sqlite3.TIMESTAMP.UUID.snapshot.bak
```

For training recovery, substitute `training_curation.sqlite3` in both paths. The
command takes the exclusive lock, validates the backup before replacement,
atomically restores it, validates the restored database, and preserves the
displaced current database in the same backup directory. It refuses a backup
outside that exact directory. Its JSON output contains only validity, schema, row,
event, and displaced-database status; it does not print row data. If recovery fails,
leave the database and backups untouched and request repository-owner assistance.

On relaunch, the normal workspace binding check confirms the restored database
still matches the selected mode and frozen inputs.

## 8. Aggregate evaluation

Evaluate components separately and keep each denominator explicit:

- prefilter: gold transaction/null counts, transaction recall, false rejections,
  null rejection, model invocation, and rejection counts by stage;
- candidate extraction: amount/account/counterparty and joint oracle coverage plus
  exact-span grounding, all conditional on human-gold transactions;
- parser: accepted/rejected status counts, rejection reasons, strict
  duplicate-key/unknown-ID/reordered-member behavior, exact reconstruction, and
  timestamp preservation;
- whole pipeline: gold/predicted transaction counts, ghosts, misses, amount,
  account, counterparty, and type accuracy on gold transactions, transaction exact,
  and whole-pipeline exact; and
- QC readiness: required, completed, agreement, disagreement, adjudicated, and
  unresolved counts by component.

Verify the evaluator contract first with the checked-in, wholly invented fixture:

```bash
python scripts/evaluate_lfm25_components.py \
  --input DATA/annotation_component_v1_synthetic.jsonl \
  --dry-run
```

This is read-only and prints aggregate JSON only. The fixture validates wiring and
denominators; it is synthetic test data, not model-quality or production evidence.
For private paired inputs, keep the JSONL under `PRIVATE_DATA` and write any
aggregate file only below `PRIVATE_DATA` or `RESULTS`.

Both the final blinded reviewed manifest and the reviewed training export retain a
`human_annotation_workbench` object with the generic complete annotation and
event-level provenance. Use
`lfm25.component_evaluation.adapt_workbench_annotation(...)` on that object's
`annotation` field when constructing private component-evaluation input; do not
reconstruct spans from legacy labels or copy raw rows into a script, terminal, or
committed fixture.

Report only aggregate values safe for the console and experiment record. Do not
print examples, row IDs, reviewer IDs, notes, spans, per-row predictions, or small
slices that reveal a message. State whether metrics use training, development, or
the newly adjudicated frozen test; training/QC agreement is not production model
evidence. Preserve Android-parser and strict-output metrics as distinct when both
are reported, and do not compare HF timing with GGUF/device timing.

## 9. Final blinded-import gates

Before final import, all of the following must pass closed:

- every frozen test row has one complete primary decision with valid reviewer and
  timestamp provenance;
- every transaction has a positive exact decimal, debit/credit type, valid exact
  amount/account spans, and either a valid exact counterparty span or explicit
  absence; every null has no extraction fields;
- no active uncertainty, unresolved disagreement, partial edit, or invalid span
  remains;
- QC covers all transactions, all noted/ever-uncertain rows, and the frozen
  deterministic sample of at least 10% of nulls;
- the source manifest, blinded mapping, metadata, row set/order, review template,
  handbook version, and workbench-state hashes still match their frozen
  provenance;
- blinded mode never exposed proposals or training state, and no test row entered a
  training pool, active queue, proposal build, or training export;
- the workbench has no live/stale write or recovery state and reports a complete,
  QC-ready aggregate summary; and
- reviewed outputs do not already exist unless replacement is a separate,
  explicitly approved and provenance-preserving operation.

Only after those gates pass, stop the workbench server so import can take the
exclusive database lock, then run from the intended repository:

```bash
python scripts/review_lfm25_blinded_test.py import \
  --workbench-db PRIVATE_DATA/lfm25/annotation_workbench/blinded_test.sqlite3
```

The database must be the one bound to this exact frozen package. Import compares
the reviewer projection with durable event history and fails closed unless every
row is complete, no uncertainty or disagreement remains, and every required QC
row has passed. `--force` can replace existing outputs when explicitly approved;
it does not bypass any workbench, provenance, or QC gate.

Import writes a separate reviewed manifest and aggregate report; it must not
mutate the frozen source manifest. Completed rows in that manifest preserve
`human_annotation_workbench`, including the validated generic annotation plus
blind-first, final, and QC event provenance. Those private annotations are suitable
for the component adapter described above, but they must never be printed or
committed. Import does not authorize model training, evaluation claims, upload,
publication, release, deployment, or changes to the Android contract. Those remain
separate explicit decisions under repository policy.
