# SMS Workbench Labeling Guide

Status: **focused canonical-label/2 editor available for synthetic verification; private corpus trial pending**

The local workbench now writes `pocketfinancer.canonical-label/2` for new
annotations. It keeps earlier `canonical-label/1` revisions in the same
append-only history. The [taxonomy](../architecture/DATA_TAXONOMY_AND_CANONICAL_LABELS.md)
and executable rules in `src/pocketfinancer_sms/labels.py` govern submitted
labels. If the guide and validator disagree, stop and resolve the policy gap.

## Start

Run from the `pF_slm_selection` checkout that owns the private canonical
corpus and secure workbench:

```bash
source scripts/activate_wsl.sh
python scripts/run_sms_processing.py serve-workbench
```

Open the printed loopback URL and enter a stable local reviewer name. The
server binds to `127.0.0.1`, uses a per-run token, and loads no remote
assets. Stop it with `Control-C`. Drafts and submitted labels remain in the
encrypted local database. Never copy private SMS into issues, notes,
screenshots, tests, or hosted tools.

Start the local UI with `serve-workbench` above and open its freshly
printed URL each time the server restarts. Keep corpus and backup files inside
the ignored private-data tree.

## Focused annotation flow

1. Choose a pool and optional filters in **Find messages**. Start with
   `annotation_training`. The middle column is the current queue; its
   ordering, position, and remaining count appear beside the editor.
   Disagreement, candidate coverage, and imported feedback filters are
   available outside blind protected pools.
2. Read the complete, unchanged SMS before opening machine suggestions.
   Choose **Posted** (`Alt+1`), **None** (`Alt+2`), or **Abstain** (`Alt+3`).
3. Verify operational class, event state, family, and payment rail. The
   decision sets default class and state, which you may correct. Weak class
   and analyzer candidates remain suggestions, never human truth. Imported
   native correction evidence appears separately; **Use native evidence**
   assigns only its verified source span, which you should inspect.
4. For **Posted**, select exact text in the SMS and assign Amount, Direction,
   Account, and optional Counterparty. Assigned fields stay highlighted in
   distinct colors. Use **Clear** beside a field to select it again.
5. Enter the decimal amount, ISO currency, debit/credit direction, and the
   account reference supported by those spans. The account reference
   defaults to the selected text and can be corrected. Add an existing
   account ID only when it is known. Counterparty may be absent.
6. Mark uncertainty and enter a short categorical note when needed. Do not
   paste SMS text, account numbers, OTPs, or names into notes.
7. Draft changes save locally after a short pause. **Save draft** is also
   available. **Submit label** validates the complete label and appends a
   revision. **Submit and next**, **Save draft and next**, **Skip**, and
   **Previous** move through the current filtered queue. The local workbench
   remembers the reviewer, filters, search, and last unfinished position across
   browser sessions. **My review → My unfinished** shows only your pending rows.

To revise a submitted v2 label, choose **Correct with new revision**.
The workbench saves a new draft and keeps the earlier submission in its
append-only history. Finish the correction and submit the new revision.

A submitted Posted label requires one exact amount, direction, and account
span. The counterparty needs a span if it is present. Offsets are half-open
Unicode scalar positions into the original SMS. The validator rejects
missing, mismatched, or ungrounded evidence. It never converts an invalid
Posted label into None. A valid None or Abstain label has no event.

**Preview extractor target** works after submission. It projects v2 truth
directly to the direct extractor contract, independently of analyzer
candidate coverage. Preview does not create a training dataset or upload
anything. Existing v1 labels still preview through the historical Candidate
Selector path; the response identifies which target contract was used.

## Historical revisions and protected pools

If a message has a v1 revision, the editor identifies it as historical.
Choose **Start v2 revision** explicitly to make a new v2 annotation. The
original v1 revision and its hash remain readable in history and are never
rewritten. The API also prevents a v2 annotation from being saved later as
legacy v1.

For `protected_test` and `later_time_holdout`, the initial decision is
blind. The source SMS is visible; weak facets, analyzer suggestions, queue
reasons, group context, and disagreement details stay hidden. Saving a
draft does not reveal them. Submit a complete first label, then explicitly
choose **Reveal after submission** if needed. Protected weak facets are
excluded from aggregate class, family, and rail coverage. Do not call a
decision made after reveal blind.

## Local safety and remaining checkpoint

The workbench keeps append-only revisions and expected-revision conflict
checks. Backups and recovery remain separate explicit operations. To export,
select specific submitted or adjudicated revisions, then choose **Export
selected** and confirm the local encrypted export. Each selection includes its
revision hash, and an outdated selection is rejected. The CLI requires a JSON
selection manifest under `PRIVATE_DATA` containing a nonempty JSON array of
`source_id`, `reviewer_id`, `revision`, and `revision_hash` objects. Native
feedback is evidence for
adjudication, not automatic canonical truth. No SFT target or model training is
enabled by this editor.

The dashboard shows your remaining count, non-protected disagreements,
validation failures, and submissions during the past hour and day. These
counts are local; they do not imply human-gold readiness.

The remaining acceptance trial must use the owner's private corpus and cover
queue selection, rapid labeling, interruption and resume across browser
sessions, conflicts, blind reveal, backup and restore, adjudication, and
consent-bound export. Cross-session resume has synthetic verification but has
not yet been exercised against the private corpus.
