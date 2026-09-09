> **Historical workbench guide.** The active workbench is documented in
> `docs/architecture/WORKBENCH_REQUIREMENTS_AND_DATA_FLOW.md` and uses the canonical
> `PRIVATE_DATA/sms_processing` manifest.

# PocketFinancer Extraction V2 local workbench

## Scope and boundary

Workbench V2 is the shared, local-only annotation and evaluation foundation for
PocketFinancer Extraction V2. It validates annotations against Semantic V2,
enforces sender/template split isolation and independent adjudication, creates a
categorically anonymized local form, scores already-mapped Semantic V2 predictions,
and applies the frozen decision policy.

It is not a model protocol. A platform or model-family adapter may use Direct V2,
Candidate V2, or another versioned protocol only in a later authorized laboratory
phase. Its output must map into the same frozen Semantic V2 boundary before this
evaluator sees it. Workbench V2 does not select a protocol, run a model, change an
application, or change a production default.

## Privacy model

- Raw messages, annotations, adjudications, row-level predictions, and even the
  anonymized row-level form stay beneath ignored `PRIVATE_DATA/`.
- Aggregate reports are written only beneath ignored `RESULTS/` and are scanned
  for message, row-ID, semantic-record, prediction, gold, and source-mapping keys.
- The CLI will not read a row-level package outside `PRIVATE_DATA/`. Its sole
  exception is the exact committed invented fixture when
  `--invented-fixture` is supplied.
- The anonymized form removes message text, timestamps, spans, exact amounts,
  account/counterparty values, reviewer identities, and split-group identities.
  Its row IDs are export-scoped HMAC-SHA256 values. It remains local row-level
  material and is not publication-approved.
- The CLI prints only structural counts, hashes, profile IDs, and aggregate safety
  status. It has no network or hosted-service integration.

No command in this guide authorizes inspecting private data. The user must first
approve the source, annotation policy, adjudication policy, data rights, and
protected split. Until then, use only the invented fixture.

## Versioned artifacts

- Workbench contract:
  `configs/contracts/pocketfinancer-workbench-v2.json`
- Frozen decision policy:
  `configs/history/pocketfinancer-extraction-v2-decision-policy.json`
- Reference implementation:
  `lfm25/workbench_v2.py` and `lfm25/evaluation_v2.py`
- Local CLI: `scripts/run_pocketfinancer_workbench_v2.py`
- Invented smoke package:
  `tests/fixtures/pocketfinancer_workbench_v2_synthetic.json`

The workbench contract pins the Semantic V2 schema, Python reference, and Phase A
conformance hashes. Annotation packages repeat those pins. Prediction packages
also bind the complete annotation package and the opaque row/split/group manifest,
plus the profile, model/runtime, prompt, parser, filter, decode, environment, and
other experiment provenance required by the program plan.

## Package lifecycle

An annotation package has these boundaries:

1. `privacy` declares either private local material or invented synthetic material.
2. `provenance` supplies an opaque package ID, annotation-policy version, source
   revision, and source digest.
3. `split_policy` requires train, development, and protected-test separation by
   pseudonymous sender and template family.
4. Each row has an opaque ID, source message, split/group assignment, safety tags,
   independent annotations, and pending/resolved/excluded adjudication state.
5. A resolved row needs at least two distinct annotators and an independent
   adjudicator. The adjudicator selects one annotation or supplies a separately
   validated Semantic V2 record.

Every submitted and adjudicated record passes the frozen Semantic V2 reference,
including exact money, host timestamp provenance, and UTF-8 byte grounding. The
validator reports row positions and contract paths, never message content.

Prediction packages contain only a profile-specific parser's mapped Semantic V2
record or an explicit invalid status, the pipeline decision (`auto_post`, `review`,
or `reject`), optional integer operational measurements, aggregate conformance
status, and complete provenance. Independent Semantic V2 validation is authoritative:
a parser-reported valid record that fails validation is scored invalid and cannot
silently escape the evaluator.

## Invented-only smoke command

From the activated WSL environment:

```bash
python scripts/run_pocketfinancer_workbench_v2.py validate \
  tests/fixtures/pocketfinancer_workbench_v2_synthetic.json \
  --invented-fixture
```

This command prints only aggregate structural counts and hashes. The normal test
suite constructs invented predictions in memory and proves correct scoring,
provenance mismatch rejection, split leakage rejection, independent adjudication,
anonymization, invalid-output fail-closed handling, false-auto-post taxonomy, and
no-selection on insufficient evidence.

After a separately authorized private workflow exists, an empty package can be
created under `PRIVATE_DATA/` with the `create` command. `anonymize` reads its HMAC
key only from an environment variable, never a command-line argument. `score`
reads both row-level inputs under `PRIVATE_DATA/` and writes one aggregate report
under `RESULTS/`. Run `--help` for exact required arguments.

## Frozen evaluation and decision rules

The policy uses two-sided 95% Wilson score intervals. Floors use the lower bound;
ceilings use the upper bound. Protected selection requires at least 1,000 rows,
250 gold-auto-post-eligible rows, 400 predicted auto-posts, 250 applicable gold
events per field, 400 invalid/fault-injection cases, 30 cases in each critical
safety category, and all 13 hash-locked conformance vectors.

The frozen semantic floors are 0.98 lower-bound accuracy for scope, posting status,
and event cardinality; 0.95 for amount/currency, direction, account, and
counterparty exactness; 0.99 for evidence validity and automatic-post precision;
and 0.70 for useful automatic-post coverage. Invalid-output upper bound is 0.02;
fail-closed lower bound is 0.99. Exact definitions and denominators are normative
in the JSON policy.

Every critical safety category has a zero observed automatic-post tolerance, and
the complete report has a zero false-automatic-post tolerance. Provenance, locked
split, conformance, reproducible baseline, and applicable platform/device gates are
hard gates. Device evidence needs a separately versioned hard-budget attachment
frozen before its first protected measurement; host timing cannot satisfy a device
gate.

Candidates are compared only within one platform and one runtime variant. Passing
candidates use the predeclared material tie order: automatic-post precision lower
bound, coverage lower bound, semantic macro lower bound, device latency p95, then
peak-memory p95. Profile ID order never breaks a tie. A failed gate, insufficient
sample, missing pin/budget/baseline, cross-scope pooling, or unresolved material
tie produces explicit no-selection and returns that platform to its laboratory
phase. Selection never changes production defaults.
