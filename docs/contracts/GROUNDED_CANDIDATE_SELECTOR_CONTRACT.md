# Grounded Candidate Selector Contract

Status: **active production-intended host/model contract**
Input schema: `configs/sms_processing/contracts/grounded-candidate-selector-input.schema.json`
Output schema: `configs/sms_processing/contracts/grounded-candidate-selector.schema.json`
Executable parser: `src/pocketfinancer_sms/selector.py`

## Model input

The host first verifies that the original message hashes to the immutable
analysis source fingerprint. It then supplies that original message, the
operation-bound analysis ID, and a candidate list under
`pocketfinancer.grounded-candidate-selector-input/1`. Each entry includes only a
compact ID, kind, clause, exact source evidence, and whether it represents
explicit absence. Direction entries also state debit or credit because the host
derived that semantic value from the displayed evidence.

Amount canonical values, currency normalization, account normalization,
counterparty normalization, dates, and offsets remain host-only. Candidate IDs
are derived from the operation-scoped analysis ID, kind, span, and metadata, so an
ID from another operation is unknown. Canonical minor units, normalized
currency/provenance, normalized account identifiers, normalized counterparties,
and source offsets are deliberately absent from the model input.

Every operation includes exactly one explicit-absent account candidate and one
explicit-absent counterparty candidate. Amount and direction never have absent
candidates.

## One-pass model output

The model performs one greedy, non-thinking pass under grammar/structured output
constraints where supported. Exactly three semantic branches exist:

```json
{"decision":"none"}
```

```json
{"decision":"abstain"}
```

```json
{"decision":"posted","amount":"amt_…","direction":"dir_…","account":"acc_…","counterparty":"cp_…"}
```

No extra keys are allowed. `posted` must contain all four nonempty IDs. Direction
must select completed source evidence; a free debit/credit token is impossible.

## Host validation

The host rejects:

- malformed/non-object JSON or unknown decisions;
- extra/missing fields;
- IDs not present in the current analysis;
- a candidate with the wrong kind;
- absent amount/direction or required evidence missing;
- amount and direction selected from different clauses;
- duplicate/ambiguous candidate IDs;
- invalid money, optional-candidate metadata, or unsupported semantic values.

On success, the host reconstructs minor units, currency/provenance, direction,
optional-field states, and exact evidence. The raw compact output, validation
stage, and reconstruction stage may be stored in the private processing trace for
truthful user-facing transparency.

## Persistence is separate

A reconstructed `posted` result is not automatically saved unless it has exactly
one event, valid positive money, approved currency provenance, grounded direction,
a present uniquely resolved account, valid timestamp provenance, normal triage,
and no failure/negation/pending/request conflict. A missing account must not fall
back to an app default.

`none`, `abstain`, invalid output, reconstruction failure, assistive selection,
and any failed persistence condition route to review rather than negative truth.

## Superseded protocols

Candidate Protocol V1 remains measured historical evidence. It was insufficient
because its intended product result did not ground direction, did not express the
current none/abstain distinction and user-review flow, and was tied to the older
candidate/data path. Candidate V2 byte-offset output is rejected outright: models
must not count UTF-8 bytes or create evidence locations.
