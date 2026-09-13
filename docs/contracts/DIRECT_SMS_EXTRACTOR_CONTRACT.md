# Direct SMS Extractor Contract

Status: **active production-intended shared host/model contract**
Input schema: `configs/sms_processing/contracts/v3/sms-extractor-input.schema.json`
Output schema: `configs/sms_processing/contracts/v3/sms-extractor.schema.json`
Validation profile:
`configs/sms_processing/contracts/v3/extractor-validation-profile.json`
Prompt: `configs/sms_processing/prompts/sms-extractor-v1.txt`
Grammar: `configs/sms_processing/grammars/sms-extractor-v1.gbnf`
Executable parser: `src/pocketfinancer_sms/extractor.py`

Android and iOS have not integrated this contract. Native ports must reproduce
the frozen v3 release and conversion rules before either app can claim parity.

## Responsibility boundary

The small language model owns the semantic decision and extraction: it decides
whether the message describes one posted event and, when it does, returns amount,
currency, direction, account reference, optional counterparty, and exact source
evidence. The deterministic analyzer is advisory only. Its candidates and cues
may help the model, but they are not an allowlist, do not constrain the answer,
and cannot override the original message.

The host owns immutable operation identity, receipt time, strict parsing,
grounding validation, money normalization, account resolution, duplicate
assessment, review routing, and persistence gating. The model never emits a
timestamp, confidence, reason code, normalized account ID, or persistence
decision.

## Input

`pocketfinancer.sms-extractor-input/1` includes the unchanged message, sender
family, primary currency snapshot, enabled profile IDs, advisory evidence, and
output rules. Each advisory item includes its analyzer contract version. The host
rejects an input whose advisory evidence was not produced by
`pocketfinancer.sms-analysis/2`.

The original message is the sole grounding authority. Advisory evidence may be
empty, incomplete, conflicting, or wrong without making a source-backed answer
invalid.

## Exactly three output branches

The model emits one JSON object and nothing else:

```json
{"decision":"none"}
```

```json
{"decision":"abstain"}
```

For the SMS body `INR 1,250.00 debited from account XX1234.`:

```json
{
  "decision": "posted",
  "amount": {"value": "1250.00", "currency": "INR", "evidence": {"start_scalar": 0, "end_scalar": 12, "text": "INR 1,250.00"}},
  "direction": {"value": "debit", "evidence": {"start_scalar": 13, "end_scalar": 20, "text": "debited"}},
  "account": {"reference": "XX1234", "evidence": {"start_scalar": 34, "end_scalar": 40, "text": "XX1234"}},
  "counterparty": null
}
```

`posted` requires amount, currency, direction, account reference, and exact
amount/direction/account spans. Counterparty and its span are both present or
both null. `none` and `abstain` permit no extraction fields.

## Strict validation and normalization

The host rejects malformed or non-object JSON, duplicate keys, trailing content,
unknown fields, type coercion, unsupported decisions, invalid decimal or
currency syntax, missing posted fields, and inconsistent optional fields. It
does not repair, retry, or reinterpret invalid output.

All model offsets are zero-based, half-open Unicode scalar indices into the
unchanged source. A span is valid only when its bounds are ordered and in range
and its `text` exactly equals the corresponding source slice. Empty evidence,
UTF-8 byte offsets, UTF-16 code-unit offsets, and normalized-text offsets are
invalid.

After validation, the host converts decimal amount to exact signed 64-bit minor
units under the currency profile, preserves the source spans, and resolves the
account reference only through the versioned account-resolution profile. Missing
or ambiguous accounts fail closed to review. Platform receipt time remains
authoritative and read-only.

## Review and persistence

Recognition does not authorize storage. Automatic persistence is disabled in
native integration release v3. A posted extraction still passes account,
duplicate, receipt-time, operation-ownership, grounding, money, and rollout
gates before any future release could persist it.

The native review screen must show the complete original message; separate
amount, direction, account, and counterparty highlights; analyzer suggestions
identified as advisory; stable reason messages; and the model suggestion when
available. Reviewers correct values by selecting exact source text, may use an
explicit debit/credit control when direction cannot be selected, and must choose
an existing account. Receipt time is displayed as read-only and is never an
editable model field. Corrections are revision-bound and append-only.

Kotlin ports convert Unicode scalar indices to UTF-16 indices by walking code
points and must reject a boundary inside a surrogate pair. Swift ports walk
`unicodeScalars` from `String.startIndex` and construct `String.Index`
boundaries from scalar positions; grapheme-cluster counts are not scalar counts.
Both ports must run the sanitized emoji and combining-mark vectors.

## Compatibility and training

The candidate-selector contracts and native releases v1/v2 remain byte-for-byte
historical compatibility artifacts. They are not rewritten, and stored
operations continue to be read with their original contract versions. New shared
operations use the direct extractor and release v3.

Future supervised targets project rich `canonical-label/2` truth directly to
`sms-extractor/1`. A posted target requires all mandatory grounded spans,
including direction. Invalid or incomplete labels fail projection; they are not
converted to `none`. No dataset publication, model training, deployment, or
automatic persistence is authorized by this contract.
