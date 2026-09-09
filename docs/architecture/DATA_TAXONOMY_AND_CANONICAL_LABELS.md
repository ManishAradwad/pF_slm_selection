# Data Taxonomy and Canonical Labels

Status: **active**
Executable authority: `src/pocketfinancer_sms/labels.py` and
`configs/sms_processing/contracts/canonical-label.schema.json`

## Weak facets are not human truth

The analyzer produces weak, revisable browsing facets. A reviewer creates a
separate canonical label. Correcting weak segregation appends a correction record;
it never edits source evidence and never impersonates a human label.

## Orthogonal taxonomy axes

### Operational class

| Value | Definition |
|---|---|
| `posted_candidate` | Evidence suggests one or more completed movements; human confirmation is still required for truth. |
| `financial_non_posted` | Financial content describes failure, pending state, request, due amount, or another non-posted state. |
| `non_financial` | No financial event is represented. |
| `ambiguous` | Available evidence does not safely resolve the operational class. This is a valid outcome. |
| `invalid_outgoing` | Invalid/non-text input or reliable outgoing metadata. |

### Event state

| Value | Definition |
|---|---|
| `posted` | A completed movement occurred. |
| `not_posted` | Financial language exists but movement did not complete. |
| `no_event` | No financial event is expressed. |
| `unknown` | The reviewer cannot determine the event state. |

### Financial family

Families describe the event for browsing and error analysis: bank transfer, bill
payment, card purchase, cash deposit/withdrawal, fee, insurance, interest,
investment, loan, merchant payment, refund, salary, UPI transfer, wallet, other,
or unknown. They do not manufacture model targets.

### Payment rail

Rails are bank-internal, card, cash, IMPS, NACH, NEFT, RTGS, UPI, wallet, other,
or unknown. A rail is descriptive and independent from posted state.

### Weak-only metadata

Weak segregation also records confidence (`high`, `medium`, `low`, `unknown`) and
aggregate-safe reason codes. Human truth records uncertainty and notes separately.

## Canonical human decisions

| Decision | Required axes | Event records |
|---|---|---:|
| `posted` | `posted_candidate` + `posted` | exactly 1 |
| `not_posted` | `financial_non_posted` + `not_posted` | 0 |
| `non_financial` | `non_financial` or `invalid_outgoing` + `no_event` | 0 |
| `ambiguous` | `ambiguous` + `unknown` | 0 |
| `multiple_event` | `posted_candidate` + `posted` | at least 2 |

Each event contains exact amount and direction evidence, currency and provenance,
direction, account and counterparty presence state plus exact evidence when
present, family, and rail. Account/counterparty state is `present`, `absent`, or
`unknown`; present requires a span, while absent/unknown forbids one.

Every submitted/adjudicated label has a reviewer, revision, timestamp, preceding
revision link, and immutable hash. Drafts may be incomplete and remain distinct
from canonical labels.

## Target projection

Rich truth is projected later:

- `not_posted` and `non_financial` → `{"decision":"none"}`;
- `ambiguous` and `multiple_event` → `{"decision":"abstain"}`;
- a valid single `posted` event → four exact candidate IDs.

An `invalid_outgoing` human outcome is deliberately excluded from selector
projection instead of becoming `none`; it is outside the incoming-model path and
must not silently become a negative training example.

Unknown optional fields, missing candidates, mismatched evidence, invalid money,
currency mismatch, or incomplete annotations fail with a field-specific reason.
No builder is allowed to catch that error and emit a negative target. SFT targets
remain disabled until sufficient submitted human labels justify them.
