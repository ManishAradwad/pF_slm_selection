# SMS Processing Decision Log

Status: **active**
Date selected: 2026-09-01

## Selected design

PocketFinancer uses one shared deterministic analyzer, a high-recall triage policy,
one non-thinking Grounded Candidate Selector pass, strict host reconstruction, a
separate persistence gate, rich human labels, group-first corpus segregation, and
one local SQLite review workbench.

## Evidence

- Phase D measured no production selection; retain that conclusion.
- Candidate Protocol V1 improved extraction accuracy in its controlled comparison
  but failed its predeclared false-positive gate on every seed.
- Existing private splits omitted source rows and overlapped on IDs, exact bodies,
  normalized templates, and sender-template groups.
- The generated Semantic package covered incoming rows only, used a fixed
  timestamp/default currency, and silently downgraded invalid annotations.
- The reviewed 1,436-row package is overwhelmingly negative and therefore useful
  for negative safety/review, not as the primary production test.
- The first clean rebuild covers all 17,830 rows, produces zero protected-boundary
  overlap, and does not discard any of the 15 mapped legacy positives.
- Initial candidate coverage and triage counts show that deterministic support is
  incomplete. Retaining ambiguity is therefore safer and more honest than forcing
  binary truth.

## Alternatives considered

### Direct semantic generation

Rejected as the production route. It asks a small model to copy/normalize money,
accounts, counterparties, currency, and dates, increasing hallucination and
validation surface. It also makes exact grounding and correction provenance harder.

### Byte-offset Candidate output

Rejected. UTF-8 byte counting is a host responsibility and is brittle for small
generative models and multilingual text. Exact spans stay in host candidates and
human truth.

### Candidate Protocol V1 as the final contract

Superseded, not erased. It proved candidate selection worth pursuing but lacked
the final none/abstain/review semantics, grounded direction requirement, currency
snapshot/provenance, and independent persistence/feedback contracts. Its measured
report remains historical evidence.

### Binary pre-filter

Rejected. A binary gate either wastes SLM work on obvious terminal messages or
silently loses ambiguous/missing-candidate cases. Tri-state storage disposition
plus selector action preserves safety and permits user assistance. Product work
should reduce `retain_review` through better analysis and feedback, not erase it.

### Body-wide OTP rejection

Rejected. Credential OTP is terminal only when no completed-event clause exists.
“Without OTP” and messages containing both posted and security clauses must remain
eligible for analysis/review.

### Regex segregation as truth

Rejected. Deterministic weak facets support browsing and sampling; only revisioned
human labels are truth. Ambiguous is a valid label.

### Random row splits after model iteration

Rejected. Template and sender repetition leaks across boundaries and biases later
choices. Whole normalized-template components are assigned before model work, with
the later-time holdout restricted to wholly new post-cutoff templates.

## Currency injection

The user's primary ISO-4217 currency is explicit configuration because sender/body
country inference is unreliable and setting changes must not rewrite history.
Explicit message currency overrides the snapshot. India-specific conventions live
in an extension profile.

## Transparency and feedback

The selected model pass is non-thinking. Product transparency therefore means
showing real deterministic evidence and real decoding/validation stages, not
displaying hidden or fabricated reasoning. `ProcessingTrace` records those stages;
`UserFeedbackEvent` binds future confirm/correct/reject actions to a trace and
canonical label revision.

## Safety and generalization implications

- All candidate values remain source-backed and operation-bound.
- Offline retention and runtime discard are separate.
- Recognition is broader than automatic persistence.
- Missing candidates and invalid labels produce review/error reasons.
- Private data, databases, source mappings, annotations, and derived rows remain
  ignored and local.
- Locale extensions can grow without baking Indian assumptions into the core.

## Repository and private-artifact disposition

| Disposition | Files or artifacts | Treatment |
| --- | --- | --- |
| Keep | `PRIVATE_DATA/all_sms.json` | Sole retained 17,830-row private source archive; ignored and never printed or committed. |
| Keep | The 1,436-row manually reviewed package, its source mapping, metadata, import report, key material, SQLite history, and backups | Human/negative-safety evidence and provenance; never regenerated or treated as the primary production test. |
| Keep | The 203-row fixture | Regression-only private evidence, not a protected product test. |
| Keep | Completed Phase D no-selection and Candidate Protocol V1 experiment reports, manifests, and compatibility code/config | Immutable historical evidence; measured results are not rewritten. |
| Adopt | `src/pocketfinancer_sms`, `configs/sms_processing`, the canonical private manifest, and SQLite storage/recovery primitives | The sole active analyzer, selector, label, corpus, and workbench path. |
| Supersede | Candidate V2 semantic byte-offset code/schema, Candidate Protocol V1, the former unified pre-filter, regex taxonomy, and active extraction-V2 plans/status/configs | Historical only. Active plans/configs were relocated under `docs/history` and `configs/history`; one compatibility symlink preserves the hash-sensitive old policy path. |
| Delete | Generated Semantic packages, overlapping random splits, old segregation outputs, duplicate extracted archives/ZIP, obsolete exploratory auto-label/build/export scripts and tests, `.DS_Store`, obsolete intermediate corpus runs, and empty workbench databases | Removed after source/member/hash and retained-copy checks. These derived outputs are not directly recoverable, but their raw source, human evidence, historical checkpoint branch, and current canonical run are retained. |

Review queues are views over the one canonical manifest. No deleted segregation is
allowed to re-enter the active path as human truth or as an independently generated
dataset.

## Unresolved risks

- The first analyzer is intentionally conservative; human review must measure
  false discard/invoke and candidate-oracle gaps by group.
- Counterparty and bare-amount enumeration need broader language/profile coverage.
- Multiple-event runtime support currently retains/abstains rather than persisting.
- Native apps do not yet implement this contract, primary-currency onboarding,
  review inbox, trace UI, or revisioned feedback.
- Protected evaluation is not human gold until blind review/adjudication completes.
- No model has been trained or deployed on this foundation.
