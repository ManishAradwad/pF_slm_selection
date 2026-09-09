# SMS Processing Architecture

Status: **active architecture and implementation authority**
Executable package: `src/pocketfinancer_sms`
Private corpus configuration: `configs/sms_processing/archive-india-inr.json`

## Outcome

PocketFinancer now has one platform-neutral processing foundation. Android and iOS
implement the frozen native release described by
`docs/contracts/NATIVE_SMS_INTEGRATION_CONTRACT.md`; the foundation remains the
behavioral authority and parity oracle.

For one incoming message, the intended runtime sequence is:

1. The host snapshots operation configuration: primary ISO-4217 currency, enabled
   locale profiles, message timestamp and provenance, direction metadata, and an
   operation identifier.
2. `DeterministicSmsAnalyzer` preserves the source text and enumerates clauses,
   exact host spans, money, direction, account, counterparty, state cues, and
   aggregate-safe reasons.
3. `evaluate_triage` applies policy to that same analysis. It returns both a
   storage disposition (`invoke`, `discard`, or `retain_review`) and an orthogonal
   selector action (`run_normal`, `run_assistive`, or `skip`).
4. When the selector action permits it, the host packages the message and its
   source-backed candidates for one greedy, non-thinking SLM pass.
5. The SLM emits only `none`, `abstain`, or one `posted` object selecting four
   candidate IDs. It never emits offsets or canonical values.
6. The host strictly validates every ID and reconstructs a rich semantic result.
7. The persistence gate independently decides whether the reconstructed result is
   safe to save automatically. Recognition does not imply persistence.
8. Anything unresolved is retained for local/user review. A processing trace can
   show deterministic analysis, the compact model stream and raw output,
   validation, reconstruction, and persistence decision without inventing chain
   of thought.

```text
platform message + configuration snapshot
                 │
                 ▼
      deterministic structural analyzer
                 │
                 ├── terminal certainty ───────────────► discard (offline row retained)
                 │
                 ├── incomplete/ambiguous ─────────────► retain_review
                 │                                      ├── run_assistive when grounded
                 │                                      └── skip when not representable
                 │
                 └── one grounded event candidate ─────► invoke / one SLM pass
                                                          │
                                    none ◄────────────────┼──────────────► abstain
                                                          │
                                                          ▼
                                               posted candidate IDs
                                                          │
                                             strict host reconstruction
                                                          │
                                  ┌───────────────────────┴──────────────────────┐
                                  ▼                                              ▼
                           auto persistence                               retained review
                         (all safety gates)                           (any unresolved gate)
```

## Executable boundaries

| Concern | Authority |
|---|---|
| Analyzer and host evidence | `analyzer.py`, `structural_text.py`, `types.py` |
| Currency snapshot and exact money | `currency.py`, `profiles.py` |
| Pre-filter policy | `triage.py` |
| Compact model contract and reconstruction | `selector.py` |
| Automatic persistence | `persistence.py` |
| Human truth and target projection | `labels.py` |
| Observability and user feedback | `trace.py`, `feedback.py` |
| Canonical private corpus and pools | `corpus/` |
| Local review UI and durable state | `workbench/` |

No taxonomy regex independently decides truth, and no second classifier model is
part of the production path.

## Analyzer output

The versioned analysis object contains:

- a message- and configuration-bound `analysis_id`;
- an NFKC/casefold/whitespace structural view used only for matching, with a
  fingerprint recorded while all returned evidence remains unchanged source text;
- unchanged source-backed clause spans in character and UTF-8 coordinates;
- amount candidates with exact minor units, currency, and provenance;
- completed direction evidence candidates;
- account/card/VPA and counterparty candidates;
- explicit absent account and counterparty candidates;
- failure, negation, pending, due, request, balance, promotional,
  administrative, credential, and payment-rail cues;
- source-safe reason codes and input/direction metadata.

UTF-8 offsets are host metadata and human truth. They are never model output.

## State transitions

```text
received
  → analyzed
  → discarded_terminal | retained_review | selector_pending
selector_pending
  → selector_none | selector_abstain | selector_posted | selector_invalid
selector_posted
  → reconstructed | reconstruction_failed
reconstructed
  → persisted_automatically | retained_review
retained_review
  → draft → submitted → revealed (protected only, explicit) → revised/adjudicated
```

Malformed, unknown, ambiguous, inconsistent, unsupported, cross-message, or
cross-clause core selections fail closed to review. They never become negative
truth and never silently persist.

## Current measured foundation state

The current configured private rebuild represents all 17,830 archive rows exactly
once. It is intentionally conservative: 1,453 rows currently receive normal
selector invocation, 125 may receive assistive invocation, and 16,252 skip the
model. This is a starting diagnostic, not a claim that
the deterministic analyzer is complete. Human segregation work will measure and
improve those boundaries without using weak outputs as ground truth.

## Frozen native release

`configs/sms_processing/contracts/releases/native-integration-v1.json` binds the
schemas, profiles, policies, selector prompt, algorithms, and sanitized golden
vectors by SHA-256. Native ports reproduce those bytes and decisions rather than
reinterpret prose. The release adds versioned operation results, durable
hash-chained traces, revision-bound feedback, explicit account resolution, and a
typed persistence gate while retaining every historical `/1` reader.

Automatic persistence is deliberately disabled. Android and iOS first operate in
shadow/review-only mode and record the same typed gate result that the foundation
would produce.
