# Documentation map

This repository owns PocketFinancer's shared SMS architecture, frozen contracts,
sanitized parity fixtures, evaluation definitions, private local workbench, and
historical model evidence. The host GGUF evaluator and encrypted native-trace
import exist; the dedicated Android and iOS native evaluation lanes are planned.
Native implementation details live in the Android and iOS repositories.

## Canonical documents

Read these in order:

1. [SMS processing architecture](architecture/SMS_PROCESSING_ARCHITECTURE.md) —
   current responsibilities, routing boundary, transparency, compatibility, and
   operating model.
2. [Cross-platform roadmap](plans/CROSS_PLATFORM_SMS_ROADMAP.md) — the one active
   implementation sequence and acceptance checkpoints.
3. [SMS evaluation strategy](plans/SMS_EVALUATION_STRATEGY.md) — contract parity,
   product quality, runtime, recovery, and device evidence.
4. [Direct SMS extractor contract](contracts/DIRECT_SMS_EXTRACTOR_CONTRACT.md) —
   model/host boundary, Unicode scalar grounding, exact money, and review rules.
5. [Native SMS integration contract](contracts/NATIVE_SMS_INTEGRATION_CONTRACT.md) —
   frozen release compatibility and model-identity provenance.
6. [SMS processing decision log](architecture/SMS_PROCESSING_DECISION_LOG.md) —
   dated decisions and rejected alternatives.
7. [Currency context and provenance](architecture/CURRENCY_CONTEXT_AND_PROVENANCE.md),
   [data taxonomy](architecture/DATA_TAXONOMY_AND_CANONICAL_LABELS.md), and
   [workbench flow](architecture/WORKBENCH_REQUIREMENTS_AND_DATA_FLOW.md).

## Platform handoffs

Each native repository has one concise next-step document:

- Android: `docs/sms-processing-next-steps.md` in `pocket-financer-android`.
- iOS: `docs/sms-processing-next-steps.md` in `pocket-financer-ios`.

The shared repository is operated from WSL, Android from Windows/Gradle and the
Android emulator/device lane, and iOS from macOS/Xcode and the simulator/iPhone
lane. Evidence is not transferable between lanes.

## Historical evidence

Use the [historical evidence index](history/SMS_PROCESSING_EVIDENCE_INDEX.md) for
dated implementation evidence, model reports, and compatibility artifacts.
Historical reports remain accurate only for the contract and runtime they
measured. The 203-row fixture is a repeatedly consulted regression set, not fresh
human gold.

## Status language

- **Implemented** means source exists and the stated local checks ran in the named
  lane; it does not imply device acceptance.
- **Planned** means the behavior requires code and usually an additive contract.
- **Historical** means preserved evidence under an older state or contract.
- **Unverified** means the required host, simulator, emulator, or physical device
  was not exercised.

Raw SMS, identifiers, private predictions, local databases, adapters, and model
weights remain ignored and local. Nothing in these documents authorizes upload,
publication, deployment, or automatic persistence.
