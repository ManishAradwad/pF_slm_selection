# PocketFinancer small-model improvement roadmap

## Product objective

The objective is not to prove that one named language model is good. It is to ship
the smallest practical on-device system that turns a filtered SMS into either
`null` or PocketFinancer's four transaction fields with very high accuracy.

The Android-facing interface stays stable while the implementation behind it may
evolve:

```text
sender + SMS
  -> PocketFinancer deterministic SMS filter
  -> selected on-device model/system
  -> null OR {amount, counterparty, type, account}
  -> PocketFinancer parser and persistence
```

Training and evaluation must use this same boundary. A research score obtained with
a different prompt, filter, parser, or decode path is diagnostic only.

## Current 2.6B diagnostic status

The 2026-08-05
[LFM2.5-2.6B Base LoRA diagnostic](../experiments/POCKETFINANCER_LFM25_2_6B_R16_S17.md)
is complete. A real BF16 rank-16 backward-pass probe passed at 7,351.9 MiB peak
allocated VRAM, so the controlled run used ordinary LoRA rather than QLoRA. On
the same 154-train / 29-dev silver materialization used for the 350M run, the
adapter did not reliably improve the untouched Base model on the reused 203-row
regression fixture.

This is diagnostic evidence, not a promotion result. Fresh human-gold evaluation,
Android-device validation, and deployment review remain open. Any 2.6B outputs
used as teacher labels must still be source-grounded and human-reviewed.

## Phase 1: establish the direct-model ceiling

Start with `LiquidAI/LFM2.5-350M` because it is small enough to make the mobile
quality/latency question interesting.

1. Build clean, source-grounded train/dev rows only from messages that pass the
   app's deterministic filter.
2. Serialize the exact system/user messages built by the app.
3. Train completion-only LoRA with a separately weighted first decision token.
4. Evaluate the untouched base and adapter on identical rows and decoding.
5. Merge, convert to GGUF, measure quantization loss, then test the target device.
6. Repeat with larger clean datasets and multiple seeds to produce a learning curve.

This phase tells us whether more clean data is still helping or whether the direct
350M formulation has reached a capacity ceiling.

## Phase 2: improve supervision before changing architecture

If direct LoRA plateaus, try the cheaper interventions first:

- human-review the highest-impact misses and false transactions;
- add hard negatives and templates unseen during training;
- add domain continued-pretraining only when enough unlabeled, privacy-safe local
  text exists and memorization controls are in place;
- distill behavior from a stronger local teacher into the 350M student;
- compare LoRA ranks/modules, QLoRA, and full fine-tuning only under the same data
  and evaluation budget; and
- train multiple seeds so a lucky checkpoint is not mistaken for an architecture
  improvement.

## Phase 3: task-specific neural design

The system does not have to remain a free-form text generator. Candidate designs
that retain the app's external JSON interface include:

### Multi-head extractor

A shared compact encoder/hybrid backbone feeds:

- a transaction-versus-null head;
- a debit-versus-credit head;
- amount start/end span heads;
- account start/end span heads; and
- optional counterparty start/end span heads plus an explicit absent value.

Deterministic code reconstructs JSON from source offsets. This naturally prevents
fabricated amounts/accounts and may need far fewer parameters than a general chat
model. It does, however, require high-quality span annotations and Android runtime
support for the chosen architecture.

### Grounded selector

Deterministic code proposes source spans and a compact model ranks/selects them.
The existing candidate-selector experiment is evidence for this direction, but a
future version must be trained and evaluated through the same app-facing pipeline.

### Teacher-distilled compact model

A larger model supplies probability targets or reviewed structured labels; a small
student learns the task-specific decision boundary. Distillation can be applied to
the direct generator, multi-head extractor, or grounded selector.

## Phase 4: modify or design the backbone

Adapting Liquid's convolution/attention mixture, reducing layers or width, changing
the tokenizer, or training a new compact backbone is possible. It is the most
expensive option because it introduces pretraining, kernel/runtime, conversion,
quantization, and Android-integration risk simultaneously.

Only start backbone design after the earlier phases provide:

- a sufficiently large, diverse, human-reviewed dataset;
- error slices showing what the current representation cannot learn;
- a strong teacher and distillation baseline;
- stable Android-level latency/RAM targets; and
- evidence that a simpler head/formulation cannot meet quality.

The output of that work would still be evaluated by the same PocketFinancer module,
so model architecture can change without moving the product goalposts.

## Decision rule

At every phase, compare exact transaction quality, false transactions, misses,
field accuracy, unseen-template performance, model size, p50/p95 latency, peak RAM,
thermals, and battery. Advance to a more complex phase only when the simpler phase
has a measured limitation that the next design specifically addresses.
