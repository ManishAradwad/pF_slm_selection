# Fine-tuning, LoRA, and QLoRA for this project

## The short version

A pretrained language model has learned general language patterns. Fine-tuning
continues training it on examples of the exact behavior we want. For PocketFinancer,
one supervised example contains the same chat messages the app will send and the
assistant answer that should follow: either literal `null` or the required JSON.

The training loop tokenizes the conversation, asks the model to predict the next
token, and computes cross-entropy loss only on the assistant completion. Prompt
tokens provide context but are masked from the loss. An optimizer changes weights
to make the correct completion more likely.

Three splits play different roles:

- **Train** examples update weights.
- **Development/tuning** examples never update weights. They select the epoch,
  hyperparameters, and stopping point.
- **Test** examples are opened only after decisions are frozen. They estimate
  generalization. Our old 203-row set is no longer suitable for this role because
  it has influenced development repeatedly.

## Full fine-tuning

Full fine-tuning updates every model parameter. It offers maximum flexibility but
needs much more optimizer memory and can overfit or damage general capabilities when
the dataset is small. Updating all 350M parameters is unnecessary for the current
160-row clean task, and full AdamW tuning of a 2.69B model would be a poor first use
of a 12 GB GPU.

## LoRA

LoRA freezes the original matrix `W` and learns a low-rank correction:

```text
effective weight = W + (alpha / rank) * B * A
```

`A` and `B` are much smaller than `W`. At inference we can keep the adapter
separate or merge the correction into the base checkpoint. Rank controls adapter
capacity; alpha controls its scale; dropout regularizes the adapter path.

Our clean 350M runs use rank 16, alpha 32, dropout 0.05, and 5,996,544 trainable
parameters - about 1.66% of the model. The base stays frozen. The clean direct run
peaked near 3.9 GiB VRAM, so ordinary BF16 LoRA is simpler than QLoRA for 350M.

LoRA is not automatically low quality. It is often the right experiment because
we need to teach a narrow behavior, not relearn language. Its limits appear when
the required transformation needs more capacity than the chosen rank/modules, or
when the underlying model cannot represent the task well enough.

## QLoRA

QLoRA combines two ideas during training:

1. Load the frozen base weights in a memory-saving 4-bit representation.
2. Train LoRA adapters in a higher precision such as BF16/FP16.

Gradients and optimizer state apply to the adapters, not to all 4-bit base weights.
This can make a 2.69B model practical on the RTX 4070. It is not the same as taking
a trained model and exporting a Q4 GGUF for Android: training-time 4-bit loading
and deployment quantization are separate steps and require separate parity tests.

QLoRA has tradeoffs: more kernel/library complexity, slower operations in some
setups, and potential architecture-specific incompatibilities. We should use it
for 2.6B only after a short real backward-pass probe confirms Liquid's hybrid
modules, PEFT targets, bitsandbytes, sequence length, and memory headroom.

## What our SFT objective changed

Literal `null` is only a few tokens, while a transaction JSON is much longer. A
plain token-average loss therefore lets transaction completions dominate the
objective even when row counts look balanced. The current trainer:

- masks prompt tokens;
- computes completion loss as a mean per example;
- applies the row's provenance/sample weight; and
- gives the first supervised decision token weight 3.0.

That first token decides transaction versus null. The design prevents long JSON
answers from drowning out hard-negative learning.

## Direct generation versus candidate selection

Direct generation asks the model to decide whether a transaction happened, infer
debit/credit, and reproduce arbitrary source spans exactly. That is demanding for
350M parameters.

The candidate path lets deterministic code extract source-backed amount, account,
and counterparty candidates. The model selects compact IDs; code reconstructs the
final PocketFinancer JSON from exact source offsets. This prevents fabricated field
values and focuses the model on semantic ranking. It performed better, but Android
does not currently implement that protocol, so it is an architecture experiment,
not a drop-in model artifact.

## Will a larger dataset improve 350M?

Probably, if it adds clean information. More duplicated or mislabeled rows can make
the model worse. The invalidated silver set is a concrete example: apparent recall
improved while 122 counterparty labels were not properly source-grounded.

High-value additions are:

- human-reviewed posted transactions and realistic hard negatives;
- template-, sender-, time-, and ideally user-disjoint coverage;
- generic `txn` messages where direction is not spelled out;
- valid transactions with no counterparty;
- multiple plausible counterparty spans, especially `to`/`by` distractors;
- OTPs, requests, mandates, reversals, failures, balances, offers, and recharges;
- long messages, multiple amounts, and unfamiliar banks/templates; and
- source-grounded field offsets and explicit label provenance.

We should build a learning curve instead of guessing. Freeze a fresh human-gold
test set, train on increasing clean subsets with three seeds, and plot exact match,
ghosts, misses, field accuracy, and seed variation. If the curve is still rising,
more data is justified. If it plateaus far below the target while training fit is
high, model capacity or the output formulation is the likely limit.

## Practical order of experiments

1. Freeze a new human-reviewed, template-held-out test set.
2. Make the evaluator reproduce the Android runtime exactly.
3. Train 350M direct and candidate variants on successively larger clean datasets.
4. Merge/export the winning adapters and test GGUF parity.
5. Compare 350M, LFM2.5-2.6B, and other finalists on identical data and decoding.
6. Measure RAM, cold load, latency, thermals, and battery on target phones.

Model size should be the variable we change after the data and contract are fixed.
