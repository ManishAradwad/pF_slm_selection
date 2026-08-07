> **Execution update (2026-08-05):** This proposal has been executed.
> [The completed run report](POCKETFINANCER_LFM25_2_6B_R16_S17.md) supersedes it for results and decisions; the original plan is preserved below.

# Future LFM2.5-2.6B evaluation plan

Status: proposed only; no model download, baseline, or fine-tuning run has been
performed in this session.

## Why it is interesting

Liquid released LFM2.5-2.6B on 2026-08-04. The checkpoint has 2.69B parameters, a
128K vocabulary, a 131,072-token trained context, and about 34T pretraining tokens.
The post-trained checkpoint was optimized for instruction following, tool use, and
agentic workflows; a separate Base checkpoint is intended for heavier customization.
The 30-layer hybrid has 22 short-convolution blocks and eight grouped-query
attention blocks. "Agentic" describes its post-training and use cases, not a
different class of neural network that prevents extraction work.

Official repositories provide Transformers, GGUF, ONNX, and MLX variants. The GGUF
Q4_0 file is 1,593,894,720 bytes and Q4_K_M is 1,674,454,848 bytes. Those download
sizes are plausible for a high-memory Android tier, but runtime memory also includes
KV cache, native buffers, prompt state, and app overhead.

## Agentic training: help or distraction?

It can cut both ways. Stronger instruction following may make the post-trained model
better at obeying the schema and rejection policy. Its mandatory reasoning behavior
and broad agent policy may also add latency or interfere with emitting one exact
JSON object. The post-trained model is a reasoning checkpoint whose template opens
a `<think>` section, so it belongs on PocketFinancer's thinking/two-pass path rather
than the 350M direct path. Generic agent benchmarks do not answer this extraction
question.

The Base checkpoint may adapt more cleanly under SFT, but it may require more data
to learn the chat/output behavior that the post-trained checkpoint already has.
Neither should be selected by intuition alone.

The official checkpoint loads through `AutoModelForCausalLM` with Transformers 5.x,
and Liquid documents TRL/PEFT LoRA plus Unsloth QLoRA. The completed 350M run used
Transformers 5.6.2, so the major version is now compatible. Keep the 2.6B run
configuration isolated and inspect the real hybrid module names and trainable
parameter count before the first backward pass.

## Fair experiment

1. Freeze the PocketFinancer contract, decoder, grammar, and a new human-gold test.
2. Run untouched post-trained and Base checkpoints under identical prompt and
   sampling settings. Record raw reasoning, final JSON, validity, and latency.
3. Fine-tune both only if their baselines justify it. Start with rank-16 or rank-32
   LoRA; use QLoRA if a measured BF16 probe cannot keep the required sequence and
   effective batch within 12 GB.
4. Train on the identical clean rows, seeds, stopping rule, and objective used for
   350M. Do not give 2.6B a curated dataset advantage.
5. Merge, export Q8/reference and Q4_K_M, verify HF-to-GGUF parity, then benchmark
   the target Android devices.
6. Compare quality, model size, cold load, p50/p95 latency, RAM, sustained thermals,
   and battery - not benchmark accuracy alone.

Use the post-trained model first as a quality ceiling and teacher. Its structured
labels still require grounding/human review, and the 350M student should learn only
PocketFinancer's direct `null`/JSON output rather than hidden reasoning. Treat 2.6B
as a premium-device deployment challenger only if its accuracy gain justifies the
roughly 1.67 GB Q4_K_M file and thinking latency.

The evaluation matrix should include direct generation and, if Android adopts it,
the grounded candidate protocol. That separates model-capacity gains from gains due
to task decomposition.

## Hardware expectation

BF16 weights alone are roughly 5.4 GB for 2.69B parameters. LoRA avoids optimizer
state for the frozen base, but activations for the long prompt and Liquid's hybrid
blocks still matter. A 12 GB RTX 4070 may fit a carefully configured BF16 LoRA run;
QLoRA is the safer initial memory plan. A short backward-pass probe must decide this,
not a paper estimate.

## License checkpoint

The model uses LFM Open License v1.0. Commercial use is licensed only while the
user/legal entity remains below USD 10 million annual revenue; redistribution also
requires retaining the license and notices. This is a technical summary, not legal
advice. Confirm the applicable entity and current terms before shipping.

## Official references

- [Liquid release article](https://www.liquid.ai/blog/lfm2-5-2-6b)
- [Post-trained model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
- [Base checkpoint](https://huggingface.co/LiquidAI/LFM2.5-2.6B-Base)
- [Official GGUF variants](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- [TRL/PEFT fine-tuning](https://docs.liquid.ai/lfm/fine-tuning/trl)
- [Unsloth/QLoRA fine-tuning](https://docs.liquid.ai/lfm/fine-tuning/unsloth)
- [LFM license guide](https://docs.liquid.ai/lfm/help/model-license)
