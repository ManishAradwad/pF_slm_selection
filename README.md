# PocketFinancer SLM selection

This repository improves the small local model used by PocketFinancer Android to
extract posted bank and card transactions from Indian financial SMS messages. The
app is the source of truth: the training and evaluation path uses its SMS filter,
prompt/messages, `null` or four-field JSON output, parser behavior, context, and
decoding settings.

The primary workflow starts with `LiquidAI/LFM2.5-350M`, but model identity is
configuration rather than a separate pipeline. The same evaluation stage can test
another HF or GGUF model after declaring its app-facing settings.

The 203-row labeled set is a locked regression benchmark, not a clean production
test set. It has been consulted repeatedly and must not be used for production
claims or training.

## One supported workflow

Run only in the canonical WSL checkout:

```bash
cd /home/tojinotzenin/pF_slm_selection
source scripts/activate_wsl.sh

# Show readiness without changing artifacts.
python scripts/run_pocketfinancer_pipeline.py check

# Show every exact argv before a long run.
python scripts/run_pocketfinancer_pipeline.py plan

# Execute one stage at a time.
python scripts/run_pocketfinancer_pipeline.py build-data
python scripts/run_pocketfinancer_pipeline.py train
python scripts/run_pocketfinancer_pipeline.py evaluate-hf
python scripts/run_pocketfinancer_pipeline.py merge
python scripts/run_pocketfinancer_pipeline.py convert
python scripts/run_pocketfinancer_pipeline.py evaluate-gguf
```

The checked-in declaration is
[pocketfinancer-lfm2.5-350m.json](configs/pipelines/pocketfinancer-lfm2.5-350m.json).
It is the reviewable record of dataset inputs, LoRA settings, outputs, and evaluation
paths. `check` reports which stage is ready; it never downloads a model or starts
training.

## Start here

- [Documentation index](docs/README.md)
- [Latest Android-aligned RTX 4070 LoRA run](docs/experiments/POCKETFINANCER_A9_LORA_R16_S17.md)
- [Current LFM2.5-350M experiment report](docs/LFM25_350M_PIPELINE_V2.md)
- [Experiment and dataset catalog](docs/experiments/EXPERIMENT_CATALOG.md)
- [Fine-tuning, LoRA, and QLoRA primer](docs/guides/FINE_TUNING_PRIMER.md)
- [Model-improvement roadmap](docs/architecture/MODEL_IMPROVEMENT_ROADMAP.md)
- [Repository layout and migration plan](docs/architecture/REPOSITORY_LAYOUT.md)
- [Script command map](scripts/README.md)

## Environment

The canonical checkout is the native WSL2 repository at
`/home/tojinotzenin/pF_slm_selection`. A Windows Codex workspace opened through
`\\wsl.localhost` is only another view of the same files. Run Python, Git, CUDA,
training, and tests inside Ubuntu; do not duplicate this repository onto NTFS.

```bash
cd /home/tojinotzenin/pF_slm_selection
source scripts/activate_wsl.sh
python scripts/verify_gpu.py
```

The prepared machine has an RTX 4070 with 12 GB VRAM. Private SMS, derived private
datasets, adapters, checkpoints, model caches, and per-row outputs must remain local
and Git-ignored.

## Current conclusion

The historical short-prompt, synthetic, candidate-selector, and broad-hybrid work
remains available for audit, but none of it is on the default command path.

The first unified Android-compatible LoRA run is complete. On the RTX 4070 it
trained a rank-16 adapter on 154 clean-silver rows, restored its best epoch-2
checkpoint, exported BF16/Q8/Q4 GGUFs, and evaluated them through the current app
prompt, filter, and parser contract. The Q4 artifact reached 120/203 overall and
31/114 exact transactions under current tokenization; a single-BOS diagnostic
reached 125/203 and 36/114 but was not statistically decisive.

LFM2.5-350M has therefore not earned a production promotion. The next work is a
fresh human-gold test, larger clean/source-grounded learning curves, an Android
single-BOS A/B, and a like-for-like LFM2.5-2.6B ceiling/teacher baseline. Candidate
selection and task-specific neural architectures remain later options if direct
tuning plateaus. The locked 203-row benchmark is too reused to establish
production generalization.

## Verification

```bash
source scripts/activate_wsl.sh
pytest -q
ruff check lfm25 scripts tests
python scripts/check_repo_safety.py
```

No dataset or model publication is authorized by this repository.
