# CLAUDE.md - SLM Evaluation for Financial SMS

## Project Overview
Evaluation playground for selecting the optimal Small Language Model (SLM) to power a mobile financial helper app called **pocket-financer**. The app tracks the user's bank accounts and credit/debit cards, then uses an SLM to process transaction SMS alerts and maintain balance/spend data. Wallets (Simpl, slice, PayTM wallet) are out of scope — only direct bank account and card transactions.

## Quick Reference

### Environment
- **Python venv (Docker)**: `pf_docker/` — use `source pf_docker/bin/activate` when running inside the dev container
- **Python venv (WSL2)**: `pf/` — use `source pf/bin/activate` when running on bare WSL2 (broken inside Docker due to hardcoded paths)
- **Hardware**: WSL2, NVIDIA RTX 4070 (12 GB VRAM), 32 GB RAM
- **Dev container**: Python 3.11 (bullseye) + Node 20, GPU passthrough confirmed working

### Key Files
- `copilot-instructions.md` — full project context and goals
- `db_analysis.ipynb` — original (v1) SMS filtering pipeline (sender filter -> amount -> OTP exclusion -> promo exclusion -> request exclusion -> financial keywords)
- `new_pipeline.py` — current (v2) SMS filtering pipeline using positive filtering: sender filter -> amount -> masked account/card number -> transaction verb -> OTP exclusion -> collect request exclusion. Produces 1,032 clean bank/card transactions from 11,659 commercial messages.
- `build_datasets.py` — heuristic financial SMS classifier, outputs `candidates_financial.json` / `candidates_non_financial.json`
- `find_sms_db.py` — extracts `sms.db` from an iPhone iTunes backup
- `DATA/sms_classification.yaml` — lm-evaluation-harness task config for classification
- `DATA/classification_ds.jsonl` — current evaluation dataset (20 samples, all labeled Non-Transaction so far)
- `RESULTS/` — old v1 pipeline outputs in root, v2 pipeline outputs in `RESULTS/new_pipeline/`

### Data Sources (gitignored)
- `sms.db` — raw iPhone SMS SQLite database
- `all_sms.csv` / `all_sms.json` — exported SMS data
- `RESULTS/` — evaluation output artifacts

### Evaluation Stack
- **lm-evaluation-harness** — cloned in `lm-evaluation-harness/` (open-source, not our code)
- Task type: `multiple_choice` with `acc` metric
- Current prompt asks model to classify SMS as "Transaction" or "Non-Transaction"

## Conventions
- Install all dependencies into the appropriate venv (`pf_docker` in Docker, `pf` on WSL2), not globally
- Keep evaluation datasets in `DATA/`, results in `RESULTS/`
- The `lm-evaluation-harness/` directory is an upstream clone — don't modify it
- Data files (csv, json, db, xlsx) are gitignored — never commit them
- Focus on evaluation/benchmarking code, not app development

## Current State & Known Issues
- v2 pipeline (`new_pipeline.py`) is the current filtering approach — uses positive filters (require masked account/card number + transaction verb) instead of v1's negative filters (exclude promos/requests). Much fewer false negatives on real bank transactions.
- The classification dataset has only 20 samples and lacks "Transaction" positive examples
- Qwen3-0.6B evaluation returned 0% accuracy — likely a prompt/config issue rather than model failure
- The `pf` venv (WSL2) has all dependencies; `pf_docker` venv (Docker) has torch, transformers, accelerate, lm-eval installed and GPU-verified
- SMS data covers 2021–2026 but tapers off after 2024 (only 66 messages in 2026, none transactional)
