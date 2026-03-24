# Copilot Instructions: SLM Evaluation for Financial SMS

## Repository Purpose
This repository is dedicated solely to evaluating and selecting the optimal Small Language Model (SLM) for a separate financial helper application. The target application will use the chosen SLM to process financial SMS messages extracted from an iPhone database. 
**This project is strictly the evaluation playground and dataset preparation pipeline, not the app itself.**

## Target Application Details
For the target application, the selected SLM will be responsible for two main tasks:
1. **Classification**: Determine whether an incoming SMS is a financial transaction or not.
2. **Extraction**: If it is a financial transaction, extract the following structured payload:
   - `amount`
   - `merchant`
   - `date`
   - `type`
   - `account`

## Key Objectives
1. **Dataset Preparation**: Process and parse iPhone SMS exports (e.g., `all_sms.csv`) into structured datasets suitable for model evaluation.
2. **SLM Benchmarking**: Utilize the open-source [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate various SLMs against our custom financial SMS dataset.
3. **Model Selection**: Compare models to find the perfect LLM/SLM that balances accuracy, speed, and efficiency for the financial SMS parsing use case.

## Current Data Processing Pipeline
To prepare the raw SMS data for SLM evaluation, we currently apply a multi-stage heuristic filtering process (documented in `db_analysis.ipynb`) to separate likely financial transactions from noise. The pipeline consists of the following filtering stages:
1. **Sender Filtering**: Exclude regular personal mobile numbers, keeping only commercial brands and numeric shortcodes.
2. **Stage 1 (Amount Requirement)**: Keep messages containing a currency indicator (`Rs.`, `INR`, `₹`) near a numeric amount.
3. **Stage 2 (OTP Exclusion)**: Remove messages containing authentication/verification keywords (`otp`, `pin`, `verification code`, etc.).
4. **Stage 3 (Promo Exclusion)**: Remove messages containing promotional keywords (`offer`, `discount`, `cashback`, `win`, etc.).
5. **Stage 4 (Request Exclusion)**: Remove messages that are requests for money (`request`, `requested`, etc.).
6. **Stage 5 (Financial Context)**: Ensure the remaining messages contain valid transactional verbs or entities (`debited`, `credited`, `account`, `balance`, `txn`, etc.).

## Guidelines for AI Assistants (Copilot)
- **Environment**: A Python virtual environment named `pf` has been created. Always use this generated environment (`pf`) when working on this project (e.g., `source pf/bin/activate`). Install any new libraries and dependencies strictly within this environment.
- **Focus**: Prioritize code related to data wrangling, dataset formatting (for `lm-evaluation-harness`), and evaluation scripting rather than app development.
- **Framework**: Assume tight integration with `lm-evaluation-harness`. Provide configurations, wrappers, and dataset tasks that align with its ecosystem.
- **Workflow Speed**: The goal is to rapidly iterate through different models. Help write automated, repeatable evaluation pipelines.
