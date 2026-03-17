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

## Guidelines for AI Assistants (Copilot)
- **Focus**: Prioritize code related to data wrangling, dataset formatting (for `lm-evaluation-harness`), and evaluation scripting rather than app development.
- **Framework**: Assume tight integration with `lm-evaluation-harness`. Provide configurations, wrappers, and dataset tasks that align with its ecosystem.
- **Workflow Speed**: The goal is to rapidly iterate through different models. Help write automated, repeatable evaluation pipelines.
