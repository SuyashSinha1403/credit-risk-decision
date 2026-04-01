# Credit Risk Decision

This repository contains a credit risk decision system that combines machine learning, explainability, API serving, and workflow automation.

## Objective

Build a simple end-to-end system that:

1. accepts borrower information
2. predicts probability of default (PD)
3. converts risk into a business decision
4. explains the decision
5. logs decisions for audit
6. sends summary updates automatically

## Solution Overview

The project is built with:

- **FastAPI** for serving predictions through an API
- **Logistic Regression** for credit risk scoring
- **SHAP** for model explanation
- **n8n** for workflow automation
- **Ollama** for rewriting technical explanations into plain language
- **Google Sheets** for decision logging
- **Gmail** and **Slack** for summary notifications

## How It Works

### 1. Model training
The model is trained on the German Credit dataset. Each row represents a borrower and the target indicates whether the borrower had a good or bad credit outcome.

### 2. Data preprocessing
Before training:

- column names are cleaned
- categorical features are encoded using one-hot encoding
- numeric features are passed through directly
- the dataset is split into train and test sets

### 3. Prediction
The trained Logistic Regression model returns **PD (Probability of Default)** for a new borrower.

### 4. Decision policy
The PD is converted into:

- `Approve`
- `Review`
- `Reject`

using business thresholds selected from candidate policy sets.

### 5. Explanation
SHAP identifies which features pushed the model prediction up or down. Ollama then rewrites that explanation into a more human-friendly format.

### 6. Workflow automation
n8n automates the operational flow:

- receives borrower data through a webhook
- calls the prediction API
- calls Ollama for explanation wording
- writes each decision to Google Sheets
- reads logged decisions
- builds a summary
- sends updates to Gmail and Slack

## Repository Structure

```text
.
|-- data/
|   |-- german.data
|   |-- german_decoded.csv
|   `-- DATA_DICTIONARY.md
|-- decode_german_dataset.py
|-- main.py
|-- requirements.txt
|-- n8n_credit_risk_combined_workflow.json
|-- n8n_workflow_setup.md
`-- .gitignore
```

## Dataset

The repository includes both the raw and readable versions of the German Credit dataset:

```text
data/german.data
data/german_decoded.csv
```

- `german.data` is the original coded dataset from the source
- `german_decoded.csv` replaces code values like `A11`, `A43`, and `A201` with human-readable meanings

This makes the project reproducible without downloading training data at API startup, while also making the data understandable for GitHub readers.

## Model Artifacts

On the first API startup, the app trains the model once and saves reusable artifacts in:

```text
artifacts/credit_risk_artifacts.joblib
```

Later API restarts load the saved artifacts instead of retraining.

## API Output

The prediction API returns:

- `pd`
- `decision`
- `review_required`
- `decision_reason`
- `applicant_expected_loss`
- `policy_version`
- `policy_low_threshold`
- `policy_high_threshold`

## Workflow Nodes

The combined n8n workflow uses:

- **Webhook**: receives borrower input
- **Predict API**: calls the FastAPI `/predict` endpoint
- **Ollama**: rewrites technical explanation text
- **Prepare Audit Record**: formats the borrower result
- **Google Sheets**: logs each borrower decision
- **Read Decisions**: reads all logged rows
- **Build Summary**: calculates totals and summary metrics
- **Gmail Summary**: sends the summary email
- **Slack Summary**: sends the summary message to Slack

## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the API:

```powershell
python -m uvicorn main:app --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

## n8n Setup

Import the main workflow file into n8n:

```text
n8n_credit_risk_combined_workflow.json
```

Then connect your credentials for:

- Google Sheets
- Gmail
- Slack
- Ollama

Update placeholders such as:

- Google Sheet URL
- Slack channel
- Gmail recipient

## Ollama Note

This project uses Ollama only for explanation wording, not for the credit decision itself.
The workflow is currently configured to use the model:

```text
gemma3:4b
```

If you prefer a different Ollama model, update the model name inside the n8n Ollama node.

## Scope and Limitations

This is a demo-oriented project, not a production system.

Current limitations:

- credentials are environment-specific
- manual review cases are flagged but not routed to a human review queue
- the system is designed for demonstration and learning, not production-scale deployment

## Summary

This project demonstrates how a structured borrower input can move through an explainable ML scoring pipeline, become a business decision, get logged for audit, and trigger automated stakeholder communication through a workflow engine.
