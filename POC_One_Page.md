# Credit Risk Decision POC

## Objective
Build a proof-of-concept system that predicts borrower default risk, converts that risk into a business decision, explains the result, and automates logging and notifications.

## Problem Statement
Manual loan screening is slow and inconsistent. This POC shows how a lightweight machine learning model and workflow automation can support faster, more explainable credit decisions.

## Data Used
The model uses the German Credit dataset, where each row represents a borrower and the target indicates whether the borrower had a good or bad credit outcome. Features include checking account status, credit history, loan amount, employment, housing, age, and related borrower attributes.

## ML Approach
- Model: Logistic Regression
- Output: Probability of Default (PD)
- Preprocessing:
  - categorical variables handled with one-hot encoding
  - numeric variables passed through directly
- Explainability: SHAP highlights which features increased or decreased model risk

## Decision Logic
The PD is converted into:
- Approve
- Review
- Reject

The policy is selected from threshold candidates using business constraints such as approval rate, missed defaults, and expected loss.

## System Architecture
1. Borrower data enters through an n8n webhook
2. n8n calls the FastAPI `/predict` API
3. FastAPI preprocesses input, scores PD, applies policy, and returns an explanation
4. Ollama rewrites technical reasons into human-friendly language
5. n8n logs each decision into Google Sheets
6. n8n reads all decisions, builds a summary, and sends updates through Gmail and Slack

## Outputs
Each borrower decision includes:
- PD
- decision
- review flag
- decision reason
- expected loss
- policy version and thresholds

## POC Value
This POC demonstrates an end-to-end credit decision workflow with:
- explainable ML scoring
- business policy control
- API serving
- audit logging
- automated stakeholder notifications

## Current POC Limitations
- built for demonstration, not production scale
- depends on configured Gmail, Slack, Sheets, and Ollama environments
- manual review cases are flagged but not yet routed to a separate review queue
