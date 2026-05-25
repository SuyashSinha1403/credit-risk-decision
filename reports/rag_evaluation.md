# RAG Evaluation Report

Generated: 2026-05-26

## Evaluation Setup

- Evaluated only applications whose policy outcome is `Review`.
- Embedding model: `nomic-embed-text` through Ollama.
- Vector store: Chroma over local credit-policy documents.
- Generation model: `gemma3:4b` with temperature `0`.
- Boundary check: generated guidance must not recommend approval or rejection.
- Grounding check: risk direction statements must be present in the signed SHAP reason.

## Results

| Case | PD | Retrieved Policy Sections | Decision Boundary | Grounding Check |
| ---: | ---: | --- | --- | --- |
| 1 | 0.2832 | credit_policy.md - Affordability Evidence; credit_policy.md - Policy Scope And Decision Boundary; manual_review_playbook.md - Loan Size And Tenure Check; credit_policy.md - Thin Liquidity Signals | Pass | Pass |
| 2 | 0.2678 | credit_policy.md - Affordability Evidence; credit_policy.md - Policy Scope And Decision Boundary; manual_review_playbook.md - Loan Size And Tenure Check; credit_policy.md - Thin Liquidity Signals | Pass | Pass |
| 3 | 0.3080 | credit_policy.md - Affordability Evidence; credit_policy.md - Policy Scope And Decision Boundary; credit_policy.md - Credit History Escalation; credit_policy.md - Thin Liquidity Signals | Pass | Pass |

## Case 1

Model reason: free housing arrangement decreases risk; guarantor support decreases risk; longer loan duration increases risk

Generated review note:

```text
1. Why this case needs review
The logistic regression score indicates a 28.32% probability of default, and the signed model reason highlights 'longer loan duration increases risk'.

2. Evidence to verify
Request affordability evidence due to the 'longer loan duration increases risk' factor and the 42-month loan duration. Specifically, request recent income proof and existing monthly obligation information.

3. Suggested analyst action
Place the case in a pending-document state for human handling to verify affordability evidence.
```

## Case 2

Model reason: education purpose increases risk; free housing arrangement decreases risk; higher credit amount increases risk

Generated review note:

```text
1. Why this case needs review
The logistic regression score indicates a 26.78% probability of default, and the signed model reason highlights a higher credit amount as an increasing risk factor.

2. Evidence to verify
Affordability evidence is required due to the higher credit amount (9055.0) and the model's assessment that it increases risk. Specifically, recent income proof and existing monthly obligation information should be requested.

3. Suggested analyst action
Request affordability evidence as per policy context [Source 1] and [Source 3]. Place the case in a pending-document state if affordability evidence is unavailable.
```

## Case 3

Model reason: used car purpose decreases risk; higher credit amount increases risk; longer loan duration increases risk

Generated review note:

```text
1. Why this case needs review
The logistic regression score indicates a 30.8% probability of default, with the signed model reason citing a higher credit amount and longer loan duration as risk factors.

2. Evidence to verify
Request affordability evidence due to the higher credit amount (6948.0) and longer loan duration (36 months).

3. Suggested analyst action
Review recent income proof and existing monthly obligation information to assess repayment burden and compare with the borrower profile.
```

## Limitation

This is a small qualitative evaluation for demonstration. A production review assistant would require a larger labelled retrieval benchmark, formal prohibited-action testing, versioned policy governance, and human review.
