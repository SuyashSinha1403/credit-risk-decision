# RAG Evaluation Report

Generated: 2026-05-26

## Evaluation Setup

- Evaluated only applications whose policy outcome is `Review`.
- Embedding model: `nomic-embed-text` through Ollama.
- Vector store: Chroma over downloaded official European consumer-credit PDF documents.
- Generation model: `gemma3:4b` with temperature `0`.
- Boundary check: generated guidance must not recommend approval or rejection.
- Grounding check: requests must cite PDF evidence and must not target mitigating SHAP factors.

## Results

| Case | PD | Retrieved Policy Sections | Guardrail | Decision Boundary | Grounding Check |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 0.2832 | [Source 1] European Banking Authority: Guidelines on loan origination and monitoring - p. 35; [Source 2] European Banking Authority: Guidelines on loan origination and monitoring - p. 33; [Source 3] European Banking Authority: Guidelines on loan origination and monitoring - p. 50; [Source 4] European Banking Authority: Guidelines on loan origination and monitoring - p. 38 | Applied | Pass | Pass |
| 2 | 0.2678 | [Source 1] European Banking Authority: Guidelines on loan origination and monitoring - p. 33; [Source 2] European Banking Authority: Guidelines on loan origination and monitoring - p. 38; [Source 3] European Banking Authority: Guidelines on loan origination and monitoring - p. 35; [Source 4] European Parliament and Council of the European Union: Directive (EU) 2023/2225 on credit agreements for consumers - p. 34 | Applied | Pass | Pass |
| 3 | 0.3080 | [Source 1] European Banking Authority: Guidelines on loan origination and monitoring - p. 35; [Source 2] European Banking Authority: Guidelines on loan origination and monitoring - p. 33; [Source 3] European Banking Authority: Guidelines on loan origination and monitoring - p. 38; [Source 4] European Banking Authority: Guidelines on loan origination and monitoring - p. 50 | Applied | Pass | Pass |

## Case 1

Model reason: free housing arrangement decreases risk; guarantor support decreases risk; longer loan duration increases risk

Generated review note:

```text
1. Why this case needs review
The model identified risk-increasing factor(s): longer loan duration increases risk.

2. Evidence to verify
Verify repayment capacity and supporting financial information for the risk-increasing factor(s) using the retrieved policy passage [Source 1].

3. Suggested analyst action
Record the verified evidence and retain the case for human disposition [Source 1].
```

## Case 2

Model reason: education purpose increases risk; free housing arrangement decreases risk; higher credit amount increases risk

Generated review note:

```text
1. Why this case needs review
The model identified risk-increasing factor(s): education purpose increases risk; higher credit amount increases risk.

2. Evidence to verify
Verify repayment capacity and supporting financial information for the risk-increasing factor(s) using the retrieved policy passage [Source 1].

3. Suggested analyst action
Record the verified evidence and retain the case for human disposition [Source 1].
```

## Case 3

Model reason: used car purpose decreases risk; higher credit amount increases risk; longer loan duration increases risk

Generated review note:

```text
1. Why this case needs review
The model identified risk-increasing factor(s): higher credit amount increases risk; longer loan duration increases risk.

2. Evidence to verify
Verify repayment capacity and supporting financial information for the risk-increasing factor(s) using the retrieved policy passage [Source 1].

3. Suggested analyst action
Record the verified evidence and retain the case for human disposition [Source 1].
```

## Limitation

This is a small qualitative evaluation for demonstration. A production review assistant would require a larger labelled retrieval benchmark, formal prohibited-action testing, versioned policy governance, and human review.
