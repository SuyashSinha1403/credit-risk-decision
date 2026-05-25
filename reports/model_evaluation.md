# Model Evaluation Report

Generated: 2026-05-26

## Evaluation Setup

- Dataset: German Credit dataset (1,000 applications)
- Target: bad credit outcome encoded as default (`1`)
- Model: Logistic Regression with one-hot encoded categorical features
- Validation: stratified 80/20 train-test split with `random_state=42`
- Test set: 200 applications (60 defaults)

## Predictive Performance

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.808 |
| Gini | 0.616 |
| KS statistic | 0.574 |
| Accuracy at PD >= 0.50 | 0.785 |
| Default precision at PD >= 0.50 | 0.673 |
| Default recall at PD >= 0.50 | 0.550 |
| Default F1-score at PD >= 0.50 | 0.606 |

## Confusion Matrix At PD >= 0.50

| Actual / Predicted | Non-default | Default |
| --- | ---: | ---: |
| Non-default | 124 | 16 |
| Default | 27 | 33 |

## Threshold Policy Comparison

| Approve / Reject thresholds | Approval rate | Missed defaults | False rejects | Booked expected loss | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.15-0.35 | 36.0% | 7 | 30 | 13,353.03 | Yes |
| 0.20-0.40 | 44.5% | 9 | 26 | 19,015.80 | No |
| 0.25-0.50 | 50.5% | 10 | 16 | 27,787.19 | No |

## Selected Policy Outcome

- Selected thresholds: `0.15-0.35`
- Configured constraints met: No
- Selection basis: No candidate met both configured approval-rate and missed-default constraints; selected as the lowest-booked-expected-loss fallback.
- Approved on held-out test set: 72
- Sent to review on held-out test set: 51
- Rejected on held-out test set: 77

## Interpretation

The model score is evaluated independently from the business decision policy. No candidate policy satisfies both configured operating constraints on the held-out set, so the application exposes its fallback selection rather than presenting it as constraint-compliant. The review band enables human assessment and RAG-supported evidence checks rather than an autonomous decision.

## Limitations Before Production Use

- The source dataset includes demographic and foreign-worker attributes. A real lending deployment would require a governed feature-eligibility decision and fairness testing.
- The dataset is small and historical; performance should not be interpreted as evidence of current portfolio calibration.
- The selected threshold policy is suitable for demonstrating workflow behavior, not for production approval decisions.
