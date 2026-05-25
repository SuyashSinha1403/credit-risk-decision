# Credit Policy Reference

## Policy Scope And Decision Boundary

The logistic regression model estimates probability of default (PD). The stored
threshold policy maps PD into `Approve`, `Review`, or `Reject`. Retrieval and
generation operate only after a `Review` result and must not change the score or
make a final lending decision.

## Review Band Rules

A `Review` result means PD is between the selected lower and upper thresholds.
The automated system must pause straight-through processing and collect human
review evidence. The analyst must record supporting evidence and a final human
disposition before the application is closed.

## Affordability Evidence

When the risk explanation includes a higher credit amount, longer loan duration,
or installment burden, the reviewer should request affordability evidence. This
includes recent income proof and existing monthly obligation information where
available. The analyst should compare repayment burden with the stated borrower
profile and document any inconsistency.

## Thin Liquidity Signals

When low savings or a weak checking-account signal contributes to review, request
recent account evidence or other liquidity support available in the process.
Missing liquidity evidence must be recorded as an unresolved review item rather
than inferred by the language model.

## Credit History Escalation

When the application indicates delayed repayment or critical credit history, the
reviewer should examine repayment evidence and escalate unresolved adverse
history to a senior credit reviewer. The note must identify the check required;
it must not characterize unverified history as resolved.
