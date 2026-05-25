# Review Explanation Guidelines

## Grounded Inputs

A reviewer note may use only the borrower application fields, the model PD and
SHAP reason, and retrieved passages from the local policy knowledge base. Each
requested review action should cite a retrieved source label.

## Required Note Structure

The note should explain why the application is in the review band, identify the
specific evidence a human should verify, and state the next analyst action. It
should be concise enough to place in an audit log.

## Prohibited Statements

The note must not provide an approval or rejection recommendation, replace the
PD score, claim missing evidence exists, or contradict threshold policy. When
the retrieved context is not sufficient, it must explicitly state that a policy
or evidence gap remains.

## Human In The Loop Positioning

The LLM summarizes retrieved guidance for ambiguous applications only. The model
remains the scoring engine and the human reviewer retains responsibility for
any reviewed disposition.
