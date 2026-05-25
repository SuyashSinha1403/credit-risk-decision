from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from main import (  # noqa: E402
    LGD,
    REVIEW_BOOKING_RATE,
    THRESHOLD_SETS,
    build_processed_frame,
    load_dataset,
    make_decision,
    train_artifacts,
)


def build_report() -> str:
    data = load_dataset()
    features = data.drop(columns=["target"])
    target = data["target"]
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    artifacts = train_artifacts()
    processed_test = build_processed_frame(
        artifacts["preprocessor"], x_test, artifacts["feature_names"]
    )
    pd_scores = artifacts["model"].predict_proba(processed_test)[:, 1]
    binary_predictions = (pd_scores >= 0.5).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, pd_scores)

    auc = roc_auc_score(y_test, pd_scores)
    gini = 2 * auc - 1
    ks = max(true_positive_rate - false_positive_rate)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, binary_predictions
    ).ravel()

    selected_policy = artifacts["best_policy"]
    policy_rows = []
    for low, high in THRESHOLD_SETS:
        decisions = [make_decision(score, low, high) for score in pd_scores]
        policy_frame = pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "pd": pd_scores,
                "decision": decisions,
                "loan_amount": x_test["credit_amount"].to_numpy(),
            }
        )
        policy_frame["expected_loss"] = (
            policy_frame["pd"] * LGD * policy_frame["loan_amount"]
        )
        approval_rate = (policy_frame["decision"] == "Approve").mean()
        missed_defaults = len(
            policy_frame[
                (policy_frame["actual"] == 1)
                & (policy_frame["decision"] == "Approve")
            ]
        )
        false_rejects = len(
            policy_frame[
                (policy_frame["actual"] == 0)
                & (policy_frame["decision"] == "Reject")
            ]
        )
        booked_expected_loss = policy_frame.loc[
            policy_frame["decision"] == "Approve", "expected_loss"
        ].sum() + REVIEW_BOOKING_RATE * policy_frame.loc[
            policy_frame["decision"] == "Review", "expected_loss"
        ].sum()
        policy_rows.append(
            (
                f"{low:.2f}-{high:.2f}",
                approval_rate,
                missed_defaults,
                false_rejects,
                booked_expected_loss,
                f"{low:.2f}-{high:.2f}" == selected_policy["thresholds"],
            )
        )

    chosen_low = float(selected_policy["low"])
    chosen_high = float(selected_policy["high"])
    selected_decisions = [make_decision(score, chosen_low, chosen_high) for score in pd_scores]
    decision_counts = pd.Series(selected_decisions).value_counts()

    lines = [
        "# Model Evaluation Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Evaluation Setup",
        "",
        "- Dataset: German Credit dataset (1,000 applications)",
        "- Target: bad credit outcome encoded as default (`1`)",
        "- Model: Logistic Regression with one-hot encoded categorical features",
        "- Validation: stratified 80/20 train-test split with `random_state=42`",
        f"- Test set: {len(y_test)} applications ({int(y_test.sum())} defaults)",
        "",
        "## Predictive Performance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| ROC-AUC | {auc:.3f} |",
        f"| Gini | {gini:.3f} |",
        f"| KS statistic | {ks:.3f} |",
        f"| Accuracy at PD >= 0.50 | {accuracy_score(y_test, binary_predictions):.3f} |",
        f"| Default precision at PD >= 0.50 | {precision_score(y_test, binary_predictions):.3f} |",
        f"| Default recall at PD >= 0.50 | {recall_score(y_test, binary_predictions):.3f} |",
        f"| Default F1-score at PD >= 0.50 | {f1_score(y_test, binary_predictions):.3f} |",
        "",
        "## Confusion Matrix At PD >= 0.50",
        "",
        "| Actual / Predicted | Non-default | Default |",
        "| --- | ---: | ---: |",
        f"| Non-default | {true_negative} | {false_positive} |",
        f"| Default | {false_negative} | {true_positive} |",
        "",
        "## Threshold Policy Comparison",
        "",
        "| Approve / Reject thresholds | Approval rate | Missed defaults | False rejects | Booked expected loss | Selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for thresholds, approval_rate, missed_defaults, false_rejects, expected_loss, selected in policy_rows:
        lines.append(
            f"| {thresholds} | {approval_rate:.1%} | {missed_defaults} | "
            f"{false_rejects} | {expected_loss:,.2f} | {'Yes' if selected else 'No'} |"
        )
    lines.extend(
        [
            "",
            "## Selected Policy Outcome",
            "",
            f"- Selected thresholds: `{selected_policy['thresholds']}`",
            f"- Configured constraints met: {'Yes' if selected_policy['constraints_met'] else 'No'}",
            f"- Selection basis: {selected_policy['selection_reason']}",
            f"- Approved on held-out test set: {int(decision_counts.get('Approve', 0))}",
            f"- Sent to review on held-out test set: {int(decision_counts.get('Review', 0))}",
            f"- Rejected on held-out test set: {int(decision_counts.get('Reject', 0))}",
            "",
            "## Interpretation",
            "",
            "The model score is evaluated independently from the business decision policy. "
            "No candidate policy satisfies both configured operating constraints on the held-out "
            "set, so the application exposes its fallback selection rather than presenting it as "
            "constraint-compliant. The review band enables human assessment and RAG-supported "
            "evidence checks rather than an autonomous decision.",
            "",
            "## Limitations Before Production Use",
            "",
            "- The source dataset includes demographic and foreign-worker attributes. A real lending "
            "deployment would require a governed feature-eligibility decision and fairness testing.",
            "- The dataset is small and historical; performance should not be interpreted as evidence "
            "of current portfolio calibration.",
            "- The selected threshold policy is suitable for demonstrating workflow behavior, not "
            "for production approval decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out credit model metrics.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    report = build_report()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
