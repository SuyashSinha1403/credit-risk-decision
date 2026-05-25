from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from review_rag import ReviewRAGService


class ReviewRAGPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ReviewRAGService(Path("knowledge_base"), Path(tempfile.gettempdir()))
        self.applicant = {
            "credit_amount": 7882,
            "duration_in_month": 42,
            "credit_history": "A32",
        }
        self.prediction = {
            "decision": "Review",
            "pd": 0.2832,
            "decision_reason": (
                "guarantor support decreases risk; longer loan duration increases risk"
            ),
        }

    def test_prompt_preserves_human_decision_boundary(self) -> None:
        prompt = self.service._build_prompt(
            self.applicant,
            self.prediction,
            "[Source 1: credit_policy.md - Affordability Evidence]",
        )

        self.assertIn("Do not make, recommend, or imply a final approve/reject decision", prompt)
        self.assertIn("mitigating context, not a concern", prompt)
        self.assertIn("checks and actions directly related to risk-increasing factors", prompt)
        self.assertNotIn("'credit_history': 'A32'", prompt)
        self.assertNotIn("'credit_amount': 7882", prompt)

    def test_non_review_case_is_rejected_before_retrieval(self) -> None:
        with self.assertRaises(ValueError):
            self.service.summarize_review_case(
                self.applicant,
                {**self.prediction, "decision": "Approve"},
            )

    def test_generated_text_is_ascii_normalized_for_audit_output(self) -> None:
        text = self.service._normalize_generated_text(
            "\u201cReview\u201d \u2013 analyst\u2019s evidence"
        )

        self.assertEqual(text, '"Review" - analyst\'s evidence')


if __name__ == "__main__":
    unittest.main()
