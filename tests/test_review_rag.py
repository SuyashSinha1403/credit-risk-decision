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
            "[Source 1: Guidelines on loan origination and monitoring, page 36]",
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

    def test_retrieval_query_excludes_mitigating_factors(self) -> None:
        query = self.service._build_query(self.applicant, self.prediction)

        self.assertIn("longer loan duration increases risk", query)
        self.assertNotIn("guarantor support decreases risk", query)

    def test_review_note_guard_blocks_consumer_irrelevant_actions(self) -> None:
        violations = self.service._note_violations(
            "2. Evidence to verify\nAssess refinancing in capital markets [Source 1].",
            self.prediction,
        )

        self.assertTrue(any("refinanc" in item for item in violations))
        self.assertTrue(any("capital market" in item for item in violations))

    def test_official_pdf_sources_are_loaded_as_the_knowledge_base(self) -> None:
        files = self.service._get_knowledge_files()

        self.assertTrue(files)
        self.assertTrue(all(path.suffix == ".pdf" for path in files))
        self.assertTrue(
            any("EBA_GL_2020_06" in path.name for path in files),
        )


if __name__ == "__main__":
    unittest.main()
