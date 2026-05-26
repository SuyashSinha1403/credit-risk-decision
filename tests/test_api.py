from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from main import app, generate_reason


APPROVE_APPLICANT = {
    "status_of_existing_checking_account": "A11",
    "duration_in_month": 6,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1169,
    "savings_account_bonds": "A65",
    "present_employment_since": "A75",
    "installment_rate_in_percentage_of_disposable_income": 4,
    "personal_status_and_sex": "A93",
    "other_debtors_guarantors": "A101",
    "present_residence_since": 4,
    "property": "A121",
    "age_in_years": 67,
    "other_installment_plans": "A143",
    "housing": "A152",
    "number_of_existing_credits_at_this_bank": 2,
    "job": "A173",
    "number_of_people_being_liable_to_provide_maintenance_for": 1,
    "telephone": "A192",
    "foreign_worker": "A201",
}

REVIEW_APPLICANT = {
    "status_of_existing_checking_account": "A11",
    "duration_in_month": 42,
    "credit_history": "A32",
    "purpose": "A42",
    "credit_amount": 7882,
    "savings_account_bonds": "A61",
    "present_employment_since": "A74",
    "installment_rate_in_percentage_of_disposable_income": 2,
    "personal_status_and_sex": "A93",
    "other_debtors_guarantors": "A103",
    "present_residence_since": 4,
    "property": "A122",
    "age_in_years": 45,
    "other_installment_plans": "A143",
    "housing": "A153",
    "number_of_existing_credits_at_this_bank": 1,
    "job": "A173",
    "number_of_people_being_liable_to_provide_maintenance_for": 2,
    "telephone": "A191",
    "foreign_worker": "A201",
}

REJECT_APPLICANT = {
    "status_of_existing_checking_account": "A12",
    "duration_in_month": 48,
    "credit_history": "A32",
    "purpose": "A43",
    "credit_amount": 5951,
    "savings_account_bonds": "A61",
    "present_employment_since": "A73",
    "installment_rate_in_percentage_of_disposable_income": 2,
    "personal_status_and_sex": "A92",
    "other_debtors_guarantors": "A101",
    "present_residence_since": 2,
    "property": "A121",
    "age_in_years": 22,
    "other_installment_plans": "A143",
    "housing": "A152",
    "number_of_existing_credits_at_this_bank": 1,
    "job": "A173",
    "number_of_people_being_liable_to_provide_maintenance_for": 1,
    "telephone": "A191",
    "foreign_worker": "A201",
}


class FakeReviewService:
    def __init__(self) -> None:
        self.calls = 0

    def summarize_review_case(self, applicant_payload, prediction_payload):
        self.calls += 1
        if prediction_payload["decision"] != "Review":
            raise ValueError("Review summaries are only available for cases with decision='Review'.")
        return {
            "review_summary": (
                "1. Why this case needs review\nPD falls in the review band.\n\n"
                "2. Evidence to verify\nRequest affordability evidence.\n\n"
                "3. Suggested analyst action\nRecord evidence before human disposition."
            ),
            "knowledge_base_sources": [
                {
                    "citation_label": "Source 1",
                    "document": "EBA_GL_2020_06_Loan_Origination_and_Monitoring.pdf",
                    "title": "Guidelines on loan origination and monitoring",
                    "authority": "European Banking Authority",
                    "page": "36",
                    "source_url": "https://www.eba.europa.eu/example.pdf",
                    "section": "Page 36",
                    "policy_version": "official-eu-credit-guidance-v1.0",
                }
            ],
            "llm_model": "fake-review-model",
            "embedding_model": "fake-embedding-model",
            "retrieval_policy_version": "review-policy-v1.0",
            "review_guardrail_applied": False,
        }


class CreditDecisionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def setUp(self) -> None:
        self.review_service = FakeReviewService()
        app.state.review_service = self.review_service

    def test_predict_returns_each_business_decision(self) -> None:
        scenarios = [
            (APPROVE_APPLICANT, "Approve", False),
            (REVIEW_APPLICANT, "Review", True),
            (REJECT_APPLICANT, "Reject", False),
        ]

        for payload, decision, review_required in scenarios:
            with self.subTest(decision=decision):
                result = self.client.post("/predict", json=payload)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["decision"], decision)
                self.assertEqual(result.json()["review_required"], review_required)
                self.assertFalse(result.json()["policy_constraints_met"])
                self.assertIn("No candidate met both", result.json()["policy_selection_reason"])

    def test_reason_does_not_describe_an_absent_categorical_value(self) -> None:
        result = self.client.post("/predict", json=REJECT_APPLICANT).json()

        self.assertNotIn("no checking account", result["decision_reason"])

    def test_reason_translates_active_dataset_category_codes(self) -> None:
        reason = generate_reason(
            [("purpose_A46", 0.5), ("duration_in_month", 0.3)],
            {"purpose_A46"},
            {"purpose_A46"},
        )

        self.assertIn("education purpose increases risk", reason)
        self.assertNotIn("purpose_A46", reason)

    def test_review_summary_returns_audit_metadata_for_review_case(self) -> None:
        prediction = self.client.post("/predict", json=REVIEW_APPLICANT).json()
        result = self.client.post(
            "/review-summary",
            json={"applicant": REVIEW_APPLICANT, "prediction": prediction},
        )

        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertEqual(self.review_service.calls, 1)
        self.assertEqual(payload["embedding_model"], "fake-embedding-model")
        self.assertFalse(payload["review_guardrail_applied"])
        self.assertEqual(
            payload["knowledge_base_sources"][0]["page"],
            "36",
        )

    def test_review_summary_blocks_non_review_case(self) -> None:
        prediction = self.client.post("/predict", json=APPROVE_APPLICANT).json()
        result = self.client.post(
            "/review-summary",
            json={"applicant": APPROVE_APPLICANT, "prediction": prediction},
        )

        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.review_service.calls, 1)


if __name__ == "__main__":
    unittest.main()
