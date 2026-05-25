from __future__ import annotations

import json
import unittest
from pathlib import Path


class N8nWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        workflow_path = Path(__file__).resolve().parents[1] / "n8n_credit_risk_combined_workflow.json"
        cls.workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        cls.nodes = {node["name"]: node for node in cls.workflow["nodes"]}
        cls.connections = cls.workflow["connections"]

    def test_review_branch_calls_review_summary_api(self) -> None:
        self.assertIn("Needs Manual Review", self.nodes)
        self.assertIn("Review Summary API", self.nodes)
        self.assertEqual(
            self.nodes["Review Summary API"]["parameters"]["url"],
            "http://127.0.0.1:8000/review-summary",
        )
        true_route = self.connections["Needs Manual Review"]["main"][0][0]["node"]
        false_route = self.connections["Needs Manual Review"]["main"][1][0]["node"]
        self.assertEqual(true_route, "Review Summary API")
        self.assertEqual(false_route, "Prepare Audit Record")

    def test_audit_record_contains_rag_trace_fields(self) -> None:
        code = self.nodes["Prepare Audit Record"]["parameters"]["jsCode"]

        for field in [
            "review_summary",
            "retrieved_policy_sources",
            "embedding_model",
            "retrieval_policy_version",
            "final_human_action",
            "policy_constraints_met",
            "policy_selection_reason",
        ]:
            with self.subTest(field=field):
                self.assertIn(field, code)


if __name__ == "__main__":
    unittest.main()
