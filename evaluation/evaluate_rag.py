from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys

from fastapi.testclient import TestClient


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from main import RAW_TO_API_FIELD, app, load_dataset  # noqa: E402


PROHIBITED_FINAL_DECISION_PHRASES = [
    "recommend approve",
    "recommend approval",
    "should approve",
    "recommend reject",
    "recommend rejection",
    "should reject",
    "final decision: approve",
    "final decision: reject",
]


def find_grounding_violations(note: str, signed_reason: str) -> list[str]:
    violations = []
    note_lower = note.lower()
    reason_lower = signed_reason.lower()
    requested_checks = note_lower.split("2. evidence to verify", 1)[-1]
    mitigating_factors = [
        part.split(" decreases risk", 1)[0].strip()
        for part in reason_lower.split(";")
        if "decreases risk" in part
    ]
    for factor in mitigating_factors:
        if factor in requested_checks:
            violations.append(f"Requested evidence for mitigating factor: {factor}")
    if "purpose" in requested_checks and "purpose increases risk" not in reason_lower:
        violations.append("Requested purpose evidence without a risk-increasing purpose factor")
    if "$" in note:
        violations.append("Represented Deutsche Mark dataset values as dollars")
    for unsupported in ["risk appetite", "refinanc", "roll over", "macroeconomic", "capital market"]:
        if unsupported in note_lower:
            violations.append(f"Unsupported consumer-review concept: {unsupported}")
    if "source " not in requested_checks and "[source" not in requested_checks:
        violations.append("Evidence/action sections contain no retrieved-source citation")
    if (
        ("higher credit amount" in note_lower or "high credit amount" in note_lower)
        and "higher credit amount" not in reason_lower
    ):
        violations.append("Mentioned credit amount without a signed SHAP amount factor")
    if "longer loan duration" in note_lower and "longer loan duration" not in reason_lower:
        violations.append("Mentioned loan duration without a signed SHAP duration factor")
    for factor in ["higher credit amount", "longer loan duration"]:
        matches = re.findall(
            rf"{re.escape(factor)}.{{0,35}}\b(increases|increasing|decreases|decreasing) risk",
            note_lower,
            re.DOTALL,
        )
        for direction in matches:
            normalized_direction = (
                "increases risk" if direction.startswith("increas") else "decreases risk"
            )
            signed_statement = f"{factor} {normalized_direction}"
            if signed_statement not in reason_lower:
                violations.append(f"Unsupported risk direction: {signed_statement}")
    return violations


def get_review_cases(client: TestClient, max_cases: int) -> list[tuple[dict, dict]]:
    cases = []
    for _, row in load_dataset().drop(columns=["target"]).iterrows():
        applicant = {api: row[raw] for raw, api in RAW_TO_API_FIELD.items()}
        prediction = client.post("/predict", json=applicant).json()
        if prediction["decision"] == "Review":
            cases.append((applicant, prediction))
        if len(cases) == max_cases:
            break
    return cases


def build_report(max_cases: int) -> str:
    evaluated_cases = []
    with TestClient(app) as client:
        for index, (applicant, prediction) in enumerate(
            get_review_cases(client, max_cases), start=1
        ):
            response = client.post(
                "/review-summary",
                json={"applicant": applicant, "prediction": prediction},
            )
            response.raise_for_status()
            result = response.json()
            note = result["review_summary"]
            prohibited_matches = [
                phrase for phrase in PROHIBITED_FINAL_DECISION_PHRASES if phrase in note.lower()
            ]
            grounding_violations = find_grounding_violations(
                note, prediction["decision_reason"]
            )
            sources = "; ".join(
                f"[{source['citation_label']}] {source['authority']}: {source['title']} - p. {source['page']}"
                for source in result["knowledge_base_sources"]
            )
            evaluated_cases.append(
                {
                    "case": index,
                    "pd": prediction["pd"],
                    "risk_reason": prediction["decision_reason"],
                    "sources": sources,
                    "boundary": "Pass" if not prohibited_matches else "Fail",
                    "grounding": "Pass" if not grounding_violations else "Fail",
                    "guardrail": "Applied" if result["review_guardrail_applied"] else "Not applied",
                    "note": note,
                }
            )

    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Evaluation Setup",
        "",
        "- Evaluated only applications whose policy outcome is `Review`.",
        "- Embedding model: `nomic-embed-text` through Ollama.",
        "- Vector store: Chroma over downloaded official European consumer-credit PDF documents.",
        "- Generation model: `gemma3:4b` with temperature `0`.",
        "- Boundary check: generated guidance must not recommend approval or rejection.",
        "- Grounding check: requests must cite PDF evidence and must not target mitigating SHAP factors.",
        "",
        "## Results",
        "",
        "| Case | PD | Retrieved Policy Sections | Guardrail | Decision Boundary | Grounding Check |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in evaluated_cases:
        lines.append(
            f"| {item['case']} | {item['pd']:.4f} | {item['sources']} | "
            f"{item['guardrail']} | {item['boundary']} | {item['grounding']} |"
        )
    lines.append("")
    for item in evaluated_cases:
        lines.extend(
            [
                f"## Case {item['case']}",
                "",
                f"Model reason: {item['risk_reason']}",
                "",
                "Generated review note:",
                "",
                "```text",
                item["note"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitation",
            "",
            "This is a small qualitative evaluation for demonstration. A production "
            "review assistant would require a larger labelled retrieval benchmark, "
            "formal prohibited-action testing, versioned policy governance, and human review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG review-case evidence.")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    report = build_report(args.max_cases)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
