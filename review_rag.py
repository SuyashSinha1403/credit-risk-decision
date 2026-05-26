from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_MODEL = "gemma3:4b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
RETRIEVAL_POLICY_VERSION = "official-eu-credit-guidance-v1.0"


class ReviewRAGService:
    def __init__(
        self,
        knowledge_base_dir: Path,
        vector_store_dir: Path,
        model_name: str = DEFAULT_REVIEW_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_dir = vector_store_dir
        self.model_name = model_name
        self.embedding_model = embedding_model
        self._retriever: Any | None = None
        self._llm: Any | None = None

    def summarize_review_case(
        self,
        applicant_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if prediction_payload.get("decision") != "Review":
            raise ValueError("Review summaries are only available for cases with decision='Review'.")

        self._ensure_runtime()
        query = self._build_query(applicant_payload, prediction_payload)
        retrieved_docs = self._retriever.invoke(query)
        context = self._format_docs(retrieved_docs)
        prompt = self._build_prompt(applicant_payload, prediction_payload, context)
        review_summary = self._normalize_generated_text(str(self._llm.invoke(prompt)).strip())
        violations = self._note_violations(review_summary, prediction_payload)
        guardrail_applied = bool(violations)
        if violations:
            correction = (
                "\n\nThe prior draft was invalid because it: "
                + "; ".join(violations)
                + ". Rewrite it and follow every rule exactly."
            )
            review_summary = self._normalize_generated_text(
                str(self._llm.invoke(prompt + correction)).strip()
            )
        if self._note_violations(review_summary, prediction_payload):
            factors = "; ".join(self._risk_increasing_factors(prediction_payload))
            review_summary = (
                "1. Why this case needs review\n"
                f"The model identified risk-increasing factor(s): {factors}.\n\n"
                "2. Evidence to verify\n"
                "Verify repayment capacity and supporting financial information for the "
                "risk-increasing factor(s) using the retrieved policy passage [Source 1].\n\n"
                "3. Suggested analyst action\n"
                "Record the verified evidence and retain the case for human disposition "
                "[Source 1]."
            )

        return {
            "review_summary": review_summary,
            "knowledge_base_sources": self._collect_sources(retrieved_docs),
            "llm_model": self.model_name,
            "embedding_model": self.embedding_model,
            "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
            "review_guardrail_applied": guardrail_applied,
        }

    def _normalize_generated_text(self, text: str) -> str:
        replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        return text

    def _ensure_runtime(self) -> None:
        if self._retriever is not None and self._llm is not None:
            return

        try:
            from langchain_chroma import Chroma
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_ollama import OllamaEmbeddings, OllamaLLM
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError(
                "Vector RAG dependencies are not installed. "
                "Install the packages listed in requirements.txt to enable /review-summary."
            ) from exc

        knowledge_files = self._get_knowledge_files()
        source_manifest = self._load_source_manifest()
        page_documents = []
        for path in knowledge_files:
            source_metadata = source_manifest.get(path.name, {})
            indexed_pages = set(source_metadata.get("indexed_pages", []))
            for document in PyPDFLoader(str(path)).load():
                page_number = int(document.metadata.get("page", 0)) + 1
                if indexed_pages and page_number not in indexed_pages:
                    continue
                document.metadata.update(
                    {
                        "document": path.name,
                        "title": source_metadata.get("title", path.stem),
                        "authority": source_metadata.get("authority", "Unknown authority"),
                        "source_url": source_metadata.get("source_url", ""),
                        "publication_date": source_metadata.get("publication_date", ""),
                        "jurisdiction": source_metadata.get("jurisdiction", ""),
                        "page": page_number,
                        "section": f"Page {page_number}",
                        "policy_version": RETRIEVAL_POLICY_VERSION,
                    }
                )
                page_documents.append(document)

        chunker = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = chunker.split_documents(page_documents)
        if not chunks:
            raise FileNotFoundError("No usable text was found in the PDF knowledge base.")

        collection_name = f"credit_review_{self._knowledge_fingerprint(knowledge_files)}"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        try:
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=str(self.vector_store_dir),
                embedding_function=embeddings,
            )
            if not vector_store.get(limit=1).get("ids"):
                vector_store.add_documents(chunks)
        except Exception as exc:
            raise RuntimeError(
                "Unable to build the policy vector index. "
                f"Ensure Ollama is running and model '{self.embedding_model}' is installed."
            ) from exc

        self._retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self._llm = OllamaLLM(model=self.model_name, temperature=0)

    def _get_knowledge_files(self) -> list[Path]:
        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found at {self.knowledge_base_dir}."
            )

        knowledge_files = sorted((self.knowledge_base_dir / "pdfs").glob("*.pdf"))
        if not knowledge_files:
            raise FileNotFoundError(
                f"No PDF knowledge-base files found in {self.knowledge_base_dir / 'pdfs'}."
            )
        return knowledge_files

    def _load_source_manifest(self) -> dict[str, dict[str, Any]]:
        manifest_path = self.knowledge_base_dir / "sources.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Knowledge-base source manifest not found at {manifest_path}."
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _knowledge_fingerprint(self, knowledge_files: list[Path]) -> str:
        hasher = hashlib.sha256()
        for path in knowledge_files:
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
        hasher.update((self.knowledge_base_dir / "sources.json").read_bytes())
        return hasher.hexdigest()[:12]

    def _build_query(
        self,
        applicant_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> str:
        case_facts = self._permitted_case_facts(applicant_payload, prediction_payload)
        return (
            "consumer creditworthiness assessment human review evidence verification "
            "income expenses financial commitments affordability audit record "
            f"risk increasing model factors {case_facts.get('risk_increasing_factors', [])} "
            f"probability of default {case_facts.get('probability_of_default', '')} "
            f"credit amount in Deutsche Mark {case_facts.get('credit_amount_in_deutsche_mark', '')} "
            f"loan duration {case_facts.get('duration_in_month', '')}"
        )

    def _build_prompt(
        self,
        applicant_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
        context: str,
    ) -> str:
        case_facts = self._permitted_case_facts(applicant_payload, prediction_payload)
        return f"""
You are assisting a human credit analyst with a manual review case.

Rules:
- Do not make, recommend, or imply a final approve/reject decision.
- Treat the logistic regression score and SHAP reason as the system of record.
- State only facts in the permitted case facts below.
- The signed model reason is authoritative: a factor marked "decreases risk" is
  mitigating context, not a concern requiring verification.
- Use only checks and actions directly related to risk-increasing factors in the
  `risk_increasing_factors` list; never request checks for mitigating factors.
- Do not request verification of a feature that is absent from `risk_increasing_factors`.
- Numeric facts are supplied only for factors identified in the signed model reason.
- Amount values are in Deutsche Mark (DM); do not convert them or write a dollar symbol.
- This is an individual consumer credit application at origination; do not discuss
  capital markets, refinancing, loan monitoring, or corporate projections.
- The case needs review because its policy outcome is Review; do not claim that
  its PD breaches an institutional risk appetite or an undocumented threshold.
- Do not mention or infer coded application categories.
- Base requested checks and next actions only on the retrieved policy context.
- Cite at least one retrieved source label such as [Source 1] in sections 2 and 3.
- Source metadata is logged separately, so do not invent source support for model facts.
- If policy context or borrower evidence is insufficient, state what is missing.

Permitted case facts:
{case_facts}

Retrieved policy context:
{context}

Write only a concise reviewer note, without a preamble, with exactly these headings:
1. Why this case needs review
2. Evidence to verify
3. Suggested analyst action
""".strip()

    def _permitted_case_facts(
        self,
        applicant_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> dict[str, Any]:
        signed_reason = str(prediction_payload.get("decision_reason", ""))
        risk_increasing_factors = self._risk_increasing_factors(prediction_payload)
        case_facts = {
            "decision": prediction_payload.get("decision"),
            "probability_of_default": prediction_payload.get("pd"),
            "risk_increasing_factors": risk_increasing_factors,
        }
        if "higher credit amount" in signed_reason:
            case_facts["credit_amount_in_deutsche_mark"] = applicant_payload.get("credit_amount")
        if "longer loan duration" in signed_reason:
            case_facts["duration_in_month"] = applicant_payload.get(
                "duration_in_month", applicant_payload.get("duration")
            )
        return case_facts

    def _risk_increasing_factors(self, prediction_payload: dict[str, Any]) -> list[str]:
        signed_reason = str(prediction_payload.get("decision_reason", ""))
        return [
            factor.strip()
            for factor in signed_reason.split(";")
            if "increases risk" in factor
        ]

    def _note_violations(
        self,
        note: str,
        prediction_payload: dict[str, Any],
    ) -> list[str]:
        requested_checks = note.lower().split("2. evidence to verify", 1)[-1]
        allowed = " ".join(self._risk_increasing_factors(prediction_payload)).lower()
        violations = []
        for term in ["purpose", "guarantor", "housing", "savings", "employment", "credit history"]:
            if term in requested_checks and term not in allowed:
                violations.append(f"requested verification of non-risk factor '{term}'")
        if "$" in note:
            violations.append("used dollars for Deutsche Mark values")
        for unsupported in ["risk appetite", "refinanc", "roll over", "macroeconomic", "capital market"]:
            if unsupported in note.lower():
                violations.append(f"used unsupported consumer-review concept '{unsupported}'")
        if "source " not in requested_checks and "[source" not in requested_checks:
            violations.append("omitted a source citation")
        return violations

    def _format_docs(self, documents: list[Any]) -> str:
        formatted_chunks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("title", document.metadata.get("document", "unknown"))
            page = document.metadata.get("page", "unknown")
            formatted_chunks.append(
                f"[Source {index}: {source}, page {page}]\n{document.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    def _collect_sources(self, documents: list[Any]) -> list[dict[str, str]]:
        sources = []
        for index, document in enumerate(documents, start=1):
            document_name = str(document.metadata.get("document", "unknown"))
            page = str(document.metadata.get("page", "unknown"))
            sources.append(
                {
                    "citation_label": f"Source {index}",
                    "document": document_name,
                    "title": str(document.metadata.get("title", document_name)),
                    "authority": str(document.metadata.get("authority", "")),
                    "page": page,
                    "source_url": str(document.metadata.get("source_url", "")),
                    "section": f"Page {page}",
                    "policy_version": str(
                        document.metadata.get(
                            "policy_version", RETRIEVAL_POLICY_VERSION
                        )
                    ),
                }
            )
        return sources
