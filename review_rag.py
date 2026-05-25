from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_MODEL = "gemma3:4b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
RETRIEVAL_POLICY_VERSION = "review-policy-v1.0"


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

        return {
            "review_summary": review_summary,
            "knowledge_base_sources": self._collect_sources(retrieved_docs),
            "llm_model": self.model_name,
            "embedding_model": self.embedding_model,
            "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
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
            from langchain_ollama import OllamaEmbeddings, OllamaLLM
            from langchain_text_splitters import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Vector RAG dependencies are not installed. "
                "Install the packages listed in requirements.txt to enable /review-summary."
            ) from exc

        knowledge_files = self._get_knowledge_files()
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "document_heading"),
                ("##", "section"),
                ("###", "subsection"),
            ],
            strip_headers=False,
        )
        section_documents = []
        for path in knowledge_files:
            for document in header_splitter.split_text(path.read_text(encoding="utf-8")):
                document.metadata.update(
                    {
                        "document": path.name,
                        "policy_version": RETRIEVAL_POLICY_VERSION,
                    }
                )
                document.metadata["section"] = self._section_name(document.metadata)
                section_documents.append(document)

        chunker = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = chunker.split_documents(section_documents)
        if not chunks:
            raise FileNotFoundError("No usable text was found in the policy knowledge base.")

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

        knowledge_files = sorted(self.knowledge_base_dir.glob("*.md"))
        if not knowledge_files:
            raise FileNotFoundError(
                f"No markdown knowledge-base files found in {self.knowledge_base_dir}."
            )
        return knowledge_files

    def _knowledge_fingerprint(self, knowledge_files: list[Path]) -> str:
        content = "\n".join(
            f"{path.name}:{path.read_text(encoding='utf-8')}" for path in knowledge_files
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def _section_name(self, metadata: dict[str, Any]) -> str:
        return str(
            metadata.get("subsection")
            or metadata.get("section")
            or metadata.get("document_heading")
            or "General"
        )

    def _build_query(
        self,
        applicant_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> str:
        case_facts = self._permitted_case_facts(applicant_payload, prediction_payload)
        return (
            "manual credit review policy evidence verification audit record "
            f"model risk drivers {case_facts.get('signed_model_reason', '')} "
            f"probability of default {case_facts.get('probability_of_default', '')} "
            f"relevant case facts {case_facts}"
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
  signed model reason; ignore unrelated retrieved sections.
- Numeric facts are supplied only for factors identified in the signed model reason.
- Do not mention or infer coded application categories.
- Base requested checks and next actions only on the retrieved policy context.
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
        case_facts = {
            "decision": prediction_payload.get("decision"),
            "probability_of_default": prediction_payload.get("pd"),
            "signed_model_reason": signed_reason,
        }
        if "higher credit amount" in signed_reason:
            case_facts["credit_amount"] = applicant_payload.get("credit_amount")
        if "longer loan duration" in signed_reason:
            case_facts["duration_in_month"] = applicant_payload.get(
                "duration_in_month", applicant_payload.get("duration")
            )
        return case_facts

    def _format_docs(self, documents: list[Any]) -> str:
        formatted_chunks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("document", "unknown")
            section = document.metadata.get("section", "General")
            formatted_chunks.append(
                f"[Source {index}: {source} - {section}]\n{document.page_content}"
            )
        return "\n\n".join(formatted_chunks)

    def _collect_sources(self, documents: list[Any]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        sources = []
        for document in documents:
            document_name = str(document.metadata.get("document", "unknown"))
            section = str(document.metadata.get("section", "General"))
            key = (document_name, section)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "document": document_name,
                        "section": section,
                        "policy_version": str(
                            document.metadata.get(
                                "policy_version", RETRIEVAL_POLICY_VERSION
                            )
                        ),
                    }
                )
        return sources
