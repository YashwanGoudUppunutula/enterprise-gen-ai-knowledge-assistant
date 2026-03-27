from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """Convert documents to a context string for prompting."""
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


def compact_source(doc: Document, max_chars: int = 280) -> str:
    """Render metadata + snippet for UI/source display."""
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "n/a")
    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return f"source={source}, page={page}\n{snippet}"
