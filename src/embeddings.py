from __future__ import annotations

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a singleton embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
