from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedding_model


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "enterprise_docs"


def get_vector_store(path: str = CHROMA_PATH) -> chromadb.PersistentClient:
    """Return a persistent Chroma client stored on disk."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def create_vector_store(
    chunks: List[Document],
    path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Tuple[Chroma, bool]:
    """
    Create (or reuse) persistent vector store.

    Returns:
        (vector_store, created_new)
    """
    if not chunks:
        raise ValueError("No chunks provided. Ingest documents before creating vector store.")

    client = get_vector_store(path=path)
    existing_collections = {c.name for c in client.list_collections()}

    # If the collection already exists and has vectors, reuse it and avoid re-ingestion.
    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)
        if collection.count() > 0:
            store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=get_embedding_model(),
                persist_directory=path,
            )
            return store, False

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        client=client,
        collection_name=collection_name,
        persist_directory=path,
    )
    store.persist()
    return store, True


def load_vector_store(
    path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load an existing persistent Chroma vector store."""
    client = get_vector_store(path=path)
    existing_collections = {c.name for c in client.list_collections()}
    if collection_name not in existing_collections:
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in '{path}'. Run ingestion first."
        )

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=path,
    )
