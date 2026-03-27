from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

from src.vector_store import create_vector_store, load_vector_store

DATA_DIR = Path("./data")
CHROMA_DIR = "./chroma_db"


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Load all PDFs from the data directory."""
    if not data_dir.exists():
        print(f"[WARN] Data directory not found: {data_dir.resolve()}")
        return []

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[INFO] No PDF files found in: {data_dir.resolve()}")
        return []

    loader = PyPDFDirectoryLoader(str(data_dir))
    documents = loader.load()
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents for better embedding and retrieval quality."""
    if not documents:
        return []

    # 1000 chars balances semantic coverage and retrieval precision for
    # typical business/technical documents. 200-char overlap keeps context
    # continuity across chunk boundaries and reduces answer fragmentation.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def query_vector_store(
    query: str, k: int = 3
) -> List[Document]:
    """Query persisted Chroma store and return top-k chunks."""
    vector_store = load_vector_store(path=CHROMA_DIR)
    return vector_store.similarity_search(query, k=k)


def main() -> None:
    documents = load_documents()
    print(f"Total raw documents loaded: {len(documents)}")

    if not documents:
        print("[INFO] Nothing to ingest. Add 3-5 PDFs to ./data and rerun.")
        return

    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("[WARN] No chunks were created from loaded documents.")
        return

    _, created_new = create_vector_store(chunks, path=CHROMA_DIR)
    if created_new:
        print(f"[OK] Persisted vector store at: {Path(CHROMA_DIR).resolve()}")
    else:
        print(f"[INFO] Reusing existing vector store at: {Path(CHROMA_DIR).resolve()}")

    sample_query = "What is the revenue for Q3?"
    print(f"\nSample query: {sample_query}")
    results = query_vector_store(sample_query, k=3)
    if not results:
        print("[INFO] No relevant chunks returned.")
        return

    for idx, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        preview = doc.page_content.strip().replace("\n", " ")
        print(f"\nResult {idx} | source={source} | page={page}")
        print(preview[:500] + ("..." if len(preview) > 500 else ""))


if __name__ == "__main__":
    main()
