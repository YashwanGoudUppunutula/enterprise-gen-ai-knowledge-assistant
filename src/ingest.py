from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Initialize a local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(
    chunks: List[Document], persist_directory: Path = CHROMA_DIR
) -> Chroma:
    """Create and persist a Chroma vector store from chunks."""
    if not chunks:
        raise ValueError("No chunks provided. Ingest documents before building vector store.")

    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vector_store.persist()
    return vector_store


def query_vector_store(
    query: str, k: int = 3, persist_directory: Path = CHROMA_DIR
) -> List[Document]:
    """Query persisted Chroma store and return top-k chunks."""
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma directory not found at {persist_directory.resolve()}. Run ingestion first."
        )

    embeddings = get_embeddings_model()
    vector_store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
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

    _ = build_vector_store(chunks)
    print(f"[OK] Persisted vector store at: {CHROMA_DIR.resolve()}")

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
