from __future__ import annotations

from pathlib import Path
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.utils import format_docs
from src.vector_store import load_vector_store


load_dotenv()

CHROMA_DIR = "./chroma_db"


class RAGResult(TypedDict):
    answer: str
    source_documents: List[Document]


def _get_vector_store():
    if not Path(CHROMA_DIR).exists():
        raise FileNotFoundError(
            f"Chroma DB not found at {Path(CHROMA_DIR).resolve()}. Run `python -m src.ingest` first."
        )
    return load_vector_store(path=CHROMA_DIR)


def _get_llm(
    backend: Literal["ollama", "openai"] = "ollama",
    ollama_model: str = "llama3",
):
    if backend == "openai":
        # Requires OPENAI_API_KEY in environment.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend selected, but langchain-openai is not installed. "
                "Install it with: pip install langchain-openai"
            ) from exc

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Default local model path via Ollama.
    return ChatOllama(model=ollama_model, temperature=0)


def get_answer(
    query: str,
    backend: Literal["ollama", "openai"] = "ollama",
    ollama_model: str = "llama3",
) -> RAGResult:
    if not query.strip():
        return {"answer": "Please enter a non-empty query.", "source_documents": []}

    vector_store = _get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = _get_llm(backend=backend, ollama_model=ollama_model)

    prompt = ChatPromptTemplate.from_template(
        """
You are an enterprise knowledge assistant. Answer the question using ONLY the context.
If the answer cannot be found in the context, say you do not have enough information.
Be concise and factual.

Context:
{context}

Question:
{question}
"""
    )

    retrieval_chain = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    chain = retrieval_chain | prompt | llm | StrOutputParser()
    answer = chain.invoke(query)

    # Query separately so caller can inspect the top documents shown to the user.
    source_docs = retriever.invoke(query)
    return {"answer": answer, "source_documents": source_docs}
