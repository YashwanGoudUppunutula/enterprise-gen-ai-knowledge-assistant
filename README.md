# enterprise-genai-knowledge-assistant

A local, enterprise-focused Retrieval-Augmented Generation (RAG) chatbot for querying private PDF documents.  
It ingests PDFs, chunks content, creates embeddings, stores vectors in ChromaDB, retrieves relevant context with LangChain, and serves an interactive chat UI via Streamlit.

## Tech Stack

- LangChain (`langchain`, `langchain-community`, `langchain-core`)
- ChromaDB (persistent local vector store)
- Sentence Transformers (`all-MiniLM-L6-v2`) for local embeddings
- Ollama (`llama3` by default) for local LLM inference
- Streamlit for chat interface
- PyPDF for PDF loading/parsing

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── .gitkeep
├── chroma_db/              # created after ingestion
└── src/
    ├── __init__.py
    ├── ingest.py
    ├── rag.py
    └── utils.py
```

## Local Setup

1. Create and activate a Python environment:
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add 3-5 PDF files to `./data/`.
4. Run ingestion and build vector DB:
   - `python -m src.ingest`
5. Start the app:
   - `streamlit run app.py`

## Ollama Setup (Default)

1. Install Ollama from [https://ollama.com](https://ollama.com).
2. Pull a model (example):
   - `ollama pull llama3`
3. Keep Ollama running locally while using the app.

## Optional: Use OpenAI Instead of Ollama

- In `src/rag.py`, call `get_answer(query, backend="openai")`.
- Set environment variable:
  - PowerShell: `$env:OPENAI_API_KEY="your_api_key_here"`

## Features Implemented

- Loads all PDFs from `./data/` using `PyPDFDirectoryLoader`
- Uses `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`)
- Creates persistent ChromaDB at `./chroma_db`
- Supports direct vector search and sample retrieval in ingestion script
- Implements LCEL retrieval + generation chain in `src/rag.py`
- Streamlit chat interface with:
  - Chat history (`st.chat_message`, `st.chat_input`)
  - Answers generated from retrieved context
  - Source chunk display in an expander for transparency

## Screenshot

Add a screenshot of your running app here after launching Streamlit (recommended path: `assets/streamlit-chat.png`):

```markdown
![Streamlit RAG Chat Screenshot](assets/streamlit-chat.png)
```
