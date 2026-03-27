from __future__ import annotations

import streamlit as st

from src.rag import get_answer
from src.utils import compact_source


st.set_page_config(page_title="Enterprise GenAI Knowledge Assistant", page_icon="🤖")
st.title("Enterprise GenAI Knowledge Assistant")
st.caption("Ask questions over your private PDF documents (local RAG with Chroma + Ollama).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for idx, source in enumerate(message["sources"], start=1):
                    st.text(f"[{idx}] {source}")

user_query = st.chat_input("Ask a question about your documents...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = get_answer(user_query)
                answer = result["answer"]
                sources = [compact_source(doc) for doc in result["source_documents"]]
            except Exception as exc:  # Keep UX friendly for missing DB/model issues.
                answer = f"Error: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for idx, source in enumerate(sources, start=1):
                    st.text(f"[{idx}] {source}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
