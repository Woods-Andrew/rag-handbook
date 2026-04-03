"""Streamlit chat interface for the PhD Handbook RAG chatbot."""

import streamlit as st
from chain import get_chain

st.set_page_config(page_title="PhD Handbook Assistant", page_icon="📚", layout="wide")
st.title("📚 PhD Handbook Assistant")
st.caption("Ask questions about the PhD handbook and get answers backed by the document.")


@st.cache_resource(show_spinner="Loading handbook and building index...")
def load_chain():
    return get_chain()


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ask a question about the handbook..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": prompt})
            answer = result["answer"]

            st.markdown(answer)

            # Show source pages in an expander
            sources = result.get("source_documents", [])
            if sources:
                with st.expander("📄 Sources"):
                    seen = set()
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:200]
                        key = (page, snippet)
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**Page {page}:** {snippet}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
