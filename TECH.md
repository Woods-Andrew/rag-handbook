# Technical Design Document
## PhD Handbook RAG Chatbot

**Version:** 1.0  
**Date:** April 2026  
**Branch:** dev

---

## 1. Overview

This document describes the specific technologies chosen to build the PhD Handbook RAG (Retrieval-Augmented Generation) chatbot, and the reasoning behind each decision. The system allows users to ask natural language questions about a PDF handbook and receive accurate, contextually grounded answers.

---

## 2. Architecture Summary

The system follows a two-phase RAG pipeline:

```
[PDF Document]
      │
      ▼
[Ingestion Pipeline]
  └── Load PDF → Split into chunks → Embed chunks → Save to vector store
      │
      ▼
[Query Pipeline]
  └── User question → Retrieve top-K chunks → Pass to LLM with context → Return answer
      │
      ▼
[Streamlit Chat UI]
```

---

## 3. Technology Decisions

### 3.1 Python
**Version:** 3.x  
**Reason:** Python is the dominant language in the ML/AI ecosystem. All major AI libraries (LangChain, OpenAI SDK, FAISS) have first-class Python support. It allows rapid development with minimal boilerplate.

---

### 3.2 LangChain
**Package:** `langchain`, `langchain-openai`, `langchain-community`  
**Role:** Orchestration framework for the RAG pipeline

**Reason:**  
LangChain provides pre-built abstractions for the full RAG lifecycle — document loading, text splitting, retrieval chains, and conversation memory — eliminating the need to wire these components manually. The `ConversationalRetrievalChain` handles chat history injection into each query automatically, which is essential for multi-turn conversations. LangChain also decouples the system from any single LLM provider, making it straightforward to swap models in future.

---

### 3.3 OpenAI API
**Embedding model:** `text-embedding-3-small`  
**Chat model:** `gpt-4o-mini` (configurable)  
**Role:** Embeddings for indexing + LLM for answer generation

**Reason:**  
OpenAI's embedding and chat models are industry-leading in quality, reliability, and documentation. `text-embedding-3-small` was chosen over the larger `text-embedding-3-large` because it delivers strong semantic search quality at significantly lower cost and latency — appropriate for a single-document use case. `gpt-4o-mini` provides GPT-4-class reasoning at a fraction of the cost of full GPT-4o, and is more than sufficient for document Q&A tasks where hallucination risk is reduced by retrieved context.

---

### 3.4 FAISS (Facebook AI Similarity Search)
**Package:** `faiss-cpu`  
**Role:** Local vector store for storing and searching document embeddings

**Reason:**  
FAISS is the most widely used high-performance vector similarity search library. The `faiss-cpu` variant was chosen over cloud vector databases (e.g. Pinecone, Weaviate) because:
- The document (single PDF) is small enough to fit entirely in a local index
- No external service dependency, no API costs, no network latency
- The index is persisted to disk and loaded instantly on subsequent runs
- Zero infrastructure to manage

For a larger multi-document system, migrating to a managed vector DB would be straightforward due to LangChain's abstraction layer.

---

### 3.5 PyPDF
**Package:** `pypdf`  
**Role:** Extract text from the handbook PDF

**Reason:**  
`pypdf` is a pure-Python, dependency-light PDF reader with native LangChain integration via `PyPDFLoader`. It preserves page metadata, which is surfaced in the UI as source citations. It was preferred over alternatives like `pdfminer` (more complex API) or `pymupdf` (requires C libraries) for its simplicity and LangChain compatibility.

---

### 3.6 RecursiveCharacterTextSplitter
**Config:** `chunk_size=1000`, `chunk_overlap=200`  
**Role:** Split document pages into indexable chunks

**Reason:**  
`RecursiveCharacterTextSplitter` is LangChain's recommended splitter because it tries to split on natural boundaries (paragraphs, sentences, words) before falling back to character-level splitting. A chunk size of 1000 characters balances two competing concerns:
- **Too large:** chunks exceed the context window or dilute relevance
- **Too small:** chunks lose necessary surrounding context

A 200-character overlap ensures that sentences or ideas spanning chunk boundaries are not lost.

---

### 3.7 Streamlit
**Package:** `streamlit`  
**Role:** Chat user interface

**Reason:**  
Streamlit allows building a fully interactive, production-quality web UI in pure Python with minimal code. The `st.chat_message` and `st.chat_input` components provide a native chat experience without any frontend development. Key benefits:
- No HTML/CSS/JavaScript required
- Session state management built-in for chat history
- `@st.cache_resource` caches the vector store and chain across user interactions, preventing redundant reloading
- Fast iteration — changes to UI require only a file save

---

### 3.8 python-dotenv
**Package:** `python-dotenv`  
**Role:** Load environment variables from `.env`

**Reason:**  
Separates secrets (API keys) from source code. The `.env` file is excluded from version control via `.gitignore`, ensuring credentials are never committed to the repository. This follows the [12-Factor App](https://12factor.net/config) principle of storing configuration in the environment.

---

### 3.9 Virtual Environment (venv)
**Tool:** Python built-in `venv`  
**Role:** Isolated Python environment

**Reason:**  
`venv` is the standard, built-in Python tool for dependency isolation. It was chosen over `conda` or `poetry` because it has no additional installation requirements and is universally available. It keeps the project's dependencies separate from the system Python and other projects.

---

## 4. Decisions Not Made (and Why)

| Alternative | Why Not Used |
|---|---|
| **Pinecone / Weaviate** (cloud vector DB) | Overkill for a single PDF; adds cost and external dependency |
| **LlamaIndex** | LangChain was sufficient; LlamaIndex adds complexity without benefit at this scale |
| **Local LLM (Ollama, LLaMA)** | Requires significant local hardware; OpenAI provides better quality with less setup |
| **React / Next.js frontend** | Streamlit delivers equivalent UX for this use case with no frontend code |
| **PostgreSQL + pgvector** | FAISS is simpler and sufficient for single-document scale |
| **Docker** | Not needed for local development; can be added for deployment later |

---

## 5. Data Flow

```
User types question
        │
        ▼
Streamlit (app.py)
        │
        ▼
ConversationalRetrievalChain (chain.py)
        │
        ├──► FAISS vector store
        │     └── Finds top-4 most relevant chunks via cosine similarity
        │
        ├──► Chat history (ConversationBufferMemory)
        │     └── Injects prior Q&A turns for context
        │
        └──► OpenAI gpt-4o-mini
              └── Generates grounded answer from retrieved chunks
                        │
                        ▼
              Answer + source pages → Streamlit UI
```

---

## 6. Configuration Reference

All tunable parameters are centralised in `config.py`:

| Parameter | Value | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Controls embedding quality/cost |
| `LLM_MODEL` | `gpt-4o-mini` | Controls answer quality/cost |
| `CHUNK_SIZE` | `1000` | Size of each indexed text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap to preserve context at boundaries |
| `PDF_PATH` | `docs/handbook.pdf` | Source document |
| `VECTORSTORE_DIR` | `vectorstore/` | Where FAISS index is persisted |
