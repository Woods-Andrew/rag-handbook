# PhD Handbook RAG Chatbot

A conversational AI assistant that answers questions about the PhD handbook using Retrieval-Augmented Generation (RAG). Built with LangChain, OpenAI, FAISS, and Streamlit.

## How It Works

1. The handbook PDF is loaded and split into chunks
2. Chunks are embedded using OpenAI and stored in a local FAISS vector index
3. When you ask a question, the most relevant chunks are retrieved and passed to GPT as context
4. The answer is displayed in a chat interface along with the source pages

## Project Structure

```
rag-handbook/
├── app.py            # Streamlit chat interface
├── chain.py          # Conversational RAG chain
├── ingest.py         # PDF loading, chunking, vector store creation
├── config.py         # Central configuration (models, paths, etc.)
├── requirements.txt  # Python dependencies
├── .env              # API key (not committed)
├── docs/
│   └── handbook.pdf  # Source document
└── vectorstore/      # FAISS index (auto-generated, not committed)
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Woods-Andrew/rag-handbook.git
cd rag-handbook
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first run, it will build the vector index from the PDF (this takes a moment). Subsequent runs load the cached index instantly.

## Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

## Rebuilding the Index

If you replace or update `docs/handbook.pdf`, delete the `vectorstore/` folder and restart the app to rebuild the index.
