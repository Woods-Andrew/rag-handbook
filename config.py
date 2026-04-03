import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
PDF_PATH = os.path.join(DOCS_DIR, "handbook.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
