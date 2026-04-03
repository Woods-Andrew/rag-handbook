"""Load the PDF, split into chunks, and build a FAISS vector store."""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import config


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Build or load the FAISS vector store from the handbook PDF."""

    if os.path.exists(config.VECTORSTORE_DIR) and not force_rebuild:
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
        return FAISS.load_local(
            config.VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )

    print(f"Loading PDF from {config.PDF_PATH}...")
    loader = PyPDFLoader(config.PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and building vector store...")
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(config.VECTORSTORE_DIR)
    print(f"Vector store saved to {config.VECTORSTORE_DIR}")

    return vectorstore


if __name__ == "__main__":
    build_vectorstore(force_rebuild=True)
