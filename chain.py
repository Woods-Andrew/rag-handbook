"""RAG chain: retrieve relevant chunks and answer with an LLM."""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import config
from ingest import build_vectorstore

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore


def get_chain():
    """Create a conversational RAG chain."""
    vectorstore = get_vectorstore()

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0,
        streaming=True,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )

    return chain
