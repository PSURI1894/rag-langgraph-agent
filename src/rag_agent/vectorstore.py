"""Shared access to the persisted Chroma collection."""

from langchain_chroma import Chroma

from rag_agent.config import Settings, get_settings
from rag_agent.embeddings import FastEmbedEmbeddings


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=FastEmbedEmbeddings(settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
    )
