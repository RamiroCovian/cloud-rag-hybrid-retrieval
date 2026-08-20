"""Acceso compartido al PineconeVectorStore."""

from __future__ import annotations

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import Settings, get_settings
from src.providers import get_embeddings


def get_vector_store(settings: Settings | None = None) -> PineconeVectorStore:
    """Conecta al índice Pinecone existente con el namespace configurado."""
    cfg = settings or get_settings()
    embeddings = get_embeddings(cfg)
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    index = pc.Index(cfg.index_name)
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=cfg.pinecone_namespace,
        text_key="text",
    )
