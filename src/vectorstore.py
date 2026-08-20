"""Acceso compartido al PineconeVectorStore."""

from __future__ import annotations

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import Settings, get_settings
from src.providers import get_embeddings


def get_vector_store(settings: Settings | None = None) -> PineconeVectorStore:
    """Devuelve un PineconeVectorStore listo para upsert y búsqueda.

    Conecta al índice existente, inyecta el modelo de embeddings activo
    y usa el namespace configurado. El texto del chunk se guarda en metadata
    bajo la clave `text`.
    """
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
