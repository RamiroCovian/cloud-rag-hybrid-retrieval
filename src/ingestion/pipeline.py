"""Pipeline de ingesta: carga → chunking → embeddings → Pinecone."""

from __future__ import annotations

from pathlib import Path

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import DOCUMENTS_DIR, Settings, get_settings
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_documents
from src.pinecone_index import ensure_pinecone_index
from src.providers import get_embeddings


def _stable_ids(chunks: list) -> list[str]:
    ids: list[str] = []
    counters: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("document_id", "doc"))
        counters[doc_id] = counters.get(doc_id, 0)
        ids.append(f"{doc_id}__chunk_{counters[doc_id]}")
        counters[doc_id] += 1
    return ids


def run_ingestion(
    documents_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 250,
) -> dict[str, int | str]:
    """Ejecuta el pipeline completo de ingesta hacia Pinecone."""
    cfg = settings or get_settings()
    source_dir = documents_dir or DOCUMENTS_DIR

    ensure_pinecone_index(cfg)

    raw_docs = load_documents(source_dir)
    chunks = split_documents(
        raw_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError("No se generaron chunks para indexar.")

    embeddings = get_embeddings(cfg)
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    index = pc.Index(cfg.index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=cfg.pinecone_namespace,
        text_key="text",
    )

    ids = _stable_ids(chunks)
    vector_store.add_documents(documents=chunks, ids=ids)

    return {
        "documents": len(raw_docs),
        "chunks": len(chunks),
        "index_name": cfg.index_name,
        "namespace": cfg.pinecone_namespace,
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
    }
