"""Sistema RAG híbrido: BM25 (léxico) + Pinecone (vectorial)."""

from __future__ import annotations

from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR, Settings, get_settings
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import load_documents
from src.vectorstore import get_vector_store


class RAGSystem:
    """Encapsula un EnsembleRetriever (BM25 + similitud vectorial).

    Devuelve los top-k documentos combinando señales léxicas y semánticas.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        documents_dir: Path | None = None,
        k: int = 5,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        chunk_size: int = 2000,
        chunk_overlap: int = 250,
    ) -> None:
        if k <= 0:
            raise ValueError("k debe ser > 0")
        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError("Los pesos deben ser >= 0")
        if bm25_weight + vector_weight == 0:
            raise ValueError("La suma de pesos debe ser > 0")

        self.settings = settings or get_settings()
        self.k = k
        self.documents_dir = documents_dir or DOCUMENTS_DIR

        local_chunks = split_documents(
            load_documents(self.documents_dir),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not local_chunks:
            raise ValueError(
                f"No hay chunks locales para BM25 en {self.documents_dir}"
            )

        self._bm25 = BM25Retriever.from_documents(local_chunks)
        self._bm25.k = k

        vector_store = get_vector_store(self.settings)
        self._vector = vector_store.as_retriever(search_kwargs={"k": k})

        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25, self._vector],
            weights=[bm25_weight, vector_weight],
        )

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Recupera los top-k documentos para una consulta."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("La consulta no puede estar vacía")

        top_k = k or self.k
        documents = self._ensemble.invoke(cleaned)
        return documents[:top_k]

    def invoke(self, query: str) -> list[Document]:
        """Alias de retrieve (compatible con la API de LangChain)."""
        return self.retrieve(query)
