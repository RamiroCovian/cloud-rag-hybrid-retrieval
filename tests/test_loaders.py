"""Tests unitarios de carga de documentos (loaders)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.loaders import (
    infer_category,
    load_documents,
    load_file,
)


def test_infer_category_known_and_default() -> None:
    assert infer_category("hybrid-retrieval-bm25") == "retrieval"
    assert infer_category("pinecone-serverless") == "vector-db"
    assert infer_category("chunking-embeddings") == "ingestion"
    assert infer_category("notas-varias") == "general"


def test_load_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "hybrid-retrieval-bm25.md"
    path.write_text("# BM25\n\nTexto de prueba sobre recuperación.", encoding="utf-8")

    docs = load_file(path, root=tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["document_id"] == "hybrid-retrieval-bm25"
    assert docs[0].metadata["category"] == "retrieval"
    assert docs[0].metadata["source"] == "hybrid-retrieval-bm25.md"
    assert "BM25" in docs[0].page_content
    assert "text" not in docs[0].metadata  # text se agrega en el chunking


def test_load_json_documents(tmp_path: Path) -> None:
    path = tmp_path / "langchain-notes.json"
    payload = {
        "documents": [
            {
                "document_id": "langchain-ensemble",
                "category": "framework",
                "tags": ["framework", "json"],
                "page": 1,
                "text": "EnsembleRetriever combina BM25 y Pinecone.",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    docs = load_file(path, root=tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata["document_id"] == "langchain-ensemble"
    assert docs[0].metadata["category"] == "framework"
    assert docs[0].metadata["page"] == 1
    assert "EnsembleRetriever" in docs[0].page_content


def test_load_documents_directory(tmp_path: Path) -> None:
    (tmp_path / "pinecone-serverless.md").write_text(
        "Namespaces en Pinecone.", encoding="utf-8"
    )
    (tmp_path / "notes.json").write_text(
        json.dumps([{"document_id": "note-1", "text": "Nota JSON"}]),
        encoding="utf-8",
    )
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")

    docs = load_documents(tmp_path)
    ids = {d.metadata["document_id"] for d in docs}
    assert "pinecone-serverless" in ids
    assert "note-1" in ids
    assert len(docs) == 2


def test_load_documents_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No hay documentos"):
        load_documents(tmp_path)


def test_load_file_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "archivo.csv"
    path.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Extensión no soportada"):
        load_file(path, root=tmp_path)
