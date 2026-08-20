"""Tests unitarios del chunking."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.ingestion.chunking import split_documents


def test_split_documents_adds_text_and_chunk_index() -> None:
    docs = [
        Document(
            page_content="Alpha. " * 50 + "\n\n" + "Beta. " * 50,
            metadata={"document_id": "sample", "category": "general", "tags": "tag"},
        )
    ]
    chunks = split_documents(docs, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["text"] == chunks[0].page_content
    assert isinstance(chunks[0].metadata["tags"], list)


def test_split_documents_invalid_overlap() -> None:
    docs = [Document(page_content="hola", metadata={})]
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents(docs, chunk_size=100, chunk_overlap=100)
