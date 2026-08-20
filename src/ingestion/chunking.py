"""Chunking de documentos con RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 250,
) -> list[Document]:
    """Divide documentos en chunks (~500–800 tokens con size≈2000)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser >= 0 y < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    enriched: list[Document] = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = index
        # Texto original en metadata para evitar una DB relacional extra.
        metadata["text"] = chunk.page_content
        if "tags" in metadata and not isinstance(metadata["tags"], list):
            metadata["tags"] = [str(metadata["tags"])]
        enriched.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )
    return enriched
