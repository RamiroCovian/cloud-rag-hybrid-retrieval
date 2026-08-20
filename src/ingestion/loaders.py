"""Carga de documentos técnicos (Markdown, PDF, JSON, TXT)."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".json"}

CATEGORY_BY_KEYWORD: dict[str, str] = {
    "pinecone": "vector-db",
    "bm25": "retrieval",
    "hybrid": "retrieval",
    "chunk": "ingestion",
    "embedding": "ingestion",
    "langchain": "framework",
    "ensemble": "retrieval",
    "evaluat": "evaluation",
    "metric": "evaluation",
}


def infer_category(stem: str) -> str:
    lowered = stem.lower()
    for keyword, category in CATEGORY_BY_KEYWORD.items():
        if keyword in lowered:
            return category
    return "general"


def _relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _load_text_file(path: Path, root: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    document_id = path.stem
    return [
        Document(
            page_content=text,
            metadata={
                "source": _relative_source(path, root),
                "document_id": document_id,
                "page": 1,
                "category": infer_category(document_id),
                "tags": [infer_category(document_id), path.suffix.lstrip(".")],
            },
        )
    ]


def _load_pdf(path: Path, root: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    document_id = path.stem
    category = infer_category(document_id)
    source = _relative_source(path, root)
    loaded = PyPDFLoader(str(path)).load()

    documents: list[Document] = []
    for doc in loaded:
        page = int(doc.metadata.get("page", 0)) + 1
        documents.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "source": source,
                    "document_id": document_id,
                    "page": page,
                    "category": category,
                    "tags": [category, "pdf"],
                },
            )
        )
    return [doc for doc in documents if doc.page_content]


def _load_json(path: Path, root: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = _relative_source(path, root)
    document_id = path.stem
    default_category = infer_category(document_id)

    if isinstance(payload, dict) and "documents" in payload:
        items = payload["documents"]
    elif isinstance(payload, list):
        items = payload
    else:
        items = [payload]

    documents: list[Document] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            content = item
            item_meta: dict = {}
        elif isinstance(item, dict):
            content = str(item.get("text") or item.get("content") or "").strip()
            item_meta = item
        else:
            continue

        if not content:
            continue

        item_id = str(item_meta.get("document_id") or f"{document_id}-{index}")
        category = str(item_meta.get("category") or default_category)
        tags = item_meta.get("tags") or [category, "json"]
        if isinstance(tags, str):
            tags = [tags]

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(item_meta.get("source") or source),
                    "document_id": item_id,
                    "page": int(item_meta.get("page") or index),
                    "category": category,
                    "tags": list(tags),
                },
            )
        )
    return documents


def load_file(path: Path, root: Path | None = None) -> list[Document]:
    """Carga un archivo individual según su extensión."""
    path = path.resolve()
    root = (root or path.parent).resolve()
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extensión no soportada: {suffix} ({path.name})")

    if suffix in {".md", ".markdown", ".txt"}:
        return _load_text_file(path, root)
    if suffix == ".pdf":
        return _load_pdf(path, root)
    return _load_json(path, root)


def load_documents(directory: Path) -> list[Document]:
    """Carga todos los documentos soportados de un directorio."""
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"No existe el directorio de documentos: {directory}")

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No hay documentos (.md, .pdf, .json, .txt) en {directory}"
        )

    documents: list[Document] = []
    for file_path in files:
        documents.extend(load_file(file_path, root=directory))
    return documents
