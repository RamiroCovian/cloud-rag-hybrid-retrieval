"""CLI de prueba del recuperador híbrido (BM25 + vectorial)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_system import RAGSystem


def build_parser() -> argparse.ArgumentParser:
    """Define los argumentos CLI de la consulta híbrida."""
    parser = argparse.ArgumentParser(
        description="Consulta el RAG híbrido (BM25 + Pinecone)."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="¿Qué es un EnsembleRetriever y cómo combina BM25?",
        help="Consulta en lenguaje natural",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k documentos")
    return parser


def main() -> None:
    """Punto de entrada: consulta el RAG e imprime el top-k con preview."""
    args = build_parser().parse_args()
    rag = RAGSystem(k=args.k)
    docs = rag.retrieve(args.query, k=args.k)

    print(f"Consulta: {args.query}")
    print(f"Resultados: {len(docs)} (top-{args.k})\n")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(
            f"[{i}] document_id={meta.get('document_id')} | "
            f"source={meta.get('source')} | category={meta.get('category')}"
        )
        preview = doc.page_content.replace("\n", " ").strip()
        print(f"    {preview[:180]}...\n" if len(preview) > 180 else f"    {preview}\n")


if __name__ == "__main__":
    main()
