"""CLI: carga documentos, genera chunks/embeddings y sube a Pinecone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DOCUMENTS_DIR
from src.ingestion import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta a Pinecone (chunking + embeddings + metadata)."
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DOCUMENTS_DIR,
        help=f"Directorio de documentos (default: {DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Tamaño de chunk en caracteres (~500-800 tokens con 2000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=250,
        help="Overlap entre chunks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_ingestion(
        documents_dir=args.documents_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        "Ingesta completada: "
        f"{result['documents']} docs → {result['chunks']} chunks | "
        f"index={result['index_name']} namespace={result['namespace']} | "
        f"embeddings={result['embedding_provider']}/{result['embedding_model']}"
    )


if __name__ == "__main__":
    main()
