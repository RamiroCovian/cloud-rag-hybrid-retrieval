"""Inicializa el índice Serverless de Pinecone si aún no existe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pinecone_index import ensure_pinecone_index


def main() -> None:
    """Punto de entrada: crea el índice Pinecone Serverless si no existe."""
    ensure_pinecone_index()


if __name__ == "__main__":
    main()
