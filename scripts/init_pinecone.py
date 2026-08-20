"""Inicializa el índice Serverless de Pinecone si aún no existe."""

from __future__ import annotations

import sys
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Settings, get_settings

METRIC = "cosine"


def ensure_pinecone_index(settings: Settings | None = None) -> str:
    """Verifica el índice y lo crea en modo Serverless si falta.

    Returns:
        Nombre del índice listo para usar.
    """
    cfg = settings or get_settings()
    pc = Pinecone(api_key=cfg.pinecone_api_key)

    if pc.has_index(cfg.index_name):
        print(
            f"El índice '{cfg.index_name}' ya existe. "
            f"Namespace configurado: '{cfg.pinecone_namespace}'."
        )
        return cfg.index_name

    print(
        f"Creando índice Serverless '{cfg.index_name}' "
        f"(provider={cfg.embedding_provider}, model={cfg.embedding_model}, "
        f"dim={cfg.embedding_dimension}, metric={METRIC}, "
        f"cloud={cfg.pinecone_cloud}, region={cfg.pinecone_region})..."
    )
    pc.create_index(
        name=cfg.index_name,
        dimension=cfg.embedding_dimension,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud=cfg.pinecone_cloud,
            region=cfg.pinecone_region,
        ),
    )
    print(
        f"Índice '{cfg.index_name}' creado correctamente. "
        f"Usá el namespace '{cfg.pinecone_namespace}' al upsert/query."
    )
    return cfg.index_name


def main() -> None:
    ensure_pinecone_index()


if __name__ == "__main__":
    main()
