"""Helpers para crear/verificar el índice Serverless de Pinecone."""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config import Settings, get_settings

METRIC = "cosine"


def ensure_pinecone_index(settings: Settings | None = None) -> str:
    """Verifica el índice y lo crea en modo Serverless si falta."""
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
