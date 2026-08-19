"""Carga y valida la configuración desde variables de entorno."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVALUATION_DIR = DATA_DIR / "evaluation"

load_dotenv(ROOT_DIR / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Falta la variable de entorno requerida: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    openai_api_key: str
    index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    embedding_model: str
    embedding_dimension: int


def get_settings() -> Settings:
    return Settings(
        pinecone_api_key=_require_env("PINECONE_API_KEY"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        index_name=_require_env("INDEX_NAME"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "docs").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
    )
