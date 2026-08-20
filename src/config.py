"""Carga y valida la configuración multi-proveedor desde variables de entorno."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVALUATION_DIR = DATA_DIR / "evaluation"

load_dotenv(ROOT_DIR / ".env")

# Proveedores soportados para embeddings y/o chat.
EMBEDDING_PROVIDERS = frozenset(
    {"openai", "gemini", "voyage", "cohere", "openai_compatible"}
)
LLM_PROVIDERS = frozenset(
    {"openai", "gemini", "anthropic", "grok", "openai_compatible"}
)

# Defaults de modelo y dimensión por proveedor de embeddings.
EMBEDDING_DEFAULTS: dict[str, tuple[str, int]] = {
    "openai": ("text-embedding-3-small", 1536),
    "gemini": ("models/gemini-embedding-001", 768),
    "voyage": ("voyage-3", 1024),
    "cohere": ("embed-english-v3.0", 1024),
    "openai_compatible": ("text-embedding-3-small", 1536),
}

LLM_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.1-flash-lite",
    "anthropic": "claude-3-5-haiku-latest",
    "grok": "grok-2-latest",
    "openai_compatible": "gpt-4o-mini",
}

# Variable de API key requerida por proveedor (embeddings / LLM).
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}


def _optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _require_env(name: str) -> str:
    value = _optional_env(name)
    if not value:
        raise ValueError(f"Falta la variable de entorno requerida: {name}")
    return value


def _normalize_provider(value: str, allowed: frozenset[str], label: str) -> str:
    provider = value.strip().lower()
    if provider not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{label} inválido: '{value}'. Opciones: {options}")
    return provider


def _resolve_api_key(provider: str, explicit_env: str | None = None) -> str:
    """Resuelve la API key del proveedor; acepta alias GEMINI_API_KEY → GOOGLE_API_KEY."""
    if explicit_env:
        return _require_env(explicit_env)

    env_name = PROVIDER_API_KEY_ENV[provider]
    value = _optional_env(env_name)

    if not value and provider == "gemini":
        value = _optional_env("GEMINI_API_KEY")

    if not value and provider == "openai_compatible":
        value = _optional_env("OPENAI_API_KEY")

    if not value:
        aliases = [env_name]
        if provider == "gemini":
            aliases.append("GEMINI_API_KEY")
        raise ValueError(
            f"Falta la API key para el proveedor '{provider}'. "
            f"Definí una de: {', '.join(aliases)}"
        )
    return value


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str

    embedding_provider: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimension: int
    embedding_base_url: str | None

    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_base_url: str | None


def get_settings() -> Settings:
    embedding_provider = _normalize_provider(
        _optional_env("EMBEDDING_PROVIDER", "gemini"),
        EMBEDDING_PROVIDERS,
        "EMBEDDING_PROVIDER",
    )
    llm_provider = _normalize_provider(
        _optional_env(
            "LLM_PROVIDER",
            embedding_provider if embedding_provider in LLM_PROVIDERS else "gemini",
        ),
        LLM_PROVIDERS,
        "LLM_PROVIDER",
    )

    default_model, default_dim = EMBEDDING_DEFAULTS[embedding_provider]
    embedding_model = _optional_env("EMBEDDING_MODEL", default_model)
    embedding_dimension = int(_optional_env("EMBEDDING_DIMENSION", str(default_dim)))

    embedding_base_url = _optional_env("EMBEDDING_BASE_URL") or None
    llm_base_url = _optional_env("LLM_BASE_URL") or None

    if embedding_provider == "openai_compatible" and not embedding_base_url:
        raise ValueError(
            "EMBEDDING_BASE_URL es obligatorio cuando EMBEDDING_PROVIDER=openai_compatible"
        )
    if llm_provider == "openai_compatible" and not llm_base_url:
        raise ValueError(
            "LLM_BASE_URL es obligatorio cuando LLM_PROVIDER=openai_compatible"
        )
    if llm_provider == "grok" and not llm_base_url:
        llm_base_url = "https://api.x.ai/v1"

    return Settings(
        pinecone_api_key=_require_env("PINECONE_API_KEY"),
        index_name=_require_env("INDEX_NAME"),
        pinecone_cloud=_optional_env("PINECONE_CLOUD", "aws"),
        pinecone_region=_optional_env("PINECONE_REGION", "us-east-1"),
        pinecone_namespace=_optional_env("PINECONE_NAMESPACE", "docs"),
        embedding_provider=embedding_provider,
        embedding_api_key=_resolve_api_key(embedding_provider),
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_base_url=embedding_base_url,
        llm_provider=llm_provider,
        llm_api_key=_resolve_api_key(llm_provider),
        llm_model=_optional_env("LLM_MODEL", LLM_DEFAULTS[llm_provider]),
        llm_base_url=llm_base_url,
    )
