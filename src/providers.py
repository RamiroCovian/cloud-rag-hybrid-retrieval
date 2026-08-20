"""Factories multi-proveedor para embeddings y modelos de chat."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Devuelve el cliente de embeddings según EMBEDDING_PROVIDER."""
    cfg = settings or get_settings()
    provider = cfg.embedding_provider

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=cfg.embedding_api_key,
            dimensions=cfg.embedding_dimension,
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=cfg.embedding_model,
            google_api_key=cfg.embedding_api_key,
        )

    if provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model=cfg.embedding_model,
            voyage_api_key=cfg.embedding_api_key,
        )

    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(
            model=cfg.embedding_model,
            cohere_api_key=cfg.embedding_api_key,
        )

    if provider == "openai_compatible":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=cfg.embedding_api_key,
            base_url=cfg.embedding_base_url,
            dimensions=cfg.embedding_dimension,
        )

    raise ValueError(f"Proveedor de embeddings no soportado: {provider}")


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Devuelve el chat model según LLM_PROVIDER."""
    cfg = settings or get_settings()
    provider = cfg.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            temperature=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.llm_api_key,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            temperature=0,
        )

    if provider in {"grok", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            temperature=0,
        )

    raise ValueError(f"Proveedor de LLM no soportado: {provider}")
