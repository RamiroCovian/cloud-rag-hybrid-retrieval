"""Tests unitarios de Precision@k y Recall@k."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    average,
    evaluate_case,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hit() -> None:
    assert recall_at_k("doc-a", ["x", "doc-a", "y"], k=5) == 1.0


def test_recall_at_k_miss() -> None:
    assert recall_at_k("doc-a", ["x", "y", "z"], k=5) == 0.0


def test_recall_at_k_outside_window() -> None:
    assert recall_at_k("doc-a", ["x", "y", "doc-a"], k=2) == 0.0


def test_precision_at_k_partial() -> None:
    useful = {"doc-a", "doc-b"}
    retrieved = ["doc-a", "noise", "doc-b", "other", "noise2"]
    assert precision_at_k(useful, retrieved, k=5) == pytest.approx(0.4)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k({"doc-a"}, [], k=5) == 0.0


def test_precision_at_k_invalid_k() -> None:
    with pytest.raises(ValueError, match="k debe ser > 0"):
        precision_at_k({"a"}, ["a"], k=0)


def test_evaluate_case_hit_and_precision() -> None:
    result = evaluate_case(
        pregunta="¿Qué es BM25?",
        documento_id_esperado="hybrid-retrieval-bm25",
        documentos_utiles=["hybrid-retrieval-bm25", "langchain-ensemble"],
        retrieved_ids=[
            "hybrid-retrieval-bm25",
            "noise",
            "langchain-ensemble",
            "other",
            "x",
        ],
        k=5,
    )
    assert result.hit is True
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == pytest.approx(0.4)
    assert result.recuperados[0] == "hybrid-retrieval-bm25"


def test_evaluate_case_miss() -> None:
    result = evaluate_case(
        pregunta="namespaces",
        documento_id_esperado="pinecone-serverless",
        documentos_utiles=["pinecone-serverless"],
        retrieved_ids=["a", "b", "c", "d", "e"],
        k=5,
    )
    assert result.hit is False
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0


def test_average_empty_and_values() -> None:
    assert average([]) == 0.0
    assert average([1.0, 0.0, 0.5]) == pytest.approx(0.5)
