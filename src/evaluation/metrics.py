"""Cálculo de Precision@k y Recall@k para el recuperador híbrido."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricResult:
    pregunta: str
    documento_id_esperado: str
    recuperados: list[str]
    recall_at_k: float
    precision_at_k: float
    hit: bool


def recall_at_k(expected_id: str, retrieved_ids: list[str], k: int) -> float:
    """1.0 si el documento esperado está en el top-k; 0.0 si no."""
    top = retrieved_ids[:k]
    return 1.0 if expected_id in top else 0.0


def precision_at_k(
    useful_ids: set[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """Porcentaje de los top-k recuperados que son útiles."""
    if k <= 0:
        raise ValueError("k debe ser > 0")
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    useful_hits = sum(1 for doc_id in top if doc_id in useful_ids)
    return useful_hits / k


def evaluate_case(
    *,
    pregunta: str,
    documento_id_esperado: str,
    documentos_utiles: list[str],
    retrieved_ids: list[str],
    k: int = 5,
) -> MetricResult:
    useful = set(documentos_utiles) | {documento_id_esperado}
    recall = recall_at_k(documento_id_esperado, retrieved_ids, k)
    precision = precision_at_k(useful, retrieved_ids, k)
    return MetricResult(
        pregunta=pregunta,
        documento_id_esperado=documento_id_esperado,
        recuperados=retrieved_ids[:k],
        recall_at_k=recall,
        precision_at_k=precision,
        hit=recall == 1.0,
    )


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
