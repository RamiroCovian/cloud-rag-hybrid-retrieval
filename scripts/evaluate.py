"""Evalúa el recuperador híbrido: Precision@5 y Recall@5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import EVALUATION_DIR
from src.evaluation import average, evaluate_case
from src.rag_system import RAGSystem

DEFAULT_GOLDEN_SET = EVALUATION_DIR / "golden_set.json"


def load_golden_set(path: Path) -> list[dict]:
    """Carga y valida el Golden Set JSON de evaluación."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el Golden Set: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("El Golden Set debe ser una lista no vacía de casos")

    cases: list[dict] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Caso #{index} inválido: se esperaba un objeto")
        pregunta = str(item.get("pregunta", "")).strip()
        expected = str(item.get("documento_id_esperado", "")).strip()
        if not pregunta or not expected:
            raise ValueError(
                f"Caso #{index} requiere 'pregunta' y 'documento_id_esperado'"
            )
        useful = item.get("documentos_utiles") or [expected]
        if isinstance(useful, str):
            useful = [useful]
        cases.append(
            {
                "pregunta": pregunta,
                "documento_id_esperado": expected,
                "documentos_utiles": [str(x) for x in useful],
            }
        )
    return cases


def document_ids_from_results(documents, k: int) -> list[str]:
    """Extrae los `document_id` de los top-k documentos recuperados."""
    ids: list[str] = []
    for doc in documents[:k]:
        doc_id = str(doc.metadata.get("document_id", "")).strip()
        ids.append(doc_id or "unknown")
    return ids


def run_evaluation(
    *,
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
    k: int = 5,
) -> dict:
    """Corre todas las preguntas del Golden Set y calcula métricas."""
    cases = load_golden_set(golden_set_path)
    rag = RAGSystem(k=k)

    results = []
    for case in cases:
        docs = rag.retrieve(case["pregunta"], k=k)
        retrieved_ids = document_ids_from_results(docs, k)
        result = evaluate_case(
            pregunta=case["pregunta"],
            documento_id_esperado=case["documento_id_esperado"],
            documentos_utiles=case["documentos_utiles"],
            retrieved_ids=retrieved_ids,
            k=k,
        )
        results.append(result)

    mean_recall = average([r.recall_at_k for r in results])
    mean_precision = average([r.precision_at_k for r in results])

    return {
        "k": k,
        "cases": len(results),
        "results": results,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
    }


def print_report(summary: dict) -> None:
    """Imprime en consola el detalle por caso y los promedios finales."""
    k = summary["k"]
    print("=" * 60)
    print(f"Evaluación del recuperador híbrido (k={k})")
    print("=" * 60)

    for index, result in enumerate(summary["results"], start=1):
        status = "HIT" if result.hit else "MISS"
        print(f"\n[{index}] {status}")
        print(f"Pregunta: {result.pregunta}")
        print(f"Esperado: {result.documento_id_esperado}")
        print(f"Top-{k}: {result.recuperados}")
        print(
            f"Recall@{k}: {result.recall_at_k:.2f} | "
            f"Precision@{k}: {result.precision_at_k:.2f}"
        )

    print("\n" + "-" * 60)
    print(
        f"Promedio Recall@{k}: {summary['mean_recall']:.2f} | "
        f"Promedio Precision@{k}: {summary['mean_precision']:.2f}"
    )
    print(f"Casos evaluados: {summary['cases']}")
    print("-" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Define los argumentos CLI del script de evaluación."""
    parser = argparse.ArgumentParser(
        description="Calcula Precision@k y Recall@k con un Golden Set."
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET,
        help=f"Ruta al Golden Set JSON (default: {DEFAULT_GOLDEN_SET})",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k a evaluar")
    return parser


def main() -> None:
    """Punto de entrada: evalúa el recuperador e imprime el reporte."""
    args = build_parser().parse_args()
    summary = run_evaluation(golden_set_path=args.golden_set, k=args.k)
    print_report(summary)


if __name__ == "__main__":
    main()
