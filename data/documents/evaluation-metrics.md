# Métricas de evaluación RAG

Para medir la calidad del recuperador se usa un Golden Set de preguntas con el
`documento_id` esperado. Dos métricas fundamentales son:

## Recall@k

Indica si el documento correcto aparece entre los k recuperados. Con k=5,
Recall@5 vale 1 si el documento fuente está en el Top-5 y 0 en caso contrario.
El recall promedio es la media sobre todas las preguntas del benchmark.

## Precision@k

Mide qué porcentaje de los k documentos recuperados son realmente útiles para
responder la pregunta. Precision@5 = útiles / 5.

Ambas métricas se suelen imprimir en consola tras ejecutar `evaluate.py`.
