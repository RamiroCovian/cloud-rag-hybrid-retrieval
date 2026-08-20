# Cloud RAG Hybrid Retrieval

Sistema RAG escalable en la nube con **Pinecone Serverless**, recuperación híbrida (**BM25 + vectorial**) y evaluación local (`Precision@5` / `Recall@5`).

## Requisitos

- Python 3.10+
- Cuenta de [Pinecone](https://www.pinecone.io/)
- API key de un proveedor de embeddings (Gemini, OpenAI, Voyage, Cohere, etc.)

## Setup

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Completá en `.env` al menos:

| Variable | Descripción |
|----------|-------------|
| `PINECONE_API_KEY` | API key de Pinecone |
| `INDEX_NAME` | Nombre del índice (ej. `rag-hybrid-retrieval`) |
| `GOOGLE_API_KEY` o `OPENAI_API_KEY` | Key del proveedor activo |

Proveedores soportados:

- **Embeddings:** `openai`, `gemini`, `voyage`, `cohere`, `openai_compatible`
- **LLM:** `openai`, `gemini`, `anthropic`, `grok`, `openai_compatible`

Defaults del proyecto (Gemini):

```env
EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIMENSION=768
PINECONE_NAMESPACE=docs
```

> **Importante:** la dimensión del índice debe coincidir con el modelo de embeddings. Gemini → `768`. OpenAI `text-embedding-3-small` → `1536`.

## Cómo replicar el índice de Pinecone

1. Creá (o verificá) el índice Serverless:

```powershell
python scripts/init_pinecone.py
```

El script:
- comprueba si existe `INDEX_NAME`
- si no existe, lo crea en modo **Serverless** con:
  - métrica `cosine`
  - dimensión = `EMBEDDING_DIMENSION`
  - cloud/region = `PINECONE_CLOUD` / `PINECONE_REGION`

2. Ingestá los documentos de `data/documents/` (Markdown, PDF, JSON, TXT):

```powershell
python scripts/ingest.py
```

Esto hace chunking con `RecursiveCharacterTextSplitter`, genera embeddings y hace upsert en el namespace configurado (`docs` por defecto), guardando en metadata `source`, `page`, `category`, `tags` y `text`.

3. (Opcional) Verificá en la consola de Pinecone que el índice tenga vectores en el namespace `docs`.

## Uso

### Consulta híbrida (BM25 + Pinecone)

```powershell
python scripts/query.py "¿Qué es BM25 en recuperación híbrida?"
```

Devuelve el top-5 combinando recuperación léxica y semántica (`RAGSystem` + `EnsembleRetriever`).

### Evaluación

```powershell
python scripts/evaluate.py
```

Usa el Golden Set en `data/evaluation/golden_set.json` (5 preguntas con documento esperado) y calcula:

- **Recall@5:** ¿está el documento correcto en el Top-5?
- **Precision@5:** ¿qué porcentaje del Top-5 es útil?

## Estructura

```text
scripts/
  init_pinecone.py   # crea el índice Serverless si no existe
  ingest.py          # pipeline de ingesta
  query.py           # consulta híbrida
  evaluate.py        # Precision@5 y Recall@5
src/
  config.py          # variables de entorno multi-proveedor
  providers.py       # factories de embeddings / LLM
  rag_system.py      # RAGSystem (EnsembleRetriever)
  ingestion/         # loaders + chunking + pipeline
  evaluation/        # métricas
data/
  documents/         # corpus técnico de ejemplo
  evaluation/        # golden_set.json
```

## Resumen de métricas

Resultado de `python scripts/evaluate.py` sobre el Golden Set de ejemplo (5 preguntas, k=5):

| Métrica | Valor |
|---------|-------|
| **Recall@5** (promedio) | **1.00** |
| **Precision@5** (promedio) | **0.32** |
| Hits | 5 / 5 |

Detalle por caso:

| # | Esperado | Recall@5 | Precision@5 |
|---|----------|----------|-------------|
| 1 | `hybrid-retrieval-bm25` | 1.00 | 0.40 |
| 2 | `pinecone-serverless` | 1.00 | 0.20 |
| 3 | `chunking-embeddings` | 1.00 | 0.40 |
| 4 | `evaluation-metrics` | 1.00 | 0.20 |
| 5 | `langchain-ensemble` | 1.00 | 0.40 |

Interpretación breve: el documento correcto siempre entra en el Top-5 (Recall perfecto). La Precision es más baja porque el corpus es pequeño y varios chunks relacionados también aparecen en el ranking.
## Flujo completo (checklist)

1. Configurar `.env`
2. `pip install -r requirements.txt`
3. `python scripts/init_pinecone.py`
4. `python scripts/ingest.py`
5. `python scripts/query.py "tu pregunta"`
6. `python scripts/evaluate.py`
