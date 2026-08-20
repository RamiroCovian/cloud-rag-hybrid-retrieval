# Recuperación híbrida con BM25

La búsqueda híbrida combina similitud vectorial semántica con recuperación léxica
BM25. BM25 destaca cuando la consulta incluye términos técnicos exactos, siglas o
nombres propios que el embedding puede diluir.

## EnsembleRetriever

En LangChain, un `EnsembleRetriever` fusiona resultados de varios recuperadores
(por ejemplo Pinecone + BM25) y reordena con pesos configurables. El sistema RAG
suele devolver los top-5 documentos combinados.

## Cuándo usar BM25

- Consultas con identificadores de API (`create_index`, `Precision@k`)
- Nombres de librerías o clases (`RecursiveCharacterTextSplitter`)
- Términos poco frecuentes en el corpus
