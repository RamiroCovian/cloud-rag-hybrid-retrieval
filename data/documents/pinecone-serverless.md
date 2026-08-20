# Pinecone Serverless

Pinecone Serverless es un índice de vectores administrado que escala automáticamente
sin gestionar pods. Se crea con una dimensión fija (por ejemplo 768 para Gemini
`gemini-embedding-001` o 1536 para OpenAI `text-embedding-3-small`) y una métrica
de similitud como `cosine`.

## Namespaces

Los namespaces permiten aislar conjuntos de vectores dentro del mismo índice.
En aplicaciones multi-inquilino o con distintos tipos de datos, usar namespaces
evita que la búsqueda sea ruidosa y lenta.

## Metadata avanzada

Al upsert se recomienda guardar en metadata:

- `source`: ruta o nombre del archivo origen
- `page`: número de página (PDFs)
- `category`: etiqueta de categoría temática
- `text`: contenido original del chunk

Guardar el texto en metadata evita consultas adicionales a una base relacional.
