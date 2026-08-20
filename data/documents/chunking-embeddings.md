# Chunking y embeddings

El chunking divide documentos largos en segmentos aptos para embedding.
`RecursiveCharacterTextSplitter` parte por separadores jerárquicos (`\n\n`, `\n`,
oraciones y espacios) preservando coherencia local.

## Tamaño recomendado

Chunks muy pequeños pierden contexto semántico; muy grandes diluyen la precisión
del embedding. Un punto medio habitual es ~500–800 tokens (aprox. 2000 caracteres
con overlap de 200–300).

## Mismatch de dimensiones

Intentar subir embeddings de 1536 dimensiones a un índice configurado con 768
(o viceversa) falla. La dimensión del índice debe coincidir con el modelo de
embedding elegido (OpenAI, Gemini, Voyage, Cohere, etc.).
