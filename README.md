# End-to-End RAG

Sistema de recuperación semántica local (RAG) desarrollado como entrega del curso AI Engineering.

## Objetivo

Consultar documentos técnicos mediante recuperación semántica y un modelo de lenguaje.

El sistema responde utilizando el contexto recuperado y devuelve las referencias de los documentos utilizados.

Si el contexto no contiene información suficiente, el sistema indica que no puede responder.

## Arquitectura

- schemas.py: define el modelo Pydantic de la respuesta.
- ingestion.py: carga documentos, los divide en fragmentos, genera embeddings y los almacena en Chroma.
- rag_chain.py: recupera fragmentos y genera respuestas.
- main.py: punto de entrada para ejecutar una consulta.
- data/: contiene los documentos de conocimiento.
- vectorstore/: almacena la base vectorial local.

## Tecnologías

- Python
- LangChain
- OpenAI
- ChromaDB
- Pydantic
- Tiktoken

## Instalación

Instalar las dependencias:

    pip install -r requirements.txt

## Configuración

Configurar la variable de entorno OPENAI_API_KEY con una API key válida de OpenAI.

No incluir la API key en el repositorio.

## Ejecución

Desde la carpeta del proyecto:

    python main.py

El programa solicitará una pregunta y devolverá la respuesta y las referencias utilizadas.

## Validación

Se probaron consultas respondibles y consultas para las cuales el contexto recuperado no contiene información suficiente.

Las respuestas se estructuran mediante el modelo Pydantic RAGResponse.

## Seguridad

- No publicar API keys.
- No publicar archivos .env.
- No publicar la base vectorial local.
