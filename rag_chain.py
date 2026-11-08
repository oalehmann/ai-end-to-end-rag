from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from schemas import RAGResponse


def create_rag_chain(vectorstore_dir: str):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embeddings,
        persist_directory=vectorstore_dir
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    async def get_rag_response(query: str) -> RAGResponse:

        results = vectorstore.similarity_search(
            query,
            k=3
        )

        context = "\n\n".join(
            result.page_content
            for result in results
        )

        references = list(dict.fromkeys(
            result.metadata["source"]
            for result in results
        ))

        prompt = f"""
        Respondé la pregunta utilizando exclusivamente
        la información del siguiente contexto.

        Si el contexto no contiene información suficiente,
        respondé exactamente:
        No encuentro información suficiente en el contexto para responder.

        CONTEXTO:
        {context}

        PREGUNTA:
        {query}
        """

        response = await llm.ainvoke(prompt)

        return RAGResponse(
            respuesta=response.content,
            referencias=references
        )

    return get_rag_response
