from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(
    data_dir: str,
    vectorstore_dir: str
):

    data_path = Path(data_dir)
    vector_path = Path(vectorstore_dir)

    documents = []

    for file_path in data_path.glob("*"):
        if file_path.suffix in [".txt", ".md"]:

            content = file_path.read_text(
                encoding="utf-8"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": file_path.name}
                )
            )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(vector_path)
    )

    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(chunks)

    return vectorstore
