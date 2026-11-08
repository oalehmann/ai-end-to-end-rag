import asyncio
from pathlib import Path

from rag_chain import create_rag_chain


PROJECT_DIR = Path(__file__).parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"


async def main():

    get_rag_response = create_rag_chain(
        vectorstore_dir=str(VECTORSTORE_DIR)
    )

    query = input("Ingresá tu pregunta: ")

    result = await get_rag_response(query)

    print("\nRESPUESTA:")
    print(result.respuesta)

    print("\nREFERENCIAS:")
    for reference in result.referencias:
        print(f"- {reference}")


if __name__ == "__main__":
    asyncio.run(main())
