from pydantic import BaseModel


class RAGResponse(BaseModel):
    respuesta: str
    referencias: list[str]
