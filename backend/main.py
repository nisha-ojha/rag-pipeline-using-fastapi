from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline import RAGPipeline

app = FastAPI()

rag = RAGPipeline()


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):
    answer = rag.generate_answer(request.query)
    return {"response": answer}