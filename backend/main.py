from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):
    context = rag.build_context(request.query)
    return {"context": context}