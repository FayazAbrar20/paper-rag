from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urljoin

import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

from generate import DEFAULT_LLM, OLLAMA_URL, answer
from retrieve import DEFAULT_INDEX, load_index, retrieve
from ui import demo


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class Source(BaseModel):
    paper: str
    page: int
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app):
    retrieve("warm up")
    yield


app = FastAPI(title="PaperRAG", description="Question answering over research papers I co-authored", lifespan=lifespan)


@app.get("/health")
def health():
    _, chunks, _ = load_index(DEFAULT_INDEX.resolve())
    try:
        ollama = requests.get(urljoin(OLLAMA_URL, "/api/tags"), timeout=3).ok
    except requests.RequestException:
        ollama = False
    return {"status": "ok" if ollama else "degraded", "chunks": len(chunks), "ollama": ollama, "llm": DEFAULT_LLM}


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    try:
        return answer(body.question)
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail=f"Language model unavailable: {error}")


app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
