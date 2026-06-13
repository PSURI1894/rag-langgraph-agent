"""FastAPI service wrapping the RAG agent.

Run with:  uv run uvicorn rag_agent.api:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_agent.config import get_settings
from rag_agent.graph import build_graph, initial_state
from rag_agent.vectorstore import get_vectorstore


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["What is a checkpointer in LangGraph?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    rewrites: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.vectorstore = get_vectorstore(settings)
    app.state.graph = build_graph(settings, vectorstore=app.state.vectorstore)
    yield


app = FastAPI(
    title="LangGraph RAG Agent",
    description="Agentic RAG over the LangGraph documentation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict:
    count = app.state.vectorstore._collection.count()
    return {"status": "ok", "indexed_chunks": count}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not get_settings().anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server.")
    state = await app.state.graph.ainvoke(initial_state(request.question))
    return AskResponse(answer=state["answer"], sources=state["sources"], rewrites=state["rewrites"])
