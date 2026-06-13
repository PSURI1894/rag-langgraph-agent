"""FastAPI service wrapping the RAG agent.

Run with:  uv run uvicorn rag_agent.api:app --reload
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


def _sse(data: dict) -> str:
    """Format a Server-Sent Events data frame."""
    return f"data: {json.dumps(data)}\n\n"


async def event_stream(graph, question: str) -> AsyncIterator[str]:
    """Stream the final answer token-by-token, then a closing frame with sources.

    Uses LangGraph's multi-mode streaming: `messages` carries per-token LLM output
    (filtered to the `generate` node so grading/rewrite tokens aren't leaked), and
    `updates` carries node state deltas, from which we pick up the final sources.
    """
    sources: list[str] = []
    async for mode, payload in graph.astream(initial_state(question), stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") == "generate":
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield _sse({"token": text})
        elif mode == "updates":
            update = payload.get("generate") or {}
            if update.get("sources"):
                sources = update["sources"]
    yield _sse({"done": True, "sources": sources})


@app.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    if not get_settings().anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured on the server.")
    return StreamingResponse(
        event_stream(app.state.graph, request.question),
        media_type="text/event-stream",
    )
