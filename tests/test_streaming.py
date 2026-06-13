"""SSE streaming logic — exercised with a fake graph, no API calls."""

import asyncio
import json

from rag_agent.api import _sse, event_stream


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGraph:
    """Mimics LangGraph's multi-mode astream output."""

    async def astream(self, state, stream_mode=None):
        yield ("messages", (_Chunk("A check"), {"langgraph_node": "generate"}))
        yield ("messages", (_Chunk("pointer ..."), {"langgraph_node": "generate"}))
        # Tokens from other nodes must NOT be streamed to the client.
        yield ("messages", (_Chunk("internal grading"), {"langgraph_node": "grade_documents"}))
        yield ("updates", {"generate": {"sources": ["https://docs.langchain.com/oss/langgraph/persistence"]}})


def _collect(question: str) -> list[str]:
    async def run() -> list[str]:
        return [frame async for frame in event_stream(_FakeGraph(), question)]

    return asyncio.run(run())


def test_sse_framing():
    assert _sse({"token": "hi"}) == 'data: {"token": "hi"}\n\n'


def test_stream_emits_answer_tokens_then_done_with_sources():
    frames = _collect("what is a checkpointer?")
    payloads = [json.loads(f.removeprefix("data: ").strip()) for f in frames]

    tokens = [p["token"] for p in payloads if "token" in p]
    assert "".join(tokens) == "A checkpointer ..."  # only generate-node tokens

    final = payloads[-1]
    assert final["done"] is True
    assert final["sources"] == ["https://docs.langchain.com/oss/langgraph/persistence"]


def test_grading_tokens_are_not_leaked():
    frames = _collect("q")
    assert not any("internal grading" in f for f in frames)
