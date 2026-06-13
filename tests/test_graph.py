"""Graph wiring tests: structure and routing logic, no API calls."""

from langchain_core.documents import Document

from rag_agent.config import Settings
from rag_agent.graph import build_graph, initial_state


class StubVectorStore:
    def similarity_search(self, query: str, k: int = 4):
        return [Document(page_content=f"stub result for {query}", metadata={"source": "stub"})]


def make_settings() -> Settings:
    return Settings(anthropic_api_key="test-key-not-real", _env_file=None)


def test_graph_compiles_with_expected_nodes():
    graph = build_graph(make_settings(), vectorstore=StubVectorStore())
    nodes = set(graph.get_graph().nodes)
    assert {"retrieve", "grade_documents", "rewrite_query", "generate"} <= nodes


def test_initial_state_shape():
    state = initial_state("What is a checkpointer?")
    assert state["question"] == state["query"] == "What is a checkpointer?"
    assert state["documents"] == [] and state["rewrites"] == 0
