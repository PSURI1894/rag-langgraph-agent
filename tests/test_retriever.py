"""Retriever logic — RRF fusion, dense, and hybrid (BM25) — all offline."""

from langchain_core.documents import Document

from rag_agent.retriever import DenseRetriever, HybridRetriever, reciprocal_rank_fusion


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": text})


class StubVectorStore:
    """Minimal stand-in: dense order is the given list; get() feeds BM25."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self._docs[:k]

    def get(self, include=None) -> dict:
        return {"documents": [d.page_content for d in self._docs], "metadatas": [d.metadata for d in self._docs]}


def test_rrf_rewards_agreement_across_lists():
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    # `b` is ranked highly in both lists; `a` and `c` only appear once each.
    fused = reciprocal_rank_fusion([[a, b], [b, c]])
    assert fused[0].page_content == "b"
    assert {d.page_content for d in fused} == {"a", "b", "c"}


def test_dense_retriever_respects_k():
    docs = [_doc("checkpointer"), _doc("reducer"), _doc("subgraph")]
    retriever = DenseRetriever(StubVectorStore(docs), k=2)
    assert [d.page_content for d in retriever.search("x")] == ["checkpointer", "reducer"]


def test_hybrid_retriever_surfaces_exact_lexical_match():
    # Dense order buries the exact-term doc; BM25 + RRF should lift it.
    docs = [
        _doc("an unrelated paragraph about graphs"),
        _doc("another tangential note on state"),
        _doc("the add_messages reducer appends messages to the list"),
    ]
    retriever = HybridRetriever(StubVectorStore(docs), k=3, fetch_k=3, reranker=None)
    results = retriever.search("what does add_messages do")
    assert any("add_messages" in d.page_content for d in results)
