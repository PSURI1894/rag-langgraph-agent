"""Retrieval strategies, all key-free.

- DenseRetriever  : vector similarity over Chroma.
- HybridRetriever : fuses dense results with BM25 lexical results via Reciprocal
                    Rank Fusion, with an optional local cross-encoder reranker.

Hybrid retrieval helps on queries that hinge on an exact term (an API name like
`add_messages`, a phrase like "durable execution") where pure dense search can
drift to semantically-similar-but-wrong chunks. The reranker then re-scores the
fused candidates with a query-aware cross-encoder.

Both expose the same interface: `.search(query, k=None) -> list[Document]`.
"""

import re

from langchain_core.documents import Document

from rag_agent.config import Settings, get_settings

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(result_lists: list[list[Document]], c: int = 60) -> list[Document]:
    """Fuse ranked lists: score each doc by sum of 1/(c + rank), dedup by content."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=lambda k: scores[k], reverse=True)]


class DenseRetriever:
    def __init__(self, vectorstore, k: int) -> None:
        self._vectorstore = vectorstore
        self._k = k

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return self._vectorstore.similarity_search(query, k=k or self._k)


class CrossEncoderReranker:
    """Local ONNX cross-encoder (via fastembed) — no API key, no torch."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs
        scores = list(self._model.rerank(query, [d.page_content for d in docs]))
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked]


class HybridRetriever:
    def __init__(self, vectorstore, k: int, fetch_k: int, reranker: CrossEncoderReranker | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self._vectorstore = vectorstore
        self._k = k
        self._fetch_k = fetch_k
        self._reranker = reranker

        # Build the BM25 index once from the documents already stored in Chroma.
        stored = vectorstore.get(include=["documents", "metadatas"])
        texts = stored.get("documents") or []
        metadatas = stored.get("metadatas") or [{}] * len(texts)
        self._docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._docs]) if self._docs else None

    def _dense(self, query: str) -> list[Document]:
        return self._vectorstore.similarity_search(query, k=self._fetch_k)

    def _sparse(self, query: str) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self._fetch_k]
        return [self._docs[i] for i in top]

    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self._k
        fused = reciprocal_rank_fusion([self._dense(query), self._sparse(query)])
        if self._reranker is not None:
            fused = self._reranker.rerank(query, fused[: self._fetch_k])
        return fused[:k]


def build_retriever(settings: Settings | None = None, vectorstore=None):
    """Construct the retriever named by settings.retrieval_mode (dense | hybrid)."""
    settings = settings or get_settings()
    if vectorstore is None:
        from rag_agent.vectorstore import get_vectorstore

        vectorstore = get_vectorstore(settings)

    if settings.retrieval_mode == "hybrid":
        reranker = CrossEncoderReranker(settings.rerank_model) if settings.use_reranker else None
        return HybridRetriever(vectorstore, settings.retrieval_k, settings.fetch_k, reranker)
    return DenseRetriever(vectorstore, settings.retrieval_k)
