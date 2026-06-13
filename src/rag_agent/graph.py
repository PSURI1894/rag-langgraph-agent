"""Agentic RAG as a LangGraph state machine.

    START -> retrieve -> grade_documents -+-> generate -> END
                ^                         |
                +------ rewrite_query <---+  (only when nothing relevant
                                              was retrieved and the rewrite
                                              budget is not exhausted)

The grading step is what makes this agentic rather than a fixed pipeline:
the model inspects its own retrieval results and decides whether to answer
or to reformulate the query and try again.
"""

from typing import Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_agent.config import Settings, get_settings


class RAGState(TypedDict):
    question: str  # the user's original question, never mutated
    query: str  # current retrieval query, may be rewritten
    documents: list[Document]
    answer: str
    sources: list[str]
    rewrites: int


class RelevanceGrade(BaseModel):
    """Which of the numbered documents are relevant to the question."""

    relevant_indices: list[int] = Field(
        description="0-based indices of the documents that contain information useful for answering the question. Empty list if none are relevant."
    )


GENERATE_SYSTEM_PROMPT = """You are a precise technical assistant answering questions about LangGraph, \
the agent-orchestration framework, using only the documentation excerpts provided.

Rules:
- Ground every claim in the provided excerpts. Cite them inline as [1], [2], ... matching the excerpt numbers.
- Include a short code example when the excerpts contain one that helps.
- If the excerpts do not contain enough information to answer, say so plainly and answer only what they support. Never invent APIs."""

GRADE_PROMPT = """Question: {question}

Below are {n} retrieved documentation excerpts, numbered from 0.

{documents}

Identify which excerpts contain information that helps answer the question."""

REWRITE_PROMPT = """A documentation search for the query below returned no relevant results.

Original question: {question}
Failed query: {query}

Write one improved search query for the LangGraph documentation. Prefer concrete API and concept terms over conversational phrasing. Reply with the query only."""


def build_graph(settings: Settings | None = None, vectorstore=None):
    """Compile the RAG graph. `vectorstore` only needs a similarity_search method."""
    settings = settings or get_settings()
    if vectorstore is None:
        from rag_agent.vectorstore import get_vectorstore

        vectorstore = get_vectorstore(settings)

    # Clients are built lazily so the graph (and the FastAPI app) can compile and
    # serve /healthz without an API key — only nodes that actually call Claude
    # require ANTHROPIC_API_KEY, and they fail at invocation time, not build time.
    clients: dict[str, object] = {}

    def get_llm() -> ChatAnthropic:
        if "llm" not in clients:
            clients["llm"] = ChatAnthropic(
                model=settings.generation_model,
                api_key=settings.api_key_or_none,
                max_tokens=settings.max_answer_tokens,
            )
        return clients["llm"]

    def get_grader():
        if "grader" not in clients:
            clients["grader"] = ChatAnthropic(
                model=settings.generation_model,
                api_key=settings.api_key_or_none,
                max_tokens=512,
            ).with_structured_output(RelevanceGrade)
        return clients["grader"]

    def retrieve(state: RAGState) -> dict:
        documents = vectorstore.similarity_search(state["query"], k=settings.retrieval_k)
        return {"documents": documents}

    def grade_documents(state: RAGState) -> dict:
        if not state["documents"]:
            return {"documents": []}
        numbered = "\n\n".join(
            f"[{i}] {doc.page_content[:1500]}" for i, doc in enumerate(state["documents"])
        )
        grade: RelevanceGrade = get_grader().invoke(
            GRADE_PROMPT.format(question=state["question"], n=len(state["documents"]), documents=numbered)
        )
        keep = [state["documents"][i] for i in grade.relevant_indices if 0 <= i < len(state["documents"])]
        return {"documents": keep}

    def rewrite_query(state: RAGState) -> dict:
        response = get_llm().invoke(REWRITE_PROMPT.format(question=state["question"], query=state["query"]))
        return {"query": response.text().strip(), "rewrites": state["rewrites"] + 1}

    def generate(state: RAGState) -> dict:
        if state["documents"]:
            context = "\n\n".join(
                f"[{i + 1}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
                for i, doc in enumerate(state["documents"])
            )
        else:
            context = "(no relevant documentation excerpts were found)"
        response = get_llm().invoke(
            [
                ("system", GENERATE_SYSTEM_PROMPT),
                ("user", f"Documentation excerpts:\n\n{context}\n\nQuestion: {state['question']}"),
            ]
        )
        sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in state["documents"]))
        return {"answer": response.text(), "sources": sources}

    def route_after_grading(state: RAGState) -> Literal["generate", "rewrite_query"]:
        if state["documents"] or state["rewrites"] >= settings.max_query_rewrites:
            return "generate"
        return "rewrite_query"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_after_grading)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "query": question,
        "documents": [],
        "answer": "",
        "sources": [],
        "rewrites": 0,
    }


def answer_question(graph, question: str) -> dict:
    """Convenience wrapper: run the graph and return a serializable result."""
    state = graph.invoke(initial_state(question))
    return {
        "answer": state["answer"],
        "sources": state["sources"],
        "rewrites": state["rewrites"],
        "documents": state["documents"],
    }
