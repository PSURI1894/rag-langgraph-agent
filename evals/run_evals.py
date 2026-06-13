"""Evaluation harness for the RAG agent.

Two layers of metrics over evals/dataset.jsonl:

  Retrieval (deterministic, offline):
    - hit@k : a retrieved chunk contains one of the item's must_retrieve phrases
    - MRR   : reciprocal rank of the first such chunk

  Generation (LLM-as-judge, needs ANTHROPIC_API_KEY):
    - correctness  : 1-5 score of the agent's answer vs the reference answer
    - faithfulness : is the answer grounded in the retrieved excerpts?

Run with:
  uv run python evals/run_evals.py                  # full suite
  uv run python evals/run_evals.py --retrieval-only # no API key needed
  uv run python evals/run_evals.py --limit 3        # quick smoke run

Results are written to evals/results/ (JSON + markdown summary).
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from rag_agent.config import get_settings
from rag_agent.graph import answer_question, build_graph
from rag_agent.retriever import build_retriever
from rag_agent.vectorstore import get_vectorstore

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"


class CorrectnessGrade(BaseModel):
    """Comparison of a candidate answer against the reference answer."""

    reasoning: str = Field(description="Brief justification for the score.")
    score: int = Field(
        ge=1,
        le=5,
        description="5 = fully correct and complete vs the reference; 3 = partially correct or missing key points; 1 = wrong or contradicts the reference.",
    )


class FaithfulnessGrade(BaseModel):
    """Whether the answer is grounded in the provided documentation excerpts."""

    reasoning: str = Field(description="Brief justification.")
    grounded: bool = Field(
        description="True only if every factual claim in the answer is supported by the excerpts (or the answer explicitly declines)."
    )


CORRECTNESS_PROMPT = """You are grading a RAG system's answer about the LangGraph framework.

Question: {question}

Reference answer (ground truth):
{reference}

Candidate answer:
{answer}

Score the candidate's correctness against the reference. Extra correct detail is fine; missing or wrong core facts lower the score."""

FAITHFULNESS_PROMPT = """You are auditing a RAG answer for hallucinations.

Documentation excerpts the system retrieved:
{context}

Answer it produced:
{answer}

Decide whether the answer is fully grounded in the excerpts."""


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def score_retrieval(documents, must_retrieve: list[str]) -> tuple[bool, float]:
    """Return (hit, reciprocal_rank) for the first chunk containing any target phrase."""
    phrases = [p.lower() for p in must_retrieve]
    for rank, doc in enumerate(documents, start=1):
        text = doc.page_content.lower()
        if any(p in text for p in phrases):
            return True, 1.0 / rank
    return False, 0.0


def run(dataset_path: Path, limit: int | None, retrieval_only: bool) -> dict:
    settings = get_settings()
    items = load_dataset(dataset_path)
    if limit:
        items = items[:limit]

    vectorstore = get_vectorstore(settings)
    retriever = build_retriever(settings, vectorstore)
    graph = None
    correctness_judge = None
    faithfulness_judge = None
    if not retrieval_only:
        if not settings.anthropic_api_key:
            sys.exit("ANTHROPIC_API_KEY is not set. Use --retrieval-only or configure the key in .env.")
        from langchain_anthropic import ChatAnthropic

        graph = build_graph(settings, retriever=retriever)
        judge = ChatAnthropic(model=settings.judge_model, api_key=settings.api_key_or_none, max_tokens=1024)
        correctness_judge = judge.with_structured_output(CorrectnessGrade)
        faithfulness_judge = judge.with_structured_output(FaithfulnessGrade)

    rows = []
    for item in items:
        row: dict = {"id": item["id"], "question": item["question"]}

        documents = retriever.search(item["question"])
        hit, rr = score_retrieval(documents, item["must_retrieve"])
        row["retrieval"] = {
            "hit": hit,
            "reciprocal_rank": rr,
            "retrieved_sources": [d.metadata.get("source", "?") for d in documents],
        }

        if not retrieval_only:
            result = answer_question(graph, item["question"])
            context = "\n\n".join(d.page_content for d in result["documents"]) or "(none)"
            correctness: CorrectnessGrade = correctness_judge.invoke(
                CORRECTNESS_PROMPT.format(
                    question=item["question"], reference=item["reference_answer"], answer=result["answer"]
                )
            )
            faithfulness: FaithfulnessGrade = faithfulness_judge.invoke(
                FAITHFULNESS_PROMPT.format(context=context, answer=result["answer"])
            )
            row["generation"] = {
                "answer": result["answer"],
                "sources": result["sources"],
                "rewrites": result["rewrites"],
                "correctness_score": correctness.score,
                "correctness_reasoning": correctness.reasoning,
                "grounded": faithfulness.grounded,
                "faithfulness_reasoning": faithfulness.reasoning,
            }

        status = "hit " if hit else "MISS"
        print(f"[{status}] {item['id']}", flush=True)
        rows.append(row)

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path.name,
        "n_items": len(rows),
        "k": settings.retrieval_k,
        "retrieval_mode": settings.retrieval_mode,
        "reranker": settings.rerank_model if (settings.retrieval_mode == "hybrid" and settings.use_reranker) else None,
        "generation_model": None if retrieval_only else settings.generation_model,
        "retrieval": {
            "hit_rate": round(sum(r["retrieval"]["hit"] for r in rows) / len(rows), 3),
            "mrr": round(statistics.mean(r["retrieval"]["reciprocal_rank"] for r in rows), 3),
        },
    }
    if not retrieval_only:
        gens = [r["generation"] for r in rows]
        summary["generation"] = {
            "mean_correctness": round(statistics.mean(g["correctness_score"] for g in gens), 2),
            "faithfulness_rate": round(sum(g["grounded"] for g in gens) / len(gens), 3),
            "rewrite_rate": round(sum(g["rewrites"] > 0 for g in gens) / len(gens), 3),
        }
    return {"summary": summary, "rows": rows}


def write_results(results: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = results["summary"]["run_at"].replace(":", "-").replace("+00-00", "Z")
    json_path = RESULTS_DIR / f"run_{stamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    s = results["summary"]
    lines = [
        "# Eval results",
        "",
        f"- Run: {s['run_at']}  |  items: {s['n_items']}  |  k: {s['k']}",
        f"- Generation model: {s['generation_model'] or 'skipped (retrieval-only)'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval hit@{s['k']} | {s['retrieval']['hit_rate']:.0%} |",
        f"| Retrieval MRR | {s['retrieval']['mrr']:.3f} |",
    ]
    if "generation" in s:
        g = s["generation"]
        lines += [
            f"| Correctness (LLM judge, 1-5) | {g['mean_correctness']} |",
            f"| Faithfulness rate | {g['faithfulness_rate']:.0%} |",
            f"| Query-rewrite rate | {g['rewrite_rate']:.0%} |",
        ]
    (RESULTS_DIR / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval suite.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N items.")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip generation + LLM judges (no API key needed)."
    )
    args = parser.parse_args()

    results = run(args.dataset, args.limit, args.retrieval_only)
    json_path = write_results(results)

    print()
    print((RESULTS_DIR / "latest.md").read_text(encoding="utf-8"))
    print(f"Full results: {json_path}")


if __name__ == "__main__":
    main()
