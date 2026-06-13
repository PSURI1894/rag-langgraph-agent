# RAG LangGraph Agent

An **agentic RAG** system over the [LangGraph](https://docs.langchain.com/oss/langgraph) documentation, built to demonstrate the full core loop of an applied-AI role: **retrieval → an agent that reasons about its own retrieval → a measurable evaluation harness → a production-style API.**

The agent answers questions about LangGraph using only the official docs, grades whether the documents it retrieved are actually relevant, rewrites its query and retries when they aren't, and cites its sources. Every part of it is measured by a versioned eval suite.

```
                ┌─────────────────────────── LangGraph state machine ───────────────────────────┐
                │                                                                                │
  question ──▶  │  retrieve ──▶ grade_documents ──┬──(relevant)────────────────▶ generate ──▶   │ ──▶ answer + sources
                │     ▲                            │                                             │
                │     │                            └──(nothing relevant)──▶ rewrite_query        │
                │     └────────────────────────────────────────────────────────┘               │
                └────────────────────────────────────────────────────────────────────────────────┘
                         Chroma (local)            Claude grades relevance          Claude answers,
                      fastembed embeddings          & decides to retry              grounded in excerpts
```

## Why this project

It exercises, end to end, the things an applied-AI/LLM-engineering role actually involves:

| Capability | Where it lives |
|---|---|
| Document ingestion & chunking | [`src/rag_agent/ingest.py`](src/rag_agent/ingest.py) |
| Vector retrieval | [`src/rag_agent/vectorstore.py`](src/rag_agent/vectorstore.py), [`embeddings.py`](src/rag_agent/embeddings.py) |
| **Agentic** orchestration (self-grading + retry) | [`src/rag_agent/graph.py`](src/rag_agent/graph.py) |
| **Evaluation discipline** (retrieval + generation metrics) | [`evals/`](evals/) |
| Production API | [`src/rag_agent/api.py`](src/rag_agent/api.py) |
| Tests | [`tests/`](tests/) |

## Architecture

The agent is a compiled `StateGraph` with four nodes:

1. **retrieve** — similarity search over the Chroma vector store.
2. **grade_documents** — Claude inspects each retrieved chunk and returns the indices that are genuinely relevant to the question (structured output). This is the step that makes the system *agentic* rather than a fixed pipeline: it reasons about the quality of its own retrieval.
3. **rewrite_query** *(conditional)* — if nothing relevant came back and the rewrite budget isn't spent, Claude reformulates the query into precise API/concept terms and the graph loops back to **retrieve**.
4. **generate** — Claude answers using only the relevant excerpts, citing them inline as `[1]`, `[2]`, and refusing to invent APIs when the context is insufficient.

**Stack:** LangGraph (orchestration) · LangChain Core · Claude (`claude-opus-4-8`) via `langchain-anthropic` · Chroma (vector store) · `fastembed`/BGE for **local, key-free embeddings** (no GPU, no torch download) · FastAPI · `uv` for packaging.

A deliberate design choice: **ingestion and retrieval need no API key** — only generation and the LLM-judge evals call Claude. That means anyone can clone the repo, build the index, and run the retrieval evals for free.

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and `git`.

```bash
# 1. Install dependencies
uv sync

# 2. Build the vector index (downloads the LangGraph docs, ~1 min, no API key needed)
uv run python -m rag_agent.ingest

# 3. (Optional) add your key to enable answering & generation evals
cp .env.example .env          # then edit .env and set ANTHROPIC_API_KEY

# 4. Run the API
uv run uvicorn rag_agent.api:app --reload
```

Then ask a question:

```bash
curl -s localhost:8000/ask -H "content-type: application/json" \
  -d '{"question": "What is a checkpointer in LangGraph and why would I use one?"}'
```

```json
{
  "answer": "A checkpointer persists a snapshot of the graph state at each super-step ... [1][2]",
  "sources": ["https://docs.langchain.com/oss/langgraph/persistence", "..."],
  "rewrites": 0
}
```

`GET /healthz` reports the number of indexed chunks. Interactive docs are at `http://localhost:8000/docs`.

## Evaluation harness

Evaluation is a first-class part of the project, not an afterthought. The dataset ([`evals/dataset.jsonl`](evals/dataset.jsonl)) is a versioned set of question / reference-answer / `must_retrieve` triples covering core LangGraph concepts. [`evals/run_evals.py`](evals/run_evals.py) scores two layers:

**Retrieval (deterministic, no API key):**
- **hit@k** — did a retrieved chunk contain the expected concept?
- **MRR** — how highly ranked was the first relevant chunk?

**Generation (LLM-as-judge, needs `ANTHROPIC_API_KEY`):**
- **correctness** — the answer graded 1–5 against the reference answer.
- **faithfulness** — is every claim grounded in the retrieved excerpts (i.e. no hallucination)?
- **rewrite rate** — how often the agent had to reformulate its query.

```bash
uv run python evals/run_evals.py --retrieval-only   # free, deterministic
uv run python evals/run_evals.py                     # full suite (uses Claude)
```

Results are written to `evals/results/` (JSON per run + a `latest.md` summary).

### Latest results

<!-- EVAL_RESULTS_START -->
14 questions, `k=4`, generation + judges on `claude-opus-4-8`:

| Metric | Value |
|---|---|
| Retrieval hit@4 | **93%** (13/14) |
| Retrieval MRR | **0.774** |
| Correctness (LLM judge, 1–5) | **4.36** |
| Faithfulness rate | **93%** (13/14) |
| Query-rewrite rate | 0% |

**What the harness surfaced** — the value here is in the misses, not the averages:

- **It caught a hallucination.** On `streaming-modes` the agent invented two stream modes that don't exist plus a fictional "typed-projection API"; the correctness judge flagged it (scored 3). This is the single most important thing an eval suite buys you.
- **It caught a faithfulness leak.** The one faithfulness failure (`parallel-send`) was the agent citing a specific error code (`INVALID_CONCURRENT_GRAPH_UPDATE`) that wasn't in the retrieved excerpts — a subtle, plausible-sounding ungrounded claim.
- **Retrieval coverage, not reasoning, is the ceiling.** On `human-in-the-loop` and `parallel-send` the agent *correctly declined* to explain mechanics it couldn't retrieve (faithful, but lower correctness). The fix is better retrieval/chunking, and the metrics point straight at it.
- The `durable-execution` retrieval miss is a metric artifact: the concept lives on the *persistence*/*checkpointers* pages (which were retrieved), but those chunks lack the literal phrase, so the strict substring check flags it.

These are left in rather than tuned away — a suite that always scores 100% isn't measuring anything.
<!-- EVAL_RESULTS_END -->

## Optional: LangSmith tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to trace every graph run in [LangSmith](https://smith.langchain.com) — useful for inspecting the retrieve → grade → rewrite → generate trajectory. The repo runs fully without it.

## Testing

```bash
uv run pytest
```

The suite validates the eval dataset's shape, the chunking/frontmatter logic, and the graph's wiring/routing — all offline (no network, no API key).

## Project layout

```
src/rag_agent/
  config.py        # pydantic-settings configuration
  embeddings.py    # fastembed → LangChain Embeddings adapter
  vectorstore.py   # Chroma access
  ingest.py        # docs → chunks → Chroma
  graph.py         # the LangGraph agent
  api.py           # FastAPI service
evals/
  dataset.jsonl    # versioned eval set
  run_evals.py     # retrieval + generation metrics
tests/             # offline unit tests
```

## License

MIT
