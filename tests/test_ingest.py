"""Chunking and frontmatter handling, without touching network or embeddings."""

from langchain_core.documents import Document

from rag_agent.ingest import FRONTMATTER_RE, TITLE_RE, chunk_documents


def test_chunking_splits_and_preserves_metadata():
    doc = Document(
        page_content="LangGraph paragraph. " * 300,
        metadata={"source": "https://docs.langchain.com/oss/langgraph/graph-api", "title": "Graph API"},
    )
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["source"] == doc.metadata["source"] for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)


def test_frontmatter_is_stripped_and_title_extracted():
    raw = '---\ntitle: "Graph API"\nsidebarTitle: Graph API\n---\n# Heading\nBody text.'
    match = FRONTMATTER_RE.match(raw)
    assert match
    assert TITLE_RE.search(match.group(1)).group(1) == "Graph API"
    assert raw[match.end() :].startswith("# Heading")
