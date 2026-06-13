"""Ingest the LangGraph documentation into a local Chroma vector store.

Pipeline: sparse-clone the langchain-ai/docs repo -> load the LangGraph .mdx
pages -> chunk -> embed locally with fastembed -> persist to Chroma.

Run with:  uv run python -m rag_agent.ingest [--force]
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import Settings, get_settings

DOCS_REPO = "https://github.com/langchain-ai/docs.git"
# Only the LangGraph-relevant slices of the (large) unified docs repo.
SPARSE_PATHS = ["src/oss/langgraph", "src/oss/concepts"]
DOCS_BASE_URL = "https://docs.langchain.com"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
TITLE_RE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.MULTILINE)


def fetch_docs(dest: Path, force: bool = False) -> Path:
    """Sparse-clone the docs repo so we only download the paths we index."""
    if dest.exists():
        if not force:
            return dest
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", DOCS_REPO, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *SPARSE_PATHS], check=True)
    return dest


def load_documents(repo_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for sparse_path in SPARSE_PATHS:
        for path in sorted((repo_dir / sparse_path).rglob("*.mdx")):
            raw = path.read_text(encoding="utf-8")
            title = ""
            match = FRONTMATTER_RE.match(raw)
            if match:
                title_match = TITLE_RE.search(match.group(1))
                title = title_match.group(1) if title_match else ""
                raw = raw[match.end() :]
            if not raw.strip():
                continue
            rel = path.relative_to(repo_dir / "src").with_suffix("")
            documents.append(
                Document(
                    page_content=raw,
                    metadata={
                        "source": f"{DOCS_BASE_URL}/{rel.as_posix()}",
                        "title": title or path.stem,
                    },
                )
            )
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: list[Document], settings: Settings) -> int:
    # Imported here so chunking stays testable without the embedding model.
    from rag_agent.vectorstore import get_vectorstore

    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    vectorstore = get_vectorstore(settings)
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[start : start + batch_size])
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download docs and rebuild the index from scratch.")
    args = parser.parse_args()

    settings = get_settings()
    repo_dir = fetch_docs(settings.raw_docs_dir, force=args.force)
    documents = load_documents(repo_dir)
    print(f"Loaded {len(documents)} pages from {repo_dir}")
    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    count = build_vectorstore(chunks, settings)
    print(f"Indexed {count} chunks into Chroma at {settings.chroma_dir}")


if __name__ == "__main__":
    main()
