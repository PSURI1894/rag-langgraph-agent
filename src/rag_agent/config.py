"""Central configuration, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""

    generation_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "langgraph_docs"
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw" / "langchain-docs"

    retrieval_k: int = 4
    max_query_rewrites: int = 1
    max_answer_tokens: int = 2048

    @property
    def api_key_or_none(self) -> str | None:
        return self.anthropic_api_key or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
