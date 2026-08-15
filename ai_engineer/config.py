"""Configuration for local-only ForgeMind services."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings with safe, small local defaults."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="FORGEMIND_", extra="ignore"
    )

    ollama_base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:7b"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_models_only: bool = True
    database_url: str = "postgresql://forgemind:forgemind@localhost:5432/forgemind"
    docker_image: str = "python:3.12-slim"
    max_iterations: int = Field(default=5, ge=1, le=10)
    retrieval_top_k: int = Field(default=8, ge=1, le=20)
    context_token_budget: int = Field(default=9000, ge=1000, le=30000)
    max_file_bytes: int = Field(default=750_000, ge=10_000, le=5_000_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
