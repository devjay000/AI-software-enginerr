"""PostgreSQL + pgvector persistence for repository, task, trace, and experience memory."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .models import CodeChunk, FinalReport, TraceEvent

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS repositories (
    id TEXT PRIMARY KEY, root TEXT NOT NULL UNIQUE, summary JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS files (
    repository_id TEXT NOT NULL, path TEXT NOT NULL, language TEXT NOT NULL, sha256 TEXT, PRIMARY KEY(repository_id, path)
);
CREATE TABLE IF NOT EXISTS code_chunks (
    id TEXT PRIMARY KEY, repository_id TEXT NOT NULL, file_path TEXT NOT NULL, symbol TEXT NOT NULL,
    kind TEXT NOT NULL, language TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL,
    content TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}', embedding vector(384),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(symbol, '') || ' ' || coalesce(content, ''))) STORED
);
CREATE INDEX IF NOT EXISTS code_chunks_repo_file_idx ON code_chunks (repository_id, file_path);
CREATE INDEX IF NOT EXISTS code_chunks_embedding_idx ON code_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS code_chunks_search_idx ON code_chunks USING GIN (search_vector);
CREATE TABLE IF NOT EXISTS symbols (
    repository_id TEXT NOT NULL, symbol TEXT NOT NULL, file_path TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY(repository_id, symbol, chunk_id)
);
CREATE TABLE IF NOT EXISTS dependencies (
    repository_id TEXT NOT NULL, source_path TEXT NOT NULL, target TEXT NOT NULL, PRIMARY KEY(repository_id, source_path, target)
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY, repository_id TEXT, issue TEXT NOT NULL, status TEXT NOT NULL, state JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY, task_id TEXT, status TEXT NOT NULL, started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS agent_steps (
    id TEXT PRIMARY KEY, run_id TEXT, timestamp TIMESTAMPTZ NOT NULL, agent TEXT NOT NULL, action TEXT NOT NULL, payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY, task_id TEXT, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, repository_id TEXT, task_id TEXT, category TEXT NOT NULL, content JSONB NOT NULL, embedding vector(384), created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS test_runs (
    id TEXT PRIMARY KEY, task_id TEXT, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class MemoryStore:
    """Uses Postgres for persistence; an in-process cache is deliberately non-persistent."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.available = False
        self.error: str | None = None
        self._ephemeral_traces: list[TraceEvent] = []
        self._ephemeral_experiences: list[str] = []
        self._initialize()

    def _initialize(self) -> None:
        try:
            import psycopg

            with psycopg.connect(self.database_url, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(SCHEMA)
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def _connection(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def save_index(
        self,
        repository_id: str,
        root: str,
        summary: dict[str, Any],
        chunks: list[CodeChunk],
        dependencies: dict[str, list[str]],
    ) -> None:
        if not self.available:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO repositories(id, root, summary) VALUES(%s,%s,%s) ON CONFLICT(root) DO UPDATE SET summary=EXCLUDED.summary",
                (repository_id, root, json.dumps(summary)),
            )
            for chunk in chunks:
                cursor.execute(
                    """INSERT INTO code_chunks(id, repository_id, file_path, symbol, kind, language, start_line, end_line, content, metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(id) DO UPDATE SET content=EXCLUDED.content, metadata=EXCLUDED.metadata""",
                    (
                        chunk.id,
                        repository_id,
                        chunk.file,
                        chunk.symbol,
                        chunk.kind,
                        chunk.language,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        json.dumps(chunk.metadata),
                    ),
                )
                cursor.execute(
                    "INSERT INTO symbols(repository_id, symbol, file_path, chunk_id) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (repository_id, chunk.symbol, chunk.file, chunk.id),
                )
            for source, targets in dependencies.items():
                for target in targets:
                    cursor.execute(
                        "INSERT INTO dependencies(repository_id, source_path, target) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                        (repository_id, source, target),
                    )
            connection.commit()

    def append_trace(self, trace: TraceEvent, run_id: str | None = None) -> None:
        self._ephemeral_traces.append(trace)
        if not self.available:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO agent_steps(id, run_id, timestamp, agent, action, payload) VALUES(%s,%s,%s,%s,%s,%s)",
                (
                    str(uuid4()),
                    run_id,
                    trace.timestamp,
                    trace.agent,
                    trace.action,
                    json.dumps(trace.model_dump(mode="json")),
                ),
            )
            connection.commit()

    def save_embeddings(
        self, repository_id: str, chunk_ids: list[str], embeddings: list[list[float]]
    ) -> None:
        """Persist normalized 384-d local embeddings for pgvector search when available."""
        if not self.available or len(chunk_ids) != len(embeddings):
            return
        rows = [
            (
                "[" + ",".join(f"{value:.8f}" for value in embedding) + "]",
                chunk_id,
                repository_id,
            )
            for chunk_id, embedding in zip(chunk_ids, embeddings)
            if len(embedding) == 384
        ]
        if not rows:
            return
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.executemany(
                    "UPDATE code_chunks SET embedding=%s::vector WHERE id=%s AND repository_id=%s",
                    rows,
                )
                connection.commit()
        except Exception:
            return

    def semantic_search(
        self, repository_id: str, embedding: list[float], limit: int
    ) -> list[tuple[str, float]]:
        if not self.available or len(embedding) != 384:
            return []
        vector = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id, 1 - (embedding <=> %s::vector) AS score FROM code_chunks
                    WHERE repository_id=%s AND embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s""",
                    (vector, repository_id, vector, limit),
                )
                return [
                    (str(row[0]), max(0.0, float(row[1]))) for row in cursor.fetchall()
                ]
        except Exception:
            return []

    def keyword_search(
        self, repository_id: str, query: str, limit: int
    ) -> list[tuple[str, float]]:
        """PostgreSQL full-text channel for hybrid retrieval; the in-memory scorer remains a graceful fallback."""
        if not self.available:
            return []
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id, ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS score
                    FROM code_chunks WHERE repository_id=%s AND search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY score DESC LIMIT %s""",
                    (query, repository_id, query, limit),
                )
                return [
                    (str(row[0]), min(1.0, float(row[1]))) for row in cursor.fetchall()
                ]
        except Exception:
            return []

    def traces(self) -> list[TraceEvent]:
        return self._ephemeral_traces.copy()

    def remember_experience(self, repository_id: str, report: FinalReport) -> None:
        content = report.model_dump(mode="json")
        text = f"Problem: {report.problem}\nRoot cause: {report.root_cause}\nFiles: {', '.join(report.files_changed)}"
        self._ephemeral_experiences.append(text)
        if not self.available:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO memories(id, repository_id, category, content) VALUES(%s,%s,%s,%s)",
                (
                    str(uuid4()),
                    repository_id,
                    "successful_engineering_experience",
                    json.dumps(content),
                ),
            )
            connection.commit()

    def similar_experiences(
        self, repository_id: str, query: str, limit: int = 5
    ) -> list[str]:
        experiences = [
            experience
            for experience in self._ephemeral_experiences
            if any(word in experience.lower() for word in query.lower().split())
        ]
        if not self.available:
            return experiences[:limit]
        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """SELECT content FROM memories WHERE repository_id=%s AND category='successful_engineering_experience'
                    AND content::text ILIKE %s ORDER BY created_at DESC LIMIT %s""",
                    (
                        repository_id,
                        f"%{query.split()[0]}%" if query.split() else "%",
                        limit,
                    ),
                )
                return [json.dumps(row[0]) for row in cursor.fetchall()]
        except Exception:
            return experiences[:limit]

    @property
    def status_message(self) -> str:
        return (
            "PostgreSQL + pgvector connected"
            if self.available
            else f"Ephemeral session only — PostgreSQL unavailable: {self.error}"
        )
