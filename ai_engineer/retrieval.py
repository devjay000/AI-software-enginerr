"""Agentic hybrid retrieval and context engineering for source code."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .config import Settings
from .models import CodeChunk, Evidence, RetrievalHit, SearchPlan
from .security import untrusted_repository_context

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def terms(value: str) -> set[str]:
    return {token.lower() for token in TOKEN.findall(value) if len(token) > 1}


@dataclass
class ContextBundle:
    prompt_context: str
    evidence: list[Evidence]
    estimated_tokens: int
    omitted_chunks: int


class HybridRetriever:
    """Combines semantic, keyword, symbol, and dependency signals before local reranking."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        dependencies: dict[str, list[str]],
        settings: Settings,
        memory: Any | None = None,
        repository_id: str | None = None,
    ):
        self.chunks = chunks
        self.dependencies = dependencies
        self.settings = settings
        self._embedding_model = None
        self._chunk_embeddings: list[list[float]] | None = None
        self._reranker = None
        self.memory = memory
        self.repository_id = repository_id

    def rewrite_query(self, issue: str, components: Iterable[str] = ()) -> SearchPlan:
        issue_terms = list(terms(issue))
        candidates = [issue]
        aliases = {
            "login": ["authentication", "session", "token"],
            "logout": ["authentication", "session", "token persistence"],
            "refresh": ["initialization", "persistence", "token refresh"],
            "slow": ["performance", "cache", "query"],
            "error": ["exception", "failure", "stack trace"],
            "database": ["repository", "query", "migration"],
        }
        for word in issue_terms:
            candidates.extend(aliases.get(word, []))
        candidates.extend(components)
        # Stable de-duplication makes retrieval traceable and testable.
        queries = list(
            dict.fromkeys(item.strip() for item in candidates if item.strip())
        )[:10]
        symbols = [
            token
            for token in issue_terms
            if any(token in chunk.symbol.lower() for chunk in self.chunks)
        ]
        return SearchPlan(
            original_query=issue,
            rewritten_queries=queries,
            symbols=symbols,
            file_hints=[],
            rationale="Issue terms were expanded into implementation and domain vocabulary.",
        )

    def search(self, plan: SearchPlan, top_k: int | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.settings.retrieval_top_k
        scores: dict[str, float] = defaultdict(float)
        channels: dict[str, set[str]] = defaultdict(set)
        query = " ".join(plan.rewritten_queries)
        query_terms = terms(query)
        for chunk in self.chunks:
            key = chunk.id
            keyword = self._keyword_score(query_terms, chunk)
            if keyword:
                scores[key] += keyword * 0.30
                channels[key].add("keyword")
            symbol = self._symbol_score(query_terms, chunk)
            if symbol:
                scores[key] += symbol * 0.25
                channels[key].add("symbol")
            dependency = self._dependency_score(plan.symbols, chunk)
            if dependency:
                scores[key] += dependency * 0.12
                channels[key].add("dependency")
        # When PostgreSQL is configured, its full-text index is a first-class hybrid-retrieval channel.
        if self.memory and self.repository_id:
            for chunk_id, score in self.memory.keyword_search(
                self.repository_id, query, max(top_k * 4, 20)
            ):
                scores[chunk_id] += score * 0.30
                channels[chunk_id].add("keyword")
        for chunk, score in self._semantic_scores(query):
            if score:
                scores[chunk.id] += score * 0.40
                channels[chunk.id].add("semantic")
        ranked = sorted(
            (chunk for chunk in self.chunks if chunk.id in scores),
            key=lambda chunk: scores[chunk.id],
            reverse=True,
        )[: max(top_k * 4, 20)]
        ranked = self._rerank(query, ranked)[:top_k]
        return [
            RetrievalHit(
                chunk=chunk,
                score=min(1.0, round(scores[chunk.id], 4)),
                channels=sorted(channels[chunk.id]),
                explanation=self._explain(channels[chunk.id], chunk),
            )
            for chunk in ranked
        ]

    @staticmethod
    def _keyword_score(query_terms: set[str], chunk: CodeChunk) -> float:
        document = terms(f"{chunk.file} {chunk.content}")
        if not document or not query_terms:
            return 0.0
        return len(query_terms & document) / math.sqrt(len(query_terms) * len(document))

    @staticmethod
    def _symbol_score(query_terms: set[str], chunk: CodeChunk) -> float:
        symbol_terms = terms(chunk.symbol)
        return len(query_terms & symbol_terms) / max(1, len(symbol_terms))

    def _dependency_score(self, symbols: list[str], chunk: CodeChunk) -> float:
        if not symbols:
            return 0.0
        imports = " ".join(self.dependencies.get(chunk.file, [])).lower()
        return min(
            1.0, sum(symbol.lower() in imports for symbol in symbols) / len(symbols)
        )

    def _semantic_scores(self, query: str) -> list[tuple[CodeChunk, float]]:
        """Use a local sentence-transformer when available; lexical cosine remains offline-safe."""
        try:
            self._ensure_embeddings()
            assert (
                self._embedding_model is not None and self._chunk_embeddings is not None
            )
            query_embedding = self._embedding_model.encode(
                [query], normalize_embeddings=True
            )[0]
            if self.memory and self.repository_id:
                persistent = dict(
                    self.memory.semantic_search(
                        self.repository_id,
                        query_embedding.tolist(),
                        max(self.settings.retrieval_top_k * 4, 20),
                    )
                )
                if not persistent:
                    self.memory.save_embeddings(
                        self.repository_id,
                        [chunk.id for chunk in self.chunks],
                        self._chunk_embeddings,
                    )
                    persistent = dict(
                        self.memory.semantic_search(
                            self.repository_id,
                            query_embedding.tolist(),
                            max(self.settings.retrieval_top_k * 4, 20),
                        )
                    )
                if persistent:
                    return [
                        (chunk, persistent.get(chunk.id, 0.0)) for chunk in self.chunks
                    ]
            return [
                (
                    chunk,
                    max(
                        0.0, float(sum(a * b for a, b in zip(query_embedding, vector)))
                    ),
                )
                for chunk, vector in zip(self.chunks, self._chunk_embeddings)
            ]
        except Exception:
            query_counts = Counter(terms(query))
            return [
                (
                    chunk,
                    self._lexical_cosine(query_counts, Counter(terms(chunk.content))),
                )
                for chunk in self.chunks
            ]

    def _ensure_embeddings(self) -> None:
        if self._embedding_model is not None:
            return
        if self.settings.local_models_only and not self._model_is_cached(
            self.settings.embedding_model
        ):
            raise RuntimeError("embedding model is not cached locally")
        from sentence_transformers import SentenceTransformer

        self._embedding_model = SentenceTransformer(
            self.settings.embedding_model,
            local_files_only=self.settings.local_models_only,
        )
        self._chunk_embeddings = self._embedding_model.encode(
            [f"{chunk.symbol}\n{chunk.content}" for chunk in self.chunks],
            normalize_embeddings=True,
        ).tolist()

    @staticmethod
    def _model_is_cached(model_name: str) -> bool:
        """Avoid a slow transformer import/download attempt when local-only mode has no cached model."""
        candidate = os.path.expanduser(model_name)
        if os.path.isdir(candidate):
            return True
        hf_home = os.getenv("HF_HOME")
        cache_root = (
            os.path.join(hf_home, "hub")
            if hf_home
            else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        )
        model_dir = "models--" + model_name.replace("/", "--")
        return os.path.isdir(os.path.join(cache_root, model_dir, "snapshots"))

    @staticmethod
    def _lexical_cosine(first: Counter[str], second: Counter[str]) -> float:
        if not first or not second:
            return 0.0
        dot = sum(first[token] * second.get(token, 0) for token in first)
        magnitude = math.sqrt(
            sum(value * value for value in first.values())
        ) * math.sqrt(sum(value * value for value in second.values()))
        return dot / magnitude if magnitude else 0.0

    def _rerank(self, query: str, candidates: list[CodeChunk]) -> list[CodeChunk]:
        if len(candidates) < 2:
            return candidates
        try:
            if self._reranker is None:
                if self.settings.local_models_only and not self._model_is_cached(
                    self.settings.reranker_model
                ):
                    return candidates
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.settings.reranker_model,
                    local_files_only=self.settings.local_models_only,
                )
            rankings = self._reranker.predict(
                [(query, candidate.content) for candidate in candidates]
            )
            return [
                candidate
                for _, candidate in sorted(
                    zip(rankings, candidates), key=lambda item: item[0], reverse=True
                )
            ]
        except Exception:
            return candidates

    @staticmethod
    def _explain(channels: set[str], chunk: CodeChunk) -> str:
        return (
            f"{chunk.symbol} in {chunk.file} matched via {', '.join(sorted(channels))}."
        )

    def references(self, symbol: str, top_k: int = 20) -> list[CodeChunk]:
        expression = re.compile(rf"\b{re.escape(symbol)}\b")
        return [chunk for chunk in self.chunks if expression.search(chunk.content)][
            :top_k
        ]

    def evidence(self, hits: list[RetrievalHit]) -> list[Evidence]:
        return [
            Evidence(
                file=hit.chunk.file,
                start_line=hit.chunk.start_line,
                end_line=hit.chunk.end_line,
                excerpt=hit.chunk.content[:1500],
                relevance=hit.score,
                source=hit.channels[0] if hit.channels else "keyword",
                rationale=hit.explanation,
            )
            for hit in hits
        ]


class ContextBuilder:
    """Token-budgeted, deduplicated, explicitly-untrusted code context."""

    def __init__(self, budget: int):
        self.budget = budget

    def build(
        self,
        issue: str,
        repository_summary: str,
        hits: list[RetrievalHit],
        memories: list[str] | None = None,
    ) -> ContextBundle:
        header = f"ISSUE:\n{issue}\n\nREPOSITORY SUMMARY:\n{repository_summary}\n"
        sections = [header]
        evidence: list[Evidence] = []
        used = self._estimate_tokens(header)
        seen_hashes: set[str] = set()
        for hit in hits:
            fingerprint = str(hit.chunk.metadata.get("sha256", hash(hit.chunk.content)))
            if fingerprint in seen_hashes:
                continue
            rendered = untrusted_repository_context(
                f"{hit.chunk.file}:{hit.chunk.start_line}-{hit.chunk.end_line} ({hit.chunk.symbol})\n{hit.chunk.content}",
                hit.chunk.file,
            )
            cost = self._estimate_tokens(rendered)
            if used + cost > self.budget:
                continue
            seen_hashes.add(fingerprint)
            sections.append(rendered)
            used += cost
            evidence.append(
                Evidence(
                    file=hit.chunk.file,
                    start_line=hit.chunk.start_line,
                    end_line=hit.chunk.end_line,
                    excerpt=hit.chunk.content[:1500],
                    relevance=hit.score,
                    source=hit.channels[0] if hit.channels else "keyword",
                    rationale=hit.explanation,
                )
            )
        if memories:
            memory = "\n".join(memories[:5])
            if used + self._estimate_tokens(memory) <= self.budget:
                sections.append(
                    "<historical_engineering_experience>\n"
                    + memory
                    + "\n</historical_engineering_experience>"
                )
        return ContextBundle(
            prompt_context="\n\n".join(sections),
            evidence=evidence,
            estimated_tokens=used,
            omitted_chunks=max(0, len(hits) - len(evidence)),
        )

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        return max(1, len(value) // 4)
