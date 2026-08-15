"""Pydantic contracts. Agent decisions never control workflow as unvalidated text."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL = "external"


class RunStatus(str, Enum):
    INVESTIGATING = "investigating"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    IMPLEMENTING = "implementing"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"
    BLOCKED = "blocked"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt: str = Field(max_length=3000)
    relevance: float = Field(ge=0, le=1)
    source: Literal["semantic", "keyword", "symbol", "dependency", "experience", "test"]
    rationale: str = Field(max_length=1000)

    @field_validator("file")
    @classmethod
    def relative_file(cls, value: str) -> str:
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("evidence file must be a safe relative path")
        return value.replace("\\", "/")


class IssueAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem: str
    expected_behavior: str
    affected_area: str
    likely_components: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class RepositoryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    languages: dict[str, int] = Field(default_factory=dict)
    frameworks: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    test_frameworks: list[str] = Field(default_factory=list)
    important_modules: list[str] = Field(default_factory=list)
    configuration_files: list[str] = Field(default_factory=list)
    database_layer: list[str] = Field(default_factory=list)
    api_layer: list[str] = Field(default_factory=list)
    summary: str = ""


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    rewritten_queries: list[str] = Field(min_length=1, max_length=12)
    symbols: list[str] = Field(default_factory=list)
    file_hints: list[str] = Field(default_factory=list)
    rationale: str


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["open", "supported", "rejected"] = "open"
    next_validation: str


class HypothesisSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=5)


class ImplementationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    root_cause: str
    files_to_modify: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    tests_required: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rollback_strategy: str
    evidence: list[Evidence] = Field(default_factory=list)


class CodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    description: str
    unified_diff: str
    tests_added: list[str] = Field(default_factory=list)
    rationale: str

    @field_validator("file")
    @classmethod
    def safe_target(cls, value: str) -> str:
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("patch target must be a safe relative path")
        return value.replace("\\", "/")


class CodeChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[CodeChange] = Field(default_factory=list, max_length=8)


class TestStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    existing_test_files: list[str] = Field(default_factory=list)
    target_test_files: list[str] = Field(default_factory=list)
    tests_to_add: list[str] = Field(default_factory=list)
    rationale: str


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    passed: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = Field(ge=0)
    summary: str = ""


class DebuggingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_type: str
    root_cause_hypothesis: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    recommended_action: str


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    solves_original_problem: bool
    regression_risks: list[str] = Field(default_factory=list)
    security_findings: list[str] = Field(default_factory=list)
    test_assessment: str
    complexity_assessment: str
    confidence: float = Field(ge=0, le=1)


class CodeChunk(BaseModel):
    """A semantic unit of code, never an arbitrary fixed-size segment by default."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    repository_id: str = ""
    file: str
    symbol: str
    kind: Literal[
        "function", "class", "method", "interface", "module", "route", "test", "config"
    ]
    language: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
    imports: list[str] = Field(default_factory=list)
    exports: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: CodeChunk
    score: float = Field(ge=0, le=1)
    channels: list[Literal["semantic", "keyword", "symbol", "dependency", "experience"]]
    explanation: str


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    action: str
    summary: str
    tool: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_files: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    problem: str
    root_cause: str
    evidence: list[Evidence] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    plan: ImplementationPlan | None = None
    files_changed: list[str] = Field(default_factory=list)
    tests: list[TestResult] = Field(default_factory=list)
    review: ReviewResult | None = None
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    iterations: int = Field(ge=0)
    next_steps: list[str] = Field(default_factory=list)
