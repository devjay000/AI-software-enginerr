"""Local, transparent RAG and agent evaluation — no external observability SaaS."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from .models import FinalReport
from .retrieval import HybridRetriever


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    issue: str
    expected_relevant_files: list[str] = Field(min_length=1)
    category: str = "general"


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    evaluated_tasks: int = Field(ge=0)


class AgentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_completion_rate: float = Field(ge=0, le=1)
    average_iterations: float = Field(ge=0)
    test_pass_rate: float = Field(ge=0, le=1)
    human_intervention_rate: float = Field(ge=0, le=1)
    incorrect_modification_rate: float = Field(ge=0, le=1)
    tool_success_rate: float = Field(ge=0, le=1)


def load_benchmarks(path: str | Path) -> list[BenchmarkTask]:
    source = Path(path)
    if not source.exists():
        return []
    raw = json.loads(source.read_text(encoding="utf-8"))
    return [BenchmarkTask.model_validate(item) for item in raw]


def evaluate_retrieval(
    retriever: HybridRetriever, tasks: list[BenchmarkTask], k: int = 8
) -> RetrievalMetrics:
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for task in tasks:
        plan = retriever.rewrite_query(task.issue)
        paths = [hit.chunk.file for hit in retriever.search(plan, top_k=k)]
        expected = set(task.expected_relevant_files)
        found = set(paths)
        recalls.append(len(expected & found) / len(expected))
        precisions.append(len(expected & found) / max(1, len(paths)))
        first = next(
            (index + 1 for index, path in enumerate(paths) if path in expected), None
        )
        reciprocal_ranks.append(1 / first if first else 0)
    return RetrievalMetrics(
        recall_at_k=mean(recalls) if recalls else 0,
        precision_at_k=mean(precisions) if precisions else 0,
        mrr=mean(reciprocal_ranks) if reciprocal_ranks else 0,
        evaluated_tasks=len(tasks),
    )


def evaluate_agent(
    reports: list[FinalReport], tool_calls: int = 0, successful_tool_calls: int = 0
) -> AgentMetrics:
    if not reports:
        return AgentMetrics(
            task_completion_rate=0,
            average_iterations=0,
            test_pass_rate=0,
            human_intervention_rate=0,
            incorrect_modification_rate=0,
            tool_success_rate=0,
        )
    completed = [report for report in reports if report.status.value == "completed"]
    all_tests = [test for report in reports for test in report.tests]
    incorrect = [
        report
        for report in reports
        if report.files_changed
        and not report.review
        or report.review
        and not report.review.approved
    ]
    intervention = [
        report
        for report in reports
        if report.status.value == "human_intervention_required"
    ]
    return AgentMetrics(
        task_completion_rate=len(completed) / len(reports),
        average_iterations=mean(report.iterations for report in reports),
        test_pass_rate=(
            sum(test.passed for test in all_tests) / len(all_tests) if all_tests else 0
        ),
        human_intervention_rate=len(intervention) / len(reports),
        incorrect_modification_rate=len(incorrect) / len(reports),
        tool_success_rate=successful_tool_calls / tool_calls if tool_calls else 0,
    )
