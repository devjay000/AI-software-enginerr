"""The central stateful LangGraph workflow and its conditional control flow."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .agents import AgentRuntime
from .models import RunStatus


class AgentState(TypedDict, total=False):
    repository_path: str
    repository_id: str
    issue: str
    approved: bool
    max_iterations: int
    run_id: str
    status: RunStatus
    issue_analysis: Any
    index: Any
    retriever: Any
    search_plan: Any
    retrieval_hits: list[Any]
    context: Any
    evidence: list[Any]
    hypotheses: list[Any]
    plan: Any
    test_strategy: Any
    code_changes: list[Any]
    patch_errors: list[str]
    test_results: list[Any]
    debugging: Any
    iteration: int
    review: Any
    report: Any
    trace: list[Any]


def _after_approval(state: AgentState) -> str:
    return "implement" if state.get("approved") else "final_report"


def _after_implementation(state: AgentState) -> str:
    return (
        "run_tests" if state.get("status") == RunStatus.IMPLEMENTING else "final_report"
    )


def _after_tests(state: AgentState) -> str:
    tests = state.get("test_results", [])
    return "review" if tests and tests[-1].passed else "debug"


def _after_debug(state: AgentState) -> str:
    return (
        "final_report"
        if state.get("status") == RunStatus.HUMAN_INTERVENTION_REQUIRED
        else "retrieve_evidence"
    )


def build_graph(runtime: AgentRuntime):
    """Build a fresh graph for a task; its state makes every agent decision auditable."""
    graph = StateGraph(AgentState)
    graph.add_node("understand_issue", runtime.understand_issue)
    graph.add_node("understand_repository", runtime.understand_repository)
    graph.add_node("retrieve_evidence", runtime.retrieve_evidence)
    graph.add_node("generate_hypotheses", runtime.generate_hypotheses)
    graph.add_node("evaluate_hypotheses", runtime.evaluate_hypotheses)
    graph.add_node("create_plan", runtime.create_plan)
    graph.add_node("human_approval", runtime.human_approval)
    graph.add_node("inspect_test_strategy", runtime.inspect_test_strategy)
    graph.add_node("implement", runtime.implement)
    graph.add_node("run_tests", runtime.run_tests)
    graph.add_node("debug", runtime.debug)
    graph.add_node("review", runtime.review)
    graph.add_node("final_report", runtime.final_report)

    graph.add_edge(START, "understand_issue")
    graph.add_edge("understand_issue", "understand_repository")
    graph.add_edge("understand_repository", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_hypotheses")
    graph.add_edge("generate_hypotheses", "evaluate_hypotheses")
    graph.add_edge("evaluate_hypotheses", "create_plan")
    graph.add_edge("create_plan", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        _after_approval,
        {"implement": "inspect_test_strategy", "final_report": "final_report"},
    )
    graph.add_edge("inspect_test_strategy", "implement")
    graph.add_conditional_edges(
        "implement",
        _after_implementation,
        {"run_tests": "run_tests", "final_report": "final_report"},
    )
    graph.add_conditional_edges(
        "run_tests", _after_tests, {"review": "review", "debug": "debug"}
    )
    graph.add_conditional_edges(
        "debug",
        _after_debug,
        {"retrieve_evidence": "retrieve_evidence", "final_report": "final_report"},
    )
    graph.add_edge("review", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile()


def initial_state(
    repository_path: str, issue: str, approved: bool, max_iterations: int
) -> AgentState:
    return AgentState(
        repository_path=repository_path,
        issue=issue,
        approved=approved,
        max_iterations=max_iterations,
        run_id=str(uuid4()),
        iteration=0,
        trace=[],
        evidence=[],
        hypotheses=[],
        code_changes=[],
        test_results=[],
        patch_errors=[],
    )
