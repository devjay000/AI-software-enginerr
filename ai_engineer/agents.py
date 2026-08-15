"""Specialized, evidence-grounded agent nodes used by the LangGraph supervisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings
from .indexing import CodeIndexer, IndexResult
from .llm import LLMProvider
from .models import (
    CodeChange,
    CodeChanges,
    DebuggingResult,
    Evidence,
    FinalReport,
    Hypothesis,
    HypothesisSet,
    ImplementationPlan,
    IssueAnalysis,
    ReviewResult,
    RunStatus,
    TestResult,
    TestStrategy,
    TraceEvent,
)
from .retrieval import ContextBuilder, ContextBundle, HybridRetriever
from .storage import MemoryStore
from .tools import ToolPermissionError, ToolRegistry


class AgentRuntime:
    """Orchestrates agents while keeping all side effects behind ToolRegistry permissions."""

    def __init__(self, settings: Settings, memory: MemoryStore | None = None):
        self.settings = settings
        self.memory = memory or MemoryStore(settings.database_url)
        self.llm = LLMProvider(settings)

    def _event(
        self,
        state: dict[str, Any],
        agent: str,
        action: str,
        summary: str,
        tool: str | None = None,
        confidence: float | None = None,
        evidence: list[Evidence] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            agent=agent,
            action=action,
            summary=summary,
            tool=tool,
            confidence=confidence,
            evidence_files=sorted({item.file for item in evidence or []}),
        )
        self.memory.append_trace(event, state.get("run_id"))
        return event

    @staticmethod
    def _append_trace(state: dict[str, Any], event: TraceEvent) -> list[TraceEvent]:
        return [*state.get("trace", []), event]

    def understand_issue(self, state: dict[str, Any]) -> dict[str, Any]:
        issue = state["issue"].strip()
        words = [word.strip(".,:;!?()") for word in issue.split()]
        components = [word for word in words if len(word) > 4][:8]
        fallback = IssueAnalysis(
            problem=issue,
            expected_behavior="The reported behavior should be corrected without regressions.",
            affected_area="unknown — establish through repository retrieval",
            likely_components=components,
            acceptance_criteria=[
                "A focused regression test passes.",
                "Existing test suite remains green.",
            ],
            search_terms=components,
            ambiguity_notes=(
                [] if len(words) > 4 else ["Issue lacks implementation detail."]
            ),
        )
        analysis = self.llm.structured(
            IssueAnalysis,
            "Convert this issue into a factual structured investigation brief. Do not guess source files.",
            issue,
            fallback,
        )
        event = self._event(
            state,
            "Issue Understanding Agent",
            "structured_issue_analysis",
            "Converted the issue into acceptance criteria and retrieval terms.",
        )
        return {
            "issue_analysis": analysis,
            "status": RunStatus.INVESTIGATING,
            "trace": self._append_trace(state, event),
        }

    def understand_repository(self, state: dict[str, Any]) -> dict[str, Any]:
        index = CodeIndexer(self.settings).index(state["repository_path"])
        repository_id = state.get("repository_id") or str(uuid4())
        self.memory.save_index(
            repository_id,
            index.analysis.root,
            index.analysis.model_dump(mode="json"),
            index.chunks,
            index.dependencies,
        )
        event = self._event(
            state,
            "Repository Intelligence Agent",
            "ast_repository_analysis",
            f"Indexed {len(index.chunks)} structural chunks across {sum(index.analysis.languages.values())} safe files.",
            tool="get_repository_summary",
        )
        return {
            "index": index,
            "repository_id": repository_id,
            "trace": self._append_trace(state, event),
        }

    def retrieve_evidence(self, state: dict[str, Any]) -> dict[str, Any]:
        index: IndexResult = state["index"]
        issue: IssueAnalysis = state["issue_analysis"]
        retriever = state.get("retriever") or HybridRetriever(
            index.chunks,
            index.dependencies,
            self.settings,
            memory=self.memory,
            repository_id=state["repository_id"],
        )
        plan = retriever.rewrite_query(
            issue.problem, issue.likely_components + issue.search_terms
        )
        tools = ToolRegistry(
            state["repository_path"],
            self.settings,
            approved=bool(state.get("approved")),
        )
        # Tool selection is deliberate: inspect manifests once, and look up symbols only when query rewriting found them.
        tool_summaries: list[str] = []
        try:
            manifests = tools.invoke("get_repository_summary")
            tool_summaries.append(f"read {len(manifests)} manifest/readme files")
            if plan.symbols:
                refs = tools.invoke("search_symbol", symbol=plan.symbols[0], limit=8)
                tool_summaries.append(
                    f"found {len(refs)} direct usages of {plan.symbols[0]}"
                )
        except (ToolPermissionError, OSError, ValueError) as exc:
            tool_summaries.append(f"safe tool skipped: {exc}")
        hits = retriever.search(plan)
        experiences = self.memory.similar_experiences(
            state["repository_id"], issue.problem
        )
        context = ContextBuilder(self.settings.context_token_budget).build(
            issue.problem, index.analysis.summary, hits, experiences
        )
        event = self._event(
            state,
            "Investigation Agent",
            "hybrid_code_retrieval",
            f"Ran query rewriting and hybrid retrieval; selected {len(hits)} chunks ({'; '.join(tool_summaries)}).",
            tool="semantic_search",
            confidence=max((hit.score for hit in hits), default=0.0),
            evidence=context.evidence,
        )
        return {
            "retriever": retriever,
            "search_plan": plan,
            "retrieval_hits": hits,
            "context": context,
            "evidence": context.evidence,
            "trace": self._append_trace(state, event),
        }

    def generate_hypotheses(self, state: dict[str, Any]) -> dict[str, Any]:
        evidence: list[Evidence] = state.get("evidence", [])
        primary = evidence[0] if evidence else None
        secondary = evidence[1] if len(evidence) > 1 else primary
        fallback = HypothesisSet(
            hypotheses=[
                Hypothesis(
                    id="H1",
                    statement=(
                        f"The defect is likely in {primary.file}:{primary.start_line}, where the most relevant related code resides."
                        if primary
                        else "Insufficient indexed evidence to identify a root cause."
                    ),
                    confidence=primary.relevance if primary else 0.15,
                    evidence=[primary] if primary else [],
                    status="open",
                    next_validation="Inspect direct references and relevant tests before modifying code.",
                ),
                Hypothesis(
                    id="H2",
                    statement="A related dependency, initialization path, or missing regression case may cause the observed behavior.",
                    confidence=(secondary.relevance * 0.72 if secondary else 0.1),
                    evidence=[secondary] if secondary else [],
                    status="open",
                    next_validation="Compare call sites and test coverage with the primary candidate.",
                ),
            ]
        )
        context: ContextBundle = state["context"]
        result = self.llm.structured(
            HypothesisSet,
            "Generate 2–4 competing root-cause hypotheses. Every hypothesis must cite only evidence provided. Do not propose a fix yet.",
            context.prompt_context,
            fallback,
        )
        hypotheses = self._ground_hypotheses(
            result.hypotheses, evidence, fallback.hypotheses
        )
        event = self._event(
            state,
            "Hypothesis Engine",
            "generate_competing_hypotheses",
            f"Generated and evidence-grounded {len(hypotheses)} competing root-cause hypotheses.",
            confidence=max(item.confidence for item in hypotheses),
            evidence=evidence,
        )
        return {"hypotheses": hypotheses, "trace": self._append_trace(state, event)}

    @staticmethod
    def _ground_hypotheses(
        candidates: list[Hypothesis],
        allowed: list[Evidence],
        fallback: list[Hypothesis],
    ) -> list[Hypothesis]:
        allowed_keys = {(item.file, item.start_line, item.end_line) for item in allowed}
        grounded: list[Hypothesis] = []
        for item in candidates:
            evidence = [
                proof
                for proof in item.evidence
                if (proof.file, proof.start_line, proof.end_line) in allowed_keys
            ]
            if evidence:
                grounded.append(item.model_copy(update={"evidence": evidence}))
        return grounded or fallback

    def evaluate_hypotheses(self, state: dict[str, Any]) -> dict[str, Any]:
        tools = ToolRegistry(
            state["repository_path"],
            self.settings,
            approved=bool(state.get("approved")),
        )
        evaluated: list[Hypothesis] = []
        for hypothesis in state["hypotheses"]:
            terms = [
                word
                for word in hypothesis.statement.split()
                if word.isidentifier() and len(word) > 3
            ]
            added_evidence: list[Evidence] = []
            if terms:
                symbol = terms[0]
                try:
                    usages = tools.invoke("find_references", symbol=symbol, limit=5)
                    for usage in usages[:2]:
                        added_evidence.append(
                            Evidence(
                                file=usage["file"],
                                start_line=usage["line"],
                                end_line=usage["line"],
                                excerpt=usage["text"],
                                relevance=0.4,
                                source="symbol",
                                rationale=f"Reference to validation term {symbol}.",
                            )
                        )
                except (ToolPermissionError, OSError, ValueError):
                    pass
            support = min(1.0, hypothesis.confidence + 0.04 * len(added_evidence))
            evaluated.append(
                hypothesis.model_copy(
                    update={
                        "confidence": support,
                        "evidence": [*hypothesis.evidence, *added_evidence],
                        "status": "supported" if hypothesis.evidence else "open",
                    }
                )
            )
        evaluated.sort(key=lambda item: item.confidence, reverse=True)
        evidence = [proof for item in evaluated for proof in item.evidence]
        event = self._event(
            state,
            "Evidence Validation Agent",
            "validate_hypotheses",
            "Validated hypothesis evidence with targeted symbol-reference lookups and ranked candidates.",
            tool="find_references",
            confidence=evaluated[0].confidence,
            evidence=evidence,
        )
        return {
            "hypotheses": evaluated,
            "evidence": evidence,
            "trace": self._append_trace(state, event),
        }

    def create_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        hypotheses: list[Hypothesis] = state["hypotheses"]
        winner = hypotheses[0]
        files = list(dict.fromkeys(proof.file for proof in winner.evidence))[:4]
        fallback = ImplementationPlan(
            objective=state["issue_analysis"].problem,
            root_cause=winner.statement,
            files_to_modify=files,
            changes=[
                f"Make the smallest behavior-preserving change in {path}."
                for path in files
            ],
            tests_required=[
                "Add or update a focused regression test.",
                "Run the repository's existing test suite in Docker.",
            ],
            risks=["Evidence is limited; do not modify unrelated modules."],
            rollback_strategy="Revert the approved unified diff with Git if validation or review identifies a regression.",
            evidence=winner.evidence,
        )
        plan = self.llm.structured(
            ImplementationPlan,
            "Create a minimal implementation plan from validated evidence. Use only evidence file paths. Include tests, risks, and rollback.",
            state["context"].prompt_context
            + "\n\nTOP HYPOTHESIS:\n"
            + winner.model_dump_json(),
            fallback,
        )
        valid_files = {proof.file for proof in state.get("evidence", [])}
        if any(path not in valid_files for path in plan.files_to_modify):
            plan = fallback
        event = self._event(
            state,
            "Planning Agent",
            "create_validated_plan",
            f"Created a minimal plan targeting {len(plan.files_to_modify)} evidence-backed file(s).",
            confidence=winner.confidence,
            evidence=plan.evidence,
        )
        return {
            "plan": plan,
            "status": RunStatus.WAITING_FOR_APPROVAL,
            "trace": self._append_trace(state, event),
        }

    def human_approval(self, state: dict[str, Any]) -> dict[str, Any]:
        approved = bool(state.get("approved"))
        event = self._event(
            state,
            "Human Approval Gate",
            "approval_check",
            (
                "Human approved the proposed write and execution actions."
                if approved
                else "Plan ready; write, execution, and external permissions remain blocked pending human approval."
            ),
        )
        return {
            "status": (
                RunStatus.IMPLEMENTING if approved else RunStatus.WAITING_FOR_APPROVAL
            ),
            "trace": self._append_trace(state, event),
        }

    def inspect_test_strategy(self, state: dict[str, Any]) -> dict[str, Any]:
        """Dedicated test-generation agent: learn the repository's test conventions before code is written."""
        index: IndexResult = state["index"]
        existing = sorted(
            {chunk.file for chunk in index.chunks if chunk.kind == "test"}
        )
        framework = ", ".join(index.analysis.test_frameworks) or "not detected"
        fallback = TestStrategy(
            framework=framework,
            existing_test_files=existing[:10],
            target_test_files=existing[:2],
            tests_to_add=state["plan"].tests_required,
            rationale="Use existing test conventions and add the smallest focused regression case.",
        )
        strategy = self.llm.structured(
            TestStrategy,
            "Inspect the repository test strategy and select only existing test files for focused regression coverage. Do not write tests yet.",
            state["context"].prompt_context
            + "\n\nPLAN:\n"
            + state["plan"].model_dump_json(),
            fallback,
        )
        valid_files = set(existing)
        if any(path not in valid_files for path in strategy.target_test_files):
            strategy = fallback
        event = self._event(
            state,
            "Test Generation Agent",
            "inspect_test_strategy",
            f"Detected {framework}; selected {len(strategy.target_test_files)} existing test target(s).",
            tool="get_tests",
        )
        return {"test_strategy": strategy, "trace": self._append_trace(state, event)}

    def implement(self, state: dict[str, Any]) -> dict[str, Any]:
        plan: ImplementationPlan = state["plan"]
        fallback = CodeChanges(changes=[])
        changes = self.llm.structured(
            CodeChanges,
            "Generate the smallest possible standard git unified diff that implements the approved plan. Only modify listed files. Return no change if evidence is insufficient. Include focused tests where appropriate.",
            state["context"].prompt_context
            + "\n\nAPPROVED PLAN:\n"
            + plan.model_dump_json()
            + "\n\nTEST STRATEGY:\n"
            + state.get(
                "test_strategy", TestStrategy(framework="unknown", rationale="")
            ).model_dump_json(),
            fallback,
        )
        valid_paths = set(plan.files_to_modify) | set(
            state.get(
                "test_strategy", TestStrategy(framework="unknown", rationale="")
            ).target_test_files
        )
        safe_changes = [
            change
            for change in changes.changes
            if change.file in valid_paths and change.unified_diff.strip()
        ]
        tools = ToolRegistry(state["repository_path"], self.settings, approved=True)
        applied: list[CodeChange] = []
        errors: list[str] = []
        for change in safe_changes:
            try:
                tools.invoke("apply_patch", diff=change.unified_diff)
                applied.append(change)
            except (ToolPermissionError, ValueError, RuntimeError, OSError) as exc:
                errors.append(str(exc))
        status = (
            RunStatus.IMPLEMENTING if applied else RunStatus.HUMAN_INTERVENTION_REQUIRED
        )
        summary = (
            f"Validated and applied {len(applied)} patch(es)."
            if applied
            else "No safe patch was applied; a human must supply or revise the implementation."
        )
        event = self._event(
            state,
            "Coding Agent",
            "generate_and_apply_patch",
            summary,
            tool="apply_patch",
            evidence=plan.evidence,
        )
        return {
            "code_changes": applied,
            "patch_errors": errors,
            "status": status,
            "trace": self._append_trace(state, event),
        }

    def run_tests(self, state: dict[str, Any]) -> dict[str, Any]:
        tools = ToolRegistry(state["repository_path"], self.settings, approved=True)
        result: TestResult = tools.invoke(
            "run_tests", repository=Path(state["repository_path"])
        )
        event = self._event(
            state,
            "Testing Agent",
            "docker_test_execution",
            result.summary,
            tool="run_tests",
            confidence=0.95 if result.passed else 0.2,
        )
        return {
            "test_results": [*state.get("test_results", []), result],
            "trace": self._append_trace(state, event),
        }

    def debug(self, state: dict[str, Any]) -> dict[str, Any]:
        test: TestResult = state["test_results"][-1]
        iteration = int(state.get("iteration", 0)) + 1
        fallback = DebuggingResult(
            failure_type="test_failure",
            root_cause_hypothesis=test.stderr[-500:]
            or test.stdout[-500:]
            or test.summary,
            evidence=state.get("evidence", [])[:3],
            confidence=0.35,
            recommended_action="Retrieve the failed symbols and test locations, then make one minimal correction.",
        )
        debug = self.llm.structured(
            DebuggingResult,
            "Analyze this Docker test failure using only the evidence. Recommend the next retrieval/debug action, not unbounded changes.",
            state["context"].prompt_context
            + "\n\nTEST RESULT:\n"
            + test.model_dump_json(),
            fallback,
        )
        terminal = (
            iteration >= int(state.get("max_iterations", self.settings.max_iterations))
            or test.exit_code == 127
        )
        status = (
            RunStatus.HUMAN_INTERVENTION_REQUIRED if terminal else RunStatus.DEBUGGING
        )
        event = self._event(
            state,
            "Debugging Agent",
            "failure_analysis_and_self_correction",
            f"Iteration {iteration}: {debug.recommended_action}",
            confidence=debug.confidence,
            evidence=debug.evidence,
        )
        return {
            "iteration": iteration,
            "debugging": debug,
            "status": status,
            "trace": self._append_trace(state, event),
        }

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        tests: list[TestResult] = state.get("test_results", [])
        passed = bool(tests and tests[-1].passed)
        fallback = ReviewResult(
            approved=passed and bool(state.get("code_changes")),
            solves_original_problem=passed,
            regression_risks=[] if passed else ["Test suite did not pass."],
            security_findings=[],
            test_assessment=(
                "Focused and existing tests passed in Docker."
                if passed
                else "Tests need human review."
            ),
            complexity_assessment="Patch is scoped to evidence-backed files.",
            confidence=0.9 if passed else 0.2,
        )
        review = self.llm.structured(
            ReviewResult,
            "Independently review whether the patch solves the issue, is secure, and is appropriately scoped. Do not approve without passing tests.",
            state["context"].prompt_context
            + "\n\nPLAN:\n"
            + state["plan"].model_dump_json()
            + "\n\nTESTS:\n"
            + str([result.model_dump() for result in tests]),
            fallback,
        )
        if not passed:
            review = fallback
        status = (
            RunStatus.COMPLETED
            if review.approved
            else RunStatus.HUMAN_INTERVENTION_REQUIRED
        )
        event = self._event(
            state,
            "Review Agent",
            "independent_solution_review",
            (
                "Patch review approved."
                if review.approved
                else "Patch requires human review."
            ),
            confidence=review.confidence,
        )
        return {
            "review": review,
            "status": status,
            "trace": self._append_trace(state, event),
        }

    def final_report(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan")
        hypotheses = state.get("hypotheses", [])
        review = state.get("review")
        confidence = (
            review.confidence
            if review
            else (hypotheses[0].confidence if hypotheses else 0.0)
        )
        report = FinalReport(
            status=state.get("status", RunStatus.BLOCKED),
            problem=state["issue"],
            root_cause=(
                plan.root_cause if plan else "No validated root cause was established."
            ),
            evidence=state.get("evidence", []),
            hypotheses=hypotheses,
            plan=plan,
            files_changed=[change.file for change in state.get("code_changes", [])],
            tests=state.get("test_results", []),
            review=review,
            risks=(plan.risks if plan else []) + state.get("patch_errors", []),
            confidence=confidence,
            iterations=int(state.get("iteration", 0)),
            next_steps=(
                []
                if state.get("status") == RunStatus.COMPLETED
                else [
                    "Review the evidence and plan, then refine or implement with human oversight."
                ]
            ),
        )
        if report.status == RunStatus.COMPLETED:
            self.memory.remember_experience(state["repository_id"], report)
        event = self._event(
            state,
            "Report Agent",
            "generate_final_report",
            f"Final status: {report.status.value}.",
            confidence=report.confidence,
            evidence=report.evidence,
        )
        return {"report": report, "trace": self._append_trace(state, event)}
