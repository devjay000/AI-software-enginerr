"""Streamlit AI engineering workstation for ForgeMind."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from ai_engineer.agents import AgentRuntime
from ai_engineer.config import Settings, get_settings
from ai_engineer.evaluation import evaluate_agent, evaluate_retrieval, load_benchmarks
from ai_engineer.graph import build_graph, initial_state
from ai_engineer.indexing import CodeIndexer
from ai_engineer.llm import LLMProvider
from ai_engineer.models import RunStatus
from ai_engineer.storage import MemoryStore
from ai_engineer.tools import DockerSandbox, ToolRegistry

st.set_page_config(
    page_title="ForgeMind · AI Software Engineer", page_icon="⚒️", layout="wide"
)
st.markdown(
    """
<style>
  .block-container { max-width: 1450px; padding-top: 1.3rem; }
  [data-testid="stMetric"] { background: #111827; border: 1px solid #263244; padding: .7rem; border-radius: .55rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 5px; }
  .stTabs [data-baseweb="tab"] { height: 45px; background: #111827; border-radius: 5px 5px 0 0; }
  code { font-size: .88em; }
</style>
""",
    unsafe_allow_html=True,
)


def workspace_settings(model: str, iterations: int, top_k: int) -> Settings:
    return get_settings().model_copy(
        update={"model": model, "max_iterations": iterations, "retrieval_top_k": top_k}
    )


def clone_github_repository(url: str) -> tuple[bool, str]:
    """Explicitly user-triggered clone restricted to GitHub and an app-owned local cache."""
    parsed = urlparse(url)
    if parsed.hostname not in {"github.com", "www.github.com"} or not parsed.path.strip(
        "/"
    ):
        return False, "Enter a public https://github.com/owner/repository URL."
    repo_name = parsed.path.strip("/").removesuffix(".git").replace("/", "--")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name):
        return False, "Repository URL contains an unsupported path."
    target = (Path.cwd() / ".forgemind" / "repositories" / repo_name).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True, str(target)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[-1200:]
    return True, str(target)


def run_graph(repository: str, issue: str, approved: bool, settings: Settings) -> dict:
    memory = MemoryStore(settings.database_url)
    runtime = AgentRuntime(settings, memory)
    app_graph = build_graph(runtime)
    state = app_graph.invoke(
        initial_state(repository, issue, approved, settings.max_iterations)
    )
    st.session_state.run_state = state
    st.session_state.runtime = runtime
    st.session_state.settings = settings
    return state


def as_rows(items):
    return [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in items
    ]


def read_repository_file(state: dict, relative_path: str) -> str:
    try:
        tools = ToolRegistry(state["repository_path"], st.session_state.settings)
        return tools.invoke("read_file", path=relative_path)
    except Exception as exc:
        return f"Unable to read {relative_path}: {exc}"


def render_investigation(state: dict) -> None:
    st.subheader("Evidence-first investigation")
    issue = state.get("issue_analysis")
    index = state.get("index")
    columns = st.columns(4)
    columns[0].metric(
        "Current status",
        str(state.get("status", "not started")).replace("_", " ").title(),
    )
    columns[1].metric("Evidence chunks", len(state.get("evidence", [])))
    columns[2].metric("Hypotheses", len(state.get("hypotheses", [])))
    columns[3].metric("Iteration", state.get("iteration", 0))
    if issue:
        st.markdown(f"**Issue:** {issue.problem}")
        st.caption(
            f"Expected behavior: {issue.expected_behavior} · Affected area: {issue.affected_area}"
        )
    st.markdown(
        "Issue → Repository understanding → Hybrid retrieval → Hypotheses → Evidence validation → Plan → Approval"
    )
    if index:
        st.info(index.analysis.summary)
        with st.expander("Repository intelligence", expanded=False):
            st.json(index.analysis.model_dump(mode="json"))
            if index.excluded_files:
                st.caption(
                    f"Excluded {len(index.excluded_files)} unsafe, secret-like, binary, ignored, or over-sized file(s)."
                )
    hits = state.get("retrieval_hits", [])
    if hits:
        st.markdown("#### Retrieved evidence")
        for hit in hits:
            with st.expander(
                f"{hit.chunk.file}:{hit.chunk.start_line} · {hit.chunk.symbol} · {hit.score:.2f}"
            ):
                st.caption(f"Channels: {', '.join(hit.channels)} — {hit.explanation}")
                st.code(
                    hit.chunk.content,
                    language=(
                        hit.chunk.language if hit.chunk.language != "text" else None
                    ),
                )
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        st.markdown("#### Ranked hypotheses")
        for hypothesis in hypotheses:
            st.progress(
                hypothesis.confidence,
                text=f"{hypothesis.id} · {hypothesis.status} · confidence {hypothesis.confidence:.0%}",
            )
            st.write(hypothesis.statement)
            st.caption(hypothesis.next_validation)


def render_plan(state: dict, repository: str, issue: str) -> None:
    plan = state.get("plan")
    if not plan:
        st.info("Run an investigation to create an evidence-backed plan.")
        return
    st.subheader("Implementation plan")
    st.json(plan.model_dump(mode="json"))
    st.markdown("#### Evidence supporting the plan")
    for evidence in plan.evidence:
        with st.expander(
            f"{evidence.file}:{evidence.start_line}-{evidence.end_line} · {evidence.relevance:.0%}"
        ):
            st.code(evidence.excerpt)
            st.caption(evidence.rationale)
    waiting = state.get("status") == RunStatus.WAITING_FOR_APPROVAL
    if waiting:
        st.warning(
            "Write, Docker execution, Git operations, and GitHub operations remain blocked until you explicitly approve."
        )
        reviewed = st.checkbox(
            "I reviewed this plan and approve the proposed local patch and Docker test execution.",
            key="approval_checkbox",
        )
        if st.button("Approve and execute", type="primary", disabled=not reviewed):
            with st.spinner(
                "Re-running the stateful workflow with approved write and execution permissions…"
            ):
                run_graph(repository, issue, True, st.session_state.settings)
            st.rerun()


def render_code(state: dict) -> None:
    index = state.get("index")
    if not index:
        st.info("Repository code appears after investigation.")
        return
    st.subheader("Structural code view")
    paths = sorted({chunk.file for chunk in index.chunks})
    selected = (
        st.selectbox("Repository file", paths, key="code_file") if paths else None
    )
    if selected:
        language = next(
            (chunk.language for chunk in index.chunks if chunk.file == selected), "text"
        )
        st.code(
            read_repository_file(state, selected),
            language=language if language != "text" else None,
            line_numbers=True,
        )
    st.markdown("#### Extracted symbols")
    st.dataframe(
        [
            {
                "file": chunk.file,
                "symbol": chunk.symbol,
                "type": chunk.kind,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "language": chunk.language,
            }
            for chunk in index.chunks
        ],
        use_container_width=True,
        hide_index=True,
        height=330,
    )


def render_diff(state: dict) -> None:
    st.subheader("Approved patch / Git diff")
    changes = state.get("code_changes", [])
    if changes:
        for change in changes:
            st.markdown(f"**{change.file}** — {change.description}")
            st.code(change.unified_diff, language="diff")
            st.caption(change.rationale)
    else:
        st.info(
            "No patch has been applied. A plan never grants write access by itself."
        )
    if state.get("repository_path"):
        try:
            current = ToolRegistry(
                state["repository_path"], st.session_state.settings
            ).invoke("get_git_diff")
            if current:
                with st.expander("Current repository Git diff"):
                    st.code(current, language="diff")
        except Exception as exc:
            st.caption(f"Git diff unavailable: {exc}")


def render_tests(state: dict) -> None:
    st.subheader("Docker sandbox test results")
    strategy = state.get("test_strategy")
    if strategy:
        with st.expander(
            "Test Generation Agent strategy",
            expanded=not bool(state.get("test_results")),
        ):
            st.json(strategy.model_dump(mode="json"))
    tests = state.get("test_results", [])
    if not tests:
        st.info(
            "Tests run only after human-approved code changes. The agent never executes generated code on the host."
        )
        return
    for result in tests:
        (st.success if result.passed else st.error)(
            f"{'Passed' if result.passed else 'Failed'} · exit {result.exit_code} · {result.duration_seconds}s"
        )
        st.caption(result.command)
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")
    debug = state.get("debugging")
    if debug:
        st.markdown("#### Latest debugging assessment")
        st.json(debug.model_dump(mode="json"))


def render_trace(state: dict) -> None:
    st.subheader("Safe agent trace")
    st.caption(
        "This is a concise audit trail of actions, evidence, and decisions; it intentionally never exposes hidden chain-of-thought."
    )
    trace = state.get("trace", [])
    if trace:
        st.dataframe(as_rows(trace), use_container_width=True, hide_index=True)
    else:
        st.info("The trace will appear after a run.")


def render_memory(state: dict) -> None:
    st.subheader("Repository, task, and experience memory")
    runtime = st.session_state.get("runtime")
    if not runtime:
        st.info("Memory activates with the first investigation.")
        return
    st.info(runtime.memory.status_message)
    if state.get("repository_id") and state.get("issue"):
        experiences = runtime.memory.similar_experiences(
            state["repository_id"], state["issue"]
        )
        if experiences:
            st.markdown("#### Retrieved engineering experiences")
            for experience in experiences:
                st.code(experience[:3000], language="text")
        else:
            st.caption("No relevant successful experience has been recorded yet.")
    st.markdown("#### Current-run memory")
    st.write(
        f"Trace events: {len(runtime.memory.traces())} · Hypotheses: {len(state.get('hypotheses', []))} · Test attempts: {len(state.get('test_results', []))}"
    )


def render_evaluation(state: dict) -> None:
    st.subheader("Local evaluation")
    retriever = state.get("retriever")
    benchmarks = load_benchmarks(Path("evaluation") / "tasks.json")
    if retriever and st.button("Run retrieval benchmark", key="evaluate_retrieval"):
        with st.spinner("Evaluating hybrid retrieval against local ground truth…"):
            st.session_state.retrieval_metrics = evaluate_retrieval(
                retriever, benchmarks, st.session_state.settings.retrieval_top_k
            )
    metrics = st.session_state.get("retrieval_metrics")
    report = state.get("report")
    agent_metrics = evaluate_agent([report] if report else [])
    columns = st.columns(5)
    columns[0].metric("Task success", f"{agent_metrics.task_completion_rate:.0%}")
    columns[1].metric("Test pass rate", f"{agent_metrics.test_pass_rate:.0%}")
    columns[2].metric("Avg iterations", f"{agent_metrics.average_iterations:.1f}")
    columns[3].metric(
        "Human escalation", f"{agent_metrics.human_intervention_rate:.0%}"
    )
    columns[4].metric("RAG Recall@K", f"{metrics.recall_at_k:.0%}" if metrics else "—")
    if metrics:
        st.json(metrics.model_dump())
    st.dataframe(
        [task.model_dump() for task in benchmarks],
        use_container_width=True,
        hide_index=True,
    )


def render_report(state: dict) -> None:
    report = state.get("report")
    if not report:
        st.info(
            "A final report is generated after the investigation or approved execution path completes."
        )
        return
    st.subheader("AI Software Engineer Report")
    st.markdown(
        f"**Status:** `{report.status.value}` · **Confidence:** {report.confidence:.0%} · **Iterations:** {report.iterations}"
    )
    st.markdown(
        f"**Problem**\n\n{report.problem}\n\n**Root cause**\n\n{report.root_cause}"
    )
    st.markdown(
        "**Files changed**\n\n"
        + (
            "\n".join(f"- `{path}`" for path in report.files_changed)
            if report.files_changed
            else "No files were changed."
        )
    )
    if report.review:
        st.markdown("**Independent review**")
        st.json(report.review.model_dump(mode="json"))
    if report.risks:
        st.markdown("**Risks / blockers**")
        for risk in report.risks:
            st.warning(risk)
    if report.next_steps:
        st.markdown("**Next steps**")
        for item in report.next_steps:
            st.write(f"- {item}")
    st.download_button(
        "Download report JSON",
        data=report.model_dump_json(indent=2),
        file_name="forgemind-report.json",
        mime="application/json",
    )


with st.sidebar:
    st.title("⚒️ ForgeMind")
    st.caption("Local Autonomous AI Software Engineer")
    source = st.radio(
        "Repository", ["Local repository", "Load GitHub repository"], horizontal=False
    )
    default_path = st.session_state.get("repository_path", str(Path.cwd()))
    if source == "Local repository":
        repository = st.text_input("Local repository path", value=default_path)
    else:
        github_url = st.text_input(
            "GitHub repository URL", placeholder="https://github.com/owner/repository"
        )
        repository = st.session_state.get("repository_path", "")
        if st.button("Clone public repository"):
            with st.spinner("Cloning the explicitly selected repository…"):
                success, outcome = clone_github_repository(github_url)
            if success:
                st.session_state.repository_path = outcome
                repository = outcome
                st.success(f"Repository ready: {outcome}")
            else:
                st.error(outcome)
    st.divider()
    issue = st.text_area(
        "Issue / custom task",
        value=st.session_state.get("issue", ""),
        height=130,
        placeholder="Describe the software problem and expected behavior.",
    )
    issue_url = st.text_input(
        "Issue URL (optional)",
        placeholder="https://github.com/owner/repository/issues/123",
    )
    st.caption(
        "GitHub issue fetching is available only through the approval-gated token integration; paste the issue details into the task to investigate locally."
    )
    st.divider()
    base = get_settings()
    model = st.text_input("Ollama model", value=base.model)
    max_iterations = st.slider(
        "Max iterations", min_value=1, max_value=10, value=base.max_iterations
    )
    retrieval_top_k = st.slider(
        "Retrieval Top-K", min_value=1, max_value=20, value=base.retrieval_top_k
    )
    st.selectbox(
        "Approval mode",
        ["Manual approval (recommended)", "Plan only"],
        index=0,
        disabled=True,
    )
    active_settings = workspace_settings(model, max_iterations, retrieval_top_k)
    healthy, health = LLMProvider(active_settings).health()
    st.caption(("● " if healthy else "○ ") + health)
    st.caption(
        "● Docker available" if DockerSandbox.available() else "○ Docker unavailable"
    )
    if st.button("Refresh index", use_container_width=True):
        if Path(repository).expanduser().is_dir():
            with st.spinner("Rebuilding secure AST index…"):
                try:
                    index = CodeIndexer(active_settings).index(repository)
                    st.session_state.preview_index = index
                    st.success(f"Indexed {len(index.chunks)} structural chunks.")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.error("Choose an existing local repository first.")
    investigate = st.button("Investigate", type="primary", use_container_width=True)

if investigate:
    path = Path(repository).expanduser()
    if not path.is_dir():
        st.error("Choose an existing local repository before investigating.")
    elif not issue.strip():
        st.error("Describe an issue or custom task before investigating.")
    else:
        st.session_state.repository_path = str(path.resolve())
        st.session_state.issue = issue.strip()
        with st.spinner("Running the evidence-first LangGraph investigation…"):
            try:
                run_graph(str(path.resolve()), issue.strip(), False, active_settings)
            except Exception as exc:
                st.exception(exc)

state = st.session_state.get("run_state", {})
st.title("AI Software Engineering Workstation")
st.caption(
    "Stateful LangGraph · agentic Code RAG · local Ollama · Docker sandbox · PostgreSQL/pgvector memory"
)

tabs = st.tabs(
    [
        "Investigation",
        "Plan",
        "Code",
        "Diff",
        "Tests",
        "Agent Trace",
        "Memory",
        "Evaluation",
        "Final Report",
    ]
)
with tabs[0]:
    render_investigation(state)
with tabs[1]:
    render_plan(
        state,
        st.session_state.get("repository_path", repository),
        st.session_state.get("issue", issue),
    )
with tabs[2]:
    render_code(state)
with tabs[3]:
    render_diff(state)
with tabs[4]:
    render_tests(state)
with tabs[5]:
    render_trace(state)
with tabs[6]:
    render_memory(state)
with tabs[7]:
    render_evaluation(state)
with tabs[8]:
    render_report(state)
