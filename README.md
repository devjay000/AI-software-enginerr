# ForgeMind — Autonomous AI Software Engineer

```text
+---------------------------+
|        User / Browser     |
|  Local repo + issue text  |
+-------------+-------------+
              |
              v
+---------------------------+
|       Streamlit UI        |
|   app.py / dashboard      |
+-------------+-------------+
              |
              v
+---------------------------+
|   LangGraph workflow      |
| understand -> retrieve -> |
| hypothesize -> plan ->    |
| approve -> patch -> test  |
+-------------+-------------+
              |
       +------v--------+
       |                |
       v                v
+----------------+  +-----------------------+
| Code Indexer   |  | Hybrid Retriever     |
| indexing.py    |  | retrieval.py         |
| AST / symbols  |  | semantic + keyword   |
+----------------+  +-----------+-----------+
                                 |
                                 v
                     +---------------------------+
                     |   LLM + Evidence layer    |
                     | Ollama / structured JSON  |
                     +-------------+-------------+
                                   |
                                   v
                     +---------------------------+
                     |  Tool Registry + Safety   |
                     | file read, patch, docker  |
                     +-------------+-------------+
                                   |
                                   v
                     +---------------------------+
                     | PostgreSQL / memory +    |
                     | pgvector + audit traces   |
                     +---------------------------+
```

ForgeMind is a local-first AI engineering workstation. It investigates an unfamiliar repository, builds structural code intelligence, retrieves evidence with hybrid search, ranks hypotheses, plans a patch, waits for human approval, validates changes in Docker, self-corrects within a bounded loop, reviews the result, and records reusable engineering experience.

It is deliberately Python- and GenAI-first: Streamlit, LangGraph, LangChain/Ollama, PostgreSQL with pgvector, sentence-transformers, AST/tree-sitter, Docker, and GitHub. No paid model, vector store, observability service, or web backend is required.

## What is implemented

- **Stateful LangGraph workflow** with conditional approval and test/debug branches.
- **Specialized nodes** for issue analysis, repository intelligence, retrieval, hypothesis evaluation, planning, patch generation, testing, debugging, review, and final reporting.
- **Code RAG** using AST/symbol-aware chunks, hybrid semantic/keyword/symbol/dependency retrieval, local reranking, context budgeting, and query rewriting.
- **Safe agent operation**: repository files are labelled untrusted, secrets are excluded, tool permissions are independently enforced, writes/external calls require approval, and test execution is Docker-only.
- **Memory and experience RAG** stored in PostgreSQL/pgvector when available; no file-backed fallback is used for persistent data.
- **Evaluation framework** for retrieval Recall@K / Precision@K / MRR and agent success metrics.
- **Streamlit workstation** with Investigation, Plan, Code, Diff, Tests, Agent Trace, Memory, Evaluation, and Final Report views.

## Prerequisites

1. Python 3.11+ (3.13 is supported).
2. [Ollama](https://ollama.com/) and a local model, for example:

   ```powershell
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ```

   For semantic retrieval and reranking, download the two sentence-transformers models once (temporarily set `FORGEMIND_LOCAL_MODELS_ONLY=false` in `.env`, run an index, then restore it to `true`). ForgeMind defaults to offline-only model loading and falls back to lexical hybrid scoring until they are present.

3. Docker Desktop, used to execute generated tests in isolation.
4. PostgreSQL with the `vector` extension. `docker compose up -d postgres` starts a local instance.

## Quick start

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the UI, choose **Local repository**, enter its absolute path, select **Refresh index**, enter an issue, and click **Investigate**. Review the plan and click **Approve and execute** only when the displayed diff is acceptable.
## Running on a different machine

Use these steps when you want to move the project to another computer and run it locally.

### 1. Copy the project files

Clone or copy the repository to the new machine. Keep the same folder structure and project files, including:

- `app.py`
- `docker-compose.yml`
- `.env` (or create it from the example)
- `requirements.txt`
- `ai_engineer/`
- `tests/`
- `evaluation/`

### 2. Install prerequisites

Make sure the following are installed on the new machine:

- Python 3.11+
- Docker Desktop
- Git
- Ollama

### 3. Create the local environment file

If `.env` is missing, create it from the example:

```powershell
Copy-Item .env.example .env
```

If no `.env.example` exists, make sure the file contains at least these values:

```dotenv
FORGEMIND_OLLAMA_BASE_URL=http://localhost:11434
FORGEMIND_MODEL=qwen2.5-coder:7b
FORGEMIND_EMBEDDING_MODEL=all-MiniLM-L6-v2
FORGEMIND_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
FORGEMIND_LOCAL_MODELS_ONLY=true
FORGEMIND_DATABASE_URL=postgresql://forgemind:forgemind@localhost:5432/forgemind
FORGEMIND_DOCKER_IMAGE=python:3.12-slim
FORGEMIND_MAX_ITERATIONS=5
FORGEMIND_RETRIEVAL_TOP_K=8
GITHUB_TOKEN=
```

If you want to use a non-default port mapping, update the Postgres port in `docker-compose.yml` and match it in `FORGEMIND_DATABASE_URL`.

### 4. Start PostgreSQL locally

From the project root:

```powershell
docker compose up -d postgres
```

Check that the container is healthy and the database is ready:

```powershell
docker ps
```

### 5. Pull local Ollama models

The app expects local models to be available through Ollama:

```powershell
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

If the embedding model is missing and you want to allow download-once setup, temporarily set:

```dotenv
FORGEMIND_LOCAL_MODELS_ONLY=false
```

Then run an index once. Restore the value to `true` after the model is cached.

### 6. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 7. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 8. Start the app

```powershell
python -m streamlit run app.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```

Then open:

```text
http://localhost:8501
```

### 9. First-run checks

After startup, verify the following:

- the app loads without onboarding prompts,
- Docker is available,
- Postgres is reachable on the configured port,
- Ollama responds to the selected model,
- the repository can be indexed successfully.

### 10. If the machine is offline or model cache is missing

If the machine does not have internet access, do one of the following:

- copy the local Ollama model files from another machine,
- or run the first setup on a machine with internet and then transfer the model cache afterward,
- or set `FORGEMIND_LOCAL_MODELS_ONLY=false` once to download the required models and then restore `true`.

### 11. If you want a portable setup checklist

When moving the project, keep this list handy:

- Python installed
- Docker installed and running
- Ollama installed and models pulled
- Postgres started via Docker Compose
- `.env` created and values verified
- virtual environment created
- dependencies installed
- Streamlit app launched

### 12. Troubleshooting on a new machine

Common issues:

- `port already in use`: another Postgres instance is already listening on 5432; change the docker port mapping or stop the conflicting service.
- `streamlit not recognized`: activate the virtual environment or run `python -m streamlit`.
- `Ollama unavailable`: ensure Ollama is installed and running locally.
- `docker: not found`: install Docker Desktop and restart the terminal.
- `database connection failed`: confirm the database host, user, password, and port match the `.env` file.

This is the safest portable setup pattern for moving ForgeMind to another local machine without changing the project logic.

## Why this is a multi-agent application

ForgeMind is a multi-agent system because it does not run as one single monolithic prompt. Instead, it splits the work into separate stages, each with a focused responsibility.

### The agent roles in this project

The workflow in `ai_engineer/graph.py` connects several specialized nodes:

- `understand_issue`: turns the user task into a structured investigation brief.
- `understand_repository`: indexes the repo and extracts structure.
- `retrieve_evidence`: searches for the most relevant code using hybrid retrieval.
- `generate_hypotheses`: proposes likely root causes from the retrieved evidence.
- `evaluate_hypotheses`: checks which hypothesis has the strongest support.
- `create_plan`: creates a patch strategy based on evidence.
- `human_approval`: waits for explicit approval before writes or external actions.
- `inspect_test_strategy`: selects the right tests or adds test strategy guidance.
- `implement`: applies code changes safely.
- `run_tests`: executes validation in the Docker sandbox.
- `debug`: handles failed tests and tries to recover.
- `review`: checks whether the patch is likely correct and safe.
- `final_report`: summarises findings, evidence, and results.

These are not just functions in a list — they are agent stages with different jobs and different state transitions. The graph decides which stage comes next based on results, approvals, and test outcomes.

### Why this qualifies as multi-agent

A multi-agent system is usually one where:

- different parts of the workflow are responsible for different tasks,
- work is routed between those parts conditionally,
- decisions are made incrementally instead of in one giant prompt,
- and each stage contributes evidence, analysis, or validation.

ForgeMind matches that pattern.

The app does not ask one model to do everything at once. Instead, it creates a pipeline of specialized agents that pass state forward. That makes the system more auditable, more controllable, and safer.

### Multi-model roles vs. multi-agent teams

This project uses multiple model roles, but the architecture is still a single orchestration workflow rather than a fully independent swarm of models:

- the main model handles structured reasoning,
- the embedding model handles semantic retrieval,
- the reranker model improves ranking quality.

So the best description is:

- multi-agent workflow
- with multiple model roles
- but not a fully decentralized multi-model agent network

This balance is intentional: it keeps the system local, explainable, and manageable while still separating responsibilities across agent stages.

## AI concepts used in the application

ForgeMind uses a set of classic AI and LLM-engineering patterns, but blends them in a local, safety-first way.

### 1. Agentic workflow / orchestration

The core design is a graph-based workflow, not a single prompt chain. This is implemented with `StateGraph` in `ai_engineer/graph.py`.

The important idea is that the app is not asking one model to do everything in one pass. Instead, it moves through a sequence of phases:

- issue understanding,
- repository indexing,
- retrieval,
- hypothesis generation,
- planning,
- approval,
- implementation,
- validation,
- review.

This is an agentic pattern: the system makes decisions step by step, updates state, and routes control based on results.

### 2. Retrieval-Augmented Generation (RAG)

The application is fundamentally a code RAG system.

It does not rely on the model remembering everything from training data. Instead, it retrieves relevant repository evidence first and then adds that evidence to the model context.

The flow is:

1. index repository code into structured chunks,
2. search for relevant files and symbols,
3. assemble the best evidence,
4. pass that evidence to the LLM,
5. let the model reason over the code rather than guessing.

This is why the project emphasizes `retrieval.py`, `indexing.py`, and the `ContextBuilder` object: they create grounded prompts based on real repository evidence.

### 3. Hybrid retrieval

ForgeMind does not use only one retrieval method. It uses a hybrid system combining:

- semantic similarity,
- keyword overlap,
- symbol matching,
- dependency/context matching,
- and occasionally stored experience memory.

This is implemented in `HybridRetriever` in `ai_engineer/retrieval.py`. The benefits are:

- semantic matching catches meaning even when vocabulary differs,
- keyword matching captures exact terms and APIs,
- symbol matching catches function names and class names,
- dependency matching follows import relationships.

This is a common AI pattern for code search: using multiple retrieval channels to improve recall.

### 4. Embeddings and vector search

Embedding-based retrieval is used for semantic similarity.

The app stores chunk embeddings and supports vector similarity queries in Postgres with pgvector via `MemoryStore` in `ai_engineer/storage.py`.

Conceptually:

- code chunks are converted into vectors,
- a user issue is converted into a comparable vector,
- the system finds nearby vectors in semantic space,
- those nearby chunks are treated as likely relevant evidence.

This is the AI concept behind semantic search, and it is especially useful when the user’s language differs from the code’s terminology.

### 5. Reranking

After a first retrieval pass, the app can rerank the best candidate chunks using a stronger model or a cross-encoder. This is implemented through the reranker path in `ai_engineer/retrieval.py`.

Reranking matters because the first search may produce many possible matches. A reranker improves the ordering so the best evidence sits at the top of the prompt.

### 6. Structured output and schema validation

The project uses Pydantic models in `ai_engineer/models.py` and a structured-output wrapper in `ai_engineer/llm.py`.

This means the model is asked to return data in a fixed schema instead of free-form text. For example:

- `IssueAnalysis`
- `SearchPlan`
- `Hypothesis`
- `ImplementationPlan`
- `TestResult`
- `FinalReport`

This is important because AI outputs can be noisy or inconsistent. Structured output gives the app a reliable contract for downstream logic.

In other words, the LLM is acting like a data generator for precise tasks rather than an uncontrolled text generator.

### 7. Prompt engineering and context assembly

The app builds a controlled prompt context using `ContextBuilder`. It focuses on:

- repository summary,
- retrieved code evidence,
- recent comparable memories,
- and a clear instruction to the model.

It also wraps repository content in an untrusted-data boundary using `untrusted_repository_context()` in `ai_engineer/security.py`.

Important AI concept here: context engineering. The model is given only the most relevant evidence, not the entire repository. This reduces noise and helps focus the model on the likely root cause.

### 8. Context budgeting and token awareness

One of the most important AI design choices is that the app limits how much repository context it includes in a prompt.

The project estimates prompt size and deliberately avoids exceeding a budget using `ContextBuilder._estimate_tokens()` and `Settings.context_token_budget`.

This matters because:

- LLMs have limited context windows,
- too much code makes reasoning worse,
- and relevant evidence must be prioritized.

This is a practical AI systems concept: graceful selection of the best information under a token budget.

### 9. Human-in-the-loop approval

The app is not fully autonomous in write mode. It uses a human approval gate before it performs sensitive actions such as:

- patch application,
- test execution,
- Git operations,
- GitHub actions.

This is a classic AI safety pattern: allow the model to propose, but require human validation before risky actions.

That safety boundary is enforced by `ToolRegistry` and `ToolPermissionError` in `ai_engineer/tools.py`.

### 10. Tool use and tool calling patterns

The project uses an agentic tool-use design. The model is not allowed to directly mutate files or run commands. Instead, it proposes actions, and the app routes those actions through permissioned tools.

Examples:

- `read_file`
- `search_code`
- `search_symbol`
- `apply_patch`
- `run_tests`
- `get_git_diff`

This is a foundational AI concept in real tool-using agents: the model chooses a tool, but the tool layer enforces policy and safety.

### 11. Safety against prompt injection and secret leakage

This project explicitly treats repository files as untrusted data.

It contains safeguards for:

- secret file detection,
- prompt-injection detection,
- path traversal prevention,
- large-file exclusion,
- and restricted execution.

This is important in AI security: an LLM may read files that include malicious or misleading instructions. The app wraps those files in an explicit warning and filters obvious dangerous content before sending them to the model.

### 12. Local model hosting and offline-first AI

The app prefers local inference through Ollama and local model caching, defined in `ai_engineer/config.py` and `ai_engineer/llm.py`.

This is an AI systems concept called local-first deployment. It gives several advantages:

- privacy,
- lower latency,
- more control,
- no cloud dependency,
- and easier reproducibility.

The embedding model and reranker are also local models, which is a common pattern when building robust private AI systems.

### 13. Memory and experience retrieval

ForgeMind stores traces, repository metadata, and successful engineering experiences in memory using `MemoryStore` and PostgreSQL/pgvector.

This is another core AI concept: memory-augmented reasoning.

The app can:

- save repository index metadata,
- keep agent traces,
- store successful experiences,
- and search past runs for similar issues.

This allows the system to reuse prior problem-solving patterns rather than starting from zero every time.

### 14. Evidence-grounded reasoning

A major theme in this project is that every stage tries to work from evidence rather than guesswork.

Examples:

- evidence is retrieved before a claim is made,
- hypotheses are grounded in evidence,
- plans must cite supporting evidence,
- tests validate the result,
- final reports include evidence and review.

This is one of the strongest AI design ideas in the repo: grounded reasoning over raw model intuition.

### 15. Test-time validation and self-correction loop

The workflow includes a bounded self-correction pattern:

- run tests,
- if they fail, enter debug mode,
- retrieve more evidence,
- form a new hypothesis,
- retry within a limited iteration count.

This is a classic AI loop for improving results over multiple attempts while still keeping the process bounded and safe.

### 16. Evaluation and benchmarking

The project also includes an evaluation layer in `ai_engineer/evaluation.py` to measure:

- retrieval recall,
- precision,
- MRR,
- task completion,
- test pass rate,
- intervention rate,
- and other quality metrics.

This is important in AI systems design because it moves the project from “it seems to work” to “we can actually measure whether it works.”

## How nested repository structures are handled

ForgeMind does not flatten a repository into a single list of files. It keeps the real directory structure and stores each file using its relative path.

### Core idea

The key logic is in `ai_engineer/indexing.py`:

- `CodeIndexer.discover()` walks the repo recursively with `repository.rglob("*")`.
- every discovered file is converted into a safe path like:

```text
src/services/auth/session.py
backend/api/routes/user.py
web/components/forms/LoginForm.tsx
```

- each path is preserved as `path.relative_to(root).as_posix()` so nested folders stay meaningful.

### Why that matters

This allows the app to understand the real layout of a project, not just a flat filename list. A nested repository can still be indexed as a tree of modules, services, tests, and config files.

### What gets stored for each file

For each valid file, ForgeMind creates structured code chunks. Each chunk keeps:

- `file`: the exact relative path inside the repo
- `symbol`: function/class/method name
- `kind`: `function`, `class`, `method`, `module`, `test`, etc.
- `start_line` and `end_line`
- `content`: the code body for that symbol
- `imports`: imported modules and dependencies

This makes later retrieval accurate. The output can point to a real path such as:

```text
src/services/auth/session.py:42-87
```

instead of just showing `session.py` with no context.

### Nested folder support

The indexer works on deeply nested repos because:

- it recurses through every directory,
- it stores relative paths, not just basenames,
- it extracts chunks per symbol within each file,
- it builds a dependency map keyed by file path,
- it keeps evidence tied to the original file location.

### Safety filtering for complex trees

Large or messy repositories still remain safe because the app filters files before indexing:

- ignored folders like `.git`, `.venv`, `node_modules`, and `dist` are excluded,
- secret files such as `.env`, key files, and certificates are blocked,
- files above `max_file_bytes` are skipped,
- binary or unsafe files are rejected,
- content containing obvious credential patterns is not embedded into prompts.

This means the index stays focused on useful code and ignores noise, secrets, or unsafe paths.

### Retrieval behavior on nested repos

When the user searches for a bug, the app does not only match words. It also tracks:

- symbol names,
- imports and dependencies,
- semantic similarity,
- keyword overlap,
- and file-level context.

So if a bug is in a nested module like `services/users/logic.py`, the returned evidence still includes the exact file path and the relevant function or class region.

### Simple example

If the repo contains:

```text
app/
  controllers/
    user_controller.py
  services/
    user_service.py

```

the indexer stores:

```text
app/controllers/user_controller.py -> controller methods
app/services/user_service.py -> service methods
```

and retrieval can say:

```text
user_service.py:120-150 -> validate_user_session
```

instead of returning an ambiguous result like just `user_service.py`.

This is how ForgeMind understands complex nested file structures without losing precision.

## Safety model

| Permission | Behaviour |
| --- | --- |
| `READ` | Automatically allowed for paths inside the selected repository. |
| `WRITE` | Requires the human approval flag and a validated unified diff. |
| `EXECUTE` | Executes only through the Docker sandbox. |
| `EXTERNAL` | Requires approval; GitHub actions are never automatic. |

Files such as `.env`, private keys, credential files, lockfiles, binaries, and over-large files are excluded from ingestion. Repository text is explicitly wrapped as untrusted data before being supplied to the model.

## Architecture

```text
Streamlit → LangGraph supervisor → specialized agent nodes
                                 ↘ Code RAG (AST + hybrid retrieval + reranking)
                                  ↘ Ollama through LangChain
                                   ↘ Docker test sandbox
                                    ↘ PostgreSQL + pgvector memory
                                     ↘ GitHub (approval-gated)
```

The graph is intentionally stateful rather than a single prompt chain:

```text
understand_issue → inspect_repository → retrieve_evidence → hypotheses
  → evaluate_hypotheses → create_plan → human_approval
  → implement → run_tests ──pass──→ review → final_report
                       └─fail→ debug → retrieve_evidence (bounded iterations)
```

## Configuration

All settings have `FORGEMIND_` environment-variable equivalents; see `.env.example`. The system can investigate and index without Ollama/PostgreSQL/Docker, but it makes the missing capability explicit. Code generation requires an Ollama model; tests require Docker.

## Evaluation

Run the local test suite with:

```powershell
python -m unittest discover -s tests -v
```

Add benchmark issues to `evaluation/tasks.json` (or via the UI) with expected relevant paths. The evaluation dashboard calculates retrieval metrics and records task-level agent metrics in PostgreSQL when configured.

## Project layout

```text
app.py                  Streamlit workstation
ai_engineer/
  graph.py              LangGraph state machine and conditional routing
  agents.py              Investigation, planning, coding, testing, review nodes
  retrieval.py           Agentic hybrid Code RAG and context builder
  indexing.py            Safe discovery, AST chunks, symbols, dependencies
  tools.py               Permissioned tool registry, Docker/Git/GitHub tools
  storage.py             PostgreSQL/pgvector memory and experiences
  models.py              Pydantic decision contracts
  llm.py                 LangChain/Ollama provider and structured generation
  evaluation.py          Local RAG and agent evaluation metrics
  security.py            Secret/prompt-injection defense
tests/                   Fast deterministic unit tests
```

# Beginner guide: how ForgeMind works

This project is a local coding agent that reads a repository, understands the code structure, searches for likely bug locations, builds a plan, asks for human approval, and optionally patches the code and runs tests in a Docker sandbox.

The important idea is this:

- It is not a single giant prompt.
- It is a workflow made of small steps.
- Each step is grounded in evidence from the repository.
- It tries to stay safe by restricting file access, blocking secret files, and running test commands only in a disposable container.

Think of ForgeMind as a careful junior engineer that:

1. reads the repository,
2. searches for useful code,
3. creates hypotheses,
4. checks the evidence,
5. makes a small plan,
6. waits for a human to approve,
7. patches code if approved,
8. runs tests in an isolated environment,
9. writes a final review and stores what it learned.

## What happens from the user's point of view

A user opens the Streamlit app, chooses a repository folder, writes a bug description or task, and presses Investigate.

Behind the scenes, the app:

- builds a safe index of the repository,
- extracts symbols and imports,
- finds code chunks and dependencies,
- retrieves the most relevant code,
- asks an LLM to propose likely root causes,
- validates those hypotheses against the repository,
- creates a plan,
- waits for approval,
- applies a patch only after approval,
- runs the test suite inside Docker,
- reports results and keeps an audit trail.

## End-to-end data flow

The simplest data flow is:

```text
User issue text
    ↓
Streamlit UI (app.py)
    ↓
initial_state() creates workflow state
    ↓
LangGraph graph (graph.py)
    ↓
Issue agent -> Repository indexer -> Hybrid retriever -> Hypothesis engine
    ↓
Evidence + search plan + context bundle
    ↓
Planner -> Human approval gate
    ↓
Patch generator -> ToolRegistry -> Git apply + Docker run tests
    ↓
Review + final report + memory store
```

### Data objects and what they mean

The project uses Pydantic models in `ai_engineer/models.py` to keep every step structured and validated.

- `IssueAnalysis`: turns a natural-language issue into a clear problem statement, scope, and search terms.
- `RepositoryAnalysis`: stores high-level facts about the repository, such as languages, frameworks, entry points, test tools, and config files.
- `SearchPlan`: stores the rewritten issue query, selected symbols, and retrieval rationale.
- `Hypothesis`: a claim like "this file likely contains the root cause" plus confidence and evidence.
- `ImplementationPlan`: the proposed patch strategy, files, risks, and rollback plan.
- `CodeChange`: a single file patch represented as a git unified diff.
- `TestStrategy`: which existing tests should be used or extended.
- `TestResult`: command, exit code, stdout/stderr, and pass/fail summary.
- `TraceEvent`: a single event in the audit history.
- `FinalReport`: final summary, confidence, root cause, findings, files changed, and next steps.

## Exact workflow, one step at a time

### 1. The Streamlit entrypoint starts the app

`app.py` is the user interface.

Key functions:

- `workspace_settings(model, iterations, top_k)`: builds a local configuration object from environment settings.
- `clone_github_repository(url)`: clones a public GitHub repository into a safe local cache under `.forgemind/repositories`.
- `run_graph(repository, issue, approved, settings)`: creates a `MemoryStore`, a runtime, builds the LangGraph, runs the workflow, stores the result in `st.session_state`, and returns the final state.
- `read_repository_file(state, relative_path)`: reads a file using the permissioned `ToolRegistry` instead of raw filesystem access.
- `render_investigation`, `render_plan`, `render_code`, `render_diff`, `render_tests`, `render_trace`, `render_memory`, `render_evaluation`, `render_report`: each tab in the app shows a different part of the agent run.

The app keeps a `run_state` in Streamlit session state so the UI can display the latest workflow state, plan, evidence, diff, tests, and final report.

### 2. The workflow graph decides what happens next

`ai_engineer/graph.py` defines the state machine.

Important pieces:

- `AgentState`: a typed dictionary that holds repository path, issue text, approval flags, state status, evidence, hypotheses, code changes, tests, trace, and final report.
- `_after_approval`: decides whether to continue to implementation or stop at final report based on the human approval flag.
- `_after_implementation`: decides whether to run tests or finish early.
- `_after_tests`: if tests pass, review; otherwise debug.
- `_after_debug`: either loop back to retrieval or finish.
- `build_graph(runtime)`: wires together all agent nodes into the LangGraph workflow.
- `initial_state(...)`: creates the starting state object for a task.

The graph order is roughly:

```text
understand_issue
  -> understand_repository
  -> retrieve_evidence
  -> generate_hypotheses
  -> evaluate_hypotheses
  -> create_plan
  -> human_approval
  -> inspect_test_strategy
  -> implement
  -> run_tests
  -> review
  -> final_report
```

If tests fail, the workflow can loop back into debug and retrieval with a bounded number of attempts.

### 3. The agent runtime orchestrates the actual reasoning steps

`ai_engineer/agents.py` is the brain of the system.

`AgentRuntime` contains the agent methods that perform each step.

#### `AgentRuntime.__init__`
- sets the project settings,
- creates or reuses a `MemoryStore`,
- creates an `LLMProvider` for local Ollama-based structured generation.

#### `AgentRuntime._event`
- creates a `TraceEvent`,
- stores the event in memory,
- attaches evidence file paths for traceability.

#### `AgentRuntime.understand_issue`
- strips and normalizes the issue text,
- extracts a short list of important keywords,
- builds a fallback `IssueAnalysis`,
- asks the LLM to convert the issue into structured categories and search terms.

#### `AgentRuntime.understand_repository`
- runs `CodeIndexer.index()` on the repository,
- saves the index into memory,
- records a repository summary and structural analysis.

#### `AgentRuntime.retrieve_evidence`
- creates or reuses a `HybridRetriever`,
- rewrites the query into better search terms,
- reads manifest files through `ToolRegistry`,
- looks up direct symbol references if symbols were found,
- runs hybrid retrieval,
- loads similar past experiences,
- builds a compact, token-budgeted `ContextBundle` for the model.

#### `AgentRuntime.generate_hypotheses`
- selects the strongest evidence,
- creates a fallback hypothesis list,
- asks the LLM for 2–4 possible root causes.
- removes any hypothesis that is not backed by actual evidence.

#### `AgentRuntime.evaluate_hypotheses`
- looks for symbol references using `ToolRegistry`,
- adds more evidence if relevant,
- ranks hypotheses by confidence.

#### `AgentRuntime.create_plan`
- chooses the top hypothesis,
- builds a minimal implementation plan,
- ensures the plan only uses files that were actually evidenced,
- pauses for human approval.

#### `AgentRuntime.human_approval`
- checks whether a human approved the patch and execution steps,
- updates the state status accordingly.

#### `AgentRuntime.inspect_test_strategy`
- reads existing repository tests,
- chooses likely test targets,
- keeps the strategy grounded in the repository.

#### `AgentRuntime.implement`
- asks the LLM for a git-style unified diff,
- limits the change set to evidence-backed files,
- validates the patch with `ToolRegistry.apply_unified_diff`,
- applies only safe changes.

#### `AgentRuntime.run_tests`
- calls the `run_tests` tool,
- executes tests in Docker,
- stores the result in the workflow state.

#### `AgentRuntime.debug`
- reads the last test result,
- creates a root-cause hypothesis for failing tests,
- recommends the next retrieval or fix action,
- decides whether the workflow should continue or stop.

#### `AgentRuntime.review`
- checks whether the patch is likely correct, secure, and appropriately scoped,
- only approves if the last test run passed.

#### `AgentRuntime.final_report`
- prepares the final report with problem, evidence, hypotheses, files changed, tests, review, risk, confidence, and next steps,
- stores successful experience in memory if the run was completed.

## How the repository is indexed and searched

`ai_engineer/indexing.py` is responsible for reading a repository without trusting the code.

### `LANGUAGES`
A lookup table that maps common file extensions to a language name such as `python`, `javascript`, `typescript`, `yaml`, and `toml`.

### `IndexResult`
A structured result containing:

- `analysis`: a high-level summary of the repository,
- `chunks`: extracted code units,
- `dependencies`: file-to-import mapping,
- `excluded_files`: files skipped because they are too large, dangerous, or secret-like.

### `language_for(path)`
Returns the language label for a file extension.

### `_line_slice(lines, start, end)`
Shows the code slice around a symbol using the starting and ending line numbers.

### `PythonAstExtractor`
This is a Python-specific extractor. It uses Python's built-in `ast` module to find:

- import statements,
- functions,
- async functions,
- classes,
- tests.

It creates `CodeChunk` objects for each meaningful structural unit.

### `TreeSitterExtractor`
This is used for JavaScript/TypeScript when tree-sitter is available.

It tries to parse the code structure before falling back to regex-based extraction. This keeps indexing useful even in stripped-down environments.

### `CodeIndexer`
This is the main repository scanner.

It does the following:

- walks all files in a repository,
- excludes ignored directories, secrets, binary files, and large files,
- reads text files safely,
- skips files with likely credentials or sensitive patterns,
- extracts code chunks and imports,
- stores dependencies and metadata,
- analyzes the repository for frameworks, test tools, entry points, important modules, API surfaces, and database files.

This step is the backbone of the RAG system. It produces the code knowledge that later retrieval uses.

## Code retrieval and context building

`ai_engineer/retrieval.py` is responsible for turning a user issue into the best possible code evidence.

### `terms(value)`
Converts raw text into lower-case words and strips punctuation. This is used for lexical matching.

### `ContextBundle`
A bundle that contains:

- `prompt_context`: the text sent to the model,
- `evidence`: chosen evidence items,
- `estimated_tokens`: approximate prompt cost,
- `omitted_chunks`: how many candidates had to be skipped because of budget.

### `HybridRetriever`
This class is the heart of Code RAG. It combines several signals:

- semantic similarity using local sentence-transformers when available,
- keyword matches from the issue text,
- symbol overlap between the issue and code symbols,
- dependency hints from imports and references,
- memory-based similarity from earlier successful experiences.

Important methods:

- `rewrite_query(issue, components)`: expands the user's issue into better search terms and likely symbols.
- `search(plan, top_k)`: scores chunks, merges multiple retrieval signals, reranks candidates, and returns `RetrievalHit` objects.
- `_keyword_score(...)`: measures lexical match quality.
- `_symbol_score(...)`: checks how strongly the symbol names match the issue.
- `_dependency_score(...)`: looks at imports and references for likely file connections.
- `_semantic_scores(...)`: creates embeddings and compares them to the query.
- `_ensure_embeddings()`: loads or creates the embedding model and vector cache.
- `_model_is_cached(...)`: checks whether the model is already downloaded locally.
- `_lexical_cosine(...)`: used as an offline fallback when semantic models are not available.
- `_rerank(...)`: uses a cross-encoder reranker when configured.
- `_explain(...)`: explains how the chunk matched.
- `references(symbol, top_k)`: returns code chunks containing a symbol name.
- `evidence(hits)`: converts retrieval hits into evidence items for the LLM and the dashboard.

### `ContextBuilder`
This class turns retrieval hits into a clean prompt context. It aggressively respects token budgets and wraps repository content as untrusted data before sending it to the model.

Important methods:

- `build(issue, repository_summary, hits, memories)`: assembles the final prompt context and evidence list while staying under the configured budget.
- `_estimate_tokens(value)`: approximates token cost per string.

## Safety model and local controls

Safety is not an afterthought in this project. It is a core design principle.

### `ai_engineer/security.py`
This file protects the app from unsafe files and prompt injection.

- `is_secret_path(path)`: rejects `.env`, private keys, certificates, and other sensitive file names.
- `contains_secret(text)`: scans file content for obvious secret-like patterns.
- `contains_prompt_injection(text)`: checks for instruction-hijacking text inside repository data.
- `safe_relative_path(repository, candidate)`: resolves a path and rejects traversal outside the repository root.
- `is_indexable(path, max_bytes)`: filters out ignored directories, big files, binary data, and sensitive content.
- `untrusted_repository_context(content, source)`: wraps repository text with an explicit warning saying the source is untrusted data and not instructions.

These functions help ensure the AI never treats repo contents as system instructions.

## Tools and permission model

`ai_engineer/tools.py` deliberately separates the LLM from side effects.

### `ToolSpec`
A description of a tool: its name, permission level, description, and the function that executes it.

### `ToolPermissionError`
Raised when an action requires higher permission than the user approved.

### `DockerSandbox`
This sandbox is used for test execution.

Important methods:

- `available()`: returns whether Docker is installed.
- `test_command(repository)`: chooses the correct test command for Python, Node, or generic projects.
- `run_tests(repository, timeout_seconds)`: copies the repo into a temporary directory, runs tests in a network-disabled container, and returns a structured `TestResult`.

This makes generated test execution deterministic and isolated from the host machine.

### `ToolRegistry`
This is the guardrail layer for file reads, writes, Git operations, and Docker execution.

Supported tool names include:

- `list_files`
- `read_file`
- `search_code`
- `search_symbol`
- `find_references`
- `find_dependencies`
- `get_repository_summary`
- `get_recent_commits`
- `get_git_diff`
- `apply_patch`
- `run_tests`
- `create_branch`
- `commit_changes`
- `push_branch`

Important methods:

- `invoke(name, **kwargs)`: checks permission and routes to the correct handler.
- `_path(candidate)`: resolves a path safely and rejects traversal and protected folders.
- `read_file(path, max_chars)`: reads a safe file with a defined byte limit.
- `search_code(...)`: performs a safe lexical search.
- `search_symbol(...)`: searches for symbol definitions/usages.
- `find_dependencies(path)`: extracts imports from code.
- `repository_summary()`: reads key repo manifest files.
- `_git(*args)`: runs a git command inside the selected repository.
- `recent_commits()`: returns recent commit history.
- `git_diff()`: returns the current git diff.
- `get_tests()`: finds likely test files.
- `apply_unified_diff(diff)`: validates a git patch before applying it.
- `create_branch(branch)`: creates a branch under Git restrictions.
- `commit_changes(files, message)`: commits only explicitly listed files.
- `push_branch(branch)`: pushes a branch if allowed.

### `GitHubTools`
This is optional external GitHub integration. It is approval-gated and requires a `GITHUB_TOKEN` environment variable.

Methods:

- `get_issue(repository, number)`: fetches a single issue.
- `get_repository(repository)`: fetches repository metadata.
- `create_pull_request(...)`: creates a pull request when approval is granted.

## LLM provider and structured output

`ai_engineer/llm.py` wraps the local Ollama API.

### `LLMProvider`
This class centralizes local LLM access.

Important methods:

- `health()`: checks whether Ollama is reachable.
- `_chat_model()`: lazily creates the ChatOllama model.
- `structured(schema, instruction, context, fallback)`: asks the model for a validated JSON object matching a Pydantic schema. If Ollama is unavailable, it smoothly falls back to predetermined safe values.
- `text(...)`: asks the model for plain text output, again with a safe fallback.

This keeps the rest of the app independent from any one model provider.

## Local memory and experience retention

`ai_engineer/storage.py` stores repository metadata, traces, and successful experiences.

### `MemoryStore`
This class persists to PostgreSQL/pgvector when available and falls back to in-memory session storage when PostgreSQL is unavailable.

Important methods:

- `__init__(database_url)`: initializes the store and connects to Postgres if available.
- `_initialize()`: runs the SQL schema setup.
- `_connection()`: creates a database connection.
- `save_index(...)`: stores repository metadata and code chunks.
- `append_trace(...)`: records the trace of agent actions.
- `save_embeddings(...)`: stores vector embeddings for semantic retrieval.
- `semantic_search(...)`: finds semantically similar chunks from the database.
- `keyword_search(...)`: does full-text search using PostgreSQL search features.
- `traces()`: returns the in-memory trace log.
- `remember_experience(...)`: remembers a successful engineering experience.
- `similar_experiences(...)`: searches earlier successful runs for related domain knowledge.
- `status_message`: tells the UI whether Postgres is connected or only in-memory mode.

The project intentionally prefers Postgres when available, but does not depend on it for basic operation.

## Configuration and environment values

`ai_engineer/config.py` defines the core settings used throughout the app.

### `Settings`
This class reads environment values starting with `FORGEMIND_` from `.env`.

Important settings:

- `ollama_base_url`: where local Ollama is listening.
- `model`: the main coding model to use.
- `embedding_model`: language embedding model for semantic retrieval.
- `reranker_model`: re-ranking model for candidate scoring.
- `local_models_only`: if true, the app prefers cached local models and avoids downloads.
- `database_url`: PostgreSQL connection string.
- `docker_image`: image used for isolated test execution.
- `max_iterations`: maximum retry loop size.
- `retrieval_top_k`: maximum results per retrieval.
- `context_token_budget`: maximum prompt budget for context.
- `max_file_bytes`: largest file size to ingest safely.

### `get_settings()`
Returns a cached settings object.

## The project’s basic mental model

At a high level, ForgeMind is a small software engineering loop:

```text
Read repo -> understand issue -> search -> rank -> plan -> human approve -> patch -> test -> review -> learn
```

This is not generic chat. It is a local, evidence-first engineering assistant with safety checks and explicit review points.

## If you are a complete beginner, read this first

- The app is designed to work on a local machine, not in the cloud.
- It indexes the repository before thinking.
- It never trusts the repo as instructions.
- It retrieves code evidence instead of guessing blindly.
- It pauses before writing to disk.
- It runs tests only inside Docker.
- It tries to be transparent by showing traces, evidence, and a final report.

The easiest way to understand the code is to follow one real run through the state graph:

1. choose a repository,
2. set an issue,
3. watch `understand_repository` create the index,
4. look at `retrieve_evidence` results,
5. inspect the hypotheses,
6. read the plan,
7. approve it,
8. inspect the diff,
9. look at test results,
10. read the final report.

That path shows the whole system from start to finish without needing to understand every helper at once.

## Main files at a glance

- `app.py`: UI, control panel, and rendering for each workflow view.
- `ai_engineer/graph.py`: state graph and rail logic.
- `ai_engineer/agents.py`: agent step implementations.
- `ai_engineer/indexing.py`: safe repository indexing and AST extraction.
- `ai_engineer/retrieval.py`: hybrid retrieval, reranking, and context builder.
- `ai_engineer/tools.py`: tool permissions and sandbox enforcement.
- `ai_engineer/security.py`: secret filtering and prompt injection defenses.
- `ai_engineer/storage.py`: Postgres or in-memory memory.
- `ai_engineer/models.py`: structured contracts used across the app.
- `ai_engineer/llm.py`: local LLM provider and validation wrapper.
- `ai_engineer/evaluation.py`: benchmark and measurement utilities.
- `tests/`: unit tests that cover indexing, retrieval, permissions, and security behavior.

## Final summary

ForgeMind is best understood as a safe, local, evidence-first software engineering loop:

- it understands the code base,
- it searches for likely root causes,
- it asks for approval before writing,
- it validates with tests in isolation,
- it records what it learned,
- and it keeps the entire run explainable.

If you are learning the project, start by reading these files in this order:

1. `app.py`
2. `ai_engineer/graph.py`
3. `ai_engineer/agents.py`
4. `ai_engineer/indexing.py`
5. `ai_engineer/retrieval.py`
6. `ai_engineer/tools.py`
7. `ai_engineer/models.py`
8. `ai_engineer/security.py`

That order follows the real runtime from user input to final output.

---

This README section is intentionally beginner-oriented. For exact implementation details, use the code plus the test suite in `tests/` as the ground truth. The complete architecture is designed to be auditable and local-first, not magical or opaque.

