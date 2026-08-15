"""Permissioned tools. The LLM suggests actions; this layer independently enforces authority."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .models import Permission, TestResult
from .security import is_secret_path, safe_relative_path


@dataclass(frozen=True)
class ToolSpec:
    name: str
    permission: Permission
    description: str
    handler: Callable[..., Any]


class ToolPermissionError(PermissionError):
    pass


class DockerSandbox:
    """Runs a repository's deterministic test command only in a short-lived, network-disabled container."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def available() -> bool:
        return shutil.which("docker") is not None

    def test_command(self, repository: Path) -> tuple[str, str]:
        if (repository / "package.json").exists():
            manager = (
                "npm ci"
                if (repository / "package-lock.json").exists()
                else "npm install"
            )
            return "node:20-bookworm-slim", f"{manager} && npm test -- --runInBand"
        if (repository / "pyproject.toml").exists() or (
            repository / "pytest.ini"
        ).exists():
            install = (
                "pip install --no-cache-dir -e ."
                if (repository / "pyproject.toml").exists()
                else "pip install --no-cache-dir -r requirements.txt"
            )
            return self.settings.docker_image, f"{install} && python -m pytest -q"
        if (repository / "requirements.txt").exists():
            return (
                self.settings.docker_image,
                "pip install --no-cache-dir -r requirements.txt && python -m unittest discover -v",
            )
        return self.settings.docker_image, "python -m unittest discover -v"

    def run_tests(self, repository: Path, timeout_seconds: int = 600) -> TestResult:
        if not self.available():
            return TestResult(
                command="docker unavailable",
                passed=False,
                exit_code=127,
                duration_seconds=0,
                summary="Docker is required for isolated execution.",
            )
        image, command = self.test_command(repository)
        started = time.monotonic()
        # Work on a disposable copy: package installation and tests cannot alter the selected repository.
        with tempfile.TemporaryDirectory(prefix="forgemind-sandbox-") as temporary:
            snapshot = Path(temporary) / "repository"
            shutil.copytree(
                repository,
                snapshot,
                ignore=shutil.ignore_patterns(
                    ".git", "node_modules", ".venv", "venv", "__pycache__"
                ),
            )
            invocation = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "2g",
                "--cpus",
                "2",
                "-v",
                f"{snapshot}:/workspace",
                "-w",
                "/workspace",
                image,
                "sh",
                "-lc",
                command,
            ]
            try:
                result = subprocess.run(
                    invocation,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return TestResult(
                    command=command,
                    passed=result.returncode == 0,
                    exit_code=result.returncode,
                    stdout=result.stdout[-12_000:],
                    stderr=result.stderr[-12_000:],
                    duration_seconds=round(time.monotonic() - started, 2),
                    summary=(
                        "Tests passed in disposable Docker sandbox."
                        if result.returncode == 0
                        else "Tests failed in disposable Docker sandbox."
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                return TestResult(
                    command=command,
                    passed=False,
                    exit_code=124,
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                    duration_seconds=round(time.monotonic() - started, 2),
                    summary="Docker test run timed out.",
                )
            except OSError as exc:
                return TestResult(
                    command=command,
                    passed=False,
                    exit_code=127,
                    stderr=str(exc),
                    duration_seconds=round(time.monotonic() - started, 2),
                    summary="Docker could not start.",
                )


class ToolRegistry:
    def __init__(
        self, repository: str | Path, settings: Settings, approved: bool = False
    ):
        self.repository = Path(repository).expanduser().resolve()
        self.settings = settings
        self.approved = approved
        self.sandbox = DockerSandbox(settings)
        self.calls: list[dict[str, str]] = []
        self.specs = {
            "list_files": ToolSpec(
                "list_files",
                Permission.READ,
                "List safe repository files",
                self.list_files,
            ),
            "read_file": ToolSpec(
                "read_file",
                Permission.READ,
                "Read a safe relative file",
                self.read_file,
            ),
            "search_code": ToolSpec(
                "search_code", Permission.READ, "Lexical code search", self.search_code
            ),
            "search_symbol": ToolSpec(
                "search_symbol",
                Permission.READ,
                "Find symbol declarations and usages",
                self.search_symbol,
            ),
            "find_references": ToolSpec(
                "find_references",
                Permission.READ,
                "Find references to a symbol",
                self.search_symbol,
            ),
            "find_dependencies": ToolSpec(
                "find_dependencies",
                Permission.READ,
                "Read import dependencies",
                self.find_dependencies,
            ),
            "get_repository_summary": ToolSpec(
                "get_repository_summary",
                Permission.READ,
                "Read project manifests",
                self.repository_summary,
            ),
            "get_recent_commits": ToolSpec(
                "get_recent_commits",
                Permission.READ,
                "Read recent Git history",
                self.recent_commits,
            ),
            "get_tests": ToolSpec(
                "get_tests", Permission.READ, "Locate test files", self.get_tests
            ),
            "get_git_diff": ToolSpec(
                "get_git_diff", Permission.READ, "Read current Git diff", self.git_diff
            ),
            "apply_patch": ToolSpec(
                "apply_patch",
                Permission.WRITE,
                "Apply validated unified diff",
                self.apply_unified_diff,
            ),
            "run_tests": ToolSpec(
                "run_tests",
                Permission.EXECUTE,
                "Run tests in Docker",
                self.sandbox.run_tests,
            ),
            "create_branch": ToolSpec(
                "create_branch",
                Permission.EXTERNAL,
                "Create a Git branch",
                self.create_branch,
            ),
            "commit_changes": ToolSpec(
                "commit_changes",
                Permission.EXTERNAL,
                "Commit explicitly listed files",
                self.commit_changes,
            ),
            "push_branch": ToolSpec(
                "push_branch",
                Permission.EXTERNAL,
                "Push branch to configured origin",
                self.push_branch,
            ),
        }

    def invoke(self, name: str, **kwargs: Any) -> Any:
        if name not in self.specs:
            raise KeyError(f"unknown tool: {name}")
        spec = self.specs[name]
        if (
            spec.permission in {Permission.WRITE, Permission.EXTERNAL}
            and not self.approved
        ):
            raise ToolPermissionError(f"{name} requires explicit human approval")
        self.calls.append({"tool": name, "permission": spec.permission.value})
        return spec.handler(**kwargs)

    def _path(self, candidate: str) -> Path:
        relative = safe_relative_path(self.repository, candidate)
        path = self.repository / relative
        if any(
            part in {".git", ".hg", ".svn", "node_modules", ".venv", "venv"}
            for part in relative.parts
        ):
            raise ToolPermissionError(
                "protected repository internals cannot be read or modified"
            )
        if is_secret_path(path):
            raise ToolPermissionError(
                "secret-protected files cannot be read or modified"
            )
        return path

    def list_files(self, limit: int = 500) -> list[str]:
        ignored = {".git", "node_modules", ".venv", "venv", "__pycache__"}
        files = [
            path.relative_to(self.repository).as_posix()
            for path in self.repository.rglob("*")
            if path.is_file()
            and not is_secret_path(path)
            and not any(part in ignored for part in path.parts)
        ]
        return sorted(files)[:limit]

    def read_file(self, path: str, max_chars: int = 60_000) -> str:
        target = self._path(path)
        if target.stat().st_size > self.settings.max_file_bytes:
            raise ValueError("file exceeds safe read limit")
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def search_code(self, query: str, limit: int = 40) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        expression = re.compile(re.escape(query), re.IGNORECASE)
        found: list[dict[str, Any]] = []
        for rel in self.list_files():
            try:
                for line_no, line in enumerate(self.read_file(rel).splitlines(), 1):
                    if expression.search(line):
                        found.append({"file": rel, "line": line_no, "text": line[:500]})
                        if len(found) >= limit:
                            return found
            except (OSError, UnicodeError, PermissionError):
                continue
        return found

    def search_symbol(self, symbol: str, limit: int = 40) -> list[dict[str, Any]]:
        expression = re.compile(rf"\b{re.escape(symbol)}\b")
        found: list[dict[str, Any]] = []
        for rel in self.list_files():
            try:
                for line_no, line in enumerate(self.read_file(rel).splitlines(), 1):
                    if expression.search(line):
                        found.append({"file": rel, "line": line_no, "text": line[:500]})
                        if len(found) >= limit:
                            return found
            except (OSError, UnicodeError, PermissionError):
                continue
        return found

    def find_dependencies(self, path: str) -> list[str]:
        text = self.read_file(path)
        return re.findall(
            r"(?:from\s+([\w./-]+)\s+import|import\s+([\w./-]+)|require\(['\"]([^'\"]+))",
            text,
        )

    def repository_summary(self) -> dict[str, str]:
        candidates = [
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "docker-compose.yml",
        ]
        return {
            path: self.read_file(path, 12_000)
            for path in candidates
            if (self.repository / path).exists()
            and not is_secret_path(self.repository / path)
        }

    def _git(self, *args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.repository), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            return (completed.stdout or completed.stderr)[-20_000:]
        except OSError as exc:
            return str(exc)

    def recent_commits(self) -> str:
        return self._git("log", "--oneline", "-10")

    def git_diff(self) -> str:
        return self._git("diff", "--", ".")

    def get_tests(self) -> list[str]:
        return [
            path
            for path in self.list_files()
            if any(token in path.lower() for token in ("test", "spec"))
        ]

    def apply_unified_diff(self, diff: str) -> str:
        """Reject path traversal/secret writes; let Git validate the exact patch before mutation."""
        targets = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
        if not targets or not diff.startswith("diff --git"):
            raise ValueError("only a standard git unified diff is accepted")
        for target in targets:
            self._path(target)
        check = subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "apply",
                "--check",
                "--whitespace=error-all",
                "-",
            ],
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            raise ValueError(f"patch validation failed: {check.stderr[-2000:]}")
        result = subprocess.run(
            ["git", "-C", str(self.repository), "apply", "--whitespace=error-all", "-"],
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"patch application failed: {result.stderr[-2000:]}")
        return "Patch applied after Git validation."

    def create_branch(self, branch: str) -> str:
        if (
            not re.fullmatch(r"[A-Za-z0-9._/-]+", branch)
            or branch.startswith("-")
            or ".." in branch
        ):
            raise ValueError("invalid branch name")
        result = subprocess.run(
            ["git", "-C", str(self.repository), "switch", "-c", branch],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
        return f"Created branch {branch}."

    def commit_changes(self, files: list[str], message: str) -> str:
        if not files or not message.strip() or len(message) > 200:
            raise ValueError("commit requires explicit files and a concise message")
        for file in files:
            self._path(file)
        add = subprocess.run(
            ["git", "-C", str(self.repository), "add", "--", *files],
            capture_output=True,
            text=True,
            check=False,
        )
        if add.returncode != 0:
            raise RuntimeError(add.stderr[-2000:])
        commit = subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-m", message.strip()],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode != 0:
            raise RuntimeError(commit.stderr[-2000:])
        return commit.stdout.strip()

    def push_branch(self, branch: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repository), "push", "-u", "origin", branch],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout.strip() or f"Pushed {branch}."


class GitHubTools:
    """Optional external integration; every operation requires the caller's approval flag."""

    def __init__(self, approved: bool):
        self.approved = approved

    def _client(self):
        if not self.approved:
            raise ToolPermissionError("GitHub actions require explicit approval")
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not configured")
        from github import Github

        return Github(token)

    def get_issue(self, repository: str, number: int) -> dict[str, str]:
        issue = self._client().get_repo(repository).get_issue(number)
        return {"title": issue.title, "body": issue.body or "", "url": issue.html_url}

    def get_repository(self, repository: str) -> dict[str, str | int | None]:
        repo = self._client().get_repo(repository)
        return {
            "name": repo.full_name,
            "description": repo.description,
            "default_branch": repo.default_branch,
            "open_issues": repo.open_issues_count,
        }

    def create_pull_request(
        self, repository: str, title: str, body: str, head: str, base: str = "main"
    ) -> str:
        pull_request = (
            self._client()
            .get_repo(repository)
            .create_pull(title=title, body=body, head=head, base=base)
        )
        return pull_request.html_url
