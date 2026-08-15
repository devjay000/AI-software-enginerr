"""Safe, structural code ingestion: discovery → AST/symbols → semantic code chunks."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .models import CodeChunk, RepositoryAnalysis
from .security import contains_secret, is_indexable

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


@dataclass
class IndexResult:
    analysis: RepositoryAnalysis
    chunks: list[CodeChunk]
    dependencies: dict[str, list[str]]
    excluded_files: list[str] = field(default_factory=list)


def language_for(path: Path) -> str:
    return LANGUAGES.get(path.suffix.lower(), "text")


def _line_slice(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[max(0, start - 1) : end]).strip()


class PythonAstExtractor:
    def extract(self, file: str, text: str) -> tuple[list[CodeChunk], list[str]]:
        lines = text.splitlines(keepends=True)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [], []
        imports: list[str] = []
        chunks: list[CodeChunk] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                imports.extend(
                    f"{prefix}.{alias.name}".strip(".") for alias in node.names
                )

        def visit(nodes: list[ast.stmt], parent: str | None = None) -> None:
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{parent}.{node.name}" if parent else node.name
                    is_test = node.name.startswith("test_") or "/test" in file.lower()
                    chunks.append(
                        CodeChunk(
                            file=file,
                            symbol=name,
                            kind=(
                                "test"
                                if is_test
                                else "method" if parent else "function"
                            ),
                            language="python",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            content=_line_slice(
                                lines, node.lineno, node.end_lineno or node.lineno
                            ),
                            imports=imports.copy(),
                            exports=(
                                [node.name] if not node.name.startswith("_") else []
                            ),
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    chunks.append(
                        CodeChunk(
                            file=file,
                            symbol=node.name,
                            kind="class",
                            language="python",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            content=_line_slice(
                                lines, node.lineno, node.end_lineno or node.lineno
                            ),
                            imports=imports.copy(),
                            exports=[node.name],
                        )
                    )
                    visit(node.body, node.name)

        visit(tree.body)
        if not chunks and text.strip():
            chunks.append(
                CodeChunk(
                    file=file,
                    symbol=Path(file).stem,
                    kind="module",
                    language="python",
                    start_line=1,
                    end_line=max(1, len(lines)),
                    content=text.strip(),
                    imports=imports,
                )
            )
        return chunks, imports


class TreeSitterExtractor:
    """Uses tree-sitter when installed; a conservative syntax fallback preserves portability."""

    declaration = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var)\s+([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    )
    import_pattern = re.compile(
        r"(?:from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))"
    )

    def extract(
        self, file: str, text: str, language: str
    ) -> tuple[list[CodeChunk], list[str]]:
        imports = [
            match.group(1) or match.group(2)
            for match in self.import_pattern.finditer(text)
        ]
        parsed = self._tree_sitter_chunks(file, text, language, imports)
        if parsed:
            return parsed, imports
        # The fallback supports constrained/offline environments where tree-sitter language bindings are absent
        # or use an incompatible ABI. It preserves symbol-aware boundaries, never arbitrary token windows.
        return self._regex_chunks(file, text, language, imports), imports

    def _tree_sitter_chunks(
        self, file: str, text: str, language: str, imports: list[str]
    ) -> list[CodeChunk]:
        try:
            from tree_sitter import Language, Parser

            if language == "typescript":
                import tree_sitter_typescript

                compiled_language = tree_sitter_typescript.language_typescript()
            else:
                import tree_sitter_javascript

                compiled_language = tree_sitter_javascript.language()
            parser = Parser(Language(compiled_language))
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            return []
        node_types = {
            "function_declaration": "function",
            "generator_function_declaration": "function",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "method_definition": "method",
            "public_field_definition": "method",
        }
        chunks: list[CodeChunk] = []
        stack = [tree.root_node]
        raw = text.encode("utf-8")
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.children))
            if node.type not in node_types:
                continue
            name = node.child_by_field_name("name")
            if name is None:
                continue
            symbol = raw[name.start_byte : name.end_byte].decode(
                "utf-8", errors="replace"
            )
            kind = (
                "test"
                if ".test." in file or ".spec." in file
                else node_types[node.type]
            )
            chunks.append(
                CodeChunk(
                    file=file,
                    symbol=symbol,
                    kind=kind,
                    language=language,
                    start_line=node.start_point[0] + 1,
                    end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    content=raw[node.start_byte : node.end_byte]
                    .decode("utf-8", errors="replace")
                    .strip(),
                    imports=imports,
                    exports=(
                        [symbol]
                        if node.parent and node.parent.type == "export_statement"
                        else []
                    ),
                )
            )
        return chunks

    def _regex_chunks(
        self, file: str, text: str, language: str, imports: list[str]
    ) -> list[CodeChunk]:
        lines = text.splitlines(keepends=True)
        matches = list(self.declaration.finditer(text))
        chunks: list[CodeChunk] = []
        for index, match in enumerate(matches):
            start = text[: match.start()].count("\n") + 1
            end = (
                text[: matches[index + 1].start()].count("\n")
                if index + 1 < len(matches)
                else len(lines)
            )
            declaration = match.group(0)
            kind = (
                "class"
                if "class" in declaration
                else "interface" if "interface" in declaration else "function"
            )
            if ".test." in file or ".spec." in file:
                kind = "test"
            chunks.append(
                CodeChunk(
                    file=file,
                    symbol=match.group(1),
                    kind=kind,
                    language=language,
                    start_line=start,
                    end_line=max(start, end),
                    content=_line_slice(lines, start, max(start, end)),
                    imports=imports,
                    exports=[match.group(1)] if "export" in declaration else [],
                )
            )
        if not chunks and text.strip():
            chunks.append(
                CodeChunk(
                    file=file,
                    symbol=Path(file).stem,
                    kind="module",
                    language=language,
                    start_line=1,
                    end_line=max(1, len(lines)),
                    content=text.strip(),
                    imports=imports,
                )
            )
        return chunks


class CodeIndexer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.python = PythonAstExtractor()
        self.tree_sitter = TreeSitterExtractor()

    def discover(self, repository: Path) -> tuple[list[Path], list[str]]:
        files: list[Path] = []
        excluded: list[str] = []
        for path in repository.rglob("*"):
            try:
                if is_indexable(path, self.settings.max_file_bytes):
                    files.append(path)
                elif path.is_file():
                    excluded.append(path.relative_to(repository).as_posix())
            except OSError:
                continue
        return files, excluded

    def index(self, repository: str | Path) -> IndexResult:
        root = Path(repository).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"repository does not exist: {root}")
        files, excluded = self.discover(root)
        chunks: list[CodeChunk] = []
        dependencies: dict[str, list[str]] = {}
        counts: Counter[str] = Counter()
        for path in files:
            rel = path.relative_to(root).as_posix()
            language = language_for(path)
            counts[language] += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                excluded.append(rel)
                continue
            # Source with probable secrets is never embedded, retrieved, or sent to the LLM.
            if contains_secret(text):
                excluded.append(rel)
                continue
            if language == "python":
                file_chunks, imports = self.python.extract(rel, text)
            elif language in {"javascript", "typescript"}:
                file_chunks, imports = self.tree_sitter.extract(rel, text, language)
            elif language in {"json", "yaml", "toml"}:
                file_chunks, imports = [
                    CodeChunk(
                        file=rel,
                        symbol=path.name,
                        kind="config",
                        language=language,
                        start_line=1,
                        end_line=max(1, len(text.splitlines())),
                        content=text,
                        metadata={"sha256": self._hash(text)},
                    )
                ], []
            else:
                file_chunks, imports = [], []
            for chunk in file_chunks:
                chunk.metadata["sha256"] = self._hash(chunk.content)
            chunks.extend(file_chunks)
            dependencies[rel] = imports
        return IndexResult(
            analysis=self.analyze(root, files, counts, chunks),
            chunks=chunks,
            dependencies=dependencies,
            excluded_files=excluded,
        )

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()

    def analyze(
        self,
        root: Path,
        files: list[Path],
        counts: Counter[str],
        chunks: list[CodeChunk],
    ) -> RepositoryAnalysis:
        relative = [path.relative_to(root).as_posix() for path in files]
        lower = {item.lower() for item in relative}
        names = " ".join(lower)
        frameworks: list[str] = []
        if "streamlit" in names:
            frameworks.append("Streamlit")
        if "django" in names or "manage.py" in lower:
            frameworks.append("Django")
        if "fastapi" in names:
            frameworks.append("FastAPI")
        if "react" in names or any(item.endswith(".tsx") for item in lower):
            frameworks.append("React")
        if "next.config" in names:
            frameworks.append("Next.js")
        tests = []
        if any("pytest" in item for item in lower):
            tests.append("pytest")
        if any("unittest" in item for item in lower):
            tests.append("unittest")
        if any("jest" in item for item in lower):
            tests.append("Jest")
        if any("vitest" in item for item in lower):
            tests.append("Vitest")
        entry_points = [
            item
            for item in relative
            if Path(item).name
            in {"main.py", "app.py", "manage.py", "index.ts", "index.js", "server.py"}
        ]
        important = sorted(
            {
                chunk.file
                for chunk in chunks
                if chunk.kind in {"class", "route", "module"}
            }
        )[:20]
        config = [
            item
            for item in relative
            if Path(item).name
            in {
                "pyproject.toml",
                "package.json",
                "docker-compose.yml",
                "Dockerfile",
                "requirements.txt",
            }
        ]
        database = [
            item
            for item in relative
            if any(
                token in item.lower()
                for token in ("database", "models", "migrations", "repository", "orm")
            )
        ][:15]
        api = [
            item
            for item in relative
            if any(
                token in item.lower()
                for token in ("api", "routes", "views", "controller", "handler")
            )
        ][:15]
        summary = (
            f"{len(relative)} safe files, {len(chunks)} structural chunks; "
            f"languages: {dict(counts)}; frameworks: {', '.join(frameworks) or 'not detected'}."
        )
        return RepositoryAnalysis(
            root=str(root),
            languages=dict(counts),
            frameworks=frameworks,
            entry_points=entry_points,
            test_frameworks=tests,
            important_modules=important,
            configuration_files=config,
            database_layer=database,
            api_layer=api,
            summary=summary,
        )
