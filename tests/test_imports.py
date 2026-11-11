from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN_ROOTS = ("web", "scripts", "tests", "eval")


def test_core_modules_do_not_import_outer_layers() -> None:
    violations: list[str] = []
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = _imported_modules(tree)
        for name in imported:
            root = name.split(".")[0]
            if root in FORBIDDEN_ROOTS:
                violations.append(f"{path.name} imports {name}")
            if path.name.startswith("lex_") and name == "solution":
                violations.append(f"{path.name} imports solution")
    assert violations == []


def test_lex_modules_do_not_read_environ() -> None:
    hits: list[str] = []
    for path in sorted(SRC.glob("lex_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                hits.append(path.name)
    assert hits == []


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names
