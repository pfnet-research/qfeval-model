"""Smoke-run every example script and check en/ja parity.

Each script runs twice and must produce bit-identical stdout —
determinism is a core promise of the library and the examples are its
user-facing demonstration. Structural and executable-AST parity checks catch
translation drift (one language's examples going stale).
"""

from __future__ import annotations

import ast
import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_FILES = sorted(
    path for lang in ("en", "ja") for path in (_EXAMPLES / lang).rglob("*.py")
)


def _run(path: Path) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        runpy.run_path(str(path), run_name="__main__")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "path", _FILES, ids=[str(path.relative_to(_EXAMPLES)) for path in _FILES]
)
def test_example_runs_and_is_deterministic(path: Path) -> None:
    # Two runs: no exception, and bit-identical stdout (determinism is the
    # library's core promise and the examples are its user-facing demo).
    assert _run(path) == _run(path)


def _cell_count(path: Path) -> int:
    return sum(
        line.startswith("# %%")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_en_ja_structural_parity() -> None:
    en = {p.relative_to(_EXAMPLES / "en"): p for p in (_EXAMPLES / "en").rglob("*.py")}
    ja = {p.relative_to(_EXAMPLES / "ja"): p for p in (_EXAMPLES / "ja").rglob("*.py")}
    assert sorted(en) == sorted(ja)
    for rel, en_path in en.items():
        assert _cell_count(en_path) == _cell_count(ja[rel]), str(rel)


def _translated_text(node: ast.expr) -> ast.expr:
    """Normalize only prose in a print/exception expression.

    Formatted values remain untouched, so translating an f-string's label is
    allowed while changing the expression interpolated into it is not.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = "<translated text>"
    elif isinstance(node, ast.JoinedStr):
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                value.value = "<translated text>"
    return node


class _NormalizeTranslatedProse(ast.NodeTransformer):
    """Remove prose that intentionally differs between bilingual examples."""

    @staticmethod
    def _strip_docstring(node: Any) -> None:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            node.args = [_translated_text(arg) for arg in node.args]
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.Raise:
        self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            node.exc.args = [_translated_text(arg) for arg in node.exc.args]
        return node


def _normalized_code(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = _NormalizeTranslatedProse().visit(tree)
    return ast.dump(normalized, include_attributes=False)


def test_en_ja_executable_code_matches() -> None:
    en = {p.relative_to(_EXAMPLES / "en"): p for p in (_EXAMPLES / "en").rglob("*.py")}
    for rel, en_path in en.items():
        ja_path = _EXAMPLES / "ja" / rel
        assert _normalized_code(en_path) == _normalized_code(ja_path), str(rel)
