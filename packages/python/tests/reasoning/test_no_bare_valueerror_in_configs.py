"""Contract: reasoning config validators must raise ReasoningInputError, not
bare ValueError. Typed errors enable precise ``except`` clauses for the SDK.

This test walks every ``models.py`` under ``src/rfnry_rag/reasoning/`` and
uses the AST to confirm that no ``raise ValueError(...)`` call exists.  Any
such call is a violation — replace it with ``raise ReasoningInputError(...)``
(which IS a ValueError via MRO, so back-compat is preserved).
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_no_bare_value_error_raised_in_reasoning_configs() -> None:
    violations: list[str] = []
    for p in Path("src/rfnry_rag/reasoning").rglob("models.py"):
        src = p.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                call = node.exc if isinstance(node.exc, ast.Call) else None
                name = call.func.id if call and isinstance(call.func, ast.Name) else None
                if name == "ValueError":
                    violations.append(f"{p}:{node.lineno} — bare ValueError raised")
    assert not violations, (
        "reasoning config raises bare ValueError:\n  "
        + "\n  ".join(violations)
        + "\n\nReplace each with ReasoningInputError (IS-A ValueError via MRO)."
    )
