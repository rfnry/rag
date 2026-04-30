"""Contract test: every user-controlled BAML prompt parameter must be fenced.

When you add a new BAML function, update USER_CONTROLLED_PARAMS below. The
test fails on any function that appears in BAML source but is NOT in
USER_CONTROLLED_PARAMS — force explicit classification on every new prompt.
"""

import re
from pathlib import Path

# For each BAML function: list USER-CONTROLLED parameters that require fence
# wrapping. An empty list means "no user-controlled parameters" (all params
# are operator-controlled and safe to interpolate directly).
#
# Fence format used throughout the codebase:
#   ======== <PARAM_UPPER> START ========
#   {{ param }}
#   ======== <PARAM_UPPER> END ========
# Preceded by an "untrusted" instruction.
#
# Classification rules (err on the side of USER-CONTROLLED):
#   USER-CONTROLLED  — flows from a public SDK API arg without transformation
#                      (e.g., query, passage, context, text, samples)
#   OPERATOR-CONTROLLED — flows from a config dataclass the SDK user sets;
#                         not end-user text (e.g., system_prompt, num_variants,
#                         knowledge_description, category names)
#
# Tree-indexing functions receive operator-loaded document content, not
# end-user query strings, but document text can still contain injection
# payloads — classify as user-controlled and fence them.
USER_CONTROLLED_PARAMS: dict[str, list[str]] = {
    # ---- retrieval / generation ----
    "CheckRelevance": ["query", "passage"],
    "SynthesizeDocument": [],
    # ---- retrieval / evaluation ----
    "JudgeAnswerQuality": ["query", "prediction", "reference"],
    # ---- retrieval / ingestion ----
    "AnalyzePage": [],
    "ExtractEntitiesFromText": ["text"],
    "GenerateSyntheticQueries": ["passage", "num_queries"],
    "AnalyzeDrawingPage": ["symbol_library", "off_page_patterns"],
}


def _find_all_baml_functions() -> dict[str, tuple[Path, str]]:
    """Return {function_name: (file_path, body)} for every BAML function in the tree."""
    root = Path("src/rfnry_rag/baml/baml_src")
    baml_files = list(root.rglob("*.baml")) if root.exists() else []

    fn_re = re.compile(r"function\s+(\w+)\s*\([^)]*\)\s*->[^{]*\{", re.DOTALL)
    out: dict[str, tuple[Path, str]] = {}
    for f in sorted(baml_files):
        src = f.read_text()
        for m in fn_re.finditer(src):
            fn_name = m.group(1)
            # Extract the function body — brace-match from the opening { after the signature.
            start = m.end() - 1  # position of the {
            depth = 0
            i = start
            while i < len(src):
                if src[i] == "{":
                    depth += 1
                elif src[i] == "}":
                    depth -= 1
                    if depth == 0:
                        out[fn_name] = (f, src[start : i + 1])
                        break
                i += 1
    return out


def _has_fence_for_param(body: str, param: str) -> bool:
    """A user-controlled param must be enclosed between START/END fence markers.

    The fence convention (established in rounds 2-3):
        ======== <TAG> START ========
        {{ param }}
        ======== <TAG> END ========

    The tag between the equals-signs is one or more whitespace-separated tokens
    (e.g., QUERY, CONTEXT, PASSAGES, SYMBOL LIBRARY, OFF PAGE CONNECTOR PATTERNS).
    Old-style delimiters like ``======== USER QUERY ========`` (no START/END sentinel)
    do NOT satisfy this contract.

    Two interpolation patterns are recognised:

      1. Direct interpolation — ``{{ param }}`` appears in the body, fenced by a
         single START … END pair. Used for ``string`` params.

      2. Jinja loop — ``{% for X in param %}`` iterates a ``string[]`` param,
         and the loop body fences the loop variable ``X`` with multi-tag
         markers like ``MEMBER_N START`` / ``MEMBER_N END`` (the ``N`` resolves
         to the iteration index). Each iteration produces one START/END pair,
         so the contract holds at the per-iteration grain rather than once
         around the array.

    Returns True if:
      - the param is not interpolated in the body at all (nothing to protect), OR
      - the direct interpolation site is enclosed by a matching START … END pair, OR
      - the param is iterated via ``{% for X in param %}`` and the loop variable
        ``X`` is fenced inside the loop body by START/END markers whose tag
        contains a ``\\d+`` placeholder (e.g. ``MEMBER_N``).
    """
    # Pattern 2: Jinja-loop iteration. Detect this first because a
    # string[] param won't have a direct ``{{ param }}`` site.
    loop_match = re.search(
        rf"\{{%\s*for\s+(\w+)\s+in\s+{re.escape(param)}\s*%\}}",
        body,
    )
    if loop_match is not None:
        loop_var = loop_match.group(1)
        loop_start = loop_match.end()
        # Find the matching ``{% endfor %}``; we don't validate nested loops
        # here (no current consumer needs them).
        endfor_match = re.search(r"\{%\s*endfor\s*%\}", body[loop_start:])
        if endfor_match is None:
            return False
        loop_body = body[loop_start : loop_start + endfor_match.start()]
        # Loop variable must be interpolated inside START/END markers whose tag
        # carries a numeric placeholder (e.g. ``MEMBER_{{ loop.index }} START``).
        var_site = re.search(rf"\{{\{{\s*{re.escape(loop_var)}\s*\}}\}}", loop_body)
        if var_site is None:
            return False
        # Look for START with Jinja-templated numeric tag before the var, and
        # matching END after.
        start_marker = re.search(
            r"========\s+\S*\{\{\s*loop\.index\s*\}\}\S*\s+START\s+========",
            loop_body[: var_site.start()],
        )
        end_marker = re.search(
            r"========\s+\S*\{\{\s*loop\.index\s*\}\}\S*\s+END\s+========",
            loop_body[var_site.end() :],
        )
        return start_marker is not None and end_marker is not None

    # Pattern 1: direct interpolation.
    param_site = re.search(rf"\{{\{{\s*{re.escape(param)}\s*\}}\}}", body)
    if not param_site:
        return True  # param not interpolated in this function body

    before = body[: param_site.start()]
    after = body[param_site.end() :]

    # Locate START and END markers in the appropriate halves.
    start_marker_ends = [m.end() for m in re.finditer(r"======== \S+(?: \S+)* START ========", before)]
    end_markers_before = [m.end() for m in re.finditer(r"======== \S+(?: \S+)* END ========", before)]
    end_markers_after = [m.start() for m in re.finditer(r"======== \S+(?: \S+)* END ========", after)]

    if not start_marker_ends or not end_markers_after:
        return False

    # The most recent START before the interpolation must come AFTER the most
    # recent END before the interpolation (i.e., no unclosed END between the
    # START and the param).
    return not (end_markers_before and end_markers_before[-1] > start_marker_ends[-1])


def test_all_user_controlled_baml_params_are_fenced() -> None:
    fns = _find_all_baml_functions()
    assert fns, "no BAML functions found — update path globs"

    violations: list[str] = []
    for fn_name, (path, body) in sorted(fns.items()):
        if fn_name not in USER_CONTROLLED_PARAMS:
            violations.append(
                f"{path.name}::{fn_name} — not classified in USER_CONTROLLED_PARAMS. "
                "Add the function with its user-controlled params (or empty list)."
            )
            continue
        for param in USER_CONTROLLED_PARAMS[fn_name]:
            if not _has_fence_for_param(body, param):
                violations.append(f"{path.name}::{fn_name}({param}) — missing fence around {{{{ {param} }}}}")

    assert not violations, "BAML fence violations:\n  " + "\n  ".join(violations)
