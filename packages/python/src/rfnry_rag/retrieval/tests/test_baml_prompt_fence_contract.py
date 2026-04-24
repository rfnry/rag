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
    "GenerateAnswer": ["context", "query"],
    "CheckRelevance": ["query", "passage"],
    "SynthesizeDocument": [],  # page_analyses is pipeline-generated
    "GenerateReasoningStep": ["query", "context"],  # prior_reasoning is pipeline-generated
    # ---- retrieval / rewriting ----
    "GenerateHypotheticalDocument": ["query"],
    "GenerateQueryVariants": ["query"],  # num_variants is an int (operator)
    "GenerateStepBackQuery": ["query"],
    # ---- retrieval / retrieval ----
    "AnalyzeQuery": ["query"],
    "RerankChunks": ["query", "passages"],
    "JudgeRetrievalNecessity": ["query"],  # knowledge_description is operator config
    "CompressRetrievedContext": ["query", "passages"],
    # ---- retrieval / evaluation ----
    "JudgeAnswerQuality": ["query", "prediction", "reference"],
    # ---- retrieval / ingestion ----
    "AnalyzePage": [],  # image type — no text injection risk
    "ExtractEntitiesFromText": ["text"],
    # ---- retrieval / tree_indexing ----
    "DetectTableOfContents": ["page_text"],
    "ParseTableOfContents": ["toc_text"],
    "FindSectionStart": ["section_title", "pages_text"],
    "VerifySectionPosition": ["title", "page_text"],
    "ExtractDocumentStructure": ["pages_text"],
    "ContinueDocumentStructure": ["pages_text"],  # existing_structure is pipeline-generated
    "GenerateNodeSummary": ["section_text"],  # title is document-extracted metadata
    "GenerateDocDescription": [],  # tree_structure is pipeline-generated
    # ---- retrieval / tree_search ----
    "TreeRetrievalStep": ["query"],  # tree_structure and accumulated_context are pipeline-generated
    # ---- reasoning / analysis ----
    "AnalyzeText": ["text"],  # instructions is operator-controlled
    "AnalyzeContext": ["messages", "roles"],  # instructions is operator-controlled
    # ---- reasoning / classification ----
    "ClassifyText": ["text"],  # categories is operator config
    "ClassifyTextSets": ["text"],  # category_sets is operator config
    # ---- reasoning / clustering ----
    "LabelCluster": ["samples"],
    # ---- reasoning / compliance ----
    "CheckCompliance": ["text", "reference"],  # dimensions is operator config
    # ---- reasoning / evaluation ----
    "JudgeOutput": ["generated", "reference", "context"],  # dimensions is operator config
}


def _find_all_baml_functions() -> dict[str, tuple[Path, str]]:
    """Return {function_name: (file_path, body)} for every BAML function in the tree."""
    baml_files: list[Path] = []
    for root_name in ("retrieval", "reasoning"):
        root = Path(f"src/rfnry_rag/{root_name}/baml/baml_src")
        if root.exists():
            baml_files.extend(root.rglob("*.baml"))

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
        ======== <ANY_TOKEN> START ========
        {{ param }}
        ======== <ANY_TOKEN> END ========

    The token between the equals-signs is a single \\S+ word (e.g., QUERY,
    CONTEXT, PASSAGES).  Old-style delimiters like ``======== USER QUERY ========``
    do NOT satisfy this contract.

    Returns True if:
      - the param is not interpolated in the body at all (nothing to protect), OR
      - the interpolation site is enclosed by a matching START … END fence pair.
    """
    param_site = re.search(rf"\{{\{{\s*{re.escape(param)}\s*\}}\}}", body)
    if not param_site:
        return True  # param not interpolated in this function body

    before = body[: param_site.start()]
    after = body[param_site.end() :]

    # Locate START and END markers in the appropriate halves.
    start_marker_ends = [m.end() for m in re.finditer(r"======== \S+ START ========", before)]
    end_markers_before = [m.end() for m in re.finditer(r"======== \S+ END ========", before)]
    end_markers_after = [m.start() for m in re.finditer(r"======== \S+ END ========", after)]

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
                violations.append(
                    f"{path.name}::{fn_name}({param}) — missing fence around {{{{ {param} }}}}"
                )

    assert not violations, "BAML fence violations:\n  " + "\n  ".join(violations)
