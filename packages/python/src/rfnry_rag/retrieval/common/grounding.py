"""Shared grounding-signal helpers.

`is_weak_chunk_signal(chunks, threshold)` returns ``True`` when the
retrieved-chunk pool's max score sits strict-below ``threshold`` — i.e.
the signal is too weak to ground a confident answer. Boundary is ``<``
not ``<=``: a chunk score equal to the threshold is grounded.

This helper is the single source of truth for the "are these chunks
weak?" check across two engine arms:

- R5.3's confidence-expansion retry loop in
  ``RagEngine._query_via_retrieval`` (escalates RETRIEVAL → DIRECT when
  expansion exhausts).
- R6.3's post-loop iterative-escalation check in
  ``RagEngine._query_via_iterative`` (escalates ITERATIVE → DIRECT when
  the multi-hop run finishes with weak accumulated chunks).

Lifting it to ``retrieval/common`` (vs duplicating per arm) keeps the
boundary semantics identical at both call sites — a divergence in the
``<`` vs ``<=`` convention here would be a real-world correctness bug
that R-series review history has flagged before.

Match ``GenerationConfig.grounding_threshold``'s existing semantics:
``grounding_threshold`` gates "is this answer grounded?", and a chunk
score equal to the threshold is grounded, not weak.
"""

from __future__ import annotations

from rfnry_rag.retrieval.common.models import RetrievedChunk


def max_chunk_score(chunks: list[RetrievedChunk]) -> float | None:
    """Max ``score`` across chunks; ``None`` for empty input.

    Returning ``None`` (not ``0.0``) for the empty case keeps "no chunks"
    distinct from "chunks but zero score" — both are weak by the
    threshold check, but the distinction surfaces in trace logs.
    """
    if not chunks:
        return None
    return max(c.score for c in chunks)


def is_weak_chunk_signal(chunks: list[RetrievedChunk], threshold: float) -> bool:
    """True when retrieval signal is weak — empty OR strict-below threshold."""
    score = max_chunk_score(chunks)
    if score is None:
        return True
    return score < threshold
