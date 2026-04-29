"""RaptorRetrieval unit tests.

Verifies the per-method behaviour in isolation: registry consultation, payload
filter shape, ``RetrievedChunk`` mapping, and short-circuit conditions
(no knowledge_id, no active tree). The end-to-end engine wiring + RRF fusion
is covered in ``test_raptor_e2e.py``.

Bias-term hygiene: fixtures use neutral identifiers (``kb-1``, ``topic_a``,
``summary-N``). No domain-specific vocabulary.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.common.models import VectorResult
from rfnry_rag.retrieval.modules.retrieval.methods.raptor import RaptorRetrieval


def _make_embeddings(dim: int = 4) -> Any:
    e = MagicMock()
    e.embed = AsyncMock(return_value=[[1.0] + [0.0] * (dim - 1)])
    return e


def _make_vector_store(results: list[VectorResult] | None = None) -> Any:
    store = MagicMock()
    store.search = AsyncMock(return_value=results if results is not None else [])
    return store


def _make_registry(active_tree_id: str | None) -> Any:
    reg = MagicMock()
    reg.get_active = AsyncMock(return_value=active_tree_id)
    return reg


def _summary_result(idx: int, *, tree_id: str, level: int, cluster_size: int) -> VectorResult:
    return VectorResult(
        point_id=f"summary-{idx}",
        score=0.9 - 0.01 * idx,
        payload={
            "vector_role": "raptor_summary",
            "knowledge_id": "kb-1",
            "raptor_tree_id": tree_id,
            "raptor_level": level,
            "raptor_cluster_size": cluster_size,
            "raptor_parent_id": None,
            "raptor_reasoning": f"reasoning-{idx}",
            "content": f"summary text {idx}",
            "contextualized": f"summary text {idx}",
        },
    )


# ---------------------------------------------------------------------------
# Test 1: registry returns None -> empty list, no vector-store call.
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_returns_empty_when_no_active_tree() -> None:
    """No tree built yet: short-circuit before embedding or searching."""
    store = _make_vector_store()
    embeddings = _make_embeddings()
    registry = _make_registry(active_tree_id=None)

    method = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)
    results = await method.search(query="topic_a query", top_k=5, knowledge_id="kb-1")

    assert results == []
    registry.get_active.assert_awaited_once_with("kb-1")
    embeddings.embed.assert_not_awaited()
    store.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 2: knowledge_id=None -> empty list, no registry call (Q2 per-kid scope).
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_returns_empty_when_knowledge_id_none() -> None:
    """Cross-scope query (no knowledge_id) skips RAPTOR by design."""
    store = _make_vector_store()
    embeddings = _make_embeddings()
    registry = _make_registry(active_tree_id="t-1")

    method = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)
    results = await method.search(query="topic_a query", top_k=5, knowledge_id=None)

    assert results == []
    registry.get_active.assert_not_awaited()
    embeddings.embed.assert_not_awaited()
    store.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test 3: payload filter scopes by raptor_tree_id == active.
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_filters_by_active_tree_id() -> None:
    """The active tree id from the registry lands in the search filter."""
    store = _make_vector_store([_summary_result(0, tree_id="kb-1__abcd1234", level=1, cluster_size=5)])
    registry = _make_registry(active_tree_id="kb-1__abcd1234")

    method = RaptorRetrieval(
        vector_store=store, embeddings=_make_embeddings(), registry=registry
    )
    await method.search(query="q", top_k=5, knowledge_id="kb-1")

    call = store.search.await_args
    assert call.kwargs["filters"]["raptor_tree_id"] == "kb-1__abcd1234"


# ---------------------------------------------------------------------------
# Test 4: payload filter restricts to vector_role="raptor_summary".
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_filters_by_summary_role() -> None:
    """Filter excludes leaf vectors (description/raw_text/table_row)."""
    store = _make_vector_store()
    registry = _make_registry(active_tree_id="kb-1__t1")

    method = RaptorRetrieval(
        vector_store=store, embeddings=_make_embeddings(), registry=registry
    )
    await method.search(query="q", top_k=5, knowledge_id="kb-1")

    call = store.search.await_args
    assert call.kwargs["filters"]["vector_role"] == "raptor_summary"


# ---------------------------------------------------------------------------
# Test 5: payload filter scopes by knowledge_id.
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_filters_by_knowledge_id() -> None:
    """Multi-scope vector store: the active knowledge_id appears in the filter."""
    store = _make_vector_store()
    registry = _make_registry(active_tree_id="kb-2__t1")

    method = RaptorRetrieval(
        vector_store=store, embeddings=_make_embeddings(), registry=registry
    )
    await method.search(query="q", top_k=5, knowledge_id="kb-2")

    call = store.search.await_args
    assert call.kwargs["filters"]["knowledge_id"] == "kb-2"


# ---------------------------------------------------------------------------
# Test 6: VectorResult -> RetrievedChunk shape (content, metadata, score, sentinel).
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_returns_retrieved_chunks_with_payload() -> None:
    """Verify the full RetrievedChunk shape so trace consumers see the keys."""
    raw = _summary_result(0, tree_id="kb-1__t1", level=2, cluster_size=7)
    store = _make_vector_store([raw])
    registry = _make_registry(active_tree_id="kb-1__t1")

    method = RaptorRetrieval(
        vector_store=store, embeddings=_make_embeddings(), registry=registry
    )
    chunks = await method.search(query="q", top_k=5, knowledge_id="kb-1")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "summary-0"
    assert chunk.content == "summary text 0"
    assert chunk.score == raw.score
    # Sentinel — summaries span sources by design; non-None to keep the
    # ``RetrievedChunk.source_id: str`` invariant intact.
    assert chunk.source_id == "raptor:kb-1__t1"
    assert chunk.metadata["raptor_tree_id"] == "kb-1__t1"
    assert chunk.metadata["raptor_level"] == 2
    assert chunk.metadata["raptor_cluster_size"] == 7
    assert chunk.source_metadata["retrieval_type"] == "raptor"


# ---------------------------------------------------------------------------
# Test 7: top_k flows through to the vector-store call.
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_top_k_passed_through() -> None:
    """Caller-provided top_k must reach the vector-store search."""
    store = _make_vector_store()
    registry = _make_registry(active_tree_id="kb-1__t1")

    method = RaptorRetrieval(
        vector_store=store, embeddings=_make_embeddings(), registry=registry
    )
    await method.search(query="q", top_k=42, knowledge_id="kb-1")

    assert store.search.await_args.kwargs["top_k"] == 42


# ---------------------------------------------------------------------------
# Test 8: BaseRetrievalMethod protocol — name / weight / top_k properties.
# ---------------------------------------------------------------------------


def test_raptor_retrieval_protocol_properties() -> None:
    """Mirror VectorRetrieval / DocumentRetrieval / GraphRetrieval shape."""
    method = RaptorRetrieval(
        vector_store=_make_vector_store(),
        embeddings=_make_embeddings(),
        registry=_make_registry(active_tree_id=None),
        weight=2.0,
        top_k=15,
    )
    assert method.name == "raptor"
    assert method.weight == 2.0
    assert method.top_k == 15


# ---------------------------------------------------------------------------
# Test 9: registry exception is contained — RAPTOR returns [] (not raise).
# ---------------------------------------------------------------------------


async def test_raptor_retrieval_returns_empty_when_registry_raises() -> None:
    """Per-method error isolation — registry failures must not break RRF fusion."""
    store = _make_vector_store()
    embeddings = _make_embeddings()
    registry = MagicMock()
    registry.get_active = AsyncMock(side_effect=RuntimeError("db down"))

    method = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)
    results = await method.search(query="q", top_k=5, knowledge_id="kb-1")

    assert results == []
    embeddings.embed.assert_not_awaited()
    store.search.assert_not_awaited()
