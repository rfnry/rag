"""R2.3 — RAPTOR end-to-end: build → query → verify summaries flow through fusion.

Closes the loop opened by R2.1 (config + registry) and R2.2 (builder + engine
API). These tests validate the integration: ``RaptorTreeBuilder.build`` writes
summary vectors into the same vector store that ``RaptorRetrieval`` searches at
query time, ``RetrievalService`` iterates the namespace including RAPTOR, and
RRF fusion includes RAPTOR's contribution.

Bias-term hygiene: fixtures use neutral identifiers (``kb-1``, ``topic_a``).
No domain-specific vocabulary.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rfnry_rag.retrieval.common.language_model import (
    LanguageModelClient,
    LanguageModelProvider,
)
from rfnry_rag.retrieval.common.models import RetrievedChunk, VectorPoint, VectorResult
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder import RaptorTreeBuilder
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import RaptorConfig
from rfnry_rag.retrieval.modules.namespace import MethodNamespace
from rfnry_rag.retrieval.modules.retrieval.methods.raptor import RaptorRetrieval
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService
from rfnry_rag.retrieval.server import (
    IngestionConfig,
    PersistenceConfig,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


_SUMMARIZE_PATH = (
    "rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder.b.SummarizeCluster"
)
_RUN_CLUSTERING_PATH = (
    "rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder.run_clustering"
)


class _FakeVectorStore:
    """In-memory vector store sufficient for build + retrieval round-trips.

    Stores written points by id; supports the small subset of the
    ``BaseVectorStore`` API the builder + ``RaptorRetrieval`` exercise:
    ``upsert`` / ``scroll`` / ``set_payload`` / ``delete`` / ``search``.
    """

    def __init__(self, leaves: list[VectorResult]) -> None:
        # Key by point_id; payloads are mutable so set_payload reflects.
        self._points: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}
        for leaf in leaves:
            self._points[leaf.point_id] = dict(leaf.payload)
            # Synthetic leaf vectors aren't surfaced by scroll, but the
            # builder re-embeds leaf text, so leaving these empty is fine.
            self._vectors[leaf.point_id] = []

    async def initialize(self, vector_size: int) -> None:
        return None

    async def upsert(self, points: list[VectorPoint]) -> None:
        for p in points:
            self._points[p.point_id] = dict(p.payload)
            self._vectors[p.point_id] = list(p.vector)

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[VectorResult], str | None]:
        out: list[VectorResult] = []
        for pid, payload in self._points.items():
            if not _matches(filters, payload):
                continue
            out.append(VectorResult(point_id=pid, score=0.0, payload=dict(payload)))
        return out, None

    async def set_payload(self, point_ids: list[str], payload: dict[str, Any]) -> None:
        for pid in point_ids:
            if pid in self._points:
                self._points[pid].update(payload)

    async def delete(self, filters: dict[str, Any]) -> int:
        to_drop = [pid for pid, p in self._points.items() if _matches(filters, p)]
        for pid in to_drop:
            self._points.pop(pid, None)
            self._vectors.pop(pid, None)
        return len(to_drop)

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        # Score by membership-rank; we don't need similarity correctness for
        # the e2e plumbing tests — we need to confirm that filtered points
        # come back at all so RAPTOR contributes to the fusion pool.
        candidates = [
            VectorResult(point_id=pid, score=1.0 - 0.01 * idx, payload=dict(payload))
            for idx, (pid, payload) in enumerate(self._points.items())
            if _matches(filters, payload)
        ]
        return candidates[:top_k]

    async def hybrid_search(
        self,
        vector: list[float],
        sparse_vector: Any,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorResult]:
        return await self.search(vector=vector, top_k=top_k, filters=filters)

    async def retrieve(self, point_ids: list[str]) -> list[VectorResult]:
        return [
            VectorResult(point_id=pid, score=0.0, payload=dict(self._points[pid]))
            for pid in point_ids
            if pid in self._points
        ]

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        return sum(1 for p in self._points.values() if _matches(filters, p))

    async def shutdown(self) -> None:
        return None


def _matches(filters: dict[str, Any] | None, payload: dict[str, Any]) -> bool:
    """Replicate Qdrant's MatchValue / MatchAny filter semantics."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = payload.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def _make_leaves(knowledge_id: str, count: int = 12) -> list[VectorResult]:
    return [
        VectorResult(
            point_id=f"leaf-{i}",
            score=0.0,
            payload={
                "vector_role": "raw_text",
                "knowledge_id": knowledge_id,
                "contextualized": f"leaf chunk {i} text content",
                "content": f"leaf chunk {i} text content",
                "source_id": f"src-{i % 3}",
            },
        )
        for i in range(count)
    ]


def _make_drawing_leaves(knowledge_id: str, count: int = 4) -> list[VectorResult]:
    """Drawing-component leaves — must NOT be summarised by RAPTOR."""
    return [
        VectorResult(
            point_id=f"draw-{i}",
            score=0.0,
            payload={
                "vector_role": "drawing_component",
                "knowledge_id": knowledge_id,
                "contextualized": f"drawing component tag {i}",
                "content": f"drawing component tag {i}",
                "source_id": f"draw-src-{i}",
            },
        )
        for i in range(count)
    ]


def _make_embeddings(dim: int = 8) -> Any:
    counter = {"i": 0}

    async def _embed(texts: list[str]) -> list[list[float]]:
        out = []
        for _ in texts:
            counter["i"] += 1
            base = counter["i"] % 3
            out.append([float(base) + 0.01 * counter["i"]] + [0.0] * (dim - 1))
        return out

    e = MagicMock()
    e.embed = AsyncMock(side_effect=_embed)
    e.embedding_dimension = AsyncMock(return_value=dim)
    return e


def _make_in_memory_registry() -> Any:
    """Async-correct registry stand-in (avoids hauling an SQLAlchemy engine)."""
    state: dict[str, str] = {}

    async def _get_active(knowledge_id: str) -> str | None:
        return state.get(knowledge_id)

    async def _set_active(
        knowledge_id: str,
        tree_id: str,
        level_counts: list[int],
        cost_usd: float | None,
    ) -> None:
        state[knowledge_id] = tree_id

    reg = MagicMock()
    reg.get_active = AsyncMock(side_effect=_get_active)
    reg.set_active = AsyncMock(side_effect=_set_active)
    reg.delete_record = AsyncMock(return_value=None)
    return reg


def _summarize_result(idx: int) -> SimpleNamespace:
    return SimpleNamespace(summary=f"synthesised summary {idx}", reasoning=f"ok-{idx}")


def _kmeans_round_robin(matrix: Any, cfg: Any) -> Any:
    n = matrix.shape[0]
    k = cfg.n_clusters
    labels = np.array([i % k for i in range(n)], dtype=np.int32)
    centroids = np.zeros((k, matrix.shape[1]), dtype=np.float32)
    return labels, centroids


# ---------------------------------------------------------------------------
# Test 1 (e2e #8): build a tree, then query — RAPTOR returns summary chunks.
# ---------------------------------------------------------------------------


async def test_e2e_build_then_query_returns_summary_chunks() -> None:
    knowledge_id = "kb-1"
    store = _FakeVectorStore(_make_leaves(knowledge_id, count=12))
    embeddings = _make_embeddings()
    registry = _make_in_memory_registry()

    cfg = RaptorConfig(
        enabled=True,
        cluster_algorithm="kmeans",
        clusters_per_level=3,
        max_levels=1,
        summary_max_tokens=128,
        summary_model=_lm_client(),
    )
    builder = RaptorTreeBuilder(
        config=cfg,
        vector_store=store,
        embeddings=embeddings,
        registry=registry,
        knowledge_manager=MagicMock(),
    )
    builder._baml_registry = MagicMock()  # noqa: SLF001

    summarize_calls = {"n": 0}

    async def _summarize(**kwargs: Any) -> SimpleNamespace:
        idx = summarize_calls["n"]
        summarize_calls["n"] += 1
        return _summarize_result(idx)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans_round_robin),
        patch(_SUMMARIZE_PATH, new=AsyncMock(side_effect=_summarize)),
    ):
        report = await builder.build(knowledge_id)

    # Build produced 3 summaries and swapped active.
    assert report.level_counts == [12, 3]
    assert (await registry.get_active(knowledge_id)) is not None

    # Now query via RaptorRetrieval — same store, same registry.
    method = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)
    chunks = await method.search(query="topic_a", top_k=5, knowledge_id=knowledge_id)

    assert len(chunks) > 0
    # All returned chunks are summaries with the active tree id.
    active_tree = await registry.get_active(knowledge_id)
    for chunk in chunks:
        assert chunk.metadata["raptor_tree_id"] == active_tree
        assert chunk.metadata["raptor_level"] == 1
        assert chunk.source_id == f"raptor:{active_tree}"


# ---------------------------------------------------------------------------
# Test 2 (e2e #9): RAPTOR participates in RRF fusion via RetrievalService trace.
# ---------------------------------------------------------------------------


async def test_e2e_raptor_chunks_participate_in_rrf_fusion() -> None:
    """Trace's per_method_results includes the 'raptor' key alongside others."""
    knowledge_id = "kb-1"
    store = _FakeVectorStore(_make_leaves(knowledge_id, count=12))
    embeddings = _make_embeddings()
    registry = _make_in_memory_registry()

    cfg = RaptorConfig(
        enabled=True,
        cluster_algorithm="kmeans",
        clusters_per_level=3,
        max_levels=1,
        summary_model=_lm_client(),
    )
    builder = RaptorTreeBuilder(
        config=cfg,
        vector_store=store,
        embeddings=embeddings,
        registry=registry,
        knowledge_manager=MagicMock(),
    )
    builder._baml_registry = MagicMock()  # noqa: SLF001

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans_round_robin),
        patch(
            _SUMMARIZE_PATH,
            new=AsyncMock(side_effect=lambda **kw: _summarize_result(0)),
        ),
    ):
        await builder.build(knowledge_id)

    # Construct a RetrievalService with RaptorRetrieval + a sibling vector
    # method so RRF has multiple inputs to fuse.
    raptor = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)

    sibling_chunk = RetrievedChunk(
        chunk_id="vec-1",
        source_id="src-0",
        content="topic_a leaf hit",
        score=0.9,
    )
    vector_method = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(return_value=[sibling_chunk]),
    )

    service = RetrievalService(retrieval_methods=[vector_method, raptor], top_k=5)
    fused, trace = await service.retrieve(
        query="topic_a", knowledge_id=knowledge_id, trace=True
    )

    assert trace is not None
    assert "raptor" in trace.per_method_results, (
        "RAPTOR contribution must appear in trace.per_method_results so "
        "operators can attribute fused output back to the RAPTOR arm"
    )
    assert "vector" in trace.per_method_results
    # RAPTOR produced at least one summary chunk in the trace.
    assert len(trace.per_method_results["raptor"]) > 0
    # And those chunks made it through fusion.
    raptor_ids = {c.chunk_id for c in trace.per_method_results["raptor"]}
    fused_ids = {c.chunk_id for c in fused}
    assert raptor_ids & fused_ids, "expected at least one RAPTOR chunk in fused output"


# ---------------------------------------------------------------------------
# Test 3 (e2e #10): raptor.enabled=False → method NOT registered.
# ---------------------------------------------------------------------------


async def test_e2e_raptor_disabled_when_method_not_registered() -> None:
    """Default-off: engine namespace contains no 'raptor' entry; queries unchanged."""
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()

    cfg = RagServerConfig(
        persistence=PersistenceConfig(document_store=document_store),
        ingestion=IngestionConfig(),  # raptor defaults enabled=False
        retrieval=RetrievalConfig(),
    )
    engine = RagEngine(cfg)
    await engine.initialize()
    try:
        # The retrieval namespace MUST NOT contain raptor when disabled.
        assert "raptor" not in engine.retrieval
        # And the eager-construction path was also skipped.
        assert engine._raptor_registry is None  # type: ignore[attr-defined]
    finally:
        await engine.shutdown()


async def test_e2e_raptor_enabled_method_registered_in_namespace() -> None:
    """Mirror of the above: raptor.enabled=True wires RaptorRetrieval into the namespace."""
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

    metadata_store = SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///:memory:")
    vector_store = _FakeVectorStore(leaves=[])

    cfg = RagServerConfig(
        persistence=PersistenceConfig(
            metadata_store=metadata_store,
            vector_store=vector_store,
        ),
        ingestion=IngestionConfig(
            embeddings=_make_embeddings(),
            raptor=RaptorConfig(enabled=True, summary_model=_lm_client()),
        ),
        retrieval=RetrievalConfig(),
    )
    engine = RagEngine(cfg)
    await engine.initialize()
    try:
        assert "raptor" in engine.retrieval
        assert isinstance(engine.retrieval.raptor, RaptorRetrieval)
        # Eager-construction (Option A): registry built at init so the
        # method can hold a stable reference.
        assert engine._raptor_registry is not None  # type: ignore[attr-defined]
        # Builder still lazy: only a build call constructs it.
        assert engine._raptor_builder is None  # type: ignore[attr-defined]
    finally:
        await engine.shutdown()


# ---------------------------------------------------------------------------
# Test 4 (e2e #11): drawing leaves skipped at build; vector retrieval still works.
# ---------------------------------------------------------------------------


async def test_e2e_drawing_corpus_skipped_at_build_then_normal_retrieval() -> None:
    """Mixed knowledge_id (chunks + drawings): drawings stay leaf-only at retrieval."""
    knowledge_id = "kb-mixed"
    chunk_leaves = _make_leaves(knowledge_id, count=12)
    drawing_leaves = _make_drawing_leaves(knowledge_id, count=4)
    store = _FakeVectorStore(chunk_leaves + drawing_leaves)
    embeddings = _make_embeddings()
    registry = _make_in_memory_registry()

    cfg = RaptorConfig(
        enabled=True,
        cluster_algorithm="kmeans",
        clusters_per_level=3,
        max_levels=1,
        summary_model=_lm_client(),
    )
    builder = RaptorTreeBuilder(
        config=cfg,
        vector_store=store,
        embeddings=embeddings,
        registry=registry,
        knowledge_manager=MagicMock(),
    )
    builder._baml_registry = MagicMock()  # noqa: SLF001

    summarize_inputs: list[list[str]] = []

    async def _summarize(**kwargs: Any) -> SimpleNamespace:
        summarize_inputs.append(list(kwargs["cluster_texts"]))
        return _summarize_result(len(summarize_inputs))

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans_round_robin),
        patch(_SUMMARIZE_PATH, new=AsyncMock(side_effect=_summarize)),
    ):
        report = await builder.build(knowledge_id)

    # Drawings (4) excluded; only the 12 chunks went into clustering.
    assert report.level_counts[0] == 12
    # No "drawing" text leaked into any summarisation call.
    for cluster_texts in summarize_inputs:
        for text in cluster_texts:
            assert "drawing" not in text.lower()
            assert "leaf chunk" in text

    # RaptorRetrieval surfaces summaries for the chunk corpus.
    raptor = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)
    chunks = await raptor.search(query="topic_a", top_k=5, knowledge_id=knowledge_id)
    assert len(chunks) > 0

    # Drawings remain reachable via their normal leaf path. Confirmed by
    # scroll: their vector_role stays "drawing_component" — RAPTOR never
    # rewrote them.
    drawing_results, _ = await store.scroll(
        filters={"knowledge_id": knowledge_id, "vector_role": "drawing_component"}
    )
    assert len(drawing_results) == 4


# ---------------------------------------------------------------------------
# Test 5 (e2e #12): per-method weight reaches RRF fusion.
# ---------------------------------------------------------------------------


async def test_e2e_method_weight_for_raptor_respected() -> None:
    """Higher RaptorRetrieval.weight pushes RAPTOR chunks higher in fused order.

    Direct test of the override-semantics risk: if the engine wiring
    silently dropped the configured weight, raising it would have no
    effect on RRF order. Two parallel RetrievalService runs with the
    same inputs but different RaptorRetrieval weights should produce
    different fused orderings (or the same content with different scores).
    """
    knowledge_id = "kb-1"

    # Build once; reuse the same store for both runs so only the weight
    # changes between them.
    store = _FakeVectorStore(_make_leaves(knowledge_id, count=12))
    embeddings = _make_embeddings()
    registry = _make_in_memory_registry()
    cfg = RaptorConfig(
        enabled=True,
        cluster_algorithm="kmeans",
        clusters_per_level=3,
        max_levels=1,
        summary_model=_lm_client(),
    )
    builder = RaptorTreeBuilder(
        config=cfg,
        vector_store=store,
        embeddings=embeddings,
        registry=registry,
        knowledge_manager=MagicMock(),
    )
    builder._baml_registry = MagicMock()  # noqa: SLF001

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans_round_robin),
        patch(
            _SUMMARIZE_PATH,
            new=AsyncMock(side_effect=lambda **kw: _summarize_result(0)),
        ),
    ):
        await builder.build(knowledge_id)

    # Sibling vector method that always returns the same chunk; RAPTOR's
    # weight is the only knob that varies between the two runs.
    sibling_chunk = RetrievedChunk(
        chunk_id="vec-1",
        source_id="src-0",
        content="topic_a leaf hit",
        score=0.9,
    )

    def _vector_mock() -> Any:
        return SimpleNamespace(
            name="vector",
            weight=1.0,
            top_k=None,
            search=AsyncMock(return_value=[sibling_chunk]),
        )

    # Run 1: weight=1.0
    raptor_low = RaptorRetrieval(
        vector_store=store, embeddings=embeddings, registry=registry, weight=1.0
    )
    service_low = RetrievalService(
        retrieval_methods=[_vector_mock(), raptor_low], top_k=5
    )
    fused_low, _ = await service_low.retrieve(
        query="topic_a", knowledge_id=knowledge_id
    )

    # Run 2: weight=10.0 — same inputs, dramatically higher RAPTOR weight.
    raptor_high = RaptorRetrieval(
        vector_store=store, embeddings=embeddings, registry=registry, weight=10.0
    )
    service_high = RetrievalService(
        retrieval_methods=[_vector_mock(), raptor_high], top_k=5
    )
    fused_high, _ = await service_high.retrieve(
        query="topic_a", knowledge_id=knowledge_id
    )

    # Look up the first RAPTOR chunk's score in each run.
    def _first_raptor_score(chunks: list[RetrievedChunk]) -> float | None:
        for c in chunks:
            if c.source_id.startswith("raptor:"):
                return c.score
        return None

    score_low = _first_raptor_score(fused_low)
    score_high = _first_raptor_score(fused_high)

    # Both runs must surface RAPTOR chunks (else the weight knob is
    # untestable — that itself would be a bug we want to catch).
    assert score_low is not None
    assert score_high is not None
    # Higher weight → higher fused score for the RAPTOR chunk.
    assert score_high > score_low, (
        f"raptor weight not honoured: low={score_low} high={score_high}"
    )


# ---------------------------------------------------------------------------
# Test 6: knowledge_id with no built tree returns no RAPTOR contribution.
# ---------------------------------------------------------------------------


async def test_e2e_query_before_build_returns_no_raptor_contribution() -> None:
    """raptor.enabled=True but no build yet: queries proceed without RAPTOR."""
    knowledge_id = "kb-1"
    store = _FakeVectorStore(_make_leaves(knowledge_id, count=8))
    embeddings = _make_embeddings()
    registry = _make_in_memory_registry()

    raptor = RaptorRetrieval(vector_store=store, embeddings=embeddings, registry=registry)

    # No build call → registry returns None → RaptorRetrieval returns [].
    chunks = await raptor.search(query="anything", top_k=5, knowledge_id=knowledge_id)
    assert chunks == []


# ---------------------------------------------------------------------------
# Test 7: namespace iteration determinism — order does not depend on RAPTOR position.
# ---------------------------------------------------------------------------


def test_namespace_iteration_includes_raptor_when_registered() -> None:
    """Regression guard: MethodNamespace iteration surfaces RAPTOR alongside siblings.

    Trace serialization keys per_method_results by method.name, so iteration
    ordering is not load-bearing for fusion correctness, but the namespace
    must include RAPTOR when registered (else it would be unreachable —
    R5.2's "unreachable feature" risk pattern).
    """
    raptor = RaptorRetrieval(
        vector_store=MagicMock(),
        embeddings=MagicMock(),
        registry=MagicMock(),
    )
    vector_method = SimpleNamespace(name="vector")
    namespace: MethodNamespace[Any] = MethodNamespace([vector_method, raptor])

    names = [m.name for m in namespace]
    assert "raptor" in names
    assert namespace.raptor is raptor


# ---------------------------------------------------------------------------
# Pytest collection marker: importable in isolation.
# ---------------------------------------------------------------------------


def test_module_imports_clean() -> None:
    """Sanity: the test module imports without circular-import or attr issues."""
    assert RaptorRetrieval is not None
    assert RetrievalService is not None


@pytest.mark.asyncio
async def test_pytest_async_collection_works() -> None:
    """asyncio_mode=auto sanity check; exists so an async-collection regression
    surfaces here rather than from mid-file e2e tests."""
    assert True
