"""R2.2 — RaptorTreeBuilder runtime: cluster→summarise→embed→persist→swap→GC.

R2.1 shipped the config + BAML stub + registry + skeleton. R2.2 lands the
runtime tree-build loop, ``RagEngine.build_raptor_index`` API, and the
atomic blue/green swap with old-tree GC. R2.3 will add ``RaptorRetrieval``
+ engine wiring on top of this.

Bias-term hygiene: fixtures use neutral identifiers (``kb-1``, ``topic_a``,
``chunk_a``, ``leaf-N``). No domain-specific vocabulary anywhere. Mocks
stand in for ``ClusteringService.run_clustering``, ``b.SummarizeCluster``,
``BaseEmbeddings.embed``, and ``BaseVectorStore.scroll/upsert/delete``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import (
    LanguageModelClient,
    LanguageModelProvider,
)
from rfnry_rag.retrieval.common.models import VectorPoint, VectorResult
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder import (
    MAX_CLUSTER_MEMBERS_PER_SUMMARY,
    RaptorTreeBuilder,
)
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import RaptorConfig

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


def _config(
    *,
    enabled: bool = True,
    cluster_algorithm: str = "kmeans",
    clusters_per_level: int = 3,
    min_cluster_size: int = 2,
    max_levels: int = 3,
    summary_max_tokens: int = 256,
) -> RaptorConfig:
    return RaptorConfig(
        enabled=enabled,
        cluster_algorithm=cluster_algorithm,
        clusters_per_level=clusters_per_level,
        min_cluster_size=min_cluster_size,
        max_levels=max_levels,
        summary_max_tokens=summary_max_tokens,
        summary_model=_lm_client() if enabled else None,
    )


def _leaf_payload(idx: int, role: str = "raw_text") -> dict[str, Any]:
    return {
        "vector_role": role,
        "knowledge_id": "kb-1",
        "contextualized": f"leaf chunk {idx} text content",
        "content": f"leaf chunk {idx} text content",
        "source_id": f"src-{idx % 3}",
    }


def _vector_result(idx: int, role: str = "raw_text") -> VectorResult:
    return VectorResult(
        point_id=f"leaf-{idx}",
        score=0.0,
        payload=_leaf_payload(idx, role=role),
    )


def _make_vector_store(leaf_count: int, drawing_count: int = 0) -> Any:
    """Mock vector store with scroll producing leaves + optional drawings."""
    leaves = [_vector_result(i, role="raw_text") for i in range(leaf_count)]
    drawings = [_vector_result(100 + i, role="drawing_component") for i in range(drawing_count)]

    async def _scroll(
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[VectorResult], str | None]:
        if filters and filters.get("vector_role") == "raptor_summary":
            # GC scroll: nothing exists yet.
            return [], None
        # Filter by vector_role list (mirrors Qdrant MatchAny).
        role_filter = filters.get("vector_role") if filters else None
        if isinstance(role_filter, list):
            allowed = set(role_filter)
            return [r for r in leaves + drawings if r.payload.get("vector_role") in allowed], None
        return leaves + drawings, None

    store = MagicMock()
    store.scroll = AsyncMock(side_effect=_scroll)
    store.upsert = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=0)
    # Optional set_payload for back-references; supplied so tests can
    # observe back-link writes.
    store.set_payload = AsyncMock(return_value=None)
    return store


def _make_embeddings(dim: int = 8) -> Any:
    """Mock embeddings: deterministic per-call vectors so cluster groups stable."""
    counter = {"i": 0}

    async def _embed(texts: list[str]) -> list[list[float]]:
        out = []
        for _ in texts:
            counter["i"] += 1
            # Spread vectors across the unit sphere so K-Means produces
            # well-separated clusters; identical vectors degenerate KMeans.
            base = counter["i"] % 3
            out.append([float(base) + 0.01 * counter["i"]] + [0.0] * (dim - 1))
        return out

    e = MagicMock()
    e.embed = AsyncMock(side_effect=_embed)
    return e


def _make_registry() -> Any:
    reg = MagicMock()
    reg.set_active = AsyncMock(return_value=None)
    return reg


def _make_knowledge_manager() -> Any:
    return MagicMock()


def _make_builder(
    *,
    cfg: RaptorConfig | None = None,
    leaf_count: int = 12,
    drawing_count: int = 0,
    vector_store: Any | None = None,
    embeddings: Any | None = None,
    registry: Any | None = None,
) -> RaptorTreeBuilder:
    builder = RaptorTreeBuilder(
        config=cfg or _config(),
        vector_store=vector_store or _make_vector_store(leaf_count, drawing_count),
        embeddings=embeddings or _make_embeddings(),
        registry=registry or _make_registry(),
        knowledge_manager=_make_knowledge_manager(),
    )
    # Replace the BAML registry with a sentinel so SummarizeCluster patches
    # cleanly without trying to materialise a real client registry.
    builder._baml_registry = MagicMock()  # noqa: SLF001
    return builder


def _summarize_result(summary: str = "synthesised summary", reasoning: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(summary=summary, reasoning=reasoning)


_SUMMARIZE_PATH = (
    "rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder.b.SummarizeCluster"
)
_RUN_CLUSTERING_PATH = (
    "rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder.run_clustering"
)


def _kmeans_labels(member_count: int, k: int) -> Any:
    """Deterministic round-robin labels with k clusters across member_count items."""
    labels = np.array([i % k for i in range(member_count)], dtype=np.int32)
    centroids = np.zeros((k, 8), dtype=np.float32)
    return labels, centroids


# ---------------------------------------------------------------------------
# Test 1: enabled=False raises ConfigurationError.
# ---------------------------------------------------------------------------


async def test_build_validates_enabled_or_raises() -> None:
    builder = _make_builder(cfg=RaptorConfig(enabled=False))
    with pytest.raises(ConfigurationError, match="enabled"):
        await builder.build("kb-1")


# ---------------------------------------------------------------------------
# Test 2: enabled=True without summary_model raises ConfigurationError.
# ---------------------------------------------------------------------------


async def test_build_validates_summary_model_or_raises() -> None:
    """RaptorConfig.__post_init__ catches this; the runtime path also defends.

    Construct a valid config, then forcibly null ``summary_model`` to mimic
    a consumer mutating the dataclass post-init.
    """
    cfg = _config()
    cfg.summary_model = None
    builder = _make_builder(cfg=cfg)
    with pytest.raises(ConfigurationError, match="summary_model"):
        await builder.build("kb-1")


# ---------------------------------------------------------------------------
# Test 3: empty knowledge_id raises ConfigurationError.
# ---------------------------------------------------------------------------


async def test_build_validates_knowledge_id_exists() -> None:
    builder = _make_builder()
    with pytest.raises(ConfigurationError, match="knowledge_id"):
        await builder.build("")
    with pytest.raises(ConfigurationError, match="knowledge_id"):
        await builder.build("   ")


# ---------------------------------------------------------------------------
# Test 4: zero-leaf knowledge_id returns empty report.
# ---------------------------------------------------------------------------


async def test_build_returns_empty_report_for_empty_knowledge_id() -> None:
    builder = _make_builder(leaf_count=0)
    report = await builder.build("kb-1")

    assert report.level_counts == [0]
    assert report.total_summaries == 0
    assert report.total_decompose_calls == 0
    assert report.knowledge_id == "kb-1"
    # Empty corpus still registers a tree pointer (so retrieval can short-
    # circuit on "tree exists but is empty" rather than "no tree built").
    cast(Any, builder._registry).set_active.assert_awaited_once()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Test 5: drawing_component leaves are never sent to SummarizeCluster.
# ---------------------------------------------------------------------------


async def test_build_skips_drawing_components() -> None:
    builder = _make_builder(leaf_count=6, drawing_count=4)

    def _capture_run(matrix: Any, cfg: Any) -> Any:
        # Force a single big cluster so SummarizeCluster receives all
        # member texts; if drawings leaked through they would appear in
        # ``cluster_texts``.
        n = matrix.shape[0]
        labels = np.zeros(n, dtype=np.int32)
        centroids = np.zeros((1, matrix.shape[1]), dtype=np.float32)
        return labels, centroids

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_capture_run),
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        await builder.build("kb-1")

    # Every cluster_texts entry must be a leaf chunk text — no drawing
    # text leaked through. Drawings have role "drawing_component" and
    # we filter at scroll-time.
    for call in summarize.await_args_list:
        cluster_texts = call.kwargs["cluster_texts"]
        for text in cluster_texts:
            assert "leaf chunk" in text
            assert "drawing" not in text.lower()


# ---------------------------------------------------------------------------
# Test 6: K-Means default produces clusters_per_level groups.
# ---------------------------------------------------------------------------


async def test_build_kmeans_default_clusters_correctly() -> None:
    cfg = _config(clusters_per_level=3, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=12)

    captured_n_clusters: list[int] = []

    def _capture_run(matrix: Any, cluster_cfg: Any) -> Any:
        captured_n_clusters.append(cluster_cfg.n_clusters)
        return _kmeans_labels(matrix.shape[0], k=cluster_cfg.n_clusters)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_capture_run),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    assert captured_n_clusters[0] == 3
    # level_counts: [12 leaves, 3 summaries]
    assert report.level_counts == [12, 3]


# ---------------------------------------------------------------------------
# Test 7: HDBSCAN dispatch is exercised when cluster_algorithm="hdbscan".
# ---------------------------------------------------------------------------


async def test_build_hdbscan_clusters_correctly() -> None:
    cfg = _config(cluster_algorithm="hdbscan", min_cluster_size=2, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=12)

    captured_algorithms: list[str] = []

    def _capture_run(matrix: Any, cluster_cfg: Any) -> Any:
        captured_algorithms.append(cluster_cfg.algorithm)
        # Two clusters of equal size, no noise.
        n = matrix.shape[0]
        labels = np.array([i // (n // 2) for i in range(n)], dtype=np.int32)
        centroids = np.zeros((2, matrix.shape[1]), dtype=np.float32)
        return labels, centroids

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_capture_run),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # Override semantics regression guard: hdbscan must dispatch to HDBSCAN.
    assert captured_algorithms[0] == "hdbscan"
    assert report.level_counts[0] == 12
    assert report.level_counts[1] == 2


# ---------------------------------------------------------------------------
# Test 8: max_levels caps the recursion depth.
# ---------------------------------------------------------------------------


async def test_build_terminates_at_max_levels() -> None:
    cfg = _config(clusters_per_level=2, max_levels=2)
    builder = _make_builder(cfg=cfg, leaf_count=20)

    def _run_kmeans(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=cluster_cfg.n_clusters)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_run_kmeans),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # leaves + at most 2 summary levels = 3 entries.
    assert len(report.level_counts) <= 3


# ---------------------------------------------------------------------------
# Test 9: K-Means terminates when next level would be degenerate.
# ---------------------------------------------------------------------------


async def test_build_terminates_when_next_level_degenerate_kmeans() -> None:
    """N <= clusters_per_level + 1 stops the recursion (single-member buckets)."""
    cfg = _config(clusters_per_level=10, max_levels=5)
    builder = _make_builder(cfg=cfg, leaf_count=5)

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=lambda *_: _kmeans_labels(5, k=2)),
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        report = await builder.build("kb-1")

    # Builder breaks out before issuing any cluster work — leaves alone.
    assert report.level_counts == [5]
    summarize.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: HDBSCAN terminates when corpus is too small for min_cluster_size.
# ---------------------------------------------------------------------------


async def test_build_terminates_when_next_level_degenerate_hdbscan() -> None:
    """N < 2 * min_cluster_size means no meaningful HDBSCAN split is possible."""
    cfg = _config(cluster_algorithm="hdbscan", min_cluster_size=5, max_levels=5)
    builder = _make_builder(cfg=cfg, leaf_count=6)  # 6 < 2 * 5

    summarize = AsyncMock(return_value=_summarize_result())
    with patch(_SUMMARIZE_PATH, new=summarize):
        report = await builder.build("kb-1")

    assert report.level_counts == [6]
    summarize.assert_not_called()


# ---------------------------------------------------------------------------
# Test 11: cluster members capped at MAX_CLUSTER_MEMBERS_PER_SUMMARY.
# ---------------------------------------------------------------------------


async def test_build_caps_cluster_members_at_centroid_nearest() -> None:
    """Cluster of 50 members -> SummarizeCluster sees exactly 20 texts."""
    assert MAX_CLUSTER_MEMBERS_PER_SUMMARY == 20  # contract sanity
    cfg = _config(clusters_per_level=2, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=50)

    def _one_cluster(matrix: Any, cluster_cfg: Any) -> Any:
        n = matrix.shape[0]
        labels = np.zeros(n, dtype=np.int32)
        # Centroid at the mean — distances vary so the slice is non-trivial.
        centroids = matrix.mean(axis=0, keepdims=True).astype(np.float32)
        return labels, centroids

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_one_cluster),
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        await builder.build("kb-1")

    summarize.assert_awaited_once()
    cluster_texts = cast(Any, summarize.await_args).kwargs["cluster_texts"]
    assert len(cluster_texts) == 20


# ---------------------------------------------------------------------------
# Test 12: cluster-of-one passthrough avoids the LLM call.
# ---------------------------------------------------------------------------


async def test_build_uses_passthrough_for_cluster_of_one() -> None:
    """One-member cluster reuses member text; no SummarizeCluster invocation."""
    cfg = _config(clusters_per_level=4, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=6)

    def _make_one_singleton(matrix: Any, cluster_cfg: Any) -> Any:
        # 1 cluster of size 1, 1 cluster of size 5.
        labels = np.array([0, 1, 1, 1, 1, 1], dtype=np.int32)
        centroids = np.array(
            [matrix[0], matrix[1:].mean(axis=0)], dtype=np.float32
        )
        return labels, centroids

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_make_one_singleton),
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        report = await builder.build("kb-1")

    # Only the multi-member cluster should reach SummarizeCluster.
    assert summarize.await_count == 1
    # decompose_calls counts only LLM-backed summaries (passthroughs excluded).
    assert report.total_decompose_calls == 1
    assert report.total_summaries == 2  # one passthrough + one summarised


# ---------------------------------------------------------------------------
# Test 13: persisted summary vectors carry the full payload schema.
# ---------------------------------------------------------------------------


async def test_build_persists_summary_vectors_with_full_payload() -> None:
    cfg = _config(clusters_per_level=2, max_levels=1)
    store = _make_vector_store(leaf_count=10)
    builder = _make_builder(cfg=cfg, vector_store=store)

    def _two_clusters(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=2)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_two_clusters),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # Find the persistence call (upsert with summary points).
    persisted_payloads: list[dict[str, Any]] = []
    for call in store.upsert.await_args_list:
        points: list[VectorPoint] = call.args[0]
        for p in points:
            persisted_payloads.append(p.payload)

    assert persisted_payloads, "no summary vectors persisted"
    for payload in persisted_payloads:
        assert payload["vector_role"] == "raptor_summary"
        assert payload["knowledge_id"] == "kb-1"
        assert payload["raptor_tree_id"] == report.tree_id
        assert payload["raptor_level"] == 1
        # parent_id is null at first persist; later level fills it.
        assert payload["raptor_parent_id"] is None
        assert payload["raptor_cluster_size"] >= 1


# ---------------------------------------------------------------------------
# Test 14: multi-level build sets parent_id back-references.
# ---------------------------------------------------------------------------


async def test_build_sets_parent_id_back_references() -> None:
    """Level-1 children get a raptor_parent_id pointing at level-2 summaries."""
    cfg = _config(clusters_per_level=2, max_levels=3)
    store = _make_vector_store(leaf_count=20)
    builder = _make_builder(cfg=cfg, vector_store=store)

    # Force the mocked clustering to produce more clusters at level 1
    # than ``clusters_per_level + 1`` so level 2 has enough inputs to
    # cluster meaningfully (otherwise the algorithm-specific termination
    # check stops the recursion before any back-references fire at level 2).
    def _kmeans(matrix: Any, cluster_cfg: Any) -> Any:
        n = matrix.shape[0]
        # Level 1: 20 inputs -> 5 clusters; level 2: 5 inputs -> 2 clusters.
        k = 5 if n > 5 else 2
        return _kmeans_labels(n, k=k)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # set_payload should have been called at least once (back-references
    # only fire at level >= 2 because level 1's children are leaves; we
    # want to see both level-1 leaves AND level-2 summaries linked).
    assert store.set_payload.await_count > 0
    # At least one back-reference write set ``raptor_parent_id`` to a
    # level-2 summary's id (not None).
    parent_ids_written = [
        call.kwargs["payload"]["raptor_parent_id"]
        for call in store.set_payload.await_args_list
    ]
    assert any(pid is not None for pid in parent_ids_written)
    # Multi-level build: level_counts should have at least 3 entries
    # (leaves + level-1 summaries + level-2 summaries).
    assert len(report.level_counts) >= 3


# ---------------------------------------------------------------------------
# Test 15: atomic swap order is registry first, then GC.
# ---------------------------------------------------------------------------


async def test_build_atomic_swap_writes_registry_then_gcs_old_tree() -> None:
    """``set_active`` is awaited BEFORE the GC delete, even when GC raises."""
    cfg = _config(clusters_per_level=2, max_levels=1)
    store = _make_vector_store(leaf_count=10)
    # Pre-existing tree under same kid — GC must target tree_id != new.
    existing_summary = VectorResult(
        point_id="old-summary-1",
        score=0.0,
        payload={
            "vector_role": "raptor_summary",
            "knowledge_id": "kb-1",
            "raptor_tree_id": "kb-1__OLD12345",
        },
    )

    async def _scroll(
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[VectorResult], str | None]:
        if filters and filters.get("vector_role") == "raptor_summary":
            return [existing_summary], None
        # Leaves
        return [_vector_result(i) for i in range(10)], None

    store.scroll = AsyncMock(side_effect=_scroll)
    registry = _make_registry()
    call_order: list[str] = []
    cast(Any, registry).set_active.side_effect = lambda *a, **k: call_order.append("set_active")

    async def _delete(filters: dict[str, Any]) -> int:
        call_order.append(f"delete:{filters['raptor_tree_id']}")
        return 1

    store.delete = AsyncMock(side_effect=_delete)
    builder = _make_builder(cfg=cfg, vector_store=store, registry=registry)

    def _two(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=2)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_two),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # Order check: set_active fires first; delete targets the OLD tree id.
    assert call_order[0] == "set_active"
    assert any(s == "delete:kb-1__OLD12345" for s in call_order)
    # The new tree id is NOT in the deleted set.
    assert not any(s == f"delete:{report.tree_id}" for s in call_order)


# ---------------------------------------------------------------------------
# Test 16: first-run with no prior tree GCs nothing (no error).
# ---------------------------------------------------------------------------


async def test_build_first_run_no_old_tree_to_gc() -> None:
    """Empty GC scroll -> zero deletes, no error."""
    cfg = _config(clusters_per_level=2, max_levels=1)
    store = _make_vector_store(leaf_count=8)
    builder = _make_builder(cfg=cfg, vector_store=store)

    def _two(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=2)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_two),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # No old summaries existed, so delete should not have been called.
    store.delete.assert_not_called()
    assert report.timings["gc_deleted_count"] == 0.0


# ---------------------------------------------------------------------------
# Test 17: per-cluster summarisation runs through run_concurrent.
# ---------------------------------------------------------------------------


async def test_build_summarize_cluster_called_concurrently() -> None:
    """5 clusters -> 5 summarise calls; verify run_concurrent dispatch."""
    cfg = _config(clusters_per_level=5, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=20)

    def _five(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=5)

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_five),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder.run_concurrent",
            wraps=__import__(
                "rfnry_rag.common.concurrency", fromlist=["run_concurrent"]
            ).run_concurrent,
        ) as concurrent_spy,
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        report = await builder.build("kb-1")

    assert summarize.await_count == 5
    # run_concurrent invoked twice per level (summarise + back-ref link).
    assert concurrent_spy.call_count == 2
    # 5 summaries persisted.
    assert report.level_counts == [20, 5]


# ---------------------------------------------------------------------------
# Test 18: level-1 cluster member texts come from ``contextualized``.
# ---------------------------------------------------------------------------


async def test_build_uses_chunk_contextualized_text_for_level_one() -> None:
    """Level-1 SummarizeCluster receives the leaf's contextualized payload field."""
    cfg = _config(clusters_per_level=2, max_levels=1)
    leaves = [
        VectorResult(
            point_id=f"leaf-{i}",
            score=0.0,
            payload={
                "vector_role": "raw_text",
                "knowledge_id": "kb-1",
                # contextualized differs from content so we can prove which is read.
                "contextualized": f"CTX-leaf-{i}",
                "content": f"RAW-leaf-{i}",
            },
        )
        for i in range(4)
    ]

    async def _scroll(filters: Any = None, limit: int = 100, offset: Any = None) -> Any:
        if filters and filters.get("vector_role") == "raptor_summary":
            return [], None
        return leaves, None

    store = MagicMock()
    store.scroll = AsyncMock(side_effect=_scroll)
    store.upsert = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=0)
    store.set_payload = AsyncMock(return_value=None)
    builder = _make_builder(cfg=cfg, vector_store=store)

    def _one(matrix: Any, cluster_cfg: Any) -> Any:
        n = matrix.shape[0]
        return np.zeros(n, dtype=np.int32), matrix.mean(axis=0, keepdims=True).astype(np.float32)

    summarize = AsyncMock(return_value=_summarize_result())
    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_one),
        patch(_SUMMARIZE_PATH, new=summarize),
    ):
        await builder.build("kb-1")

    cluster_texts = cast(Any, summarize.await_args).kwargs["cluster_texts"]
    for text in cluster_texts:
        assert text.startswith("CTX-")
        assert "RAW-" not in text


# ---------------------------------------------------------------------------
# Test 19: level-2 cluster members are level-1 summary texts (not leaf texts).
# ---------------------------------------------------------------------------


async def test_build_uses_summary_text_for_higher_levels() -> None:
    cfg = _config(clusters_per_level=2, max_levels=3)
    builder = _make_builder(cfg=cfg, leaf_count=20)

    summarize_texts_seen: list[list[str]] = []

    # Force level 1 to produce > clusters_per_level + 1 outputs so level 2
    # can run; otherwise the termination guard halts the recursion.
    def _kmeans(matrix: Any, cluster_cfg: Any) -> Any:
        n = matrix.shape[0]
        k = 5 if n > 5 else 2
        return _kmeans_labels(n, k=k)

    summary_counter = {"i": 0}

    async def _summarize(**kwargs: Any) -> Any:
        summarize_texts_seen.append(list(kwargs["cluster_texts"]))
        summary_counter["i"] += 1
        return _summarize_result(summary=f"summary-{summary_counter['i']}")

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_kmeans),
        patch(_SUMMARIZE_PATH, new=AsyncMock(side_effect=_summarize)),
    ):
        await builder.build("kb-1")

    # First level's summaries must feed the second level's cluster texts.
    # Find the call where every text matches "summary-N" pattern.
    assert any(
        all(t.startswith("summary-") for t in seen) for seen in summarize_texts_seen
    )


# ---------------------------------------------------------------------------
# Test 20: report carries timings + counts populated.
# ---------------------------------------------------------------------------


async def test_build_returns_report_with_timings_and_counts() -> None:
    cfg = _config(clusters_per_level=2, max_levels=1)
    builder = _make_builder(cfg=cfg, leaf_count=10)

    def _two(matrix: Any, cluster_cfg: Any) -> Any:
        return _kmeans_labels(matrix.shape[0], k=2)

    with (
        patch(_RUN_CLUSTERING_PATH, side_effect=_two),
        patch(_SUMMARIZE_PATH, new=AsyncMock(return_value=_summarize_result())),
    ):
        report = await builder.build("kb-1")

    # All declared report fields must be populated.
    assert report.knowledge_id == "kb-1"
    assert report.tree_id.startswith("kb-1__")
    assert report.level_counts == [10, 2]
    assert report.total_summaries == 2
    assert report.total_decompose_calls == 2  # both clusters had > 1 members
    assert report.duration_seconds >= 0.0

    # Trace-data-dropped-at-boundary regression guard (R5.3 lesson):
    # timings must include per-stage breakdown, not just totals.
    assert "load_leaves" in report.timings
    assert "level_1_cluster" in report.timings
    assert "level_1_summarize" in report.timings
    assert "level_1_embed" in report.timings
    assert "level_1_persist" in report.timings
    assert "swap" in report.timings
    assert "gc" in report.timings
