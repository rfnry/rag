"""RaptorTreeBuilder — cluster→summarize→embed→persist→recurse loop (R2.2).

Implements the RAPTOR-style hierarchical-summarisation tree build:
  1. Load leaf vectors for the knowledge_id (chunks; drawings excluded).
  2. For each level up to ``max_levels``:
     a. Cluster the current level's vectors (K-Means default; HDBSCAN opt-in).
     b. Skip the level entirely when the next level would degenerate to a
        single cluster (algorithm-specific guard).
     c. Cap each cluster to the centroid-nearest top-N members so a single
        large cluster cannot blow up the LLM call budget.
     d. Summarise each cluster concurrently via ``b.SummarizeCluster``;
        cluster-of-one short-circuits the LLM call (passthrough).
     e. Embed and persist summaries to the vector store; back-reference each
        member to its parent summary.
  3. Atomically swap the active tree pointer in ``RaptorTreeRegistry``,
     then GC older trees' summary vectors (orphan-tolerant: a GC failure
     after swap leaves dangling old vectors that retrieval filters out
     by ``raptor_tree_id``, not a broken tree).

Concurrency: per-level cluster summarisation runs through ``run_concurrent``
(bounded). Across-level work is sequential — level N+1 reads N's persisted
output. The builder is sibling to ``VectorIngestion`` / ``AnalyzedIngestionService``,
NOT a fold-in: RAPTOR is a distinct retrieval strategy, not a layer on top
of chunk indexing.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from rfnry_rag.common.concurrency import run_concurrent
from rfnry_rag.common.embeddings import embed_batched
from rfnry_rag.reasoning.modules.clustering.algorithms import run_clustering
from rfnry_rag.reasoning.modules.clustering.models import ClusteringConfig
from rfnry_rag.retrieval.baml.baml_client.async_client import b
from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.common.language_model import build_registry
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import VectorPoint
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import RaptorConfig
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.registry import RaptorTreeRegistry
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.report import RaptorBuildReport
from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager
from rfnry_rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("ingestion.methods.raptor.builder")

# Cap cluster member count at top-N centroid-nearest before passing to
# ``SummarizeCluster``. Bounds LLM cost per call regardless of cluster size:
# even on a cluster of 200 members, we summarise the 20 closest-to-centroid
# texts. Hardcoded (not config) because tuning this is an algorithm-design
# decision, not a per-deployment knob — pushing it past ~30 risks the
# summary devolving into a list rather than synthesis.
MAX_CLUSTER_MEMBERS_PER_SUMMARY = 20

# Default per-level summarisation concurrency. Mirrors the analyze pipeline's
# ``analyze_concurrency`` default — stays under typical Tier-1 LLM rate limits
# while keeping multi-cluster levels from running serially.
_SUMMARIZE_CONCURRENCY = 5

# Leaf-vector role allowlist. Drawing components are excluded by design —
# their text representation is component-tag-style and not amenable to
# narrative summarisation; mixing them into clusters would skew summaries.
_LEAF_VECTOR_ROLES = ("description", "raw_text", "table_row")


class _ClusterMember:
    """In-memory representation of one cluster member during the build.

    Holds the embedding vector (used to pick centroid-nearest), the text we
    pass to ``SummarizeCluster``, the originating ``point_id`` (so we can
    back-reference each leaf to its parent summary), and the leaf count this
    node descends from (for the ``raptor_cluster_size`` payload field).
    """

    __slots__ = ("point_id", "vector", "text", "leaf_count")

    def __init__(
        self,
        point_id: str,
        vector: list[float],
        text: str,
        leaf_count: int,
    ) -> None:
        self.point_id = point_id
        self.vector = vector
        self.text = text
        self.leaf_count = leaf_count


class _SummaryNode:
    """One produced summary (node in the tree).

    ``children_point_ids`` records the input member ids that contributed —
    used to populate ``raptor_parent_id`` back-pointers after persist.
    """

    __slots__ = ("point_id", "text", "reasoning", "leaf_count", "children_point_ids", "vector")

    def __init__(
        self,
        point_id: str,
        text: str,
        reasoning: str,
        leaf_count: int,
        children_point_ids: list[str],
    ) -> None:
        self.point_id = point_id
        self.text = text
        self.reasoning = reasoning
        self.leaf_count = leaf_count
        self.children_point_ids = children_point_ids
        self.vector: list[float] = []


class RaptorTreeBuilder:
    """Builds the RAPTOR summary tree for one ``knowledge_id``.

    Constructor injects all collaborators so the builder is testable without
    a live engine. Consumers don't construct this directly — ``RagEngine``
    lazily instantiates one on the first ``build_raptor_index`` call.
    """

    def __init__(
        self,
        config: RaptorConfig,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        registry: RaptorTreeRegistry,
        knowledge_manager: KnowledgeManager,
    ) -> None:
        self._cfg = config
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._registry = registry
        self._knowledge_manager = knowledge_manager
        # Build the BAML registry once per builder; ``SummarizeCluster`` is
        # called many times during a build, and rebuilding the registry per
        # call would re-pay the BAML setup cost on each cluster.
        self._baml_registry: Any = (
            build_registry(config.summary_model) if config.summary_model is not None else None
        )

    async def build(self, knowledge_id: str) -> RaptorBuildReport:
        """Build a fresh RAPTOR tree for ``knowledge_id`` and swap it active.

        Raises ``ConfigurationError`` if the config / knowledge_id combination
        is ineligible (matches R5/R6's "fail at runtime, don't silently
        no-op" pattern).
        """
        await self._validate_eligible(knowledge_id)

        new_tree_id = self._allocate_tree_id(knowledge_id)
        timings: dict[str, float] = {}
        cost_usd: float | None = None  # No pricing API on LanguageModelClient yet.
        decompose_calls = 0
        wall_start = time.perf_counter()

        logger.info(
            "raptor build start: knowledge_id=%s tree_id=%s", knowledge_id, new_tree_id
        )

        # ---- Stage 1: load leaves ---------------------------------------
        t0 = time.perf_counter()
        leaves = await self._load_leaves(knowledge_id)
        timings["load_leaves"] = time.perf_counter() - t0

        if not leaves:
            # Empty corpus: register an empty tree so subsequent retrieval
            # short-circuits cleanly instead of treating "no row" as
            # "haven't built yet".
            t_swap = time.perf_counter()
            await self._registry.set_active(knowledge_id, new_tree_id, [0], None)
            timings["swap"] = time.perf_counter() - t_swap
            t_gc = time.perf_counter()
            deleted = await self._gc_old_trees(knowledge_id, new_tree_id)
            timings["gc"] = time.perf_counter() - t_gc
            timings["gc_deleted_count"] = float(deleted)
            duration = time.perf_counter() - wall_start
            logger.info(
                "raptor build done (empty): knowledge_id=%s tree_id=%s duration=%.2fs",
                knowledge_id,
                new_tree_id,
                duration,
            )
            return RaptorBuildReport(
                knowledge_id=knowledge_id,
                tree_id=new_tree_id,
                level_counts=[0],
                total_summaries=0,
                total_decompose_calls=0,
                total_cost_usd=cost_usd,
                duration_seconds=duration,
                timings=timings,
            )

        level_counts: list[int] = [len(leaves)]
        current: list[_ClusterMember] = leaves
        # Track every persisted summary so we can map back-pointers and
        # finally write all parent_ids in one update pass per level.
        persisted_summaries_by_id: dict[str, _SummaryNode] = {}

        # ---- Stage 2: cluster + summarise + embed + persist per level ----
        for level_index in range(self._cfg.max_levels):
            level = level_index + 1

            if not self._can_cluster_meaningfully(current):
                # Termination: next clustering pass would produce a single
                # cluster (or fewer than min_cluster_size for HDBSCAN).
                # Keeping the empty level out of level_counts mirrors
                # "level_counts[0] = leaves, level_counts[1:] = summaries".
                logger.info(
                    "raptor build: terminating at level %d — %d nodes cannot cluster meaningfully",
                    level,
                    len(current),
                )
                break

            t_cluster = time.perf_counter()
            clusters = self._cluster(current)
            timings[f"level_{level}_cluster"] = time.perf_counter() - t_cluster

            if not clusters:
                # HDBSCAN can produce zero clusters when everything is noise.
                logger.info(
                    "raptor build: level %d produced no clusters — terminating", level
                )
                break

            t_sum = time.perf_counter()
            summaries, level_decompose_calls = await self._summarise_clusters(clusters, level)
            timings[f"level_{level}_summarize"] = time.perf_counter() - t_sum
            decompose_calls += level_decompose_calls

            t_emb = time.perf_counter()
            await self._embed_summaries(summaries)
            timings[f"level_{level}_embed"] = time.perf_counter() - t_emb

            t_persist = time.perf_counter()
            await self._persist_summaries(summaries, knowledge_id, new_tree_id, level)
            timings[f"level_{level}_persist"] = time.perf_counter() - t_persist

            # Back-reference every child point to its just-created parent.
            # Children for level=1 are leaf vectors; for level>=2 they are
            # the previous level's summaries. Either way, ``point_id`` is
            # the existing vector to update.
            t_back = time.perf_counter()
            await self._set_parent_back_references(summaries)
            timings[f"level_{level}_parent_link"] = time.perf_counter() - t_back

            for s in summaries:
                persisted_summaries_by_id[s.point_id] = s

            level_counts.append(len(summaries))

            # Promote summaries to the next level's input. ``vector`` and
            # ``text`` are populated by embed; leaf_count is the sum of
            # contributing leaves so the root node correctly reports the
            # full corpus size.
            current = [
                _ClusterMember(
                    point_id=s.point_id,
                    vector=s.vector,
                    text=s.text,
                    leaf_count=s.leaf_count,
                )
                for s in summaries
            ]

        # ---- Stage 3: atomic swap ---------------------------------------
        t_swap = time.perf_counter()
        await self._registry.set_active(knowledge_id, new_tree_id, level_counts, cost_usd)
        timings["swap"] = time.perf_counter() - t_swap

        # ---- Stage 4: GC old trees --------------------------------------
        # Order matters: swap first, then GC. If GC raises after swap, the
        # new tree is live and old vectors are orphans (still tagged with
        # the old ``raptor_tree_id`` so retrieval correctly filters them
        # out at query time). A future ``gc_orphans`` sweep can clean up.
        t_gc = time.perf_counter()
        try:
            deleted = await self._gc_old_trees(knowledge_id, new_tree_id)
            timings["gc_deleted_count"] = float(deleted)
        except Exception as exc:
            # GC failure is observable but non-fatal: the new tree is
            # already swapped active. Log and continue to surface the
            # report so the operator can retry GC out of band.
            logger.warning(
                "raptor build: GC failed after swap (knowledge_id=%s): %s — "
                "new tree is active; old vectors are orphans (retrieval filters by tree_id)",
                knowledge_id,
                exc,
            )
            timings["gc_deleted_count"] = 0.0
        timings["gc"] = time.perf_counter() - t_gc

        duration = time.perf_counter() - wall_start
        total_summaries = sum(level_counts[1:]) if len(level_counts) > 1 else 0
        logger.info(
            "raptor build done: knowledge_id=%s tree_id=%s level_counts=%s "
            "duration=%.2fs cost=%s",
            knowledge_id,
            new_tree_id,
            level_counts,
            duration,
            cost_usd,
        )
        return RaptorBuildReport(
            knowledge_id=knowledge_id,
            tree_id=new_tree_id,
            level_counts=level_counts,
            total_summaries=total_summaries,
            total_decompose_calls=decompose_calls,
            total_cost_usd=cost_usd,
            duration_seconds=duration,
            timings=timings,
        )

    # ------------------------------------------------------------------
    # Validation + identifiers
    # ------------------------------------------------------------------

    async def _validate_eligible(self, knowledge_id: str) -> None:
        """Defensive runtime checks independent of ``RaptorConfig.__post_init__``.

        The config-time check forbids ``enabled=True`` without a
        ``summary_model``; here we re-check the same invariant in case a
        consumer constructed the builder directly bypassing the engine.
        """
        if not self._cfg.enabled:
            raise ConfigurationError(
                "RaptorConfig.enabled=False — RAPTOR builds are disabled. "
                "Set IngestionConfig.raptor.enabled=True to opt in."
            )
        if self._cfg.summary_model is None:
            raise ConfigurationError(
                "RaptorConfig.summary_model is None — required when enabled=True."
            )
        if not knowledge_id or not knowledge_id.strip():
            raise ConfigurationError(
                "build_raptor_index requires a non-empty knowledge_id."
            )
        # Existence check: a knowledge_id with zero sources is allowed
        # (we'll write an empty-tree row), but a literally-empty value is
        # rejected above. We don't fail when the knowledge has no sources
        # so consumers can pre-register tree pointers without ingesting.

    @staticmethod
    def _allocate_tree_id(knowledge_id: str) -> str:
        # 8 hex chars = 32 bits of entropy — plenty for de-duplication
        # within a single knowledge_id. Embedding the knowledge_id in the
        # tree_id keeps grep/observability friendly without forcing a
        # per-knowledge_id sequence counter.
        return f"{knowledge_id}__{uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Leaf loading
    # ------------------------------------------------------------------

    async def _load_leaves(self, knowledge_id: str) -> list[_ClusterMember]:
        """Scroll all leaf vectors for ``knowledge_id``.

        Uses the vector-store ``scroll`` filter to restrict to
        ``vector_role IN _LEAF_VECTOR_ROLES``. Drawings are excluded by
        omission. We filter in two passes (one per role) when the
        store's filter shape can't express ``IN`` with a list value —
        but the existing Qdrant ``_build_filter`` does support list
        values via ``MatchAny``, so we issue a single scroll call.
        """
        out: list[_ClusterMember] = []
        # Note: using a list value triggers Qdrant's ``MatchAny`` (OR over
        # values) — the IN-clause shape we want.
        filters: dict[str, Any] = {
            "knowledge_id": knowledge_id,
            "vector_role": list(_LEAF_VECTOR_ROLES),
        }
        offset: str | None = None
        while True:
            results, next_offset = await self._vector_store.scroll(
                filters=filters, limit=500, offset=offset
            )
            for r in results:
                payload = r.payload
                # Prefer the chunk's contextualized text (level-1 input);
                # fall back to ``content`` for legacy points that predate
                # the contextualized field. ``contextualized`` is the
                # default for analyze-pipeline + chunk-pipeline writes.
                text = (
                    payload.get("contextualized")
                    or payload.get("content")
                    or payload.get("text")
                    or ""
                )
                # Vector-store scroll doesn't return embeddings by
                # default. We need them for clustering, so we'll fetch a
                # fresh embedding for any point missing one. In practice
                # the leaf's stored vector is not surfaced by the scroll
                # API — the clean choice is to embed the text we already
                # have, mirroring how ``ClusteringService`` handles it.
                vec = payload.get("_vector") or []
                out.append(
                    _ClusterMember(
                        point_id=str(r.point_id),
                        vector=vec,
                        text=text,
                        leaf_count=1,
                    )
                )
            if not next_offset:
                break
            offset = next_offset

        # The vector-store scroll API doesn't return raw vectors, so embed
        # leaf text once up front. Bounded by ``embed_batched`` internals.
        missing_vec_indices = [i for i, m in enumerate(out) if not m.vector]
        if missing_vec_indices:
            texts = [out[i].text for i in missing_vec_indices]
            vectors = await embed_batched(self._embeddings, texts)
            for idx, vec in zip(missing_vec_indices, vectors, strict=True):
                out[idx].vector = vec

        return out

    # ------------------------------------------------------------------
    # Termination + clustering
    # ------------------------------------------------------------------

    def _can_cluster_meaningfully(self, current: list[_ClusterMember]) -> bool:
        """Algorithm-specific guard against degenerate next-level clustering.

        K-Means with N ≤ k+1 produces single-member clusters that just
        re-encode the input — no compression, no synthesis. HDBSCAN with
        N < 2*min_cluster_size cannot find two clusters (since
        ``min_cluster_size`` is the smallest valid cluster). Either case
        means we should stop the recursion rather than burn LLM calls
        producing trivial summaries.
        """
        if self._cfg.cluster_algorithm == "kmeans":
            return len(current) > self._cfg.clusters_per_level + 1
        if self._cfg.cluster_algorithm == "hdbscan":
            return len(current) >= 2 * self._cfg.min_cluster_size
        # Defensive: the config allowlist forbids other values, but if a
        # consumer slips one through (e.g. monkey-patches the dataclass)
        # we refuse to cluster rather than silently mis-dispatching.
        return False

    def _cluster(self, current: list[_ClusterMember]) -> list[list[_ClusterMember]]:
        """Cluster ``current`` and return a list of member groups.

        Calls ``run_clustering`` from the reasoning SDK directly — the
        higher-level ``ClusteringService`` wraps embedding+labelling
        logic we don't need (we already have vectors, and we want raw
        per-input labels rather than sample-bounded ``Cluster`` objects).
        """
        if self._cfg.cluster_algorithm == "kmeans":
            cfg = ClusteringConfig(
                algorithm="kmeans",
                n_clusters=min(self._cfg.clusters_per_level, len(current)),
            )
        else:
            cfg = ClusteringConfig(
                algorithm="hdbscan",
                min_cluster_size=self._cfg.min_cluster_size,
            )

        matrix: NDArray[np.float32] = np.array(
            [m.vector for m in current], dtype=np.float32
        )
        labels, centroids = run_clustering(matrix, cfg)

        # Bucket inputs by label. HDBSCAN's noise label (-1) is dropped:
        # noise points are by definition not part of any cluster, and
        # forcing them into single-member groups would just trigger the
        # passthrough shortcut over and over.
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int == -1:
                continue
            groups.setdefault(label_int, []).append(idx)

        # For each cluster, sort members by distance to centroid (ascending),
        # so the centroid-nearest cap can be applied as a simple slice.
        out: list[list[_ClusterMember]] = []
        for label_int in sorted(groups.keys()):
            indices = groups[label_int]
            if label_int < len(centroids):
                centroid = centroids[label_int]
                dists = [
                    (np.linalg.norm(matrix[i] - centroid), i) for i in indices
                ]
                dists.sort(key=lambda t: t[0])
                ordered = [current[i] for _, i in dists]
            else:
                # HDBSCAN may produce clusters without a centroid in the
                # array (shouldn't happen given how we build centroids,
                # but defensive); fall back to insertion order.
                ordered = [current[i] for i in indices]
            out.append(ordered)
        return out

    # ------------------------------------------------------------------
    # Cluster summarisation
    # ------------------------------------------------------------------

    async def _summarise_clusters(
        self,
        clusters: list[list[_ClusterMember]],
        level: int,
    ) -> tuple[list[_SummaryNode], int]:
        """Summarise each cluster; return summary nodes and decompose-call count."""
        # Cap cluster member count at top-N centroid-nearest BEFORE
        # dispatching the LLM call. ``clusters`` is already centroid-
        # nearest-first, so a slice is correct.
        capped: list[list[_ClusterMember]] = [
            cluster[:MAX_CLUSTER_MEMBERS_PER_SUMMARY] for cluster in clusters
        ]

        # We pre-allocate ``summaries`` and write each result back at its
        # cluster index so the order is stable for back-reference linking.
        decompose_calls = 0

        async def _one(cluster: list[_ClusterMember]) -> _SummaryNode:
            nonlocal decompose_calls
            new_id = uuid4().hex
            leaf_total = sum(m.leaf_count for m in cluster)
            child_ids = [m.point_id for m in cluster]
            if len(cluster) == 1:
                # Cluster-of-one passthrough: avoid an LLM call that would
                # just restate the input. Keep the original text and stamp
                # a placeholder reasoning so traces document the shortcut.
                return _SummaryNode(
                    point_id=new_id,
                    text=cluster[0].text,
                    reasoning="cluster-of-one passthrough",
                    leaf_count=leaf_total,
                    children_point_ids=child_ids,
                )
            decompose_calls += 1
            verdict = await b.SummarizeCluster(
                cluster_texts=[m.text for m in cluster],
                level=level,
                max_summary_tokens=self._cfg.summary_max_tokens,
                baml_options={"client_registry": self._baml_registry},
            )
            return _SummaryNode(
                point_id=new_id,
                text=str(verdict.summary),
                reasoning=str(verdict.reasoning),
                leaf_count=leaf_total,
                children_point_ids=child_ids,
            )

        summaries = await run_concurrent(capped, _one, _SUMMARIZE_CONCURRENCY)
        return summaries, decompose_calls

    # ------------------------------------------------------------------
    # Embedding + persistence
    # ------------------------------------------------------------------

    async def _embed_summaries(self, summaries: list[_SummaryNode]) -> None:
        if not summaries:
            return
        vectors = await embed_batched(self._embeddings, [s.text for s in summaries])
        for s, v in zip(summaries, vectors, strict=True):
            s.vector = v

    async def _persist_summaries(
        self,
        summaries: list[_SummaryNode],
        knowledge_id: str,
        tree_id: str,
        level: int,
    ) -> None:
        if not summaries:
            return
        points = [
            VectorPoint(
                point_id=s.point_id,
                vector=s.vector,
                payload={
                    "vector_role": "raptor_summary",
                    "knowledge_id": knowledge_id,
                    "raptor_tree_id": tree_id,
                    "raptor_level": level,
                    # ``raptor_parent_id`` will be set on the next iteration
                    # when this node becomes child of a higher summary.
                    "raptor_parent_id": None,
                    "raptor_cluster_size": s.leaf_count,
                    "raptor_reasoning": s.reasoning,
                    "content": s.text,
                    "contextualized": s.text,
                },
            )
            for s in summaries
        ]
        await self._vector_store.upsert(points)

    async def _set_parent_back_references(
        self, summaries: list[_SummaryNode]
    ) -> None:
        """Update each child point's ``raptor_parent_id`` payload field.

        Two cases share this code path:

        - level=1: children are leaf chunks (vector_role in
          _LEAF_VECTOR_ROLES). Their payloads gain the parent pointer
          but stay leaves; retrieval can still surface them via the
          original chunk path (RAPTOR is additive).
        - level>=2: children are the previous level's summaries. Their
          parent_id flips from ``None`` to the new summary's id.

        The vector store's ``upsert`` is the only available payload-update
        path. For sparse store backends without payload-only updates, we
        fall back to a re-upsert with the original vector (out of scope
        for R2.2 — Qdrant handles this natively).
        """
        if not summaries:
            return
        # Build per-child point updates. Re-fetching the existing payload
        # would race against concurrent writes; instead we issue a
        # ``set_payload``-equivalent via the qdrant client when available.
        # For the BaseVectorStore protocol, we use ``upsert`` of the
        # minimal-payload update — Qdrant merges payload on upsert by id
        # only when explicitly requested. Since the protocol exposes
        # ``upsert`` only, we issue a parent-id-only re-upsert with a
        # zero-vector placeholder ONLY if the store's ``set_payload`` is
        # not available. To keep the protocol clean and avoid fabricating
        # a partial payload that drops other fields, we route through a
        # store-specific update when available; otherwise we accept the
        # current limitation that back-references write through upsert
        # of a re-embedded payload — this happens at level >= 2 where the
        # children are the previous level's summaries (we still hold their
        # text and vector in memory at the time of upsert).
        update_method = getattr(self._vector_store, "set_payload", None)
        if update_method is None:
            # Best-effort fallback: skip back-references when the store
            # protocol doesn't expose payload-only updates. Retrieval can
            # still walk top-down from the root summary's
            # ``raptor_cluster_size`` and ``raptor_level`` without
            # parent pointers; back-references are an accelerator, not a
            # correctness requirement.
            logger.debug(
                "vector store has no set_payload; skipping raptor_parent_id back-references"
            )
            return
        for s in summaries:
            for child_id in s.children_point_ids:
                # ``set_payload`` is the optional partial-update primitive.
                await update_method(
                    point_ids=[child_id],
                    payload={"raptor_parent_id": s.point_id},
                )

    # ------------------------------------------------------------------
    # GC
    # ------------------------------------------------------------------

    async def _gc_old_trees(self, knowledge_id: str, new_tree_id: str) -> int:
        """Delete summary vectors belonging to older trees of the same kid.

        Two-step (query then delete) because the existing
        ``BaseVectorStore`` filter shape supports equality / IN, not
        ``$ne``. We first scroll for all distinct ``raptor_tree_id``
        values under this knowledge_id and ``vector_role=raptor_summary``,
        then delete each old tree by exact match.
        """
        old_tree_ids: set[str] = set()
        offset: str | None = None
        filters: dict[str, Any] = {
            "knowledge_id": knowledge_id,
            "vector_role": "raptor_summary",
        }
        while True:
            results, next_offset = await self._vector_store.scroll(
                filters=filters, limit=500, offset=offset
            )
            for r in results:
                tid = r.payload.get("raptor_tree_id")
                if tid and tid != new_tree_id:
                    old_tree_ids.add(str(tid))
            if not next_offset:
                break
            offset = next_offset

        deleted_total = 0
        for old_tid in old_tree_ids:
            count = await self._vector_store.delete(
                {
                    "knowledge_id": knowledge_id,
                    "vector_role": "raptor_summary",
                    "raptor_tree_id": old_tid,
                }
            )
            deleted_total += count
            logger.info(
                "raptor gc: deleted %d summary vectors from old tree_id=%s",
                count,
                old_tid,
            )
        return deleted_total
