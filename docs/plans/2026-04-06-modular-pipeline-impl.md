# Modular Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor retrieval and ingestion into protocol-based plugin architectures with self-contained methods, per-method error isolation, and a clean namespace API.

**Architecture:** Each retrieval/ingestion path becomes a standalone method class conforming to a Protocol. Services receive a list of methods and dispatch generically. `RagServer` exposes methods via `MethodNamespace` for both attribute access (`rag.retrieval.vector`) and iteration.

**Tech Stack:** Python 3.12, Protocol typing, asyncio, dataclasses, pytest with AsyncMock

**Design doc:** `docs/plans/2026-04-06-modular-pipeline.md`

---

### Task 1: MethodNamespace Generic Container

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/namespace.py`
- Test: `src/rfnry_rag/retrieval/tests/test_namespace.py`

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_namespace.py
from types import SimpleNamespace

import pytest

from x64rag.retrieval.modules.namespace import MethodNamespace


def _method(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_attribute_access():
    ns = MethodNamespace([_method("vector"), _method("document")])
    assert ns.vector.name == "vector"
    assert ns.document.name == "document"


def test_attribute_access_missing_raises():
    ns = MethodNamespace([_method("vector")])
    with pytest.raises(AttributeError, match="No method 'graph' configured"):
        ns.graph


def test_iteration():
    methods = [_method("vector"), _method("document")]
    ns = MethodNamespace(methods)
    names = [m.name for m in ns]
    assert names == ["vector", "document"]


def test_len():
    ns = MethodNamespace([_method("a"), _method("b"), _method("c")])
    assert len(ns) == 3


def test_contains():
    ns = MethodNamespace([_method("vector")])
    assert "vector" in ns
    assert "graph" not in ns


def test_empty():
    ns = MethodNamespace([])
    assert len(ns) == 0
    assert list(ns) == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_namespace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'x64rag.retrieval.modules.namespace'`

**Step 3: Write the implementation**

```python
# src/rfnry_rag/retrieval/modules/namespace.py
from __future__ import annotations

from typing import Any


class MethodNamespace[T]:
    """Exposes pipeline methods as attributes and supports iteration.

    Methods must have a ``name`` attribute used as the access key.
    """

    def __init__(self, methods: list[T]) -> None:
        self._methods: dict[str, T] = {}
        for method in methods:
            self._methods[method.name] = method  # type: ignore[union-attr]

    def __getattr__(self, name: str) -> T:
        try:
            return self._methods[name]
        except KeyError:
            raise AttributeError(f"No method '{name}' configured") from None

    def __iter__(self) -> Any:
        return iter(self._methods.values())

    def __len__(self) -> int:
        return len(self._methods)

    def __contains__(self, name: object) -> bool:
        return name in self._methods
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_namespace.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```
feat(pipeline): add MethodNamespace generic container
```

---

### Task 2: BaseRetrievalMethod Protocol

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/base.py`

**Step 1: Write the protocol**

```python
# src/rfnry_rag/retrieval/modules/retrieval/base.py
from __future__ import annotations

from typing import Any, Protocol

from x64rag.retrieval.common.models import RetrievedChunk


class BaseRetrievalMethod(Protocol):
    """Protocol for pluggable retrieval methods."""

    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]: ...
```

No test needed — this is a Protocol definition. It will be validated structurally when concrete classes implement it.

**Step 2: Run typecheck**

Run: `poe typecheck`
Expected: PASS

**Step 3: Commit**

```
feat(pipeline): add BaseRetrievalMethod protocol
```

---

### Task 3: VectorRetrieval (refactor VectorSearch + absorb BM25)

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/__init__.py`
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py`
- Test: `src/rfnry_rag/retrieval/tests/test_vector_retrieval.py`
- Reference: `src/rfnry_rag/retrieval/modules/retrieval/search/vector.py` (VectorSearch — source of dense/hybrid logic)
- Reference: `src/rfnry_rag/retrieval/modules/retrieval/search/keyword.py` (KeywordSearch — BM25 logic to absorb)

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_vector_retrieval.py
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import SparseVector, VectorResult
from x64rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval


async def test_dense_search():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        weight=1.0,
    )
    assert method.name == "vector"
    assert method.weight == 1.0

    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.search.assert_called_once()


async def test_hybrid_search_with_sparse():
    vector_store = AsyncMock()
    vector_store.hybrid_search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "test", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])
    sparse = AsyncMock()
    sparse.embed_sparse_query = AsyncMock(return_value=SparseVector(indices=[1], values=[0.8]))

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=sparse,
        weight=1.5,
    )
    results = await method.search(query="test", top_k=5)
    assert len(results) == 1
    vector_store.hybrid_search.assert_called_once()


async def test_bm25_enabled_fuses_results():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            VectorResult(
                point_id="p1",
                score=0.9,
                payload={"content": "matching content", "source_id": "s1", "chunk_type": "child", "parent_id": None},
            ),
        ]
    )
    vector_store.scroll = AsyncMock(
        return_value=(
            [
                VectorResult(
                    point_id="p1",
                    score=0.0,
                    payload={
                        "content": "matching content",
                        "source_id": "s1",
                        "chunk_type": "child",
                        "source_type": None,
                        "source_weight": 1.0,
                        "source_name": "",
                        "file_url": "",
                        "tags": [],
                        "page_number": None,
                        "section": None,
                    },
                ),
            ],
            None,
        )
    )
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        bm25_enabled=True,
        bm25_max_indexes=16,
        weight=1.0,
    )
    results = await method.search(query="matching content", top_k=5)
    assert len(results) >= 1


async def test_error_returns_empty():
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(side_effect=RuntimeError("connection lost"))
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1, 0.2]])

    method = VectorRetrieval(
        vector_store=vector_store,
        embeddings=embeddings,
        weight=1.0,
    )
    results = await method.search(query="test", top_k=5)
    assert results == []


async def test_name_and_weight_properties():
    method = VectorRetrieval(
        vector_store=AsyncMock(),
        embeddings=AsyncMock(),
        weight=2.5,
    )
    assert method.name == "vector"
    assert method.weight == 2.5


async def test_invalidate_cache():
    """VectorRetrieval with BM25 should support cache invalidation."""
    method = VectorRetrieval(
        vector_store=AsyncMock(),
        embeddings=AsyncMock(),
        bm25_enabled=True,
        weight=1.0,
    )
    # Should not raise
    await method.invalidate_cache(knowledge_id=None)
    await method.invalidate_cache(knowledge_id="kb-1")
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_vector_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create empty `__init__.py`:
```python
# src/rfnry_rag/retrieval/modules/retrieval/methods/__init__.py
```

`VectorRetrieval` combines logic from `VectorSearch` (dense/hybrid search, parent expansion, result-to-chunk conversion) and `KeywordSearch` (BM25 index building, caching, LRU eviction, search).

```python
# src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk, VectorResult
from x64rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from x64rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion
from x64rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("retrieval.methods.vector")

_GLOBAL_KEY = "__global__"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class _BM25Entry:
    index: BM25Okapi | None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = 0.0


class VectorRetrieval:
    """Chunk-level retrieval: dense + optional SPLADE + optional BM25, fused via RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        parent_expansion: bool = False,
        bm25_enabled: bool = False,
        bm25_max_indexes: int = 16,
        weight: float = 1.0,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._parent_expansion = parent_expansion
        self._bm25_enabled = bm25_enabled
        self._bm25_max_indexes = bm25_max_indexes
        self._weight = weight

        # BM25 internals
        self._bm25_cache: dict[str, _BM25Entry] = {}
        self._bm25_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "vector"

    @property
    def weight(self) -> float:
        return self._weight

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            result_lists: list[list[RetrievedChunk]] = []

            # Dense or hybrid search
            dense_results = await self._dense_search(query, top_k, filters)
            result_lists.append(dense_results)

            # BM25 search (if enabled)
            if self._bm25_enabled:
                bm25_results = await self._bm25_search(query, top_k, knowledge_id)
                if bm25_results:
                    result_lists.append(bm25_results)
                    logger.info("%d dense + %d bm25 candidates", len(dense_results), len(bm25_results))

            # Fuse if multiple result lists
            if len(result_lists) > 1:
                results = reciprocal_rank_fusion(result_lists)
            else:
                results = result_lists[0] if result_lists else []

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(results), elapsed)
            return results

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []

    async def invalidate_cache(self, knowledge_id: str | None = None) -> None:
        """Invalidate BM25 index cache for a knowledge_id."""
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        async with self._bm25_lock:
            self._bm25_cache.pop(key, None)

    # -- Dense / hybrid search (from VectorSearch) --

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        if self._sparse:
            dense_result, sparse_vector = await asyncio.gather(
                self._embeddings.embed([query]),
                self._sparse.embed_sparse_query(query),
            )
            query_vector = dense_result[0] if dense_result else None
            if not query_vector:
                logger.warning("embedding returned no vectors for query")
                return []
            results = await self._store.hybrid_search(
                vector=query_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filters=filters,
            )
            logger.info("%d candidates from hybrid search", len(results))
        else:
            vectors = await self._embeddings.embed([query])
            if not vectors:
                logger.warning("embedding returned no vectors for query")
                return []
            query_vector = vectors[0]
            results = await self._store.search(vector=query_vector, top_k=top_k, filters=filters)
            logger.info("%d candidates from dense search", len(results))

        results = [r for r in results if r.payload.get("chunk_type", "child") == "child"]

        if self._parent_expansion and results:
            results = await self._expand_parents(results)

        return [self._result_to_chunk(r) for r in results]

    async def _expand_parents(self, results: list[VectorResult]) -> list[VectorResult]:
        parent_ids = set()
        for r in results:
            pid = r.payload.get("parent_id")
            if pid:
                parent_ids.add(pid)

        if not parent_ids:
            return results

        parents = await self._store.retrieve(list(parent_ids))
        parent_map = {p.point_id: p for p in parents}

        expanded = []
        seen_parents: set[str] = set()

        for r in results:
            pid = r.payload.get("parent_id")
            if pid and pid in parent_map:
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                parent = parent_map[pid]
                expanded.append(
                    VectorResult(
                        point_id=r.point_id,
                        score=r.score,
                        payload={**parent.payload, "expanded_from_child": r.point_id},
                    )
                )
            else:
                expanded.append(r)

        return expanded

    @staticmethod
    def _result_to_chunk(r: VectorResult) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r.point_id,
            content=r.payload.get("content", ""),
            score=r.score,
            page_number=r.payload.get("page_number"),
            section=r.payload.get("section"),
            source_id=r.payload.get("source_id", ""),
            source_type=r.payload.get("source_type"),
            source_weight=r.payload.get("source_weight", 1.0),
            source_metadata={
                "name": r.payload.get("source_name", ""),
                "file_url": r.payload.get("file_url", ""),
                "tags": r.payload.get("tags", []),
                "chunk_type": r.payload.get("chunk_type", "child"),
                "parent_id": r.payload.get("parent_id"),
            },
        )

    # -- BM25 search (from KeywordSearch) --

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        knowledge_id: str | None,
    ) -> list[RetrievedChunk]:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        if key not in self._bm25_cache:
            await self._build_bm25_index(knowledge_id)

        entry = self._bm25_cache.get(key)
        if entry is None or entry.index is None or not entry.chunks:
            return []

        entry.last_used = time.monotonic()
        tokenized_query = _tokenize(query)
        scores = entry.index.get_scores(tokenized_query)

        scored = sorted(
            zip(scores, entry.chunks, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored:
            if score <= 0:
                break
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["point_id"],
                    content=chunk["content"],
                    score=float(score),
                    page_number=chunk["page_number"],
                    section=chunk["section"],
                    source_id=chunk["source_id"],
                    source_type=chunk["source_type"],
                    source_weight=chunk["source_weight"],
                    source_metadata={
                        "name": chunk["source_name"],
                        "file_url": chunk["file_url"],
                        "tags": chunk["tags"],
                    },
                )
            )
            if len(results) >= top_k:
                break

        return results

    async def _build_bm25_index(self, knowledge_id: str | None) -> None:
        key = knowledge_id if knowledge_id is not None else _GLOBAL_KEY
        if key in self._bm25_cache:
            return
        async with self._bm25_lock:
            if key in self._bm25_cache:
                return

            filters = {"knowledge_id": knowledge_id} if knowledge_id is not None else None
            all_chunks: list[dict[str, Any]] = []
            offset = None

            while True:
                results, next_offset = await self._store.scroll(filters=filters, limit=500, offset=offset)
                for r in results:
                    all_chunks.append(
                        {
                            "point_id": r.point_id,
                            "content": r.payload.get("content", ""),
                            "page_number": r.payload.get("page_number"),
                            "section": r.payload.get("section"),
                            "source_id": r.payload.get("source_id", ""),
                            "source_type": r.payload.get("source_type"),
                            "source_weight": r.payload.get("source_weight", 1.0),
                            "source_name": r.payload.get("source_name", ""),
                            "file_url": r.payload.get("file_url", ""),
                            "tags": r.payload.get("tags", []),
                        }
                    )
                if next_offset is None or not results:
                    break
                offset = next_offset

            self._evict_lru()

            if not all_chunks:
                self._bm25_cache[key] = _BM25Entry(index=None, last_used=time.monotonic())
                return

            loop = asyncio.get_running_loop()
            tokenized = await loop.run_in_executor(None, lambda: [_tokenize(c["content"]) for c in all_chunks])
            index = await loop.run_in_executor(None, lambda: BM25Okapi(tokenized))
            self._bm25_cache[key] = _BM25Entry(index=index, chunks=all_chunks, last_used=time.monotonic())
            logger.info("built bm25 index for knowledge_id=%s: %d chunks", knowledge_id, len(all_chunks))

    def _evict_lru(self) -> None:
        if len(self._bm25_cache) < self._bm25_max_indexes:
            return
        oldest_key = min(self._bm25_cache, key=lambda k: self._bm25_cache[k].last_used)
        del self._bm25_cache[oldest_key]
        logger.info("evicted bm25 index for key=%s (lru)", oldest_key)
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_vector_retrieval.py -v`
Expected: PASS

**Step 5: Run full suite to verify no regressions**

Run: `poe test && poe typecheck && poe check`
Expected: PASS

**Step 6: Commit**

```
feat(pipeline): add VectorRetrieval method (dense + sparse + BM25)
```

---

### Task 4: DocumentRetrieval Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/document.py`
- Test: `src/rfnry_rag/retrieval/tests/test_document_retrieval.py`
- Reference: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py:202-219` (`_content_matches_to_chunks` — extract this logic)

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_document_retrieval.py
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import ContentMatch
from x64rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval


async def test_search_converts_matches():
    store = AsyncMock()
    store.search_content = AsyncMock(
        return_value=[
            ContentMatch(
                source_id="src-1",
                title="Manual",
                excerpt="Excerpt text",
                score=0.85,
                match_type="fulltext",
                source_type="manuals",
            ),
        ]
    )
    method = DocumentRetrieval(document_store=store, weight=0.8)
    assert method.name == "document"
    assert method.weight == 0.8

    results = await method.search(query="test", top_k=10, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "fulltext:src-1"
    assert results[0].content == "Excerpt text"
    assert results[0].source_metadata["match_type"] == "fulltext"
    store.search_content.assert_called_once_with(query="test", knowledge_id="kb-1", top_k=10)


async def test_search_empty_results():
    store = AsyncMock()
    store.search_content = AsyncMock(return_value=[])

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.search_content = AsyncMock(side_effect=RuntimeError("db down"))

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_document_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/rfnry_rag/retrieval/modules/retrieval/methods/document.py
from __future__ import annotations

import time
from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import ContentMatch, RetrievedChunk
from x64rag.retrieval.stores.document.base import BaseDocumentStore

logger = get_logger("retrieval.methods.document")


class DocumentRetrieval:
    """Full-text / substring search on documents via the document store."""

    def __init__(
        self,
        document_store: BaseDocumentStore,
        weight: float = 1.0,
    ) -> None:
        self._store = document_store
        self._weight = weight

    @property
    def name(self) -> str:
        return "document"

    @property
    def weight(self) -> float:
        return self._weight

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            matches = await self._store.search_content(
                query=query, knowledge_id=knowledge_id, top_k=top_k
            )
            results = self._convert(matches)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(results), elapsed)
            return results
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []

    @staticmethod
    def _convert(matches: list[ContentMatch]) -> list[RetrievedChunk]:
        chunks = []
        for match in matches:
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"fulltext:{match.source_id}",
                    source_id=match.source_id,
                    content=match.excerpt,
                    score=match.score,
                    source_type=match.source_type,
                    source_metadata={
                        "title": match.title,
                        "match_type": match.match_type,
                    },
                )
            )
        return chunks
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_document_retrieval.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(pipeline): add DocumentRetrieval method
```

---

### Task 5: GraphRetrieval Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/retrieval/methods/graph.py`
- Test: `src/rfnry_rag/retrieval/tests/test_graph_retrieval_method.py`
- Reference: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py:166-200` (`_graph_results_to_chunks` — extract this logic)

Note: test file is `test_graph_retrieval_method.py` to avoid collision with existing `test_graph_retrieval.py`.

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_graph_retrieval_method.py
from unittest.mock import AsyncMock

from x64rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval
from x64rag.retrieval.stores.graph.models import GraphEntity, GraphPath, GraphResult


def _make_graph_results():
    return [
        GraphResult(
            entity=GraphEntity(
                name="Motor M1",
                entity_type="motor",
                category="electrical",
                value="480V 3-phase",
                properties={"source_id": "src-2"},
            ),
            connected_entities=[
                GraphEntity(name="Breaker CB-3", entity_type="breaker", category="electrical"),
                GraphEntity(name="VFD-3", entity_type="vfd", category="electrical"),
            ],
            paths=[
                GraphPath(
                    entities=["Motor M1", "Breaker CB-3", "Panel MCC-1"],
                    relationships=["POWERED_BY", "FEEDS"],
                ),
            ],
            relevance_score=0.95,
        ),
    ]


async def test_search_converts_graph_results():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=_make_graph_results())

    method = GraphRetrieval(graph_store=store, weight=0.7)
    assert method.name == "graph"
    assert method.weight == 0.7

    results = await method.search(query="Motor M1", top_k=5, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "graph:Motor M1:motor"
    assert results[0].score == 0.95
    assert "480V 3-phase" in results[0].content
    assert "POWERED_BY" in results[0].content
    assert results[0].source_metadata["retrieval_type"] == "graph"
    store.query_graph.assert_called_once_with(query="Motor M1", knowledge_id="kb-1", max_hops=2, top_k=5)


async def test_search_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=[])

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(side_effect=RuntimeError("neo4j down"))

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_graph_retrieval_method.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/rfnry_rag/retrieval/modules/retrieval/methods/graph.py
from __future__ import annotations

import time
from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.stores.graph.base import BaseGraphStore
from x64rag.retrieval.stores.graph.models import GraphResult

logger = get_logger("retrieval.methods.graph")


class GraphRetrieval:
    """Entity lookup + N-hop graph traversal via the graph store."""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        weight: float = 1.0,
    ) -> None:
        self._store = graph_store
        self._weight = weight

    @property
    def name(self) -> str:
        return "graph"

    @property
    def weight(self) -> float:
        return self._weight

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            results = await self._store.query_graph(
                query=query, knowledge_id=knowledge_id, max_hops=2, top_k=top_k
            )
            chunks = self._convert(results)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d results in %.1fms", len(chunks), elapsed)
            return chunks
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            return []

    @staticmethod
    def _convert(results: list[GraphResult]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for result in results:
            lines = [f"{result.entity.name} ({result.entity.entity_type})"]
            if result.entity.value:
                lines.append(f"  Specifications: {result.entity.value}")

            for path in result.paths:
                parts: list[str] = []
                for i, entity_name in enumerate(path.entities):
                    if i > 0 and i - 1 < len(path.relationships):
                        parts.append(f"-[{path.relationships[i - 1]}]->")
                    parts.append(entity_name)
                lines.append(f"  Path: {' '.join(parts)}")

            for connected in result.connected_entities[:5]:
                lines.append(f"  Connected: {connected.name} ({connected.entity_type})")

            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph:{result.entity.name}:{result.entity.entity_type}",
                    source_id=result.entity.properties.get("source_id", ""),
                    content="\n".join(lines),
                    score=result.relevance_score,
                    source_metadata={
                        "retrieval_type": "graph",
                        "entity_name": result.entity.name,
                        "entity_type": result.entity.entity_type,
                        "connected_count": len(result.connected_entities),
                    },
                )
            )
        return chunks
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_graph_retrieval_method.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(pipeline): add GraphRetrieval method
```

---

### Task 6: Refactor RetrievalService to Method List Dispatch

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`
- Test: `src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py`

**Step 1: Write the failing tests for the new interface**

```python
# src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService


def _mock_method(name: str, results: list[RetrievedChunk]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=1.0,
        search=AsyncMock(return_value=results),
    )


async def test_dispatch_single_method():
    vector = _mock_method("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
    ])
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    vector.search.assert_called_once()


async def test_dispatch_multiple_methods_fused():
    vector = _mock_method("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="vector", score=0.9),
    ])
    document = _mock_method("document", [
        RetrievedChunk(chunk_id="c2", source_id="s2", content="doc", score=0.8),
    ])
    service = RetrievalService(retrieval_methods=[vector, document], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 2
    ids = {r.chunk_id for r in results}
    assert "c1" in ids
    assert "c2" in ids


async def test_empty_method_list():
    service = RetrievalService(retrieval_methods=[], top_k=5)
    results = await service.retrieve(query="test")
    assert results == []


async def test_failed_method_returns_empty_others_succeed():
    """A method returning empty (simulating caught error) doesn't break others."""
    vector = _mock_method("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
    ])
    graph = _mock_method("graph", [])  # Simulates a failed method
    service = RetrievalService(retrieval_methods=[vector, graph], top_k=5)
    results = await service.retrieve(query="test")
    assert len(results) == 1


async def test_tree_chunks_injected():
    vector = _mock_method("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
    ])
    tree_chunk = RetrievedChunk(chunk_id="tree-1", source_id="s2", content="tree", score=0.7)
    service = RetrievalService(retrieval_methods=[vector], top_k=5)
    results = await service.retrieve(query="test", tree_chunks=[tree_chunk])
    ids = {r.chunk_id for r in results}
    assert "tree-1" in ids


async def test_reranker_applied():
    vector = _mock_method("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.5),
        RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.3),
    ])
    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=[
        RetrievedChunk(chunk_id="c2", source_id="s2", content="text2", score=0.95),
    ])
    service = RetrievalService(retrieval_methods=[vector], reranking=reranker, top_k=1)
    results = await service.retrieve(query="test")
    assert len(results) == 1
    assert results[0].chunk_id == "c2"
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py -v`
Expected: FAIL with `TypeError` (constructor signature mismatch)

**Step 3: Refactor RetrievalService**

Replace the full file `src/rfnry_rag/retrieval/modules/retrieval/search/service.py` with:

```python
# src/rfnry_rag/retrieval/modules/retrieval/search/service.py
import asyncio
from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.base import BaseRetrievalMethod
from x64rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefiner
from x64rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion
from x64rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking
from x64rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriter

logger = get_logger("retrieval.search.service")


class RetrievalService:
    def __init__(
        self,
        retrieval_methods: list[BaseRetrievalMethod],
        reranking: BaseReranking | None = None,
        top_k: int = 5,
        source_type_weights: dict[str, float] | None = None,
        query_rewriter: BaseQueryRewriter | None = None,
        chunk_refiner: BaseChunkRefiner | None = None,
    ) -> None:
        self._retrieval_methods = retrieval_methods
        self._reranking = reranking
        self._top_k = top_k
        self._source_type_weights = source_type_weights
        self._query_rewriter = query_rewriter
        self._chunk_refiner = chunk_refiner

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None = None,
        top_k: int | None = None,
        tree_chunks: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        top_k = top_k if top_k is not None else self._top_k
        fetch_k = top_k * 4
        filters = self._build_filters(knowledge_id)

        logger.info('query: "%s" (knowledge_id=%s)', query[:80], knowledge_id)

        queries = [query]
        if self._query_rewriter:
            try:
                rewritten = await self._query_rewriter.rewrite(query)
                queries.extend(rewritten)
                if rewritten:
                    logger.info(
                        "query rewriting: %d total queries (1 original + %d rewritten)",
                        len(queries),
                        len(rewritten),
                    )
            except Exception as exc:
                logger.exception("query rewriter failed: %s — proceeding with original query", exc)

        search_tasks = [self._search_single_query(q, fetch_k, filters, knowledge_id) for q in queries]
        query_results = await asyncio.gather(*search_tasks)

        all_result_lists: list[list[RetrievedChunk]] = []
        for result_lists in query_results:
            all_result_lists.extend(result_lists)

        if tree_chunks:
            all_result_lists.append(tree_chunks)
            logger.info("%d tree search candidates added to fusion", len(tree_chunks))

        if len(all_result_lists) > 1:
            fused = reciprocal_rank_fusion(all_result_lists, source_type_weights=self._source_type_weights)
            logger.info("%d unique after reciprocal rank fusion", len(fused))
        elif all_result_lists:
            fused = self._apply_source_weights(all_result_lists[0])
        else:
            fused = []

        if self._reranking and fused:
            fused = await self._reranking.rerank(query, fused, top_k=top_k)
            logger.info("top %d selected after reranking", len(fused))
        else:
            fused = fused[:top_k]

        if self._chunk_refiner and fused:
            fused = await self._chunk_refiner.refine(query, fused)
            logger.info("chunk refinement: %d chunks after refinement", len(fused))

        return fused

    async def _search_single_query(
        self,
        query: str,
        fetch_k: int,
        filters: dict[str, Any] | None,
        knowledge_id: str | None,
    ) -> list[list[RetrievedChunk]]:
        """Run all retrieval methods in parallel for a single query."""
        if not self._retrieval_methods:
            return []

        gathered = await asyncio.gather(*(
            method.search(query=query, top_k=fetch_k, filters=filters, knowledge_id=knowledge_id)
            for method in self._retrieval_methods
        ))
        return [results for results in gathered if results]

    def _apply_source_weights(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self._source_type_weights:
            return results
        from dataclasses import replace

        weighted = []
        for r in results:
            weighted.append(replace(r, score=r.score * r.source_weight))
        weighted.sort(key=lambda x: x.score, reverse=True)
        return weighted

    @staticmethod
    def _build_filters(knowledge_id: str | None) -> dict[str, Any] | None:
        if knowledge_id is None:
            return None
        return {"knowledge_id": knowledge_id}
```

**Step 4: Run new tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py -v`
Expected: PASS

**Step 5: Commit**

```
refactor(pipeline): RetrievalService uses method list dispatch
```

---

### Task 7: Migrate Existing Retrieval Tests

**Files:**
- Modify: `src/rfnry_rag/retrieval/tests/test_fulltext_retrieval.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_graph_retrieval.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_hybrid_retrieval.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_query_rewriting.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_server_query.py`

**Step 1: Migrate test_fulltext_retrieval.py**

Update `_make_service` to use method list. Move `_content_matches_to_chunks` tests to test `DocumentRetrieval._convert` directly.

- Replace `VectorSearch` mock with a `SimpleNamespace` mock method matching `BaseRetrievalMethod`
- Replace `RetrievalService(vector_search=..., document_store=...)` with `RetrievalService(retrieval_methods=[mock_vector, mock_document])`
- Replace `RetrievalService._content_matches_to_chunks(...)` call with `DocumentRetrieval._convert(...)`

**Step 2: Migrate test_graph_retrieval.py**

Same pattern:
- Replace `_make_service(graph_store=...)` with method list including a `GraphRetrieval` mock or `SimpleNamespace`
- Replace `RetrievalService._graph_results_to_chunks(...)` with `GraphRetrieval._convert(...)`

**Step 3: Migrate test_hybrid_retrieval.py**

- Replace `VectorSearch(...)` with `VectorRetrieval(...)`, add `weight=1.0`
- Constructor now takes `weight` param

**Step 4: Migrate test_query_rewriting.py**

- Replace `_make_retrieval_service` to use `retrieval_methods=[mock_vector]`
- Replace `RetrievalService(vector_search=..., keyword_search=None, ...)` with `RetrievalService(retrieval_methods=[mock_vector], ...)`

**Step 5: Migrate test_server_query.py**

- Replace `server._unstructured_retrieval` with `server._retrieval_service`
- Remove `server._keyword_search = None`

**Step 6: Run full test suite**

Run: `poe test`
Expected: All 487+ tests pass

**Step 7: Commit**

```
refactor(pipeline): migrate retrieval tests to method list interface
```

---

### Task 8: BaseIngestionMethod Protocol

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/base.py`

**Step 1: Write the protocol**

```python
# src/rfnry_rag/retrieval/modules/ingestion/base.py
from __future__ import annotations

from typing import Any, Protocol

from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage


class BaseIngestionMethod(Protocol):
    """Protocol for pluggable ingestion methods."""

    @property
    def name(self) -> str: ...

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None: ...

    async def delete(self, source_id: str) -> None: ...
```

**Step 2: Run typecheck**

Run: `poe typecheck`
Expected: PASS

**Step 3: Commit**

```
feat(pipeline): add BaseIngestionMethod protocol
```

---

### Task 9: VectorIngestion Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/methods/__init__.py`
- Create: `src/rfnry_rag/retrieval/modules/ingestion/methods/vector.py`
- Test: `src/rfnry_rag/retrieval/tests/test_vector_ingestion.py`
- Reference: `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py:105-193` (`_embed_and_store_incremental`, `_build_points`, `_embed_sparse_safe`)

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_vector_ingestion.py
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.common.models import SparseVector
from x64rag.retrieval.modules.ingestion.methods.vector import VectorIngestion


def _make_chunks(n=1):
    chunks = []
    for i in range(n):
        chunk = MagicMock()
        chunk.content = f"chunk {i}"
        chunk.embedding_text = f"chunk {i}"
        chunk.context = ""
        chunk.contextualized = ""
        chunk.page_number = 1
        chunk.section = None
        chunk.chunk_type = "child"
        chunk.parent_id = None
        chunks.append(chunk)
    return chunks


async def test_ingest_embeds_and_upserts():
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=embeddings,
        embedding_model_name="test:model",
    )
    assert method.name == "vector"

    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        source_weight=1.0,
        title="Test",
        full_text="chunk 0",
        chunks=_make_chunks(1),
        tags=[],
        metadata={},
    )
    embeddings.embed.assert_called_once()
    vector_store.upsert.assert_called_once()


async def test_ingest_with_sparse():
    embeddings = AsyncMock()
    embeddings.embed = AsyncMock(return_value=[[0.1] * 128])
    sparse = AsyncMock()
    sparse.embed_sparse = AsyncMock(return_value=[SparseVector(indices=[1], values=[0.8])])
    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=embeddings,
        sparse_embeddings=sparse,
        embedding_model_name="test:model",
    )
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="chunk 0",
        chunks=_make_chunks(1),
        tags=[],
        metadata={},
    )
    sparse.embed_sparse.assert_called_once()
    points = vector_store.upsert.call_args[0][0]
    assert points[0].sparse_vector is not None


async def test_delete():
    vector_store = AsyncMock()
    vector_store.delete = AsyncMock()

    method = VectorIngestion(
        vector_store=vector_store,
        embeddings=AsyncMock(),
        embedding_model_name="test:model",
    )
    await method.delete("src-1")
    vector_store.delete.assert_called_once_with({"source_id": "src-1"})
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_vector_ingestion.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create empty `__init__.py`:
```python
# src/rfnry_rag/retrieval/modules/ingestion/methods/__init__.py
```

```python
# src/rfnry_rag/retrieval/modules/ingestion/methods/vector.py
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import SparseVector, VectorPoint
from x64rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.utils import embed_batched
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("ingestion.methods.vector")

INGESTION_BATCH_SIZE = 20


class VectorIngestion:
    """Embed chunks and store as vector points."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        embedding_model_name: str,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._sparse = sparse_embeddings
        self._embedding_model_name = embedding_model_name

    @property
    def name(self) -> str:
        return "vector"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            texts = [c.embedding_text for c in chunks]
            vectors = await embed_batched(self._embeddings, texts)
            sparse_vectors = await self._embed_sparse_safe(texts)

            if vectors:
                await self._store.initialize(len(vectors[0]))

            points = self._build_points(
                source_id, chunks, vectors, sparse_vectors,
                tags, metadata, knowledge_id, source_type, source_weight,
            )
            await self._store.upsert(points)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d chunks embedded in %.1fms", len(chunks), elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            raise

    async def delete(self, source_id: str) -> None:
        await self._store.delete({"source_id": source_id})

    async def _embed_sparse_safe(self, texts: list[str]) -> list[SparseVector] | None:
        if not self._sparse:
            return None
        try:
            return await self._sparse.embed_sparse(texts)
        except Exception as exc:
            logger.warning("sparse embedding failed, continuing without: %s", exc)
            return None

    @staticmethod
    def _build_points(
        source_id: str,
        chunks: list[ChunkedContent],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None,
        tags: list[str],
        metadata: dict[str, Any],
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        chunk_index_offset: int = 0,
    ) -> list[VectorPoint]:
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            sparse = sparse_vectors[idx] if sparse_vectors else None
            point_id = chunk.parent_id if chunk.chunk_type == "parent" and chunk.parent_id else str(uuid4())
            points.append(
                VectorPoint(
                    point_id=point_id,
                    vector=vector,
                    sparse_vector=sparse,
                    payload={
                        "content": chunk.content,
                        "context": chunk.context,
                        "contextualized": chunk.contextualized,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "chunk_index": chunk_index_offset + idx,
                        "source_id": source_id,
                        "knowledge_id": knowledge_id,
                        "source_type": source_type,
                        "source_weight": source_weight,
                        "chunk_type": chunk.chunk_type,
                        "parent_id": chunk.parent_id,
                        "tags": tags,
                        "source_name": metadata.get("name", ""),
                        "file_url": metadata.get("file_url", ""),
                    },
                )
            )
        return points
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_vector_ingestion.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(pipeline): add VectorIngestion method
```

---

### Task 10: DocumentIngestion Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/methods/document.py`
- Test: `src/rfnry_rag/retrieval/tests/test_document_ingestion.py`

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_document_ingestion.py
from unittest.mock import AsyncMock

from x64rag.retrieval.modules.ingestion.methods.document import DocumentIngestion


async def test_ingest_stores_content():
    store = AsyncMock()
    store.store_content = AsyncMock()

    method = DocumentIngestion(document_store=store)
    assert method.name == "document"

    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        source_weight=1.0,
        title="Test Doc",
        full_text="Full document text here.",
        chunks=[],
        tags=[],
        metadata={},
    )
    store.store_content.assert_called_once_with(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="manuals",
        title="Test Doc",
        content="Full document text here.",
    )


async def test_delete():
    store = AsyncMock()
    store.delete_content = AsyncMock()

    method = DocumentIngestion(document_store=store)
    await method.delete("src-1")
    store.delete_content.assert_called_once_with("src-1")
```

**Step 2: Run tests to verify they fail**

Run: `pytest src/rfnry_rag/retrieval/tests/test_document_ingestion.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# src/rfnry_rag/retrieval/modules/ingestion/methods/document.py
from __future__ import annotations

import time
from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.stores.document.base import BaseDocumentStore

logger = get_logger("ingestion.methods.document")


class DocumentIngestion:
    """Store full document text in the document store."""

    def __init__(self, document_store: BaseDocumentStore) -> None:
        self._store = document_store

    @property
    def name(self) -> str:
        return "document"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None:
        start = time.perf_counter()
        try:
            await self._store.store_content(
                source_id=source_id,
                knowledge_id=knowledge_id,
                source_type=source_type,
                title=title,
                content=full_text,
            )
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("stored in %.1fms", elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)
            raise

    async def delete(self, source_id: str) -> None:
        await self._store.delete_content(source_id)
```

**Step 4: Run tests to verify they pass**

Run: `pytest src/rfnry_rag/retrieval/tests/test_document_ingestion.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(pipeline): add DocumentIngestion method
```

---

### Task 11: GraphIngestion Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py`
- Test: `src/rfnry_rag/retrieval/tests/test_graph_ingestion.py`

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_graph_ingestion.py
from unittest.mock import AsyncMock

from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion


async def test_ingest_delegates_to_store():
    store = AsyncMock()
    store.store_entities = AsyncMock()

    method = GraphIngestion(graph_store=store)
    assert method.name == "graph"

    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="Entity A connects to Entity B.",
        chunks=[],
        tags=[],
        metadata={},
    )
    store.store_entities.assert_called_once()


async def test_delete():
    store = AsyncMock()
    store.delete_entities = AsyncMock()

    method = GraphIngestion(graph_store=store)
    await method.delete("src-1")
    store.delete_entities.assert_called_once_with("src-1")
```

Note: The exact `GraphIngestion.ingest()` implementation depends on how entity extraction currently works in `StructuredIngestionService`. The test above validates the store delegation pattern. The implementation will need to be refined when wiring to the actual graph extraction logic — see the design doc section on `GraphIngestion`.

**Step 2: Run tests, implement, verify pass**

Follow same pattern as previous tasks.

**Step 3: Commit**

```
feat(pipeline): add GraphIngestion method
```

---

### Task 12: TreeIngestion Method

**Files:**
- Create: `src/rfnry_rag/retrieval/modules/ingestion/methods/tree.py`
- Test: `src/rfnry_rag/retrieval/tests/test_tree_ingestion.py`

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_tree_ingestion.py
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.methods.tree import TreeIngestion


async def test_ingest_with_pages():
    tree_service = AsyncMock()
    tree_service.build_tree_index = AsyncMock(return_value=MagicMock())
    tree_service.save_tree_index = AsyncMock()

    method = TreeIngestion(tree_service=tree_service)
    assert method.name == "tree"

    pages = [MagicMock(page_number=1, content="Page 1 text")]
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test Doc",
        full_text="Page 1 text",
        chunks=[],
        tags=[],
        metadata={},
        pages=pages,
    )
    tree_service.build_tree_index.assert_called_once()
    tree_service.save_tree_index.assert_called_once()


async def test_ingest_without_pages_skips():
    tree_service = AsyncMock()
    method = TreeIngestion(tree_service=tree_service)

    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="text",
        chunks=[],
        tags=[],
        metadata={},
        pages=None,
    )
    tree_service.build_tree_index.assert_not_called()


async def test_delete_is_noop():
    tree_service = AsyncMock()
    method = TreeIngestion(tree_service=tree_service)
    await method.delete("src-1")  # Should not raise
```

**Step 2: Run tests, implement, verify pass**

```python
# src/rfnry_rag/retrieval/modules/ingestion/methods/tree.py
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage

if TYPE_CHECKING:
    from x64rag.retrieval.modules.ingestion.tree.service import TreeIndexingService

logger = get_logger("ingestion.methods.tree")


class TreeIngestion:
    """Build and persist tree index from parsed pages."""

    def __init__(self, tree_service: TreeIndexingService) -> None:
        self._service = tree_service

    @property
    def name(self) -> str:
        return "tree"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None:
        if not pages:
            logger.info("skipped — no pages provided")
            return

        start = time.perf_counter()
        try:
            from x64rag.retrieval.modules.ingestion.tree.toc import PageContent

            page_contents = [
                PageContent(index=p.page_number, text=p.content, token_count=len(p.content) // 4)
                for p in pages
            ]
            tree_idx = await self._service.build_tree_index(
                source_id=source_id,
                doc_name=title,
                pages=page_contents,
            )
            await self._service.save_tree_index(tree_idx)

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("built tree index (%d pages) in %.1fms", len(pages), elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)

    async def delete(self, source_id: str) -> None:
        pass  # Tree indexes are cleaned up via metadata store
```

**Step 3: Commit**

```
feat(pipeline): add TreeIngestion method
```

---

### Task 13: Refactor IngestionService to Method List Dispatch

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py`
- Test: `src/rfnry_rag/retrieval/tests/test_ingestion_service_methods.py`

This is the most complex refactor. The key change: `IngestionService` no longer owns embedding, vector store, or document store directly. It delegates to `ingestion_methods`.

**Step 1: Write the failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_ingestion_service_methods.py
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _mock_method(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        ingest=AsyncMock(),
        delete=AsyncMock(),
    )


def _make_service(methods=None, metadata_store=None):
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(
                content="chunk text",
                embedding_text="chunk text",
                page_number=1,
                section=None,
                chunk_index=0,
                context="",
                contextualized="",
                parent_id=None,
                chunk_type="child",
            ),
        ]
    )
    return IngestionService(
        chunker=chunker,
        ingestion_methods=methods or [],
        metadata_store=metadata_store,
    )


async def test_ingest_text_delegates_to_methods():
    vector = _mock_method("vector")
    document = _mock_method("document")
    service = _make_service(methods=[vector, document])

    await service.ingest_text(content="Hello world", metadata={"name": "test"})
    vector.ingest.assert_called_once()
    document.ingest.assert_called_once()


async def test_ingest_text_no_methods():
    service = _make_service(methods=[])
    source = await service.ingest_text(content="Hello world")
    assert source is not None
```

**Step 2: Refactor IngestionService**

This is a significant refactor. The new constructor signature:
```python
def __init__(
    self,
    chunker: SemanticChunker,
    ingestion_methods: list[BaseIngestionMethod],
    embedding_model_name: str = "",
    source_type_weights: dict[str, float] | None = None,
    metadata_store: BaseMetadataStore | None = None,
    on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
    vision_parser: BaseVision | None = None,
    contextual_chunking: bool = True,
) -> None:
```

The `ingest()` and `ingest_text()` methods change: after chunking, iterate `ingestion_methods` and call `method.ingest(...)` for each. Remove `_embed_and_store_incremental`, `_build_points`, `_embed_sparse_safe` (moved to `VectorIngestion`). Remove `document_store` blocks (moved to `DocumentIngestion`).

**Important:** `_check_duplicate()` and `_resolve_weight()` stay (orchestration concerns). `ingest()` still handles file parsing, chunking, and contextual chunking. Metadata store interactions (create source) stay.

**Step 3: Run new + existing tests**

Run: `poe test`
Expected: All pass

**Step 4: Commit**

```
refactor(pipeline): IngestionService uses method list dispatch
```

---

### Task 14: Migrate Existing Ingestion Tests

**Files:**
- Modify: `src/rfnry_rag/retrieval/tests/test_ingestion_advanced.py`

Update `_make_service` to use `ingestion_methods=[mock_vector]` instead of `embeddings=..., vector_store=...`.

**Step 1: Migrate, run, verify pass**

Run: `poe test`
Expected: PASS

**Step 2: Commit**

```
refactor(pipeline): migrate ingestion tests to method list interface
```

---

### Task 15: Config Validation and Optional Fields

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py`

**Step 1: Make fields optional**

In `PersistenceConfig`: change `vector_store: BaseVectorStore` to `vector_store: BaseVectorStore | None = None`

In `IngestionConfig`: change `embeddings: BaseEmbeddings` to `embeddings: BaseEmbeddings | None = None`

**Step 2: Add `_validate_config()` method**

Add to `RagServer` class, before `initialize()`:

```python
def _validate_config(self) -> None:
    cfg = self._config
    p = cfg.persistence
    i = cfg.ingestion

    has_vector = p.vector_store is not None and i.embeddings is not None
    has_document = p.document_store is not None
    has_graph = p.graph_store is not None

    if not any([has_vector, has_document, has_graph]):
        raise ConfigurationError(
            "At least one retrieval path must be configured: "
            "vector (vector_store + embeddings), "
            "document (document_store), or graph (graph_store)"
        )

    if p.vector_store and not i.embeddings:
        raise ConfigurationError("vector_store requires embeddings")
    if i.embeddings and not p.vector_store:
        raise ConfigurationError("embeddings requires vector_store")

    if has_graph and not i.lm_config:
        raise ConfigurationError(
            "graph_store requires ingestion.lm_config for entity extraction"
        )

    if cfg.tree_indexing.enabled and not p.metadata_store:
        raise ConfigurationError("tree_indexing requires metadata_store")
    if cfg.tree_search.enabled and not p.metadata_store:
        raise ConfigurationError("tree_search requires metadata_store")
```

**Step 3: Call `_validate_config()` at top of `initialize()`**

**Step 4: Run typecheck and tests**

Run: `poe typecheck && poe test`
Expected: PASS

**Step 5: Commit**

```
feat(pipeline): make vector_store and embeddings optional, add config validation
```

---

### Task 16: Refactor RagServer.initialize() — Dynamic Assembly + Namespaces

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py`

This is the final wiring task. Replace the hardcoded service construction in `initialize()` with dynamic assembly of method lists.

**Step 1: Update imports**

Remove:
- `from x64rag.retrieval.modules.retrieval.search.keyword import KeywordSearch`
- `from x64rag.retrieval.modules.retrieval.search.vector import VectorSearch`

Add:
- `from x64rag.retrieval.modules.namespace import MethodNamespace`
- `from x64rag.retrieval.modules.retrieval.base import BaseRetrievalMethod`
- `from x64rag.retrieval.modules.retrieval.methods.vector import VectorRetrieval`
- `from x64rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval`
- `from x64rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval`
- `from x64rag.retrieval.modules.ingestion.base import BaseIngestionMethod`
- `from x64rag.retrieval.modules.ingestion.methods.vector import VectorIngestion`
- `from x64rag.retrieval.modules.ingestion.methods.document import DocumentIngestion`
- `from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion`
- `from x64rag.retrieval.modules.ingestion.methods.tree import TreeIngestion`

**Step 2: Update `__init__` fields**

Remove:
- `self._keyword_search`
- `self._unstructured_retrieval`
- `self._unstructured_ingestion`

Add:
- `self._retrieval_service: RetrievalService | None = None`
- `self._ingestion_service: IngestionService | None = None`
- `self._retrieval_namespace: MethodNamespace[BaseRetrievalMethod] | None = None`
- `self._ingestion_namespace: MethodNamespace[BaseIngestionMethod] | None = None`

**Step 3: Rewrite `initialize()` body**

Follow the assembly pattern from the design doc's "Server Assembly" section. Build `ingestion_methods` and `retrieval_methods` lists conditionally, then construct `MethodNamespace`, `IngestionService`, and `RetrievalService`.

**Step 4: Add namespace properties**

```python
@property
def retrieval(self) -> MethodNamespace[BaseRetrievalMethod]:
    self._check_initialized()
    assert self._retrieval_namespace is not None
    return self._retrieval_namespace

@property
def ingestion(self) -> MethodNamespace[BaseIngestionMethod]:
    self._check_initialized()
    assert self._ingestion_namespace is not None
    return self._ingestion_namespace
```

**Step 5: Update `_on_ingestion_complete` and `_on_source_removed`**

```python
async def _on_ingestion_complete(self, knowledge_id: str | None) -> None:
    if self._retrieval_namespace and "vector" in self._retrieval_namespace:
        vector = self._retrieval_namespace.vector
        if hasattr(vector, "invalidate_cache"):
            await vector.invalidate_cache(knowledge_id)

async def _on_source_removed(self, knowledge_id: str | None) -> None:
    if self._retrieval_namespace and "vector" in self._retrieval_namespace:
        vector = self._retrieval_namespace.vector
        if hasattr(vector, "invalidate_cache"):
            await vector.invalidate_cache(knowledge_id)
```

**Step 6: Update `_get_retrieval()` and `_get_ingestion()`**

Replace references to `self._unstructured_retrieval` with `self._retrieval_service`.
Replace references to `self._unstructured_ingestion` with `self._ingestion_service`.

**Step 7: Update `_build_retrieval_pipeline()` and `_build_ingestion_service()`**

These per-collection builders now construct method lists instead of individual services.

**Step 8: Update `_enabled_flows()`**

```python
def _enabled_flows(self) -> list[str]:
    flows = []
    if self._retrieval_namespace:
        flows.extend(m.name for m in self._retrieval_namespace)
    if self._structured_ingestion:
        flows.append("structured")
    if self._generation_service:
        flows.append("generation")
    if self._tree_search_service:
        flows.append("tree-search")
    return flows
```

**Step 9: Run full suite**

Run: `poe test && poe typecheck && poe check`
Expected: All pass

**Step 10: Commit**

```
refactor(pipeline): dynamic server assembly with method namespaces
```

---

### Task 17: Update Public Exports

**Files:**
- Modify: `src/rfnry_rag/retrieval/__init__.py`

**Step 1: Add new exports**

Add imports and `__all__` entries for:
- `BaseRetrievalMethod`
- `BaseIngestionMethod`
- `VectorRetrieval`, `DocumentRetrieval`, `GraphRetrieval`
- `VectorIngestion`, `DocumentIngestion`, `GraphIngestion`, `TreeIngestion`
- `MethodNamespace`

**Step 2: Run typecheck**

Run: `poe typecheck && poe check`
Expected: PASS

**Step 3: Commit**

```
feat(pipeline): export new method protocols and classes
```

---

### Task 18: Final Verification

**Step 1: Run everything**

```bash
poe test && poe typecheck && poe check
```

Expected: All tests pass, no type errors, no lint issues.

**Step 2: Review test count**

Run: `pytest src/rfnry_rag/retrieval/tests/ -v --tb=short | tail -5`

Expected: Original 487+ tests plus new tests all passing.

**Step 3: Final commit if any cleanup needed**

```
chore(pipeline): final cleanup after modular pipeline refactor
```
