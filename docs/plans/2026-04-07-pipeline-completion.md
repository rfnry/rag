# Pipeline Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the modular pipeline: wire GraphIngestion with LLM entity extraction, refactor AnalyzedIngestionService to use method list, and make `weight`/`top_k` work in retrieval dispatch.

**Architecture:** GraphIngestion gets a new BAML function (`ExtractEntitiesFromText`) that extracts entities from plain text, then uses existing graph mappers to store them. AnalyzedIngestionService phase 3 delegates to ingestion methods. RRF fusion uses per-method weights, and each method can override top_k.

**Tech Stack:** Python 3.12, BAML (structured LLM output), asyncio, Protocol typing, pytest

---

### Task 1: Create ExtractEntitiesFromText BAML Function

**Files:**
- Modify: `src/rfnry_rag/retrieval/baml/baml_src/ingestion/functions.baml`
- No test — BAML functions are validated by regenerating the client

The existing `AnalyzePage(image) -> PageAnalysis` is vision-only. We need a text equivalent that reuses the same `PageAnalysis` output type (entities, description, page_type). This lets us reuse the `page_entities_to_graph()` mapper unchanged.

**Step 1: Add the new BAML function**

Append to `src/rfnry_rag/retrieval/baml/baml_src/ingestion/functions.baml`:

```baml
function ExtractEntitiesFromText(text: string) -> PageAnalysis {
  client Default
  prompt #"
    You are a technical document analyst. Extract all identifiable entities and structured information from this text.

    ======== TEXT ========
    {{ text }}
    ======== END TEXT ========

    Instructions:
    - Describe what this text contains in the description field
    - Extract all identifiable entities (components, parts, equipment, materials, specifications, etc.)
    - For each entity, provide:
      - name: the identifier or label (e.g., "Motor M1", "SAE 30", "RV-2201")
      - category: descriptive type (component, valve, motor, material, specification, part_number, etc.)
      - value: any associated measurement or specification (e.g., "480V", "175 PSI", "3450 RPM")
      - context: brief note about how this entity appears in the text
    - Extract any tables with column headers and row data
    - Set page_type to "text"
    - Note any reference marks, part numbers, or cross-references in annotations

    ======== IMPORTANT ========
    - Extract what you actually see — do NOT invent or hallucinate entities or values
    - If a value is unclear, use the entity's context field to note uncertainty
    - Categories should be descriptive (component, valve, motor, material, specification, part_number, etc.)
    ======== END IMPORTANT ========

    {{ ctx.output_format }}
  "#
}
```

**Step 2: Regenerate BAML client**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run poe baml:generate:retrieval`
Expected: Success, new `ExtractEntitiesFromText` function available in `baml_client`

**Step 3: Verify the generated client has the function**

Run: `grep -n "ExtractEntitiesFromText" src/rfnry_rag/retrieval/baml/baml_client/async_client.py | head -3`
Expected: Function definition found

**Step 4: Commit**

```
feat(baml): add ExtractEntitiesFromText function for text-based entity extraction
```

---

### Task 2: Wire GraphIngestion with Entity Extraction

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_graph_ingestion_method.py`

GraphIngestion is currently a stub. Replace it with actual entity extraction using the new BAML function + existing mappers.

**Step 1: Write the failing tests**

Replace `src/rfnry_rag/retrieval/tests/test_graph_ingestion_method.py` with:

```python
from unittest.mock import AsyncMock, MagicMock, patch

from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion
from x64rag.retrieval.stores.graph.models import GraphEntity


async def test_ingest_extracts_entities_and_stores():
    store = AsyncMock()
    store.add_entities = AsyncMock()
    lm_config = MagicMock()

    # Mock BAML response
    mock_entity = MagicMock()
    mock_entity.name = "Motor M1"
    mock_entity.category = "motor"
    mock_entity.value = "480V"
    mock_entity.context = "main motor"

    mock_result = MagicMock()
    mock_result.description = "Technical specifications"
    mock_result.entities = [mock_entity]
    mock_result.tables = []
    mock_result.annotations = []
    mock_result.page_type = "text"

    with (
        patch("x64rag.retrieval.modules.ingestion.methods.graph.b") as mock_b,
        patch("x64rag.retrieval.modules.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        method = GraphIngestion(graph_store=store, lm_config=lm_config)
        assert method.name == "graph"

        await method.ingest(
            source_id="src-1",
            knowledge_id="kb-1",
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="Motor M1 operates at 480V.",
            chunks=[],
            tags=[],
            metadata={},
        )

    store.add_entities.assert_called_once()
    call_kwargs = store.add_entities.call_args.kwargs
    assert call_kwargs["source_id"] == "src-1"
    assert call_kwargs["knowledge_id"] == "kb-1"
    assert len(call_kwargs["entities"]) == 1
    assert call_kwargs["entities"][0].name == "Motor M1"


async def test_ingest_skips_when_no_entities():
    store = AsyncMock()
    store.add_entities = AsyncMock()
    lm_config = MagicMock()

    mock_result = MagicMock()
    mock_result.description = "General text"
    mock_result.entities = []
    mock_result.tables = []
    mock_result.annotations = []
    mock_result.page_type = "text"

    with (
        patch("x64rag.retrieval.modules.ingestion.methods.graph.b") as mock_b,
        patch("x64rag.retrieval.modules.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()

        method = GraphIngestion(graph_store=store, lm_config=lm_config)
        await method.ingest(
            source_id="src-1", knowledge_id=None, source_type=None,
            source_weight=1.0, title="Test", full_text="No entities here.",
            chunks=[], tags=[], metadata={},
        )

    store.add_entities.assert_not_called()


async def test_ingest_error_does_not_raise():
    store = AsyncMock()
    lm_config = MagicMock()

    with (
        patch("x64rag.retrieval.modules.ingestion.methods.graph.b") as mock_b,
        patch("x64rag.retrieval.modules.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_registry.return_value = MagicMock()

        method = GraphIngestion(graph_store=store, lm_config=lm_config)
        # Should not raise — logs warning and continues
        await method.ingest(
            source_id="src-1", knowledge_id=None, source_type=None,
            source_weight=1.0, title="Test", full_text="Some text.",
            chunks=[], tags=[], metadata={},
        )


async def test_delete():
    store = AsyncMock()
    store.delete_by_source = AsyncMock()
    method = GraphIngestion(graph_store=store, lm_config=MagicMock())
    await method.delete("src-1")
    store.delete_by_source.assert_called_once_with("src-1")
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/test_graph_ingestion_method.py -v`
Expected: FAIL (constructor signature mismatch — no `lm_config` param yet)

**Step 3: Implement GraphIngestion**

Replace `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py`:

```python
from __future__ import annotations

import time
from typing import Any

from x64rag.retrieval.common.language_model import LanguageModelConfig, build_registry
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.modules.ingestion.analyze.models import DiscoveredEntity, PageAnalysis
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.stores.graph.base import BaseGraphStore
from x64rag.retrieval.stores.graph.mapper import page_entities_to_graph

logger = get_logger("ingestion.methods.graph")

# Lazy import — avoid circular dependency and heavy BAML import at module level
b: Any = None


def _get_baml_client() -> Any:
    global b
    if b is None:
        from x64rag.retrieval.baml.baml_client.async_client import b as _b
        b = _b
    return b


class GraphIngestion:
    """Extract entities from text via LLM and store in graph store.

    Uses the ``ExtractEntitiesFromText`` BAML function to extract entities,
    then maps them to ``GraphEntity`` via ``page_entities_to_graph()`` and
    stores via ``graph_store.add_entities()``.
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        lm_config: LanguageModelConfig | None = None,
    ) -> None:
        self._store = graph_store
        self._registry = build_registry(lm_config) if lm_config else None

    @property
    def name(self) -> str:
        return "graph"

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
        if not self._registry:
            logger.warning("graph ingestion skipped — no lm_config provided")
            return

        start = time.perf_counter()
        try:
            client = _get_baml_client()
            result = await client.ExtractEntitiesFromText(
                full_text,
                baml_options={"client_registry": self._registry},
            )

            if not result.entities:
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("no entities found in %.1fms", elapsed)
                return

            # Convert BAML output to internal model
            analysis = PageAnalysis(
                page_number=1,
                description=result.description,
                entities=[
                    DiscoveredEntity(
                        name=e.name,
                        category=e.category,
                        value=e.value,
                        context=e.context,
                    )
                    for e in result.entities
                ],
                tables=[],
                annotations=result.annotations if result.annotations else [],
                page_type=result.page_type or "text",
            )

            # Reuse existing mapper
            graph_entities = page_entities_to_graph(analysis, source_id)

            await self._store.add_entities(
                source_id=source_id,
                knowledge_id=knowledge_id,
                entities=graph_entities,
            )

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("%d entities extracted and stored in %.1fms", len(graph_entities), elapsed)

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)

    async def delete(self, source_id: str) -> None:
        await self._store.delete_by_source(source_id)
```

**Step 4: Run tests**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/test_graph_ingestion_method.py -v`
Expected: PASS (4 tests)

**Step 5: Update server.py to pass lm_config when constructing GraphIngestion**

In `src/rfnry_rag/retrieval/server.py`, find the graph path section in `initialize()` (around line 323-326). Currently it does not add `GraphIngestion` (removed in review fix). Re-add it with `lm_config`:

```python
# Graph path
if persistence.graph_store:
    if ingestion.lm_config:
        ingestion_methods.append(GraphIngestion(
            graph_store=persistence.graph_store,
            lm_config=ingestion.lm_config,
        ))
    retrieval_methods.append(GraphRetrieval(graph_store=persistence.graph_store, weight=0.7))
```

**Step 6: Run full test suite**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/ -q --tb=short`
Expected: All pass

**Step 7: Commit**

```
feat(pipeline): wire GraphIngestion with LLM entity extraction
```

---

### Task 3: Add top_k to BaseRetrievalMethod Protocol

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/base.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/vector.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/document.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/methods/graph.py`

**Step 1: Update protocol**

Add `top_k` property to `BaseRetrievalMethod`:

```python
class BaseRetrievalMethod(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    @property
    def top_k(self) -> int | None: ...

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        knowledge_id: str | None = None,
    ) -> list[RetrievedChunk]: ...
```

**Step 2: Add top_k to all three method constructors**

Each method gets `top_k: int | None = None` in `__init__` and a matching `@property`:

For `VectorRetrieval`:
```python
def __init__(self, ..., top_k: int | None = None, weight: float = 1.0) -> None:
    ...
    self._top_k = top_k

@property
def top_k(self) -> int | None:
    return self._top_k
```

Same pattern for `DocumentRetrieval` and `GraphRetrieval`.

**Step 3: Run tests**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/ -q --tb=short`
Expected: All pass (existing tests don't pass top_k, default is None)

**Step 4: Commit**

```
feat(pipeline): add top_k property to BaseRetrievalMethod protocol
```

---

### Task 4: Wire weight and top_k into Retrieval Dispatch

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/fusion.py`
- Test: `src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py` (add tests)

**Step 1: Write failing tests**

Add to `src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py`:

```python
def _mock_method_with_config(name: str, results: list[RetrievedChunk], weight: float = 1.0, top_k: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        weight=weight,
        top_k=top_k,
        search=AsyncMock(return_value=results),
    )


async def test_method_weight_affects_fusion_scores():
    """Higher-weight method should produce higher fused scores."""
    shared_chunk_id = "shared-1"
    high_weight = _mock_method_with_config("vector", [
        RetrievedChunk(chunk_id=shared_chunk_id, source_id="s1", content="text", score=0.9),
    ], weight=2.0)
    low_weight = _mock_method_with_config("document", [
        RetrievedChunk(chunk_id="doc-1", source_id="s2", content="doc", score=0.9),
    ], weight=0.5)

    service = RetrievalService(retrieval_methods=[high_weight, low_weight], top_k=5)
    results = await service.retrieve(query="test")

    # Both methods contribute, but vector (weight=2.0) should rank higher
    assert len(results) == 2
    # The shared-1 chunk from weight=2.0 method should have a higher fused score
    scores = {r.chunk_id: r.score for r in results}
    assert scores[shared_chunk_id] > scores["doc-1"]


async def test_method_top_k_override():
    """Method with top_k should be called with its own top_k, not service fetch_k."""
    vector = _mock_method_with_config("vector", [
        RetrievedChunk(chunk_id="c1", source_id="s1", content="text", score=0.9),
    ], top_k=50)  # Method wants 50 candidates
    document = _mock_method_with_config("document", [
        RetrievedChunk(chunk_id="c2", source_id="s2", content="doc", score=0.8),
    ], top_k=None)  # Use service default

    service = RetrievalService(retrieval_methods=[vector, document], top_k=5)
    await service.retrieve(query="test")

    # Vector should be called with top_k=50 (its override)
    vector_call = vector.search.call_args
    assert vector_call.kwargs["top_k"] == 50

    # Document should be called with fetch_k = 5 * 4 = 20 (service default)
    doc_call = document.search.call_args
    assert doc_call.kwargs["top_k"] == 20
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/test_retrieval_service_methods.py -v -k "weight_affects or top_k_override"`
Expected: FAIL

**Step 3: Update `_search_single_query` to use per-method top_k**

In `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`, change `_search_single_query`:

```python
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

    gathered = await asyncio.gather(
        *(
            method.search(
                query=query,
                top_k=method.top_k if method.top_k is not None else fetch_k,
                filters=filters,
                knowledge_id=knowledge_id,
            )
            for method in self._retrieval_methods
        )
    )
    return [results for results in gathered if results]
```

**Step 4: Update `reciprocal_rank_fusion` to accept method weights**

In `src/rfnry_rag/retrieval/modules/retrieval/search/fusion.py`, the current signature is:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    source_type_weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
```

Add `method_weights` parameter:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    source_type_weights: dict[str, float] | None = None,
    method_weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    results_by_key: dict[str, RetrievedChunk] = {}

    for list_idx, result_list in enumerate(result_lists):
        list_weight = method_weights[list_idx] if method_weights and list_idx < len(method_weights) else 1.0
        for rank, result in enumerate(result_list):
            key = result.chunk_id
            if key not in scores:
                scores[key] = 0
                results_by_key[key] = result
            scores[key] += list_weight / (k + rank + 1)

    if source_type_weights:
        for key, result in results_by_key.items():
            scores[key] *= result.source_weight

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for key in sorted_keys:
        fused.append(replace(results_by_key[key], score=scores[key]))

    return fused
```

**Step 5: Pass method weights from RetrievalService to RRF**

In `RetrievalService.retrieve()`, when calling `reciprocal_rank_fusion`, build method weights from the result lists. The challenge: `all_result_lists` is flat (multiple queries × multiple methods). We need to track which weight belongs to which result list.

Simpler approach: build weights in `_search_single_query` and return them alongside results.

Change `_search_single_query` return type to include weights:

```python
async def _search_single_query(
    self, query: str, fetch_k: int, filters: dict[str, Any] | None, knowledge_id: str | None,
) -> tuple[list[list[RetrievedChunk]], list[float]]:
    if not self._retrieval_methods:
        return [], []

    gathered = await asyncio.gather(*(
        method.search(
            query=query,
            top_k=method.top_k if method.top_k is not None else fetch_k,
            filters=filters,
            knowledge_id=knowledge_id,
        )
        for method in self._retrieval_methods
    ))

    result_lists = []
    weights = []
    for method, results in zip(self._retrieval_methods, gathered):
        if results:
            result_lists.append(results)
            weights.append(method.weight)
    return result_lists, weights
```

Update `retrieve()` to collect weights:

```python
all_result_lists: list[list[RetrievedChunk]] = []
all_weights: list[float] = []
for result_lists, weights in query_results:
    all_result_lists.extend(result_lists)
    all_weights.extend(weights)

if tree_chunks:
    all_result_lists.append(tree_chunks)
    all_weights.append(1.0)  # Tree chunks use default weight

if len(all_result_lists) > 1:
    fused = reciprocal_rank_fusion(
        all_result_lists,
        source_type_weights=self._source_type_weights,
        method_weights=all_weights,
    )
```

**Step 6: Run tests**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/ -q --tb=short`
Expected: All pass

**Step 7: Commit**

```
feat(pipeline): wire weight and top_k into retrieval dispatch and RRF fusion
```

---

### Task 5: Refactor AnalyzedIngestionService Phase 3 to Use Method List

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/service.py`
- Modify: `src/rfnry_rag/retrieval/server.py`
- Modify: `src/rfnry_rag/retrieval/tests/test_graph_ingestion.py`

The `AnalyzedIngestionService.ingest()` (phase 3) currently:
1. Embeds page descriptions → upserts to vector_store
2. Maps entities → graph_store.add_entities
3. Maps relations → graph_store.add_relations

After refactor, phase 3 delegates embedding+storage to an ingestion method list, same pattern as `IngestionService`. Phases 1-2 (analyze, synthesize) stay unchanged — they're unique to this service.

**Step 1: Write failing test**

Add to `src/rfnry_rag/retrieval/tests/test_graph_ingestion.py` (which tests AnalyzedIngestionService):

```python
async def test_ingest_delegates_to_methods():
    """Phase 3 should delegate to ingestion methods."""
    from types import SimpleNamespace
    mock_method = SimpleNamespace(name="vector", ingest=AsyncMock(), delete=AsyncMock())

    embeddings = MagicMock()
    embeddings.model = "test-model"
    metadata_store = AsyncMock()

    service = AnalyzedIngestionService(
        embeddings=embeddings,
        metadata_store=metadata_store,
        embedding_model_name="test:model",
        ingestion_methods=[mock_method],
    )

    source = _make_source_with_analysis()
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    await service.ingest(source.source_id)

    mock_method.ingest.assert_called_once()
```

**Step 2: Refactor AnalyzedIngestionService**

Changes to constructor:
- Add `ingestion_methods: list = field(default_factory=list)` parameter (or plain list)
- Remove direct `vector_store` and `graph_store` parameters
- Keep `embeddings` (still needed for phase 3 text embedding before passing to methods)
- Keep `metadata_store` (needed for all 3 phases)

Changes to `ingest()` (phase 3):
- After embedding page descriptions, instead of directly upserting to vector_store and graph_store, call each ingestion method with the embedded content
- Build a synthetic `full_text` from page descriptions and pass to methods
- `VectorIngestion` handles the actual vector upsert
- `GraphIngestion` handles entity extraction + graph storage
- `DocumentIngestion` handles document store (if configured)

Wait — there's a subtlety. `AnalyzedIngestionService.ingest()` builds custom `VectorPoint` payloads that are different from what `VectorIngestion._build_points()` produces (different payload fields: `page_type`, `entities`, `cross_references` vs standard chunk fields). This means we can't just delegate to `VectorIngestion` as-is.

**Revised approach:** Phase 3 for vector storage stays in `AnalyzedIngestionService` (the payload format is unique). Only graph storage delegates to `GraphIngestion` (or directly to graph methods). This is the minimal, correct refactor.

Actually, even simpler: the key problem was that `AnalyzedIngestionService` takes `graph_store` directly. We can replace that with a `GraphIngestion` method reference and call its `ingest()`. But `GraphIngestion.ingest()` calls `ExtractEntitiesFromText` — which is redundant since `AnalyzedIngestionService` already extracted entities in phase 1.

**Cleanest approach:** `AnalyzedIngestionService` keeps its vector upsert (unique payload) but delegates graph storage to the mapper functions directly (which it already does). The real refactor is: remove `graph_store` and `document_store` from the constructor, pass them via an `ingestion_methods` list, and call method.ingest() for non-vector work.

But this gets complex fast with diminishing returns. Let me simplify.

**Final approach:** Replace `graph_store` and `document_store` direct references with the same method list. For phase 3:
- Vector embedding stays in the service (unique payload)
- Graph: call `graph_store.add_entities()` / `add_relations()` through a stored reference or via GraphIngestion method
- Document: call DocumentIngestion.ingest() with the synthesized full_text

Simplest, most honest change:

```python
class AnalyzedIngestionService:
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        metadata_store: BaseMetadataStore,
        embedding_model_name: str,
        vision: BaseVision | None = None,
        dpi: int = 300,
        source_type_weights: dict[str, float] | None = None,
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
        lm_config: LanguageModelConfig | None = None,
        # NEW: vector_store still needed for unique payload
        vector_store: BaseVectorStore | None = None,
        # NEW: optional ingestion methods for delegation
        ingestion_methods: list | None = None,
    ) -> None:
```

Phase 3 `ingest()`:
- Vector upsert stays (unique structured payload with page_type, entities, cross_references)
- Graph: if any method with name="graph" exists in ingestion_methods and has a `_store` with `add_entities`, use existing mapper + store. Otherwise skip.
- Document: iterate methods, call `method.ingest()` for "document" method with synthesized full_text

This is getting over-engineered. Let me step back.

**Pragmatic approach:** The `AnalyzedIngestionService` has a fundamentally different flow from `IngestionService`. Forcing method list into it adds complexity for little gain. The honest refactor is:

1. Remove `graph_store` param — pass graph entities through an optional `GraphIngestion` method
2. Remove `document_store` param — pass full text through an optional `DocumentIngestion` method  
3. Keep `vector_store` — the structured payload is too different from standard chunking
4. Accept `ingestion_methods: list` for graph + document delegation

**Step 3: Implement**

Update constructor to replace `document_store` and `graph_store` with `ingestion_methods`:

```python
def __init__(
    self,
    embeddings: BaseEmbeddings,
    vector_store: BaseVectorStore,
    metadata_store: BaseMetadataStore,
    embedding_model_name: str,
    vision: BaseVision | None = None,
    dpi: int = 300,
    source_type_weights: dict[str, float] | None = None,
    on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
    lm_config: LanguageModelConfig | None = None,
    ingestion_methods: list | None = None,
) -> None:
    ...
    self._ingestion_methods = ingestion_methods or []
```

In phase 3 `ingest()`, replace graph_store blocks with:

```python
# Delegate to ingestion methods (document store, graph store, etc.)
full_text = "\n\n".join(texts)
for method in self._ingestion_methods:
    try:
        if method.name == "graph":
            # Graph already has entities from analysis — use mapper directly
            all_entities = []
            for pa in page_analyses:
                all_entities.extend(page_entities_to_graph(pa, source.source_id))
            relations = cross_refs_to_graph_relations(synthesis, page_analyses, source.knowledge_id)
            if all_entities:
                await method._store.add_entities(
                    source_id=source.source_id,
                    knowledge_id=source.knowledge_id,
                    entities=all_entities,
                )
            if relations:
                await method._store.add_relations(
                    source_id=source.source_id,
                    relations=relations,
                )
            logger.info("graph: %d entities, %d relations", len(all_entities), len(relations))
        elif method.name == "document":
            await method.ingest(
                source_id=source.source_id,
                knowledge_id=source.knowledge_id,
                source_type=source.source_type,
                source_weight=source.source_weight,
                title=source.metadata.get("file_name", ""),
                full_text=full_text,
                chunks=[],
                tags=[],
                metadata=source.metadata,
            )
    except Exception as exc:
        logger.warning("ingestion method '%s' failed: %s", method.name, exc)
```

**Step 4: Update server.py**

In `initialize()`, pass `ingestion_methods` to `AnalyzedIngestionService` instead of `document_store` and `graph_store`:

```python
# Build analyzed ingestion method list (subset — no VectorIngestion, handled internally)
analyzed_methods: list = []
if persistence.document_store:
    analyzed_methods.append(DocumentIngestion(document_store=persistence.document_store))
if persistence.graph_store and ingestion.lm_config:
    analyzed_methods.append(GraphIngestion(
        graph_store=persistence.graph_store,
        lm_config=ingestion.lm_config,
    ))

self._structured_ingestion = AnalyzedIngestionService(
    embeddings=ingestion.embeddings,
    vector_store=persistence.vector_store,
    metadata_store=persistence.metadata_store,
    embedding_model_name=self._embedding_model_name,
    vision=ingestion.vision,
    dpi=ingestion.dpi,
    source_type_weights=retrieval.source_type_weights,
    on_ingestion_complete=self._on_ingestion_complete,
    lm_config=ingestion.lm_config,
    ingestion_methods=analyzed_methods,
)
```

**Step 5: Update existing tests**

`test_graph_ingestion.py` constructs `AnalyzedIngestionService(graph_store=...)`. Update to pass `ingestion_methods` with a mock graph method that exposes `_store`.

**Step 6: Run full suite**

Run: `cd /home/frndvrgs/software/rfnry/rag && uv run pytest src/rfnry_rag/retrieval/tests/ -q --tb=short`
Expected: All pass

**Step 7: Commit**

```
refactor(pipeline): AnalyzedIngestionService delegates graph and document to methods
```

---

### Task 6: Final Verification

**Step 1: Run everything**

```bash
cd /home/frndvrgs/software/rfnry/rag
uv run pytest src/ -q --tb=short
uv run ruff check src/
uv run mypy src/ --ignore-missing-imports 2>&1 | tail -10
```

Expected: All tests pass, lint clean.

**Step 2: Verify new BAML function works with generated client**

```bash
grep -c "ExtractEntitiesFromText" src/rfnry_rag/retrieval/baml/baml_client/async_client.py
```

Expected: At least 1 match.

**Step 3: Commit if any cleanup needed**

```
chore(pipeline): final cleanup after pipeline completion
```

---

## Implementation Order

1. BAML function (foundation — everything depends on this)
2. GraphIngestion wiring (uses the new BAML function)
3. top_k on protocol (small, independent)
4. weight + top_k in dispatch (uses the protocol change)
5. AnalyzedIngestionService refactor (uses GraphIngestion)
6. Final verification
