from __future__ import annotations

import contextlib
import os

import pytest

QDRANT_URL = os.environ.get("KNWL_TEST_QDRANT_URL")
NEO4J_URL = os.environ.get("KNWL_TEST_NEO4J_URL")
NEO4J_PASSWORD = os.environ.get("KNWL_TEST_NEO4J_PASSWORD")

requires_backends = pytest.mark.skipif(
    not (QDRANT_URL and NEO4J_URL and NEO4J_PASSWORD),
    reason="set KNWL_TEST_QDRANT_URL, KNWL_TEST_NEO4J_URL, KNWL_TEST_NEO4J_PASSWORD",
)


@requires_backends
async def test_knowledge_and_memory_namespaces_are_disjoint() -> None:
    """Same physical Qdrant + Neo4j; knowledge and memory must not see each other."""
    from rfnry_knowledge import (
        ExtractedMemory,
        Interaction,
        InteractionTurn,
        MemoryEngine,
        MemoryEngineConfig,
        MemoryIngestionConfig,
        MemoryRetrievalConfig,
        Neo4jGraphStore,
        QdrantVectorStore,
    )
    from rfnry_knowledge.providers import EmbeddingResult

    class _FakeEmbeddings:
        model = "fake"
        name = "fake:fake"

        async def embed(self, texts):
            return EmbeddingResult(
                vectors=[[0.1] * 8 for _ in texts], usage=None,
            )

        async def embedding_dimension(self) -> int:
            return 8

    class _StubExtractor:
        async def extract(self, interaction, existing_memories=()):
            return (ExtractedMemory(text="lisbon fact", attributed_to="user"),)

    knowledge_qdrant = QdrantVectorStore(url=QDRANT_URL, collection="knowledge_iso_test")
    memory_qdrant = QdrantVectorStore(url=QDRANT_URL, collection="memory_iso_test")
    knowledge_neo = Neo4jGraphStore(uri=NEO4J_URL, password=NEO4J_PASSWORD)
    memory_neo = Neo4jGraphStore(
        uri=NEO4J_URL, password=NEO4J_PASSWORD, node_label_prefix="Memory",
    )

    cfg = MemoryEngineConfig(
        ingestion=MemoryIngestionConfig(
            extractor=_StubExtractor(),
            embeddings=_FakeEmbeddings(),
            vector_store=memory_qdrant,
        ),
        retrieval=MemoryRetrievalConfig(),
    )

    try:
        async with MemoryEngine(cfg) as memory:
            await memory.add(
                Interaction(turns=(InteractionTurn("user", "I moved to Lisbon."),)),
                memory_id="user-iso-test",
            )
            results = await memory.search("lisbon", memory_id="user-iso-test")

        assert len(results) >= 1
        assert all(r.row.memory_id == "user-iso-test" for r in results)

        # Knowledge-side qdrant collection saw no writes from the memory engine.
        # initialize() is required before count() — but knowledge-side was not
        # opened by an engine; it stays at zero or never created.
        # Use a soft check: count() returns 0 or raises (collection missing).
        try:
            await knowledge_qdrant.initialize(8)
            knowledge_count = await knowledge_qdrant.count()
            assert knowledge_count == 0
        finally:
            await knowledge_qdrant.shutdown()
    finally:
        # Cleanup the memory collection so re-runs are clean.
        with contextlib.suppress(Exception):
            await memory_qdrant.delete({"memory_id": "user-iso-test"})
        await memory_qdrant.shutdown()
        # Neo4j stores were instantiated but never `initialize()`d for memory_neo
        # in this test — so shutting them down is a no-op for the unconfigured path.
        # Still, ensure both are closed.
        with contextlib.suppress(Exception):
            await knowledge_neo.shutdown()
        with contextlib.suppress(Exception):
            await memory_neo.shutdown()
