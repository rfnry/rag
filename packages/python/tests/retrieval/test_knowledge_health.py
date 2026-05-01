"""KnowledgeManager.health fuses ingestion notes, retrieval stats, and embedding
freshness into a single read-side surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_rag.knowledge.manager import KnowledgeManager
from rfnry_rag.models import HealthSummary, RetrievalHealth, Source, SourceStats


def _source(
    sid: str,
    *,
    stale: bool = False,
    notes: list[str] | None = None,
    embedding_model: str = "openai:text-embedding-3-small",
) -> Source:
    metadata: dict[str, object] = {}
    if notes:
        metadata["ingestion_notes"] = list(notes)
    return Source(
        source_id=sid,
        knowledge_id="k",
        status="completed",
        stale=stale,
        embedding_model=embedding_model,
        metadata=metadata,
    )


def _store(*, sources: list[Source], stats: dict[str, SourceStats] | None = None):
    by_id = {s.source_id: s for s in sources}
    return SimpleNamespace(
        list_sources=AsyncMock(return_value=sources),
        get_source=AsyncMock(side_effect=lambda sid: by_id.get(sid)),
        get_source_stats=AsyncMock(side_effect=lambda sid: (stats or {}).get(sid)),
    )


async def test_health_returns_none_for_unknown_source() -> None:
    metadata_store = _store(sources=[])
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    assert await km.health("missing") is None


async def test_health_with_clean_ingest_no_retrieval() -> None:
    src = _source("a")
    metadata_store = _store(sources=[src])
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    h = await km.health("a")
    assert isinstance(h, HealthSummary)
    assert h.fully_ingested is True
    assert h.ingestion_notes == []
    assert h.stale_embedding is False
    assert h.retrieval is None


async def test_health_with_notes_and_grounded_queries() -> None:
    notes = [
        "vision:warn:page_3:invalid_output(missing field)",
        "graph:warn:extraction_failed(rate limit)",
    ]
    src = _source("b", notes=notes)
    stats = SourceStats(
        source_id="b",
        total_hits=10,
        grounded_hits=7,
        ungrounded_hits=3,
    )
    metadata_store = _store(sources=[src], stats={"b": stats})
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    h = await km.health("b")
    assert h is not None
    assert h.fully_ingested is False
    assert h.ingestion_notes == notes
    assert h.stale_embedding is False
    assert h.retrieval == RetrievalHealth(
        total_hits=10,
        grounded_hits=7,
        ungrounded_hits=3,
        grounding_rate=0.7,
    )


async def test_health_with_stale_embedding() -> None:
    src = _source("c", stale=True)
    metadata_store = _store(sources=[src])
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    h = await km.health("c")
    assert h is not None
    assert h.stale_embedding is True


async def test_health_zero_hits_grounding_rate_is_none() -> None:
    src = _source("d")
    stats = SourceStats(source_id="d", total_hits=0, grounded_hits=0, ungrounded_hits=0)
    metadata_store = _store(sources=[src], stats={"d": stats})
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]
    h = await km.health("d")
    assert h is not None
    assert h.retrieval is not None
    assert h.retrieval.total_hits == 0
    assert h.retrieval.grounding_rate is None
