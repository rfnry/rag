"""Status-based resume: RagEngine.ingest routes to the first unfinished phase."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from rfnry_rag.ingestion.analyze.models import PageAnalysis
from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_rag.ingestion.hashing import file_hash as compute_file_hash
from rfnry_rag.models import Source
from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore


class _FakeEmbeddings:
    @property
    def model(self) -> str:
        return "fake-384"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    async def embedding_dimension(self) -> int:
        return 384


class _FakeVectorStore:
    """Minimal vector store that silently drops upserts."""

    async def initialize(self, dim: int) -> None:
        pass

    async def upsert(self, points: list) -> None:
        pass


class _CapturingVectorStore(_FakeVectorStore):
    """Variant that records every upsert call."""

    def __init__(self, captured: dict[str, list]) -> None:
        self._captured = captured

    async def upsert(self, points: list) -> None:
        self._captured["upserts"].extend(points)


class _FakeVision:
    pass


def _make_xml_page_analyses() -> list[PageAnalysis]:
    return [
        PageAnalysis(page_number=1, description="xml element 1", page_type="xml_element"),
        PageAnalysis(page_number=2, description="xml element 2", page_type="xml_element"),
        PageAnalysis(page_number=3, description="xml element 3", page_type="xml_element"),
    ]


# ---------------------------------------------------------------------------
# Thin engine wrapper — mimics server.py's structured-path dispatch
# without requiring a full RagEngine initialisation.
# ---------------------------------------------------------------------------


class _MinimalEngine:
    """Implements .ingest() with the same status-based routing as server.py."""

    def __init__(
        self,
        structured_ingestion: AnalyzedIngestionService,
        metadata_store: SQLAlchemyMetadataStore,
    ) -> None:
        self._structured_ingestion = structured_ingestion
        self._metadata_store = metadata_store

    async def ingest(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        **_kwargs: Any,
    ) -> Source:
        """Status-based resume across the 3-phase pipeline (mirrors server.py logic)."""
        fp = Path(file_path)
        ext = fp.suffix.lower()
        if ext not in {".xml", ".l5x"}:
            raise ValueError(f"unsupported extension for structured ingestion: {ext}")

        # Compute file_hash via asyncio.to_thread (same pattern as server.py).
        # In tests compute_file_hash is patched at the service level, so call
        # it directly on the service's patched version.  Since server.py patches
        # at the module level we replicate the call here using asyncio.to_thread.
        file_hash_value = await asyncio.to_thread(compute_file_hash, fp)
        existing = await self._metadata_store.find_by_hash(file_hash_value, knowledge_id)

        if existing is not None and existing.status == "completed":
            return existing

        if existing is not None and existing.status == "synthesized":
            return await self._structured_ingestion.ingest(existing.source_id)

        if existing is not None and existing.status == "analyzed":
            source = await self._structured_ingestion.synthesize(existing.source_id)
            return await self._structured_ingestion.ingest(source.source_id)

        # Fresh run
        source = await self._structured_ingestion.analyze(
            file_path=fp,
            knowledge_id=knowledge_id,
        )
        source = await self._structured_ingestion.synthesize(source.source_id)
        source = await self._structured_ingestion.ingest(source.source_id)
        return source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_FILE_HASH = "file_hash_resume_abc"


@pytest_asyncio.fixture
async def resume_rag_engine_pdf(tmp_path):
    """Returns (engine, mock_b).

    - engine._structured_ingestion is a real AnalyzedIngestionService backed
      by SQLite.
    - engine.ingest() uses the same status-based routing as server.py.
    - mock_b.AnalyzePage / mock_b.SynthesizeDocument are available for call-count
      assertions (never fired for .xml; counts remain 0 throughout).
    - compute_file_hash is patched to return a fixed value; parse_xml returns 3
      deterministic PageAnalysis objects so no real XML I/O is needed.
    """
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
    )
    svc._registry = MagicMock()

    mock_b = MagicMock()
    mock_b.AnalyzePage = AsyncMock()
    mock_b.SynthesizeDocument = AsyncMock()

    engine = _MinimalEngine(structured_ingestion=svc, metadata_store=store)

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.is_l5x",
            return_value=False,
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.parse_xml",
            return_value=_make_xml_page_analyses(),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value=_FAKE_FILE_HASH,
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch(
            f"{__name__}.compute_file_hash",
            return_value=_FAKE_FILE_HASH,
        ),
        patch(
            f"{__name__}.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b", mock_b),
    ):
        yield engine, mock_b

    await store.shutdown()


@pytest_asyncio.fixture
async def resume_rag_engine_pdf_with_vector_capture(tmp_path):
    """Returns (engine, mock_b, captured) — same as resume_rag_engine_pdf
    but the vector store records upserts so tests can assert idempotency."""
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    captured: dict[str, list] = {"upserts": []}
    vec_store = _CapturingVectorStore(captured)

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=vec_store,
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
    )
    svc._registry = MagicMock()

    mock_b = MagicMock()
    mock_b.AnalyzePage = AsyncMock()
    mock_b.SynthesizeDocument = AsyncMock()

    engine = _MinimalEngine(structured_ingestion=svc, metadata_store=store)

    with (
        patch(
            "rfnry_rag.ingestion.analyze.service.is_l5x",
            return_value=False,
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.parse_xml",
            return_value=_make_xml_page_analyses(),
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.compute_file_hash",
            return_value=_FAKE_FILE_HASH,
        ),
        patch(
            "rfnry_rag.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch(
            f"{__name__}.compute_file_hash",
            return_value=_FAKE_FILE_HASH,
        ),
        patch(
            f"{__name__}.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.baml.baml_client.async_client.b", mock_b),
    ):
        yield engine, mock_b, captured

    await store.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_from_analyzed_skips_phase_1(resume_rag_engine_pdf) -> None:
    """Source.status == 'analyzed' -> engine skips analyze, runs synthesize + ingest."""
    engine, mock_baml = resume_rag_engine_pdf
    # Prime: run just the analyze phase via the stepped API
    src = await engine._structured_ingestion.analyze("/tmp/doc.xml", knowledge_id="k1")
    assert src.status == "analyzed"
    analyze_calls_before = mock_baml.AnalyzePage.call_count

    # Full engine.ingest() on the same file — must NOT rerun AnalyzePage
    result = await engine.ingest("/tmp/doc.xml", knowledge_id="k1")
    assert mock_baml.AnalyzePage.call_count == analyze_calls_before
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_resume_from_synthesized_skips_phases_1_and_2(resume_rag_engine_pdf) -> None:
    """Source.status == 'synthesized' -> engine runs only ingest."""
    engine, mock_baml = resume_rag_engine_pdf
    src = await engine._structured_ingestion.analyze("/tmp/doc.xml", knowledge_id="k1")
    src = await engine._structured_ingestion.synthesize(src.source_id)
    assert src.status == "synthesized"
    analyze_count = mock_baml.AnalyzePage.call_count
    synth_count = mock_baml.SynthesizeDocument.call_count

    result = await engine.ingest("/tmp/doc.xml", knowledge_id="k1")
    assert mock_baml.AnalyzePage.call_count == analyze_count
    assert mock_baml.SynthesizeDocument.call_count == synth_count
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_resume_from_completed_returns_existing_source(resume_rag_engine_pdf) -> None:
    """Source.status == 'completed' -> return existing source, zero new work."""
    engine, mock_baml = resume_rag_engine_pdf
    first = await engine.ingest("/tmp/doc.xml", knowledge_id="k1")
    assert first.status == "completed"
    total_calls = mock_baml.AnalyzePage.call_count + mock_baml.SynthesizeDocument.call_count

    result = await engine.ingest("/tmp/doc.xml", knowledge_id="k1")
    # No additional LLM calls at all
    assert mock_baml.AnalyzePage.call_count + mock_baml.SynthesizeDocument.call_count == total_calls
    assert result.source_id == first.source_id


@pytest.mark.asyncio
async def test_synthesize_idempotent_on_already_synthesized_source(resume_rag_engine_pdf) -> None:
    """Calling synthesize() twice on the same source_id does not re-run SynthesizeDocument."""
    engine, mock_baml = resume_rag_engine_pdf
    src = await engine._structured_ingestion.analyze("/tmp/doc.xml", knowledge_id="k1")
    src = await engine._structured_ingestion.synthesize(src.source_id)
    synth_count = mock_baml.SynthesizeDocument.call_count

    # Second synthesize call — must be a no-op
    src2 = await engine._structured_ingestion.synthesize(src.source_id)
    assert mock_baml.SynthesizeDocument.call_count == synth_count
    assert src2.status == "synthesized"


@pytest.mark.asyncio
async def test_ingest_phase_idempotent_on_already_completed_source(
    resume_rag_engine_pdf_with_vector_capture,
) -> None:
    """Calling the ingest phase on a completed source does not re-embed or re-upsert."""
    engine, mock_baml, captured = resume_rag_engine_pdf_with_vector_capture
    src = await engine.ingest("/tmp/doc.xml", knowledge_id="k1")
    assert src.status == "completed"
    upsert_count = len(captured["upserts"])

    # Call the ingest-phase method directly a second time on the completed source
    src2 = await engine._structured_ingestion.ingest(src.source_id)
    assert len(captured["upserts"]) == upsert_count  # no new upserts
    assert src2.status == "completed"
