"""AnalyzedIngestionService now stores page_analyses in the dedicated knowledge_page_analyses table."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from rfnry_knowledge.ingestion.analyze.models import (
    DiscoveredEntity,
    PageAnalysis,
)
from rfnry_knowledge.ingestion.analyze.service import (
    AnalyzedIngestionService,
    _serialize_analysis,
)
from rfnry_knowledge.models import Source
from rfnry_knowledge.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_page_analyses() -> list[PageAnalysis]:
    return [
        PageAnalysis(
            page_number=1,
            description="Page 1 description",
            entities=[DiscoveredEntity(name="E1", category="component", context="", value=None)],
            tables=[],
            annotations=[],
            page_type="text",
            raw_text="RAW_PAGE_1",
        ),
        PageAnalysis(
            page_number=2,
            description="Page 2 description",
            entities=[DiscoveredEntity(name="E2", category="component", context="", value=None)],
            tables=[],
            annotations=[],
            page_type="text",
            raw_text="RAW_PAGE_2",
        ),
    ]


class FakeEmbeddings:
    @property
    def model(self) -> str:
        return "fake-embed-384"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    async def embedding_dimension(self) -> int:
        return 384


class FakeVectorStore:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def initialize(self, dim: int) -> None:
        pass

    async def upsert(self, points) -> None:
        self._captured["upserts"].extend(points)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fake_analyzed_service_from_multi_vector(tmp_path):
    """Service + captured dict. Runs analyze() on a real XML file so phase 1 is exercised."""
    captured: dict = {"upserts": []}

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(captured),
        metadata_store=store,
        embedding_model_name="fake-embed-384",
    )

    xml = tmp_path / "sample.xml"
    xml.write_text("<root><item>hello</item></root>")

    source = await svc.analyze(file_path=xml)
    # Expose the post-analyze source id so the test can look it up
    svc._post_analyze_source_id = source.source_id

    yield svc, captured
    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_synthesizable(tmp_path):
    """Service pre-seeded with a source in 'analyzed' status and page_analyses in the table."""
    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore({}),
        metadata_store=store,
        embedding_model_name="fake-embed-384",
    )

    source_id = str(uuid4())
    page_analyses = _make_page_analyses()

    source = Source(
        source_id=source_id,
        knowledge_id="k1",
        source_type=None,
        status="analyzed",
        embedding_model="fake-embed-384",
        file_hash="deadbeef",
        created_at=datetime.now(UTC),
        source_weight=1.0,
        # page_analyses must NOT be in metadata
        metadata={"file_type": "xml", "file_name": "seed.xml"},
    )
    await store.create_source(source)

    # Seed page analyses directly into the dedicated table (bypass analyze phase)
    await store.upsert_page_analyses(
        source_id,
        [{"page_number": pa.page_number, "data": _serialize_analysis(pa)} for pa in page_analyses],
    )

    svc._seeded_source_id = source_id
    yield svc
    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_ingestable(tmp_path):
    """Service pre-seeded with a source in 'synthesized' status and page_analyses in the table."""
    captured: dict = {"upserts": []}

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(captured),
        metadata_store=store,
        embedding_model_name="fake-embed-384",
    )

    source_id = str(uuid4())
    page_analyses = _make_page_analyses()

    source = Source(
        source_id=source_id,
        knowledge_id="k1",
        source_type=None,
        status="synthesized",
        embedding_model="fake-embed-384",
        file_hash="deadbeef",
        created_at=datetime.now(UTC),
        source_weight=1.0,
        # page_analyses must NOT be in metadata; synthesis blob is still in metadata
        metadata={
            "file_type": "xml",
            "file_name": "seed.xml",
            "synthesis": {
                "cross_references": [],
                "page_clusters": [],
                "document_summary": "",
            },
        },
    )
    await store.create_source(source)

    # Seed page analyses directly into the dedicated table
    await store.upsert_page_analyses(
        source_id,
        [{"page_number": pa.page_number, "data": _serialize_analysis(pa)} for pa in page_analyses],
    )

    svc._seeded_source_id = source_id
    yield svc, captured
    await store.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_writes_to_page_analyses_table(fake_analyzed_service_from_multi_vector) -> None:
    """After analyze phase, page data lives in knowledge_page_analyses, NOT source.metadata['page_analyses']."""
    svc, _captured = fake_analyzed_service_from_multi_vector

    source = await svc._metadata_store.get_source(svc._post_analyze_source_id)
    assert source is not None

    # source.metadata must NOT contain the page_analyses key
    assert "page_analyses" not in source.metadata, (
        "analyze() must not write page_analyses into source.metadata; use knowledge_page_analyses table"
    )
    # The dedicated table holds them
    rows = await svc._metadata_store.get_page_analyses(source.source_id)
    assert len(rows) >= 1, "analyze() must write at least 1 row to knowledge_page_analyses"


@pytest.mark.asyncio
async def test_synthesize_reads_page_analyses_from_table(fake_analyzed_service_synthesizable) -> None:
    """If page_analyses live in the table (not metadata), synthesize must still work."""
    svc = fake_analyzed_service_synthesizable

    # Verify the source has no page_analyses in metadata (fixture guarantee)
    source = await svc._metadata_store.get_source(svc._seeded_source_id)
    assert source is not None
    assert "page_analyses" not in source.metadata

    result = await svc.synthesize(source_id=svc._seeded_source_id)
    # Must not raise. Result status transitions to 'synthesized'.
    assert result.status == "synthesized"


@pytest.mark.asyncio
async def test_ingest_reads_page_analyses_from_table(fake_analyzed_service_ingestable) -> None:
    """Ingest phase reads from the table. Vectors are produced."""
    svc, captured = fake_analyzed_service_ingestable

    # Verify the source has no page_analyses in metadata (fixture guarantee)
    source = await svc._metadata_store.get_source(svc._seeded_source_id)
    assert source is not None
    assert "page_analyses" not in source.metadata

    result = await svc.ingest(source_id=svc._seeded_source_id)
    assert result.status == "completed"
    assert len(captured["upserts"]) >= 1, "ingest() must produce at least one vector point"
