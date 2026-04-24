"""AnalyzedIngestionService: multi-vector per page + raw OCR reaches BM25."""
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.analyze.models import (
    DiscoveredEntity,
    DiscoveredTable,
    PageAnalysis,
)


def _serialize_page_for_test(pa: PageAnalysis) -> dict:
    """Match the internal _serialize_analysis helper's shape."""
    return {
        "page_number": pa.page_number,
        "description": pa.description,
        "entities": [
            {"name": e.name, "category": e.category, "context": e.context, "value": e.value}
            for e in pa.entities
        ],
        "tables": [{"title": t.title, "columns": t.columns, "rows": t.rows} for t in pa.tables],
        "annotations": pa.annotations,
        "page_type": pa.page_type,
        "metadata": pa.metadata,
        "raw_text": pa.raw_text,
    }


@pytest_asyncio.fixture
async def fake_analyzed_service(tmp_path):
    """Returns (service, captured_dict) for assertion.

    Uses in-memory mocks for embeddings, vector_store, metadata_store.
    Seeds 2 pages of PageAnalysis: page 1 has a table with 3 rows + raw_text
    "RAW_PAGE_1_CONTENT", page 2 has raw_text "RAW_PAGE_2_CONTENT".
    """
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService
    from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

    captured: dict = {"upserts": [], "method_calls": []}

    class FakeEmbeddings:
        @property
        def model(self) -> str:
            return "fake-embed-384"

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 384 for _ in texts]

        async def embedding_dimension(self) -> int:
            return 384

    class FakeVectorStore:
        async def initialize(self, dim: int) -> None:
            pass

        async def upsert(self, points) -> None:
            captured["upserts"].extend(points)

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    class FakeMethod:
        name = "document"

        async def ingest(self, **kwargs) -> None:
            captured["method_calls"].append(kwargs)

    svc = AnalyzedIngestionService(
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-embed-384",
        ingestion_methods=[FakeMethod()],
    )

    # Seed a source as if phase 1 + 2 already ran
    source_id = str(uuid4())
    page_analyses = [
        PageAnalysis(
            page_number=1,
            description="llm-written description of page 1",
            entities=[DiscoveredEntity(name="E1", category="component", context="", value=None)],
            tables=[
                DiscoveredTable(
                    title="Spec table",
                    columns=["Part", "Torque"],
                    rows=[
                        {"Part": "M8", "Torque": "24"},
                        {"Part": "M10", "Torque": "48"},
                        {"Part": "M12", "Torque": "84"},
                    ],
                )
            ],
            annotations=[],
            page_type="text",
            raw_text="RAW_PAGE_1_CONTENT",
        ),
        PageAnalysis(
            page_number=2,
            description="llm-written description of page 2",
            entities=[],
            tables=[],
            annotations=[],
            page_type="text",
            raw_text="RAW_PAGE_2_CONTENT",
        ),
    ]
    source = Source(
        source_id=source_id,
        knowledge_id="k1",
        source_type=None,
        status="synthesized",
        embedding_model="fake-embed-384",
        file_hash="deadbeef",
        created_at=datetime.now(UTC),
        source_weight=1.0,
        metadata={
            "file_type": "pdf",
            "file_name": "seed.pdf",
            "page_analyses": [_serialize_page_for_test(pa) for pa in page_analyses],
            "synthesis": {
                "cross_references": [],
                "page_clusters": [],
                "document_summary": "",
            },
        },
    )
    await store.create_source(source)
    svc._seeded_source_id = source_id

    yield svc, captured
    await store.shutdown()


@pytest.mark.asyncio
async def test_ingest_writes_multi_vector_per_page(fake_analyzed_service) -> None:
    svc, captured = fake_analyzed_service
    await svc.ingest(source_id=svc._seeded_source_id)
    points = captured["upserts"]

    roles = {p.payload.get("vector_role") for p in points}
    # At least description + raw_text + table_row
    assert "description" in roles
    assert "raw_text" in roles
    assert "table_row" in roles


@pytest.mark.asyncio
async def test_full_text_passed_to_methods_is_raw_text(fake_analyzed_service) -> None:
    svc, captured = fake_analyzed_service
    await svc.ingest(source_id=svc._seeded_source_id)

    # method.ingest received full_text = concatenation of raw_text values
    method_call = captured["method_calls"][0]
    assert "RAW_PAGE_1_CONTENT" in method_call["full_text"]
    assert "RAW_PAGE_2_CONTENT" in method_call["full_text"]
    # Must NOT be the LLM description
    assert "llm-written description" not in method_call["full_text"]


@pytest.mark.asyncio
async def test_table_rows_become_individual_vectors(fake_analyzed_service) -> None:
    svc, captured = fake_analyzed_service
    await svc.ingest(source_id=svc._seeded_source_id)
    points = captured["upserts"]

    table_vectors = [p for p in points if p.payload.get("vector_role") == "table_row"]
    # 3 rows in the seeded table
    assert len(table_vectors) == 3
    # Column header must be prefixed in each row vector content
    for p in table_vectors:
        assert "Part" in p.payload["content"]
        assert "Torque" in p.payload["content"]


@pytest.mark.asyncio
async def test_page_without_raw_text_still_produces_description_vector(
    fake_analyzed_service,
) -> None:
    """If raw_text is empty (older docs), skip raw_text vector gracefully — do NOT crash."""
    svc, captured = fake_analyzed_service
    # Override seed to have empty raw_text on page 2
    source = await svc._metadata_store.get_source(svc._seeded_source_id)
    pas = source.metadata["page_analyses"]
    pas[1]["raw_text"] = ""
    source.metadata["page_analyses"] = pas
    await svc._metadata_store.update_source(
        source.source_id,
        metadata=source.metadata,
    )
    await svc.ingest(source_id=svc._seeded_source_id)

    # Description vectors always exist
    descs = [p for p in captured["upserts"] if p.payload.get("vector_role") == "description"]
    assert len(descs) == 2
    # Only page 1 has a raw_text vector
    raws = [p for p in captured["upserts"] if p.payload.get("vector_role") == "raw_text"]
    assert len(raws) == 1


@pytest.mark.asyncio
async def test_full_text_for_non_pdf_falls_back_to_description_with_entities(
    fake_analyzed_service,
) -> None:
    """L5X/XML pages have empty raw_text; full_text must still include entity names
    so document-store search on 'Motor M1' still matches.
    """
    svc, captured = fake_analyzed_service
    # Blank out raw_text on both seeded pages to simulate L5X/XML
    source = await svc._metadata_store.get_source(svc._seeded_source_id)
    for pa in source.metadata["page_analyses"]:
        pa["raw_text"] = ""
    await svc._metadata_store.update_source(
        source.source_id, metadata=source.metadata,
    )

    await svc.ingest(source_id=svc._seeded_source_id)

    method_call = captured["method_calls"][0]
    full_text = method_call["full_text"]
    # Description is present
    assert "llm-written description" in full_text
    # Entity names from page 1 are present (Entities: E1)
    assert "E1" in full_text
