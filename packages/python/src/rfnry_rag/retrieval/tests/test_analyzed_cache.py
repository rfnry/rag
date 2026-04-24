"""Analyzed ingest caching: file-hash short-circuit + per-page-hash reuse."""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_baml_result(page_num: int) -> SimpleNamespace:
    """Minimal BAML AnalyzePage result shaped like the generated type."""
    return SimpleNamespace(
        description=f"page {page_num} description",
        entities=[],
        tables=[],
        annotations=[],
        page_type="text",
    )


def _fake_source(
    source_id: str,
    knowledge_id: str,
    file_hash: str,
    status: str = "analyzed",
) -> Source:
    return Source(
        source_id=source_id,
        knowledge_id=knowledge_id,
        source_type=None,
        status=status,
        embedding_model="fake-384",
        file_hash=file_hash,
        created_at=datetime.now(UTC),
        source_weight=1.0,
        metadata={"file_type": "pdf", "file_name": "doc.pdf"},
    )


class _FakeEmbeddings:
    @property
    def model(self) -> str:
        return "fake-384"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    async def embedding_dimension(self) -> int:
        return 384


class _FakeVectorStore:
    async def initialize(self, dim: int) -> None:
        pass

    async def upsert(self, points) -> None:
        pass


class _FakeVision:
    pass


# ---------------------------------------------------------------------------
# Three-page mock page list (shared across fixtures)
# ---------------------------------------------------------------------------

def _make_pages() -> list[dict]:
    return [
        {"page_number": 1, "image_base64": "aW1nMQ==", "raw_text": "text1", "page_hash": "hash_1"},
        {"page_number": 2, "image_base64": "aW1nMg==", "raw_text": "text2", "page_hash": "hash_2"},
        {"page_number": 3, "image_base64": "aW1nMw==", "raw_text": "text3", "page_hash": "hash_3"},
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def fake_analyzed_service_pdf(tmp_path):
    """Returns (service, mock_b) for the file-hash short-circuit + page-hash tests.

    iter_pdf_page_images is patched to return 3 deterministic pages.
    BAML b.AnalyzePage is patched to return minimal results.
    compute_file_hash is patched to return a fixed file_hash so we control it.
    """
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
    )
    svc._registry = MagicMock()  # inject registry without going through build_registry

    mock_b = MagicMock()
    mock_b.AnalyzePage = AsyncMock(side_effect=lambda img, **kw: _make_fake_baml_result(0))

    pages = _make_pages()

    with (
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.compute_file_hash",
            return_value="file_hash_abc",
        ),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.retrieval.baml.baml_client.async_client.b", mock_b),
    ):
        yield svc, mock_b

    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_pdf_mutable(tmp_path):
    """Returns (service, mock_b, modify_page) where modify_page(n) changes page n's hash.

    Allows per-page cache tests: call modify_page(2) to simulate page 2 changing.
    """
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
    )
    svc._registry = MagicMock()  # inject registry without going through build_registry

    mock_b = MagicMock()
    mock_b.AnalyzePage = AsyncMock(side_effect=lambda img, **kw: _make_fake_baml_result(0))

    pages = _make_pages()

    def modify_page(page_num: int) -> None:
        """Change page_num's hash to simulate content change."""
        for p in pages:
            if p["page_number"] == page_num:
                p["page_hash"] = f"changed_hash_{page_num}"
                break

    file_hash_state = ["file_hash_fixed"]

    with (
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.iter_pdf_page_images",
            side_effect=lambda fp, **_kw: iter([dict(p) for p in pages]),
        ),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.compute_file_hash",
            side_effect=lambda fp: file_hash_state[0],
        ),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.retrieval.baml.baml_client.async_client.b", mock_b),
    ):
        def change_file_hash():
            file_hash_state[0] = "file_hash_different"

        # Expose modify_page to the test; test must call change_file_hash separately if needed.
        # For the page-level cache test we keep the file hash different per call by toggling.
        svc._change_file_hash = change_file_hash
        yield svc, mock_b, modify_page

    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_pdf_partial(tmp_path):
    """Returns (service, mock_b) pre-seeded with a Source in status='analyzed'."""
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = AnalyzedIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
    )
    svc._registry = MagicMock()  # inject registry without going through build_registry

    # Seed a Source with status='analyzed' for hash 'deadbeef_partial'
    existing = _fake_source(
        source_id="src_partial",
        knowledge_id="k1",
        file_hash="file_hash_partial",
        status="analyzed",
    )
    await store.create_source(existing)
    # Seed page_analyses rows for the source
    await store.upsert_page_analyses(
        "src_partial",
        [
            {"page_number": 1, "data": {"description": "p1", "page_hash": "hash_1", "entities": [],
                                        "tables": [], "annotations": [], "page_type": "text",
                                        "metadata": {}, "raw_text": "text1"}},
            {"page_number": 2, "data": {"description": "p2", "page_hash": "hash_2", "entities": [],
                                        "tables": [], "annotations": [], "page_type": "text",
                                        "metadata": {}, "raw_text": "text2"}},
            {"page_number": 3, "data": {"description": "p3", "page_hash": "hash_3", "entities": [],
                                        "tables": [], "annotations": [], "page_type": "text",
                                        "metadata": {}, "raw_text": "text3"}},
        ],
    )

    mock_b = MagicMock()
    mock_b.AnalyzePage = AsyncMock(side_effect=lambda img, **kw: _make_fake_baml_result(0))

    with (
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.compute_file_hash",
            return_value="file_hash_partial",
        ),
        patch(
            "rfnry_rag.retrieval.modules.ingestion.analyze.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_rag.retrieval.baml.baml_client.async_client.b", mock_b),
    ):
        yield svc, mock_b

    await store.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reingesting_same_file_short_circuits_via_file_hash(
    fake_analyzed_service_pdf,
) -> None:
    """When re-calling analyze() on the same file, zero additional LLM calls fire."""
    svc, mock_baml = fake_analyzed_service_pdf
    src1 = await svc.analyze("/tmp/doc.pdf", knowledge_id="k1")
    first_count = mock_baml.AnalyzePage.call_count
    assert first_count > 0

    # Re-ingest: file_hash match on k1 should return early
    src2 = await svc.analyze("/tmp/doc.pdf", knowledge_id="k1")
    assert mock_baml.AnalyzePage.call_count == first_count
    # Same source_id returned (not a new analyze)
    assert src2.source_id == src1.source_id


@pytest.mark.asyncio
async def test_page_level_cache_skips_unchanged_pages(
    fake_analyzed_service_pdf_mutable,
) -> None:
    """If page 2 changes but pages 1 and 3 don't, only 1 new LLM call fires on re-analyze."""
    svc, mock_baml, modify_page = fake_analyzed_service_pdf_mutable

    await svc.analyze("/tmp/doc.pdf", knowledge_id="k1")
    count_1 = mock_baml.AnalyzePage.call_count  # 3 for a 3-page fixture

    # Modify page 2 so its image_hash changes; use a DIFFERENT knowledge_id so
    # the file-hash short-circuit doesn't fire. Per-page cache should still kick
    # in for pages 1 and 3.
    modify_page(2)
    # Also change file hash so file-hash short-circuit doesn't fire for k2
    svc._change_file_hash()

    await svc.analyze("/tmp/doc.pdf", knowledge_id="k2")
    # Only 1 additional LLM call — for page 2
    assert mock_baml.AnalyzePage.call_count == count_1 + 1


@pytest.mark.asyncio
async def test_page_hash_stored_on_each_row(fake_analyzed_service_pdf) -> None:
    """After analyze, each rag_page_analyses row has a non-empty page_hash."""
    svc, _ = fake_analyzed_service_pdf
    src = await svc.analyze("/tmp/doc.pdf", knowledge_id="k1")
    rows = await svc._metadata_store.get_page_analyses(src.source_id)
    assert all(r["page_hash"] for r in rows), [r["page_hash"] for r in rows]


@pytest.mark.asyncio
async def test_file_hash_short_circuit_respects_status(
    fake_analyzed_service_pdf_partial,
) -> None:
    """A prior Source with status='analyzed' triggers the short-circuit — no LLM calls."""
    svc, mock_baml = fake_analyzed_service_pdf_partial
    # The store is pre-seeded with a Source having file_hash='file_hash_partial', status='analyzed'
    src = await svc.analyze("/tmp/doc.pdf", knowledge_id="k1")
    # No new LLM calls — reused the analyzed source
    assert mock_baml.AnalyzePage.call_count == 0
    assert src.status == "analyzed"


@pytest.mark.asyncio
async def test_empty_page_hash_does_not_match_cache(
    fake_analyzed_service_pdf_mutable,
) -> None:
    """A PageAnalysis with empty/NULL page_hash must not be matched to anything."""
    svc, mock_baml, _ = fake_analyzed_service_pdf_mutable

    # Seed: source with page 1 row having page_hash=None (legacy data)
    legacy_source = _fake_source("s_legacy", "k_legacy", "hash_legacy_file")
    await svc._metadata_store.create_source(legacy_source)
    await svc._metadata_store.upsert_page_analyses(
        "s_legacy",
        [
            {
                "page_number": 1,
                "data": {
                    "description": "legacy",
                    "page_hash": None,  # NULL hash — legacy row
                    "entities": [],
                    "tables": [],
                    "annotations": [],
                    "page_type": "text",
                    "metadata": {},
                    "raw_text": "",
                },
            },
        ],
    )

    # Query with a specific hash — should not match the NULL-hash row
    hits = await svc._metadata_store.get_page_analyses_by_hash(
        page_hashes=["any_hash"],
        knowledge_id="k_legacy",
    )
    assert hits == {}
