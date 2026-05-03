"""Text-density pre-filter: text-heavy + image-free pages skip the vision LLM call."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from rfnry_knowledge.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_baml_result(page_num: int) -> SimpleNamespace:
    return SimpleNamespace(
        description=f"vision description for page {page_num}",
        entities=[],
        tables=[],
        annotations=[],
        page_type="diagram",
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
# Page lists used by fixtures
# ---------------------------------------------------------------------------


def _make_mixed_pages() -> list[dict]:
    """3-page PDF: pages 1 and 3 are image-heavy; page 2 is text-only (500 chars, no images)."""
    return [
        {
            "page_number": 1,
            "image_base64": "aW1nMQ==",
            "raw_text": "short",  # 5 chars — below 300 threshold
            "raw_text_char_count": 5,
            "has_images": False,
            "page_hash": "hash_p1",
        },
        {
            "page_number": 2,
            "image_base64": "aW1nMg==",
            "raw_text": "x" * 500,  # 500 chars — above threshold, no images
            "raw_text_char_count": 500,
            "has_images": False,
            "page_hash": "hash_p2",
        },
        {
            "page_number": 3,
            "image_base64": "aW1nMw==",
            "raw_text": "short",  # 5 chars — below threshold
            "raw_text_char_count": 5,
            "has_images": False,
            "page_hash": "hash_p3",
        },
    ]


def _make_text_with_images_pages() -> list[dict]:
    """1-page PDF: lots of text (500 chars) but also contains embedded images."""
    return [
        {
            "page_number": 1,
            "image_base64": "aW1nMQ==",
            "raw_text": "x" * 500,  # 500 chars — above threshold
            "raw_text_char_count": 500,
            "has_images": True,  # but has embedded images → must NOT skip vision
            "page_hash": "hash_img1",
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fake_analyzed_service_mixed_pdf(tmp_path):
    """Service + mock_baml for a 3-page PDF where page 2 is text-only (no images).

    Default threshold is 300 chars (the service default). Vision should only
    fire on pages 1 and 3.
    """
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = StructuredIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
        analyze_text_skip_threshold_chars=300,
    )
    svc._registry = MagicMock()

    mock_b = MagicMock()
    mock_b.AnalyzeStructuredPage = AsyncMock(
        side_effect=lambda img, **kw: _make_fake_baml_result(0),
    )

    pages = _make_mixed_pages()

    with (
        patch(
            "rfnry_knowledge.ingestion.structured.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.compute_file_hash",
            return_value="file_hash_mixed",
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_knowledge.baml.baml_client.async_client.b", mock_b),
    ):
        yield svc, mock_b

    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_mixed_pdf_thresh0(tmp_path):
    """Same 3-page PDF but with threshold=0 (pre-filter disabled)."""
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = StructuredIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
        analyze_text_skip_threshold_chars=0,  # disabled
    )
    svc._registry = MagicMock()

    mock_b = MagicMock()
    mock_b.AnalyzeStructuredPage = AsyncMock(
        side_effect=lambda img, **kw: _make_fake_baml_result(0),
    )

    pages = _make_mixed_pages()

    with (
        patch(
            "rfnry_knowledge.ingestion.structured.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.compute_file_hash",
            return_value="file_hash_mixed_thresh0",
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_knowledge.baml.baml_client.async_client.b", mock_b),
    ):
        yield svc, mock_b

    await store.shutdown()


@pytest_asyncio.fixture
async def fake_analyzed_service_text_with_images(tmp_path):
    """1-page PDF: 500 chars of text but also embedded images — vision must still fire."""
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    store = SQLAlchemyMetadataStore(url=f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    await store.initialize()

    svc = StructuredIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=store,
        embedding_model_name="fake-384",
        vision=_FakeVision(),
        analyze_text_skip_threshold_chars=300,
    )
    svc._registry = MagicMock()

    mock_b = MagicMock()
    mock_b.AnalyzeStructuredPage = AsyncMock(
        side_effect=lambda img, **kw: _make_fake_baml_result(0),
    )

    pages = _make_text_with_images_pages()

    with (
        patch(
            "rfnry_knowledge.ingestion.structured.service.iter_pdf_page_images",
            return_value=iter(pages),
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.compute_file_hash",
            return_value="file_hash_text_img",
        ),
        patch(
            "rfnry_knowledge.ingestion.structured.service.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=lambda fn, *args: fn(*args),
        ),
        patch("rfnry_knowledge.baml.baml_client.async_client.b", mock_b),
    ):
        yield svc, mock_b

    await store.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_heavy_page_skips_vision(fake_analyzed_service_mixed_pdf) -> None:
    """Fixture: 3-page PDF where page 2 is text-only (no images). Vision only fires on pages 1 and 3."""
    svc, mock_baml = fake_analyzed_service_mixed_pdf
    await svc.analyze("/tmp/mixed.pdf", knowledge_id="k1")
    assert mock_baml.AnalyzeStructuredPage.call_count == 2  # only pages 1 and 3, not page 2


@pytest.mark.asyncio
async def test_text_only_page_produces_page_analysis_with_page_type_text(
    fake_analyzed_service_mixed_pdf,
) -> None:
    svc, _ = fake_analyzed_service_mixed_pdf
    src = await svc.analyze("/tmp/mixed.pdf", knowledge_id="k1")
    rows = await svc._metadata_store.get_page_analyses(src.source_id)
    page_2 = next(r for r in rows if r["page_number"] == 2)
    assert page_2["data"]["page_type"] == "text"
    # The raw text ends up in the description
    assert page_2["data"]["description"]


@pytest.mark.asyncio
async def test_threshold_zero_disables_prefilter(fake_analyzed_service_mixed_pdf_thresh0) -> None:
    """When analyze_text_skip_threshold_chars=0, all pages go to vision (pre-filter off)."""
    svc, mock_baml = fake_analyzed_service_mixed_pdf_thresh0
    await svc.analyze("/tmp/mixed.pdf", knowledge_id="k1")
    assert mock_baml.AnalyzeStructuredPage.call_count == 3  # all pages, none skipped


@pytest.mark.asyncio
async def test_page_with_images_does_not_skip_vision(fake_analyzed_service_text_with_images) -> None:
    """A page that has abundant text BUT also embedded images must NOT be skipped
    (the images may carry content not captured by text)."""
    svc, mock_baml = fake_analyzed_service_text_with_images
    # Fixture: 1-page PDF where the page has 500 chars text + 1 embedded image
    await svc.analyze("/tmp/text_with_img.pdf", knowledge_id="k1")
    assert mock_baml.AnalyzeStructuredPage.call_count == 1  # vision still fires


def test_wrapper_bounds_rejected() -> None:
    from unittest.mock import MagicMock

    from rfnry_knowledge.exceptions import ConfigurationError
    from rfnry_knowledge.ingestion.methods.structured import StructuredIngestion

    with pytest.raises(ConfigurationError):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=-1)
    with pytest.raises(ConfigurationError):
        StructuredIngestion(store=MagicMock(), embeddings=MagicMock(), analyze_text_skip_threshold_chars=100_001)
