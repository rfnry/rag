"""Required-vs-optional ingestion methods.

Regression: when a required method (vector/document) fails, the service must
raise IngestionError AND skip the metadata commit. Previously every failure
was caught as a warning and the source row was committed anyway, causing
silent data loss (e.g. vector upsert failed but user sees a valid source)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rfnry_rag.exceptions import IngestionError
from rfnry_rag.ingestion.chunk.chunker import SemanticChunker
from rfnry_rag.ingestion.chunk.service import IngestionService


def _make_method(name: str, *, required: bool, fails: bool) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.required = required
    m.ingest = AsyncMock(side_effect=RuntimeError("boom") if fails else None)
    m.delete = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_required_method_failure_aborts_and_does_not_commit_source(tmp_path):
    metadata_store = MagicMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()
    svc = IngestionService(
        chunker=SemanticChunker(chunk_size=100, chunk_overlap=10),
        ingestion_methods=[
            _make_method("vector", required=True, fails=True),
            _make_method("document", required=True, fails=False),
        ],
        embedding_model_name="test:model",
        source_type_weights=None,
        metadata_store=metadata_store,
        on_ingestion_complete=None,
        vision_parser=None,
        chunk_context_headers=False,
    )
    fp = tmp_path / "a.txt"
    fp.write_text("hello world " * 50)

    with pytest.raises(IngestionError, match="required ingestion method failed"):
        await svc.ingest(file_path=fp)

    metadata_store.create_source.assert_not_called()


@pytest.mark.asyncio
async def test_optional_method_failure_is_logged_and_ingest_succeeds(tmp_path):
    metadata_store = MagicMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()
    svc = IngestionService(
        chunker=SemanticChunker(chunk_size=100, chunk_overlap=10),
        ingestion_methods=[
            _make_method("vector", required=True, fails=False),
            _make_method("graph", required=False, fails=True),
        ],
        embedding_model_name="test:model",
        source_type_weights=None,
        metadata_store=metadata_store,
        on_ingestion_complete=None,
        vision_parser=None,
        chunk_context_headers=False,
    )
    fp = tmp_path / "a.txt"
    fp.write_text("hello world " * 50)

    await svc.ingest(file_path=fp)

    metadata_store.create_source.assert_awaited_once()


@pytest.mark.asyncio
async def test_method_without_required_attribute_defaults_to_required(tmp_path):
    """Back-compat: if a protocol-conforming third-party method doesn't expose
    `required`, we treat it as required so failures are not silently swallowed."""
    metadata_store = MagicMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()

    legacy_method = MagicMock()
    legacy_method.name = "legacy"
    legacy_method.ingest = AsyncMock(side_effect=RuntimeError("boom"))
    legacy_method.delete = AsyncMock()
    del legacy_method.required  # MagicMock auto-creates attrs; delete to simulate "missing"

    svc = IngestionService(
        chunker=SemanticChunker(chunk_size=100, chunk_overlap=10),
        ingestion_methods=[legacy_method],
        embedding_model_name="test:model",
        source_type_weights=None,
        metadata_store=metadata_store,
        on_ingestion_complete=None,
        vision_parser=None,
        chunk_context_headers=False,
    )
    fp = tmp_path / "a.txt"
    fp.write_text("hello world " * 50)

    with pytest.raises(IngestionError):
        await svc.ingest(file_path=fp)


@pytest.mark.asyncio
async def test_required_methods_multiple_failures_surface_all_messages(tmp_path) -> None:
    """When multiple required methods fail concurrently, the IngestionError must
    include all failure messages, not just the first one picked from the ExceptionGroup."""
    from unittest.mock import AsyncMock

    from rfnry_rag.ingestion.chunk.service import IngestionService

    def _mock_method(name: str, required: bool = True):
        from types import SimpleNamespace

        return SimpleNamespace(name=name, required=required, ingest=AsyncMock(), delete=AsyncMock())

    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(
                content="chunk text",
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
    metadata_store = MagicMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()

    a = _mock_method(name="a", required=True)
    a.ingest = AsyncMock(side_effect=RuntimeError("boom-a"))
    b = _mock_method(name="b", required=True)
    b.ingest = AsyncMock(side_effect=RuntimeError("boom-b"))

    service = IngestionService(
        chunker=chunker,
        ingestion_methods=[a, b],
        embedding_model_name="test:model",
        metadata_store=metadata_store,
        chunk_context_headers=False,
    )

    fp = tmp_path / "sample.txt"
    fp.write_text("hello world " * 50)

    with pytest.raises(IngestionError) as excinfo:
        await service.ingest(file_path=fp)

    msg = str(excinfo.value)
    assert "boom-a" in msg
    assert "boom-b" in msg
