from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.modules.ingestion.chunk.service import IngestionService


def _mock_method(name: str, required: bool = True) -> SimpleNamespace:
    return SimpleNamespace(name=name, required=required, ingest=AsyncMock(), delete=AsyncMock())


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

    source = await service.ingest_text(content="Hello world", metadata={"name": "test"})
    vector.ingest.assert_called_once()
    document.ingest.assert_called_once()
    assert source.chunk_count == 1


async def test_ingest_text_no_methods():
    service = _make_service(methods=[])
    source = await service.ingest_text(content="Hello world")
    assert source is not None
    assert source.chunk_count == 1


async def test_ingest_text_creates_metadata_source():
    meta_store = SimpleNamespace(
        create_source=AsyncMock(),
        list_sources=AsyncMock(return_value=[]),
    )
    service = _make_service(methods=[], metadata_store=meta_store)
    source = await service.ingest_text(content="Hello world")
    meta_store.create_source.assert_called_once()
    created = meta_store.create_source.call_args[0][0]
    assert created.source_id == source.source_id
    assert created.chunk_count == 1


async def test_ingest_text_fires_on_complete_callback():
    callback = AsyncMock()
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
    service = IngestionService(
        chunker=chunker,
        ingestion_methods=[],
        on_ingestion_complete=callback,
    )
    source = await service.ingest_text(content="Hello world", knowledge_id="k1")
    callback.assert_called_once_with("k1")
    assert source is not None


async def test_ingest_text_method_receives_correct_args():
    method = _mock_method("vector")
    service = _make_service(methods=[method])

    source = await service.ingest_text(
        content="Hello world",
        knowledge_id="k1",
        source_type="document",
        metadata={"name": "test-doc"},
    )

    call_kwargs = method.ingest.call_args[1]
    assert call_kwargs["source_id"] == source.source_id
    assert call_kwargs["knowledge_id"] == "k1"
    assert call_kwargs["source_type"] == "document"
    assert call_kwargs["title"] == "test-doc"
    assert call_kwargs["full_text"] == "Hello world"
    assert len(call_kwargs["chunks"]) == 1
    assert call_kwargs["tags"] == []
    assert call_kwargs["metadata"] == {"name": "test-doc"}


async def test_ingest_text_empty_chunks_raises():
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[])
    service = IngestionService(
        chunker=chunker,
        ingestion_methods=[],
    )
    import pytest

    from rfnry_rag.retrieval.common.errors import EmptyDocumentError

    with pytest.raises(EmptyDocumentError):
        await service.ingest_text(content="Hello world")


# --- Tests merged from test_fulltext_ingestion.py ---


def _make_method_with_store(name="document"):
    return SimpleNamespace(
        name=name,
        ingest=AsyncMock(),
        delete=AsyncMock(),
    )


def _make_service_with_metadata(ingestion_methods=None):
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[
            MagicMock(content="chunk 1", page_number=1, section=None, chunk_index=0),
        ]
    )

    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.create_source = AsyncMock()

    if ingestion_methods is None:
        ingestion_methods = [_make_method_with_store("vector")]

    return IngestionService(
        chunker=chunker,
        ingestion_methods=ingestion_methods,
        embedding_model_name="test:model",
        metadata_store=metadata_store,
    )


async def test_ingest_calls_document_method(tmp_path):
    doc_method = _make_method_with_store("document")
    service = _make_service_with_metadata(ingestion_methods=[doc_method])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Page one content.\n\nPage two content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals")

    doc_method.ingest.assert_called_once()
    kwargs = doc_method.ingest.call_args[1]
    assert kwargs["knowledge_id"] == "kb-1"
    assert kwargs["source_type"] == "manuals"
    assert "Page one content" in kwargs["full_text"]


async def test_ingest_text_calls_document_method():
    doc_method = _make_method_with_store("document")
    service = _make_service_with_metadata(ingestion_methods=[doc_method])

    await service.ingest_text(
        content="Full manual text with FBD-20254.",
        knowledge_id="kb-1",
        source_type="manuals",
    )

    doc_method.ingest.assert_called_once()
    kwargs = doc_method.ingest.call_args[1]
    assert "FBD-20254" in kwargs["full_text"]


async def test_ingest_without_methods(tmp_path):
    """Ingestion succeeds with an empty method list (no methods to dispatch)."""
    service = _make_service_with_metadata(ingestion_methods=[])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(file_path=test_file, knowledge_id="kb-1")
    assert source is not None


async def test_structured_ingestion_has_document_method():
    """AnalyzedIngestionService stores document method in ingestion_methods."""
    from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService

    doc_method = SimpleNamespace(name="document", ingest=AsyncMock(), delete=AsyncMock())
    metadata_store = AsyncMock()
    embeddings = AsyncMock()
    embeddings.model = "test-model"

    service = AnalyzedIngestionService(
        embeddings=embeddings,
        vector_store=AsyncMock(),
        metadata_store=metadata_store,
        embedding_model_name="test:model",
        ingestion_methods=[doc_method],
    )
    assert doc_method in service._ingestion_methods
    assert any(m.name == "document" for m in service._ingestion_methods)


# --- Tests merged from test_ingestion_advanced.py ---


def _make_service_advanced(ingestion_methods=None, chunk_context_headers=True):
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

    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.create_source = AsyncMock()

    if ingestion_methods is None:
        ingestion_methods = [_mock_method("vector")]

    return IngestionService(
        chunker=chunker,
        ingestion_methods=ingestion_methods,
        embedding_model_name="test:model",
        metadata_store=metadata_store,
        chunk_context_headers=chunk_context_headers,
    )


async def test_ingestion_creates_source(tmp_path):
    """The service creates a Source object and stores it via metadata_store."""
    method = _mock_method("vector")
    service = _make_service_advanced(ingestion_methods=[method])

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    source = await service.ingest(
        file_path=test_file,
        knowledge_id="kb-1",
        source_type="manuals",
        metadata={"name": "Test Doc"},
    )

    assert source.source_id is not None
    assert source.knowledge_id == "kb-1"
    assert source.source_type == "manuals"
    assert source.chunk_count == 1
    assert source.embedding_model == "test:model"

    service._metadata_store.create_source.assert_called_once()


async def test_ingestion_payload_has_context_fields(tmp_path):
    """Contextual chunking adds context fields to chunks before passing to methods."""
    method = _mock_method("vector")
    service = _make_service_advanced(ingestion_methods=[method], chunk_context_headers=True)

    test_file = tmp_path / "test.txt"
    test_file.write_text("Some content.")

    await service.ingest(file_path=test_file, knowledge_id="kb-1", source_type="manuals", metadata={"name": "Test Doc"})

    method.ingest.assert_called_once()
    kwargs = method.ingest.call_args[1]
    chunks = kwargs["chunks"]
    chunk = chunks[0]
    assert hasattr(chunk, "context")
    assert hasattr(chunk, "contextualized")
    assert hasattr(chunk, "chunk_type")


# --- Partial failure test ---


async def test_method_failure_does_not_abort_pipeline():
    """If an OPTIONAL method raises, others still run."""
    failing = SimpleNamespace(
        name="graph", required=False, ingest=AsyncMock(side_effect=RuntimeError("boom")), delete=AsyncMock()
    )
    succeeding = SimpleNamespace(name="document", required=True, ingest=AsyncMock(), delete=AsyncMock())
    service = _make_service(methods=[failing, succeeding])
    source = await service.ingest_text(content="Hello world")
    # Graph failed but document still called
    succeeding.ingest.assert_called_once()
    assert source is not None


# --- on_progress callback tests ---


async def test_on_progress_fires_at_group_boundaries(tmp_path) -> None:
    """Progress fires once after required group and once after optional group."""
    calls: list[tuple[int, int]] = []

    async def progress(done: int, total: int) -> None:
        calls.append((done, total))

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world " * 50)

    # Two required methods, zero optional → progress fires ONCE after the required
    # group: (2, 2). The empty optional group is skipped entirely.
    method_a = _mock_method(name="a", required=True)
    method_b = _mock_method(name="b", required=True)
    service = _make_service_advanced(ingestion_methods=[method_a, method_b])

    await service.ingest(file_path=file_path, on_progress=progress)

    assert calls == [(2, 2)]


async def test_on_progress_fires_at_both_group_boundaries(tmp_path) -> None:
    """Progress fires after required group and after optional group (even when optional fails)."""
    calls: list[tuple[int, int]] = []

    async def progress(done: int, total: int) -> None:
        calls.append((done, total))

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world " * 50)

    # One required, one failing optional (total=2).
    # After required group: (1, 2). After optional group: (2, 2).
    failing_optional = _mock_method(name="opt", required=False)
    failing_optional.ingest = AsyncMock(side_effect=RuntimeError("boom"))
    required = _mock_method(name="req", required=True)

    service = _make_service_advanced(ingestion_methods=[required, failing_optional])
    await service.ingest(file_path=file_path, on_progress=progress)

    assert calls == [(1, 2), (2, 2)]


async def test_required_methods_run_concurrently_within_group(tmp_path) -> None:
    """Required methods execute in parallel inside the TaskGroup."""
    import asyncio

    concurrent = 0
    max_concurrent = 0

    async def slow_ingest(**kwargs: object) -> None:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world " * 50)

    a = _mock_method(name="a", required=True)
    a.ingest = AsyncMock(side_effect=slow_ingest)
    b = _mock_method(name="b", required=True)
    b.ingest = AsyncMock(side_effect=slow_ingest)

    service = _make_service_advanced(ingestion_methods=[a, b])
    await service.ingest(file_path=file_path)

    assert max_concurrent >= 2


async def test_optional_methods_run_concurrently_within_group(tmp_path) -> None:
    """Optional methods execute in parallel inside gather."""
    import asyncio

    concurrent = 0
    max_concurrent = 0

    async def slow_ingest(**kwargs: object) -> None:
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world " * 50)

    a = _mock_method(name="a", required=False)
    a.ingest = AsyncMock(side_effect=slow_ingest)
    b = _mock_method(name="b", required=False)
    b.ingest = AsyncMock(side_effect=slow_ingest)

    service = _make_service_advanced(ingestion_methods=[a, b])
    await service.ingest(file_path=file_path)

    assert max_concurrent >= 2
