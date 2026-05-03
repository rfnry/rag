"""Per-step soft-fail notes propagation through IngestionService and graph method.

Pins the contract that:

- ``EntityIngestion.ingest()`` writes a ``graph:warn:extraction_failed(...)`` note
  to the caller-supplied ``notes`` list when the BAML call raises.
- ``IngestionService`` instantiates a fresh notes list per ingest, threads it
  through ``_dispatch_methods``, and merges the result into
  ``metadata["ingestion_notes"]`` at the end.
- A clean ingest produces no notes and ``Source.fully_ingested`` is True.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_knowledge.ingestion.chunk.service import IngestionService
from rfnry_knowledge.ingestion.methods.entity import EntityIngestion
from rfnry_knowledge.ingestion.models import ChunkedContent


def _service_chunker(contents: list[str]) -> MagicMock:
    chunker = MagicMock()
    chunker.chunk = MagicMock(
        return_value=[ChunkedContent(content=c, page_number=1, chunk_index=i) for i, c in enumerate(contents)]
    )
    return chunker


def _required_method(name: str = "vector") -> SimpleNamespace:
    return SimpleNamespace(name=name, required=True, ingest=AsyncMock(), delete=AsyncMock())


def _optional_method(name: str = "graph") -> SimpleNamespace:
    return SimpleNamespace(name=name, required=False, ingest=AsyncMock(), delete=AsyncMock())


async def test_graph_ingestion_soft_fail_writes_note() -> None:
    store = AsyncMock()
    lm_client = MagicMock()
    notes: list[str] = []
    with (
        patch("rfnry_knowledge.ingestion.methods.entity.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.entity.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_registry.return_value = MagicMock()
        method = EntityIngestion(store=store, provider_client=lm_client)
        await method.ingest(
            source_id="src-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="Some text.",
            chunks=[],
            tags=[],
            metadata={},
            notes=notes,
        )
    assert len(notes) == 1
    assert notes[0].startswith("graph:warn:extraction_failed(")
    assert "LLM down" in notes[0]


async def test_graph_ingestion_no_note_when_notes_is_none() -> None:
    store = AsyncMock()
    lm_client = MagicMock()
    with (
        patch("rfnry_knowledge.ingestion.methods.entity.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.entity.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_registry.return_value = MagicMock()
        method = EntityIngestion(store=store, provider_client=lm_client)
        await method.ingest(
            source_id="src-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="Some text.",
            chunks=[],
            tags=[],
            metadata={},
        )


async def test_graph_clean_run_no_note() -> None:
    store = AsyncMock()
    store.add_entities = AsyncMock()
    lm_client = MagicMock()
    mock_entity = MagicMock()
    mock_entity.name = "X"
    mock_entity.category = "thing"
    mock_entity.value = None
    mock_entity.context = ""
    mock_result = MagicMock()
    mock_result.description = "desc"
    mock_result.entities = [mock_entity]
    mock_result.tables = []
    mock_result.annotations = []
    mock_result.page_type = "text"
    notes: list[str] = []
    with (
        patch("rfnry_knowledge.ingestion.methods.entity.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.entity.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()
        method = EntityIngestion(store=store, provider_client=lm_client)
        await method.ingest(
            source_id="src-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="X is a thing.",
            chunks=[],
            tags=[],
            metadata={},
            notes=notes,
        )
    assert notes == []


async def test_optional_method_soft_fail_writes_note_to_metadata() -> None:
    """Optional ingestion method that produces a note via the notes list:
    IngestionService merges it into ``metadata['ingestion_notes']``."""

    vector = _required_method("vector")
    graph = _optional_method("graph")

    async def fake_graph_ingest(**kwargs):
        notes = kwargs.get("notes")
        if notes is not None:
            notes.append("graph:warn:extraction_failed(LLM down)")

    graph.ingest = AsyncMock(side_effect=fake_graph_ingest)

    service = IngestionService(
        chunker=_service_chunker(["alpha", "beta"]),
        ingestion_methods=[vector, graph],
    )

    source = await service.ingest_text(content="DOCBODY", metadata={"name": "src"})

    assert source.fully_ingested is False
    assert source.ingestion_notes == ["graph:warn:extraction_failed(LLM down)"]


async def test_clean_ingest_no_notes() -> None:
    vector = _required_method("vector")
    graph = _optional_method("graph")
    service = IngestionService(
        chunker=_service_chunker(["alpha", "beta"]),
        ingestion_methods=[vector, graph],
    )

    source = await service.ingest_text(content="DOCBODY", metadata={"name": "src"})

    assert source.fully_ingested is True
    assert source.ingestion_notes == []


async def test_analyzed_method_soft_fail_writes_method_failed_note() -> None:
    """When a delegate method raises during the analyzed-pipeline ingest phase,
    the service catches the error, logs a warning, and appends a
    ``<method>:warn:method_failed(...)`` note to source.metadata."""
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService
    from rfnry_knowledge.models import Source

    page_rows = [
        {
            "page_number": 1,
            "data": {
                "page_number": 1,
                "description": "page1",
                "entities": [],
                "tables": [],
                "annotations": [],
                "page_type": "text",
                "metadata": {},
                "raw_text": "raw",
            },
        }
    ]

    embeddings = MagicMock()
    embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 8] * len(texts))
    vector_store = AsyncMock()
    metadata_store = AsyncMock()
    metadata_store.get_page_analyses = AsyncMock(return_value=page_rows)

    service = StructuredIngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_model_name="t",
    )

    source = Source(
        source_id="s1",
        knowledge_id=None,
        source_type=None,
        status="synthesized",
        embedding_model="t",
        metadata={"file_type": "pdf", "file_name": "f.pdf"},
    )
    metadata_store.get_source = AsyncMock(return_value=source)
    metadata_store.update_source = AsyncMock()

    boom_method = SimpleNamespace(
        name="document",
        required=True,
        ingest=AsyncMock(side_effect=RuntimeError("doc store down")),
        delete=AsyncMock(),
    )
    service._ingestion_methods = [boom_method]

    await service.ingest("s1")

    update_calls = metadata_store.update_source.await_args_list
    assert update_calls, "expected update_source to be called"
    last = update_calls[-1]
    written_metadata = last.kwargs.get("metadata")
    assert written_metadata is not None
    notes = written_metadata.get("ingestion_notes", [])
    assert any(n.startswith("document:warn:method_failed(") for n in notes), notes


async def test_analyzed_clean_pipeline_no_notes_no_metadata_write() -> None:
    """Clean analyzed ingest finishes without writing metadata (pre-Phase-A behavior)."""
    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService
    from rfnry_knowledge.models import Source

    page_rows = [
        {
            "page_number": 1,
            "data": {
                "page_number": 1,
                "description": "page1",
                "entities": [],
                "tables": [],
                "annotations": [],
                "page_type": "text",
                "metadata": {},
                "raw_text": "raw",
            },
        }
    ]

    embeddings = MagicMock()
    embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 8] * len(texts))
    vector_store = AsyncMock()
    metadata_store = AsyncMock()
    metadata_store.get_page_analyses = AsyncMock(return_value=page_rows)

    service = StructuredIngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_model_name="t",
    )

    source = Source(
        source_id="s1",
        knowledge_id=None,
        source_type=None,
        status="synthesized",
        embedding_model="t",
        metadata={"file_type": "pdf", "file_name": "f.pdf"},
    )
    metadata_store.get_source = AsyncMock(return_value=source)
    metadata_store.update_source = AsyncMock()

    await service.ingest("s1")

    update_calls = metadata_store.update_source.await_args_list
    assert update_calls
    last = update_calls[-1]
    # No notes → metadata not written; only status + chunk_count.
    assert "metadata" not in last.kwargs
