"""Document synthesis failures soft-skip with a note instead of raising.

The PDF synthesize phase calls SynthesizeDocument; on failure we record a
note and proceed with an empty DocumentSynthesis. Downstream consumers
(graph mapper, xref_map builder) already iterate over empty cross-reference
lists without breaking.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_rag.ingestion.analyze.models import DocumentSynthesis
from rfnry_rag.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_rag.models import Source

_PAGE_ROWS = [
    {
        "page_number": 1,
        "data": {
            "page_number": 1,
            "description": "page 1",
            "entities": [],
            "tables": [],
            "annotations": [],
            "page_type": "diagram",
            "metadata": {},
            "raw_text": "",
        },
    },
    {
        "page_number": 2,
        "data": {
            "page_number": 2,
            "description": "page 2",
            "entities": [],
            "tables": [],
            "annotations": [],
            "page_type": "diagram",
            "metadata": {},
            "raw_text": "",
        },
    },
]


def _make_service():
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 4] * len(texts))
    vector_store = AsyncMock()
    metadata_store = AsyncMock()
    metadata_store.get_page_analyses = AsyncMock(return_value=_PAGE_ROWS)
    svc = AnalyzedIngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_model_name="t",
    )
    svc._registry = MagicMock()
    return svc


def _analyzed_source() -> Source:
    return Source(
        source_id="s1",
        knowledge_id=None,
        source_type=None,
        status="analyzed",
        embedding_model="t",
        metadata={"file_type": "pdf", "file_name": "f.pdf"},
    )


async def test_synthesis_failure_records_note_continues() -> None:
    svc = _make_service()
    svc._metadata_store.get_source = AsyncMock(return_value=_analyzed_source())
    svc._metadata_store.update_source = AsyncMock()

    with patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b:
        mock_b.SynthesizeDocument = AsyncMock(side_effect=RuntimeError("LLM down"))
        await svc.synthesize("s1")

    update_call = svc._metadata_store.update_source.await_args
    written_metadata = update_call.kwargs["metadata"]
    notes = written_metadata.get("ingestion_notes", [])
    assert any(n.startswith("document_synthesis:warn:RuntimeError(") for n in notes), notes
    # Synthesis is empty (default) and serialized into metadata.
    assert written_metadata["synthesis"]["cross_references"] == []


async def test_synthesis_clean_no_notes() -> None:
    svc = _make_service()
    svc._metadata_store.get_source = AsyncMock(return_value=_analyzed_source())
    svc._metadata_store.update_source = AsyncMock()

    fake_result = MagicMock()
    fake_result.cross_references = []
    fake_result.page_clusters = []
    fake_result.document_summary = "summary"

    with patch("rfnry_rag.baml.baml_client.async_client.b") as mock_b:
        mock_b.SynthesizeDocument = AsyncMock(return_value=fake_result)
        await svc.synthesize("s1")

    update_call = svc._metadata_store.update_source.await_args
    written_metadata = update_call.kwargs["metadata"]
    assert "ingestion_notes" not in written_metadata


async def test_synthesis_consumer_tolerates_empty_synthesis() -> None:
    """The ingest phase reads metadata['synthesis'] and feeds it to consumers
    (xref_map and cross_refs_to_graph_relations). Empty synthesis must not
    block ingest completion."""
    svc = _make_service()

    source = _analyzed_source()
    source.status = "synthesized"
    # Empty synthesis path: serialize an empty DocumentSynthesis into metadata.
    from rfnry_rag.ingestion.analyze.service import _serialize_synthesis

    source.metadata["synthesis"] = _serialize_synthesis(DocumentSynthesis())
    source.metadata["ingestion_notes"] = ["document_synthesis:warn:RuntimeError(boom)"]

    svc._metadata_store.get_source = AsyncMock(return_value=source)
    svc._metadata_store.update_source = AsyncMock()

    await svc.ingest("s1")

    svc._vector_store.upsert.assert_awaited()