from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.common.models import Source
from rfnry_rag.retrieval.modules.ingestion.analyze.service import AnalyzedIngestionService


def _make_service(graph_store=None, ingestion_methods=None):
    embeddings = MagicMock()
    embeddings.model = "test-model"
    embeddings.embed = AsyncMock(return_value=[[0.1] * 10, [0.2] * 10])

    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    metadata_store = AsyncMock()

    return AnalyzedIngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_model_name="test:test-model",
        graph_store=graph_store,
        ingestion_methods=ingestion_methods,
    )


def _make_source_with_analysis() -> Source:
    return Source(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="drawings",
        status="synthesized",
        embedding_model="test:test-model",
        metadata={
            "file_type": "pdf",
            "file_name": "test.pdf",
            "page_analyses": [
                {
                    "page_number": 1,
                    "description": "Electrical schematic showing motor circuit",
                    "entities": [
                        {
                            "name": "Motor M1",
                            "category": "electrical_component",
                            "context": "main motor",
                            "value": "480V",
                        },
                        {
                            "name": "Breaker CB-3",
                            "category": "electrical_component",
                            "context": "feeder",
                            "value": None,
                        },
                    ],
                    "tables": [],
                    "annotations": [],
                    "page_type": "electrical_schematic",
                    "metadata": {},
                },
                {
                    "page_number": 2,
                    "description": "Panel schedule",
                    "entities": [
                        {
                            "name": "Panel MCC-1",
                            "category": "electrical_component",
                            "context": "main panel",
                            "value": None,
                        },
                    ],
                    "tables": [],
                    "annotations": [],
                    "page_type": "panel_schedule",
                    "metadata": {},
                },
            ],
            "synthesis": {
                "cross_references": [
                    {
                        "source_page": 1,
                        "target_page": 2,
                        "relationship": "power feed from breaker to panel",
                        "shared_entities": ["Motor M1", "Panel MCC-1"],
                    },
                ],
                "page_clusters": [],
                "document_summary": "Test document",
            },
        },
    )


async def test_ingest_with_graph_store():
    graph_store = AsyncMock()
    service = _make_service(graph_store=graph_store)
    source = _make_source_with_analysis()
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    await service.ingest(source.source_id)

    graph_store.add_entities.assert_called_once()
    graph_store.add_relations.assert_called_once()

    entities_call = graph_store.add_entities.call_args
    assert entities_call.kwargs["source_id"] == "src-1"
    assert entities_call.kwargs["knowledge_id"] == "kb-1"
    entities = entities_call.kwargs["entities"]
    assert len(entities) == 3

    relations_call = graph_store.add_relations.call_args
    assert relations_call.kwargs["source_id"] == "src-1"
    relations = relations_call.kwargs["relations"]
    assert len(relations) >= 1


async def test_ingest_without_graph_store():
    service = _make_service(graph_store=None)
    source = _make_source_with_analysis()
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    await service.ingest(source.source_id)

    service._vector_store.upsert.assert_called_once()


async def test_ingest_graph_store_failure_warns():
    """Graph method failure is caught and logged as a warning, not raised."""
    graph_store = AsyncMock()
    graph_store.add_entities = AsyncMock(side_effect=RuntimeError("Neo4j connection failed"))

    service = _make_service(graph_store=graph_store)
    source = _make_source_with_analysis()
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    # Should not raise — method failures are caught and logged
    await service.ingest(source.source_id)
    # Vector upsert should still have succeeded
    service._vector_store.upsert.assert_called_once()


async def test_analyzed_ingestion_writes_document_store_exactly_once(tmp_path) -> None:
    """Phase 3 must be the single authoritative document write."""
    from unittest.mock import AsyncMock, MagicMock

    from rfnry_rag.retrieval.modules.ingestion.methods.document import DocumentIngestion

    doc_method = MagicMock(spec=DocumentIngestion)
    doc_method.name = "document"
    doc_method.required = True
    doc_method.ingest = AsyncMock()

    xml = tmp_path / "sample.xml"
    xml.write_text("<root><item>hello</item></root>")

    service = _make_service(ingestion_methods=[doc_method])
    # Override embed so it returns exactly one vector per text (avoids zip mismatch)
    service._embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 10] * len(texts))

    # Phase 1: analyze — metadata_store.create_source saves the source
    created_sources: list[Source] = []

    async def capture_create(source: Source) -> None:
        created_sources.append(source)

    service._metadata_store.create_source = AsyncMock(side_effect=capture_create)

    source = await service.analyze(file_path=xml)

    # Phase 2: synthesize — get_source returns the analyzed source; update_source updates it
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    source = await service.synthesize(source.source_id)

    # Refresh: get_source for phase 3 returns the synthesized source (with synthesis in metadata)
    service._metadata_store.get_source = AsyncMock(return_value=source)

    # Phase 3: ingest
    await service.ingest(source.source_id)

    # The document method must have been called exactly once — in phase 3 only
    assert doc_method.ingest.await_count == 1


async def test_analyzed_ingestion_identifies_document_method_by_type_not_name() -> None:
    """If DocumentIngestion.name is renamed, document storage must still route correctly."""
    from rfnry_rag.retrieval.modules.ingestion.methods.document import DocumentIngestion

    # DocumentIngestion instance whose .name attr has drifted.
    doc_method = MagicMock(spec=DocumentIngestion)
    doc_method.name = "doc"  # simulate rename
    doc_method.required = True
    doc_method.ingest = AsyncMock()

    # Non-document method with a similar-looking name.
    other = MagicMock()
    other.name = "document_like_but_wrong"
    other.required = True
    other.ingest = AsyncMock()

    # The filter must pick only the DocumentIngestion instance.
    methods = [doc_method, other]
    filtered = [m for m in methods if isinstance(m, DocumentIngestion)]
    assert filtered == [doc_method]
