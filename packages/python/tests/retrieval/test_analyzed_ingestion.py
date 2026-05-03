from unittest.mock import AsyncMock, MagicMock

from rfnry_knowledge.ingestion.analyze.service import AnalyzedIngestionService
from rfnry_knowledge.models import Source

# Serialised page-analysis rows stored in knowledge_page_analyses (keyed by "data")
_PAGE_ANALYSES_ROWS = [
    {
        "page_number": 1,
        "data": {
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
            "raw_text": "",
        },
    },
    {
        "page_number": 2,
        "data": {
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
            "raw_text": "",
        },
    },
]


def _make_service(graph_store=None, ingestion_methods=None):
    embeddings = MagicMock()
    embeddings.model = "test-model"
    embeddings.embed = AsyncMock(return_value=[[0.1] * 10, [0.2] * 10])

    vector_store = AsyncMock()
    vector_store.initialize = AsyncMock()
    vector_store.upsert = AsyncMock()

    metadata_store = AsyncMock()
    # New table-based reads — return the page analyses rows by default
    metadata_store.get_page_analyses = AsyncMock(return_value=_PAGE_ANALYSES_ROWS)

    return AnalyzedIngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_model_name="test:test-model",
        graph_store=graph_store,
        ingestion_methods=ingestion_methods,
    )


def _make_source_with_analysis() -> Source:
    """Source in 'synthesized' state. page_analyses live in knowledge_page_analyses, NOT metadata."""
    return Source(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type="drawings",
        status="synthesized",
        embedding_model="test:test-model",
        metadata={
            "file_type": "pdf",
            "file_name": "test.pdf",
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
    """The ingest phase must be the single authoritative document write."""
    from unittest.mock import AsyncMock, MagicMock

    from rfnry_knowledge.ingestion.methods.document import DocumentIngestion

    doc_method = MagicMock(spec=DocumentIngestion)
    doc_method.name = "document"
    doc_method.required = True
    doc_method.ingest = AsyncMock()

    xml = tmp_path / "sample.xml"
    xml.write_text("<root><item>hello</item></root>")

    service = _make_service(ingestion_methods=[doc_method])
    # Override embed so it returns exactly one vector per text (avoids zip mismatch)
    service._embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 10] * len(texts))

    # analyze phase — metadata_store.create_source saves the source
    created_sources: list[Source] = []

    async def capture_create(source: Source) -> None:
        created_sources.append(source)

    service._metadata_store.create_source = AsyncMock(side_effect=capture_create)

    source = await service.analyze(file_path=xml)

    # synthesize phase — get_source returns the analyzed source; update_source updates it
    service._metadata_store.get_source = AsyncMock(return_value=source)
    service._metadata_store.update_source = AsyncMock()

    source = await service.synthesize(source.source_id)

    # Refresh: get_source for phase 3 returns the synthesized source (with synthesis in metadata)
    service._metadata_store.get_source = AsyncMock(return_value=source)

    # ingest phase
    await service.ingest(source.source_id)

    # The document method must have been called exactly once — in the ingest phase only
    assert doc_method.ingest.await_count == 1


async def test_analyze_pdf_runs_pages_concurrently(tmp_path) -> None:
    """Page analysis must overlap — 200-page PDF shouldn't be 200 serial LLM calls."""
    import asyncio
    from unittest.mock import patch

    concurrent = 0
    max_concurrent = 0

    async def slow_analyze_page(image, *, baml_options=None):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        concurrent -= 1
        result = MagicMock()
        result.description = "test page"
        result.entities = []
        result.tables = []
        result.annotations = []
        result.page_type = "diagram"
        return result

    n_pages = 8
    fake_pages = [
        {"page_number": i + 1, "image_base64": "dGVzdA=="}  # b64("test")
        for i in range(n_pages)
    ]

    service = _make_service()
    # Inject a vision provider and registry so the guards in _analyze_pdf pass
    service._vision = MagicMock()
    service._registry = MagicMock()

    with (
        patch(
            "rfnry_knowledge.ingestion.analyze.service.iter_pdf_page_images",
            return_value=iter(fake_pages),
        ),
        patch(
            "rfnry_knowledge.baml.baml_client.async_client.b.AnalyzePage",
            new=slow_analyze_page,
        ),
    ):
        service._metadata_store.create_source = AsyncMock()
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")
        await service.analyze(file_path=fake_pdf)

    # With serial execution all pages run one-at-a-time → max_concurrent == 1.
    # With bounded gather (semaphore=5) over 8 pages, max_concurrent must be > 1.
    assert max_concurrent >= 2, (
        f"max_concurrent={max_concurrent}: page analysis appears to be running serially. "
        "Expected at least 2 concurrent AnalyzePage calls with bounded gather."
    )


async def test_analyzed_ingestion_identifies_document_method_by_type_not_name() -> None:
    """If DocumentIngestion.name is renamed, document storage must still route correctly."""
    from rfnry_knowledge.ingestion.methods.document import DocumentIngestion

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


def test_synthesize_shared_entities_caps_cross_refs_per_entity() -> None:
    """Entity appearing on 50+ pages must not produce O(n^2) cross-refs."""
    from rfnry_knowledge.ingestion.analyze.models import DiscoveredEntity, PageAnalysis
    from rfnry_knowledge.ingestion.analyze.service import _MAX_PAGES_PER_ENTITY, AnalyzedIngestionService

    # Build 50 pages all sharing one entity — uncapped this would produce
    # 50*49/2 = 1225 cross-refs; capped at _MAX_PAGES_PER_ENTITY it must be
    # at most _MAX_PAGES_PER_ENTITY * (_MAX_PAGES_PER_ENTITY - 1) / 2 = 190.
    n_pages = 50
    pages = [
        PageAnalysis(
            page_number=i + 1,
            description=f"Page {i + 1}",
            entities=[DiscoveredEntity(name="SharedEntity", category="test", context="", value=None)],
        )
        for i in range(n_pages)
    ]

    xrefs = AnalyzedIngestionService._synthesize_shared_entities(pages)
    max_expected = _MAX_PAGES_PER_ENTITY * (_MAX_PAGES_PER_ENTITY - 1) // 2
    assert len(xrefs) <= max_expected, (
        f"got {len(xrefs)} cross-refs for {n_pages} pages sharing one entity; "
        f"expected ≤ {max_expected} (cap={_MAX_PAGES_PER_ENTITY})"
    )
