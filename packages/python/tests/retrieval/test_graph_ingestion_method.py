from unittest.mock import AsyncMock, MagicMock, patch

from rfnry_knowledge.ingestion.methods.graph import GraphIngestion


async def test_ingest_extracts_entities_and_stores():
    store = AsyncMock()
    store.add_entities = AsyncMock()
    lm_client = MagicMock()
    mock_entity = MagicMock()
    mock_entity.name = "Motor M1"
    mock_entity.category = "motor"
    mock_entity.value = "480V"
    mock_entity.context = "main motor"
    mock_result = MagicMock()
    mock_result.description = "Technical specifications"
    mock_result.entities = [mock_entity]
    mock_result.tables = []
    mock_result.annotations = []
    mock_result.page_type = "text"
    with (
        patch("rfnry_knowledge.ingestion.methods.graph.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()
        method = GraphIngestion(store=store, provider_client=lm_client)
        assert method.name == "graph"
        await method.ingest(
            source_id="src-1",
            knowledge_id="kb-1",
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="Motor M1 operates at 480V.",
            chunks=[],
            tags=[],
            metadata={},
        )
    store.add_entities.assert_called_once()
    call_kwargs = store.add_entities.call_args.kwargs
    assert call_kwargs["source_id"] == "src-1"
    assert call_kwargs["knowledge_id"] == "kb-1"
    assert len(call_kwargs["entities"]) == 1
    assert call_kwargs["entities"][0].name == "Motor M1"


async def test_ingest_skips_when_no_entities():
    store = AsyncMock()
    store.add_entities = AsyncMock()
    lm_client = MagicMock()
    mock_result = MagicMock()
    mock_result.description = "General text"
    mock_result.entities = []
    mock_result.tables = []
    mock_result.annotations = []
    mock_result.page_type = "text"
    with (
        patch("rfnry_knowledge.ingestion.methods.graph.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(return_value=mock_result)
        mock_registry.return_value = MagicMock()
        method = GraphIngestion(store=store, provider_client=lm_client)
        await method.ingest(
            source_id="src-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="Test",
            full_text="No entities here.",
            chunks=[],
            tags=[],
            metadata={},
        )
    store.add_entities.assert_not_called()


async def test_ingest_error_does_not_raise():
    store = AsyncMock()
    lm_client = MagicMock()
    with (
        patch("rfnry_knowledge.ingestion.methods.graph.b") as mock_b,
        patch("rfnry_knowledge.ingestion.methods.graph.build_registry") as mock_registry,
    ):
        mock_b.ExtractEntitiesFromText = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_registry.return_value = MagicMock()
        method = GraphIngestion(store=store, provider_client=lm_client)
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


async def test_ingest_skips_without_lm_client():
    store = AsyncMock()
    method = GraphIngestion(store=store, provider_client=None)
    await method.ingest(
        source_id="src-1",
        knowledge_id=None,
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="Text.",
        chunks=[],
        tags=[],
        metadata={},
    )
    store.add_entities.assert_not_called()


async def test_delete():
    store = AsyncMock()
    store.delete_by_source = AsyncMock()
    with patch("rfnry_knowledge.ingestion.methods.graph.build_registry") as mock_registry:
        mock_registry.return_value = MagicMock()
        method = GraphIngestion(store=store, provider_client=MagicMock())
    await method.delete("src-1")
    store.delete_by_source.assert_called_once_with("src-1")
