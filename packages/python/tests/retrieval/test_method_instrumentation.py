"""Per-method retrieval/ingestion timings: methods record their durations on
the active row and emit retrieval.method.* / ingestion.method.* events.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _recording import RecordingObservabilitySink as RecordingSink

from rfnry_knowledge.ingestion.methods.keyword import KeywordIngestion
from rfnry_knowledge.ingestion.methods.semantic import SemanticIngestion
from rfnry_knowledge.models import VectorResult
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.observability.context import _reset_obs, _set_obs
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval
from rfnry_knowledge.telemetry import IngestTelemetryRow, QueryTelemetryRow
from rfnry_knowledge.telemetry.context import _reset_row, _set_row


def _query_row() -> QueryTelemetryRow:
    return QueryTelemetryRow(query_id="q-1", mode="retrieval", routing_decision="retrieval", outcome="success")


def _ingest_row() -> IngestTelemetryRow:
    return IngestTelemetryRow(source_id="s-1", ingest_id="i-1", outcome="success")


@pytest.mark.asyncio
async def test_vector_retrieval_records_method_duration_and_event() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search = AsyncMock(
        return_value=[VectorResult(point_id="p1", score=0.9, payload={"content": "x", "source_id": "s"})]
    )
    fake_embeddings = AsyncMock()
    fake_embeddings.embed = AsyncMock(return_value=[[0.0]])

    retriever = SemanticRetrieval(store=fake_store, embeddings=fake_embeddings)
    try:
        results = await retriever.search("q", top_k=5)
        assert len(results) == 1
        assert "semantic" in row.methods_used
        assert "semantic" in row.method_durations_ms
        assert row.method_durations_ms["semantic"] >= 0
        assert row.chunks_retrieved == 1
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.success" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_vector_retrieval_failure_increments_method_errors() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search = AsyncMock(side_effect=RuntimeError("store down"))
    fake_embeddings = AsyncMock()
    fake_embeddings.embed = AsyncMock(return_value=[[0.0]])

    retriever = SemanticRetrieval(store=fake_store, embeddings=fake_embeddings)
    try:
        results = await retriever.search("q", top_k=5)
        assert results == []  # error-isolated
        assert row.method_errors == 1
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.error" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_document_retrieval_records_method_duration_and_event() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _query_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.search_content = AsyncMock(return_value=[])

    retriever = KeywordRetrieval(backend="postgres_fts", document_store=fake_store)
    try:
        await retriever.search("q")
        assert "keyword" in row.method_durations_ms
        kinds = [r.kind for r in obs_sink.records]
        assert "retrieval.method.success" in kinds
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_document_ingestion_emits_method_success() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _ingest_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.store_content = AsyncMock(return_value=None)

    ing = KeywordIngestion(store=fake_store)
    try:
        await ing.ingest(
            source_id="s-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="t",
            full_text="body",
            chunks=[],
            tags=[],
            metadata={},
        )
        kinds = [r.kind for r in obs_sink.records]
        assert "ingestion.method.success" in kinds
        assert row.persist_ms >= 0
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_ingest_row_counts_contextual_chunk_calls() -> None:
    from unittest.mock import patch

    from pydantic import SecretStr

    from rfnry_knowledge.config import ContextualChunkConfig
    from rfnry_knowledge.ingestion.chunk.contextualize import contextualize_chunks_with_llm
    from rfnry_knowledge.ingestion.models import ChunkedContent
    from rfnry_knowledge.providers import ProviderClient

    class _Counter:
        name = "test"
        model = "test"

        def count(self, text: str) -> int:
            return len(text.split())

    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _ingest_row()
    row_token = _set_row(row)

    chunks = [ChunkedContent(content=f"passage_{i}", chunk_index=i, contextualized=f"passage_{i}") for i in range(3)]
    cfg = ContextualChunkConfig(
        enabled=True,
        provider_client=ProviderClient(name="anthropic", model="m", api_key=SecretStr("k")),
        token_counter=_Counter(),
    )

    try:
        with patch(
            "rfnry_knowledge.ingestion.chunk.contextualize.instrument_baml_call",
            new=AsyncMock(return_value="ctx"),
        ):
            await contextualize_chunks_with_llm(chunks, document_text="doc", config=cfg)
        assert row.contextual_chunk_calls == 3
        assert row.contextual_chunk_skipped is False
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_ingest_row_marks_contextual_chunk_skipped_on_oversized() -> None:

    from pydantic import SecretStr

    from rfnry_knowledge.config import ContextualChunkConfig, IngestionConfig
    from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker
    from rfnry_knowledge.ingestion.chunk.service import IngestionService
    from rfnry_knowledge.providers import ProviderClient

    class _Counter:
        name = "test"
        model = "test"

        def count(self, text: str) -> int:
            return len(text.split())

    counter = _Counter()
    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _ingest_row()
    row_token = _set_row(row)

    cfg_ingest = IngestionConfig(token_counter=counter)
    chunker = SemanticChunker(
        chunk_size=cfg_ingest.chunk_size,
        chunk_overlap=cfg_ingest.chunk_overlap,
        token_counter=counter,
    )

    contextual_cfg = ContextualChunkConfig(
        enabled=True,
        provider_client=ProviderClient(name="anthropic", model="m", api_key=SecretStr("k"), context_size=32_000),
        token_counter=counter,
        max_context_tokens=100,
    )

    svc = IngestionService(
        chunker=chunker,
        ingestion_methods=[],
        contextual_chunk=contextual_cfg,
        chunk_context_headers=False,
        token_counter=counter,
    )

    huge_doc = "word " * 20_000

    try:
        await svc.ingest_text(huge_doc, knowledge_id="k")
        assert row.contextual_chunk_skipped is True
        assert row.contextual_chunk_calls == 0
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_ingest_row_counts_document_expansion_calls_and_failures() -> None:
    from unittest.mock import patch

    from rfnry_knowledge.config import DocumentExpansionConfig
    from rfnry_knowledge.ingestion.chunk.expand import expand_chunks
    from rfnry_knowledge.ingestion.models import ChunkedContent

    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _ingest_row()
    row_token = _set_row(row)

    from pydantic import SecretStr

    from rfnry_knowledge.providers import ProviderClient

    fake_registry = object()
    cfg = DocumentExpansionConfig(
        enabled=True,
        num_queries=3,
        provider_client=ProviderClient(name="anthropic", model="m", api_key=SecretStr("k")),
    )

    chunks = [ChunkedContent(content=f"p{i}", chunk_index=i) for i in range(4)]
    call_count = {"n": 0}

    async def selective_baml(*, operation, call):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("expansion broke")
        return SimpleNamespace(queries=[f"q{call_count['n']}_a", f"q{call_count['n']}_b"])

    try:
        with patch("rfnry_knowledge.ingestion.chunk.expand.instrument_baml_call", side_effect=selective_baml):
            await expand_chunks(chunks, cfg, fake_registry, notes=[])  # type: ignore[arg-type]
        assert row.document_expansion_calls == 3
        assert row.document_expansion_chunk_failures == 1
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_ingest_row_counts_vision_pages_analyzed_and_skipped(tmp_path) -> None:
    from unittest.mock import patch

    from rfnry_knowledge.ingestion.structured.service import StructuredIngestionService

    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _ingest_row()
    row_token = _set_row(row)

    metadata_store = AsyncMock()
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()
    metadata_store.upsert_page_analyses = AsyncMock()
    metadata_store.get_page_analyses_by_hash = AsyncMock(return_value={})

    class _FakeEmbeddings:
        @property
        def model(self) -> str:
            return "fake"

        async def embed(self, texts):
            return [[0.1] * 4 for _ in texts]

        async def embedding_dimension(self) -> int:
            return 4

    class _FakeVectorStore:
        async def initialize(self, dim: int) -> None:
            pass

        async def upsert(self, points) -> None:
            pass

    svc = StructuredIngestionService(
        embeddings=_FakeEmbeddings(),
        vector_store=_FakeVectorStore(),
        metadata_store=metadata_store,
        embedding_model_name="fake",
        vision=SimpleNamespace(),
        analyze_text_skip_threshold_chars=0,
    )
    svc._registry = SimpleNamespace()  # truthy

    pages = [
        {
            "page_number": i + 1,
            "image_base64": "aW1n",
            "raw_text": "short",
            "raw_text_char_count": 5,
            "has_images": True,
            "page_hash": f"h{i + 1}",
        }
        for i in range(3)
    ]

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    counts = {"n": 0}

    async def selective_baml(image, *, baml_options=None):
        counts["n"] += 1
        if counts["n"] == 2:
            raise RuntimeError("page 2 broke")
        return SimpleNamespace(
            description=f"page {counts['n']}",
            entities=[],
            tables=[],
            annotations=[],
            page_type="diagram",
        )

    try:
        with (
            patch(
                "rfnry_knowledge.ingestion.structured.service.iter_pdf_page_images",
                return_value=iter(pages),
            ),
            patch(
                "rfnry_knowledge.ingestion.structured.service.compute_file_hash",
                return_value="hashv",
            ),
            patch(
                "rfnry_knowledge.ingestion.structured.service.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=lambda fn, *args: fn(*args),
            ),
            patch("rfnry_knowledge.baml.baml_client.async_client.b") as mock_b,
        ):
            mock_b.AnalyzeStructuredPage = AsyncMock(side_effect=selective_baml)
            await svc.analyze(file_path=pdf)
        assert row.vision_pages_analyzed == 2
        assert row.vision_pages_skipped == 1
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_ingest_row_marks_graph_extraction_failed() -> None:
    from unittest.mock import patch

    from rfnry_knowledge.ingestion.methods.entity import EntityIngestion

    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _ingest_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    method = EntityIngestion(store=fake_store)
    method._registry = SimpleNamespace()  # truthy bypasses early return

    async def boom(*, operation, call):
        raise RuntimeError("graph_extraction_blew_up")

    try:
        with patch("rfnry_knowledge.ingestion.methods.entity.instrument_baml_call", side_effect=boom):
            await method.ingest(
                source_id="s",
                knowledge_id="k",
                source_type=None,
                source_weight=1.0,
                title="t",
                full_text="body",
                chunks=[],
                tags=[],
                metadata={},
                notes=[],
            )
        assert row.graph_extraction_failed is True
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_vector_ingestion_emits_method_event_and_records_duration() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _ingest_row()
    row_token = _set_row(row)

    fake_store = AsyncMock()
    fake_store.upsert = AsyncMock(return_value=None)

    fake_embeddings = SimpleNamespace(
        embed=AsyncMock(return_value=[[0.1, 0.2]]),
        embedding_dimension=AsyncMock(return_value=2),
        name="x",
        model="m",
    )

    from rfnry_knowledge.ingestion.models import ChunkedContent

    chunk = ChunkedContent(content="body", chunk_index=0)

    ing = SemanticIngestion(store=fake_store, embeddings=fake_embeddings, embedding_model_name="x")
    try:
        await ing.ingest(
            source_id="s-1",
            knowledge_id=None,
            source_type=None,
            source_weight=1.0,
            title="t",
            full_text="body",
            chunks=[chunk],
            tags=[],
            metadata={},
        )
        kinds = [r.kind for r in obs_sink.records]
        assert "ingestion.method.success" in kinds
        assert row.embed_ms >= 0
        assert row.chunks_count == 1
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)
