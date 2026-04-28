"""R1.1 — token counting at ingest + corpus-token / corpus-text loaders.

Plumbing for R1.2/R1.3/R1.4 routing modes. No user-facing behavior change in
this slice — `mode="retrieval"` (the default) does not exercise any of this.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from rfnry_rag.retrieval.common.models import Source, VectorResult
from rfnry_rag.retrieval.modules.ingestion.chunk.service import IngestionService
from rfnry_rag.retrieval.modules.ingestion.chunk.token_counter import count_tokens
from rfnry_rag.retrieval.modules.knowledge.manager import KnowledgeManager


def _mock_method(name: str, required: bool = True) -> SimpleNamespace:
    return SimpleNamespace(name=name, required=required, ingest=AsyncMock(), delete=AsyncMock())


def _make_chunker():
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
    return chunker


def test_source_estimated_tokens_field_defaults_to_none() -> None:
    source = Source(source_id="source_a")
    assert source.estimated_tokens is None


async def test_ingest_file_populates_estimated_tokens(tmp_path) -> None:
    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()

    service = IngestionService(
        chunker=_make_chunker(),
        ingestion_methods=[_mock_method("vector")],
        embedding_model_name="test:model",
        metadata_store=metadata_store,
    )

    file_path = tmp_path / "doc_a.txt"
    file_text = "Page one content.\n\nPage two content."
    file_path.write_text(file_text)

    source = await service.ingest(file_path=file_path, knowledge_id="kb-1")

    # TextParser splits on blank lines; sum across the parsed pages must match
    # the per-page count_tokens contribution.
    expected = count_tokens("Page one content.") + count_tokens("Page two content.")
    assert source.estimated_tokens == expected
    # Persisted on the metadata blob, not a dedicated column.
    assert source.metadata["estimated_tokens"] == expected


async def test_ingest_text_populates_estimated_tokens() -> None:
    metadata_store = AsyncMock()
    metadata_store.list_sources = AsyncMock(return_value=[])
    metadata_store.find_by_hash = AsyncMock(return_value=None)
    metadata_store.create_source = AsyncMock()

    service = IngestionService(
        chunker=_make_chunker(),
        ingestion_methods=[],
        embedding_model_name="test:model",
        metadata_store=metadata_store,
    )

    content = "hello world from chunk_0"
    source = await service.ingest_text(content=content, knowledge_id="kb-1")

    assert source.estimated_tokens == count_tokens(content)


async def test_knowledge_manager_get_corpus_tokens_sums_across_sources() -> None:
    sources = [
        Source(source_id="source_a", knowledge_id="kb-1", metadata={"estimated_tokens": 10}),
        Source(source_id="source_b", knowledge_id="kb-1", metadata={"estimated_tokens": 25}),
        Source(source_id="source_c", knowledge_id="kb-1", metadata={"estimated_tokens": 7}),
    ]
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=sources),
        update_source=AsyncMock(),
    )
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]

    total = await km.get_corpus_tokens(knowledge_id="kb-1")

    assert total == 42
    # No legacy lazy-compute writes when every source already has a count.
    metadata_store.update_source.assert_not_called()


async def test_get_corpus_tokens_lazy_computes_for_legacy_sources() -> None:
    legacy = Source(source_id="source_a", knowledge_id="kb-1", metadata={})
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=[legacy]),
        update_source=AsyncMock(),
    )
    legacy_text = "legacy corpus body for chunk_0"
    document_store = SimpleNamespace(get=AsyncMock(return_value=legacy_text))

    km = KnowledgeManager(
        metadata_store=metadata_store,  # type: ignore[arg-type]
        document_store=document_store,  # type: ignore[arg-type]
    )

    total = await km.get_corpus_tokens(knowledge_id="kb-1")

    expected = count_tokens(legacy_text)
    assert total == expected
    # Cached back via metadata-blob update so the next call short-circuits.
    metadata_store.update_source.assert_awaited_once()
    args, kwargs = metadata_store.update_source.call_args
    assert args == ("source_a",)
    assert kwargs["metadata"]["estimated_tokens"] == expected


async def test_load_full_corpus_prefers_document_store_over_vector_scroll() -> None:
    from rfnry_rag.retrieval.server import RagEngine

    sources = [Source(source_id="source_a", knowledge_id="kb-1", metadata={"name": "doc_a"})]
    metadata_store = SimpleNamespace(list_sources=AsyncMock(return_value=sources))
    document_store = SimpleNamespace(get=AsyncMock(return_value="doc-text body"))
    vector_store = SimpleNamespace(
        scroll=AsyncMock(
            return_value=(
                [VectorResult(point_id="chunk_0", score=1.0, payload={"content": "scrolled"})],
                None,
            )
        )
    )

    engine = RagEngine.__new__(RagEngine)
    engine._config = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(
            metadata_store=metadata_store,
            document_store=document_store,
            vector_store=vector_store,
        )
    )

    corpus = await engine._load_full_corpus(knowledge_id="kb-1")

    assert "doc-text body" in corpus
    assert "[Source: doc_a]" in corpus
    document_store.get.assert_awaited_once_with("source_a")
    vector_store.scroll.assert_not_called()


async def test_get_corpus_tokens_returns_zero_for_empty_knowledge_scope() -> None:
    metadata_store = SimpleNamespace(
        list_sources=AsyncMock(return_value=[]),
        update_source=AsyncMock(),
    )
    km = KnowledgeManager(metadata_store=metadata_store)  # type: ignore[arg-type]

    total = await km.get_corpus_tokens(knowledge_id="kb-empty")

    assert total == 0
    metadata_store.update_source.assert_not_called()


async def test_load_full_corpus_returns_empty_string_for_empty_knowledge_scope() -> None:
    from rfnry_rag.retrieval.server import RagEngine

    metadata_store = SimpleNamespace(list_sources=AsyncMock(return_value=[]))
    document_store = SimpleNamespace(get=AsyncMock())
    vector_store = SimpleNamespace(scroll=AsyncMock())

    engine = RagEngine.__new__(RagEngine)
    engine._config = SimpleNamespace(  # type: ignore[assignment]
        persistence=SimpleNamespace(
            metadata_store=metadata_store,
            document_store=document_store,
            vector_store=vector_store,
        )
    )

    corpus = await engine._load_full_corpus(knowledge_id="kb-empty")

    assert corpus == ""
    document_store.get.assert_not_called()
    vector_store.scroll.assert_not_called()
