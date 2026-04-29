# src/rfnry-rag/retrieval/tests/test_document_retrieval.py
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rfnry_rag.retrieval.common.models import ContentMatch, RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.methods.document import DocumentRetrieval
from rfnry_rag.retrieval.modules.retrieval.search.service import RetrievalService


async def test_search_converts_matches():
    store = AsyncMock()
    store.search_content = AsyncMock(
        return_value=[
            ContentMatch(
                source_id="src-1",
                title="Manual",
                excerpt="Excerpt text",
                score=0.85,
                match_type="fulltext",
                source_type="manuals",
            ),
        ]
    )
    method = DocumentRetrieval(document_store=store, weight=0.8)
    assert method.name == "document"
    assert method.weight == 0.8

    results = await method.search(query="test", top_k=10, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "fulltext:src-1"
    assert results[0].content == "Excerpt text"
    assert results[0].source_metadata["match_type"] == "fulltext"
    store.search_content.assert_called_once_with(query="test", knowledge_id="kb-1", top_k=10)


async def test_search_empty_results():
    store = AsyncMock()
    store.search_content = AsyncMock(return_value=[])

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.search_content = AsyncMock(side_effect=RuntimeError("db down"))

    method = DocumentRetrieval(document_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []


# --- Integration tests (RetrievalService with document method) ---


def _make_service(document_method=None):
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Some chunk content", score=0.8),
            ]
        ),
    )
    methods = [mock_vector]
    if document_method is not None:
        methods.append(document_method)
    return RetrievalService(
        retrieval_methods=methods,
        reranking=None,
        top_k=5,
    )


async def test_retrieve_with_document_store():
    mock_document = SimpleNamespace(
        name="document",
        weight=1.0,
        top_k=None,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="fulltext:src-2",
                    source_id="src-2",
                    content="The FBD-20254 filter specs...",
                    score=0.9,
                    source_type="manuals",
                    source_metadata={"title": "Manual X", "match_type": "exact"},
                ),
            ]
        ),
    )
    service = _make_service(document_method=mock_document)
    results, _ = await service.retrieve(query="FBD-20254", knowledge_id="kb-1")
    assert len(results) == 2
    mock_document.search.assert_called_once()


async def test_retrieve_without_document_store():
    service = _make_service(document_method=None)
    results, _ = await service.retrieve(query="test query", knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
