"""Public API surface tests — ensure lower-level building blocks are importable
at the package root so users can compose custom pipelines without touching
RagEngine."""

import rfnry_rag
import rfnry_rag.retrieval as pkg


def test_top_level_does_not_shadow_builtin_BaseException() -> None:
    """Regression: rfnry_rag must not export a symbol named 'BaseException'
    because it shadows the Python builtin for any user doing
    `from rfnry_rag import *` or `from rfnry_rag import BaseException`."""
    assert "BaseException" not in vars(rfnry_rag), (
        "rfnry_rag exports a symbol named 'BaseException' — this shadows "
        "the Python builtin and silently narrows user except clauses."
    )


def test_retrieval_service_exported_at_package_root() -> None:
    assert "RetrievalService" in pkg.__all__
    assert pkg.RetrievalService is not None


def test_ingestion_service_exported_at_package_root() -> None:
    assert "IngestionService" in pkg.__all__
    assert pkg.IngestionService is not None


def test_semantic_chunker_exported_at_package_root() -> None:
    assert "SemanticChunker" in pkg.__all__
    assert pkg.SemanticChunker is not None


def test_base_retrieval_method_exported() -> None:
    assert "BaseRetrievalMethod" in pkg.__all__
    assert pkg.BaseRetrievalMethod is not None


def test_base_ingestion_method_exported() -> None:
    assert "BaseIngestionMethod" in pkg.__all__
    assert pkg.BaseIngestionMethod is not None
