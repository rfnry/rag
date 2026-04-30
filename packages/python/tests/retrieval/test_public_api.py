"""Public API surface tests — ensure lower-level building blocks are importable
at the package root so users can compose custom pipelines without touching
RagEngine."""

import rfnry_rag


def test_top_level_does_not_shadow_builtin_BaseException() -> None:
    """Regression: rfnry_rag must not export a symbol named 'BaseException'
    because it shadows the Python builtin for any user doing
    `from rfnry_rag import *` or `from rfnry_rag import BaseException`."""
    assert "BaseException" not in vars(rfnry_rag), (
        "rfnry_rag exports a symbol named 'BaseException' — this shadows "
        "the Python builtin and silently narrows user except clauses."
    )


def test_retrieval_service_exported_at_package_root() -> None:
    assert "RetrievalService" in rfnry_rag.__all__
    assert rfnry_rag.RetrievalService is not None


def test_ingestion_service_exported_at_package_root() -> None:
    assert "IngestionService" in rfnry_rag.__all__
    assert rfnry_rag.IngestionService is not None


def test_semantic_chunker_exported_at_package_root() -> None:
    assert "SemanticChunker" in rfnry_rag.__all__
    assert rfnry_rag.SemanticChunker is not None


def test_base_retrieval_method_exported() -> None:
    assert "BaseRetrievalMethod" in rfnry_rag.__all__
    assert rfnry_rag.BaseRetrievalMethod is not None


def test_base_ingestion_method_exported() -> None:
    assert "BaseIngestionMethod" in rfnry_rag.__all__
    assert rfnry_rag.BaseIngestionMethod is not None


def test_top_level_exports_method_classes() -> None:
    from rfnry_rag import (
        AnalyzedIngestion,
        DocumentIngestion,
        DrawingIngestion,
        GraphIngestion,
        VectorIngestion,
    )

    assert AnalyzedIngestion.__name__ == "AnalyzedIngestion"
    assert DocumentIngestion.__name__ == "DocumentIngestion"
    assert DrawingIngestion.__name__ == "DrawingIngestion"
    assert GraphIngestion.__name__ == "GraphIngestion"
    assert VectorIngestion.__name__ == "VectorIngestion"
    for name in (
        "AnalyzedIngestion",
        "DocumentIngestion",
        "DrawingIngestion",
        "GraphIngestion",
        "VectorIngestion",
    ):
        assert name in rfnry_rag.__all__
