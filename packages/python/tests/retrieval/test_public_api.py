"""Public API surface tests — ensure lower-level building blocks are importable
at the package root so users can compose custom pipelines without touching
KnowledgeEngine."""

import rfnry_knowledge


def test_top_level_does_not_shadow_builtin_BaseException() -> None:
    """Regression: rfnry_knowledge must not export a symbol named 'BaseException'
    because it shadows the Python builtin for any user doing
    `from rfnry_knowledge import *` or `from rfnry_knowledge import BaseException`."""
    assert "BaseException" not in vars(rfnry_knowledge), (
        "rfnry_knowledge exports a symbol named 'BaseException' — this shadows "
        "the Python builtin and silently narrows user except clauses."
    )


def test_retrieval_service_exported_at_package_root() -> None:
    assert "RetrievalService" in rfnry_knowledge.__all__
    assert rfnry_knowledge.RetrievalService is not None


def test_ingestion_service_exported_at_package_root() -> None:
    assert "IngestionService" in rfnry_knowledge.__all__
    assert rfnry_knowledge.IngestionService is not None


def test_semantic_chunker_exported_at_package_root() -> None:
    assert "SemanticChunker" in rfnry_knowledge.__all__
    assert rfnry_knowledge.SemanticChunker is not None


def test_base_retrieval_method_exported() -> None:
    assert "BaseRetrievalMethod" in rfnry_knowledge.__all__
    assert rfnry_knowledge.BaseRetrievalMethod is not None


def test_base_ingestion_method_exported() -> None:
    assert "BaseIngestionMethod" in rfnry_knowledge.__all__
    assert rfnry_knowledge.BaseIngestionMethod is not None


def test_top_level_exports_method_classes() -> None:
    from rfnry_knowledge import (
        DrawingIngestion,
        EntityIngestion,
        KeywordIngestion,
        SemanticIngestion,
        StructuredIngestion,
    )

    assert StructuredIngestion.__name__ == "StructuredIngestion"
    assert KeywordIngestion.__name__ == "KeywordIngestion"
    assert DrawingIngestion.__name__ == "DrawingIngestion"
    assert EntityIngestion.__name__ == "EntityIngestion"
    assert SemanticIngestion.__name__ == "SemanticIngestion"
    for name in (
        "StructuredIngestion",
        "KeywordIngestion",
        "DrawingIngestion",
        "EntityIngestion",
        "SemanticIngestion",
    ):
        assert name in rfnry_knowledge.__all__
