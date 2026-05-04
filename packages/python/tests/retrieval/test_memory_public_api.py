def test_top_level_imports() -> None:
    from rfnry_knowledge import (  # noqa: F401
        BaseExtractor,
        DefaultMemoryExtractor,
        ExtractedMemory,
        Interaction,
        InteractionTurn,
        MemoryEngine,
        MemoryEngineConfig,
        MemoryEngineError,
        MemoryExtractionError,
        MemoryIngestionConfig,
        MemoryNotFoundError,
        MemoryRetrievalConfig,
        MemoryRow,
        MemorySearchResult,
    )


def test_all_lists_memory_names() -> None:
    import rfnry_knowledge as pkg
    expected = {
        "BaseExtractor", "DefaultMemoryExtractor", "ExtractedMemory",
        "Interaction", "InteractionTurn", "MemoryEngine", "MemoryEngineConfig",
        "MemoryEngineError", "MemoryExtractionError", "MemoryIngestionConfig",
        "MemoryNotFoundError", "MemoryRetrievalConfig", "MemoryRow",
        "MemorySearchResult",
    }
    assert expected.issubset(set(pkg.__all__))


def test_memory_subpackage_imports() -> None:
    from rfnry_knowledge.memory import (  # noqa: F401
        BaseExtractor,
        DefaultMemoryExtractor,
        ExtractedMemory,
        Interaction,
        InteractionTurn,
        MemoryEngine,
        MemoryRow,
        MemorySearchResult,
    )
