from rfnry_knowledge.exceptions import (
    IngestionError,
    KnowledgeEngineError,
    MemoryEngineError,
    MemoryExtractionError,
    MemoryNotFoundError,
    StoreError,
)


def test_memory_engine_error_is_engine_error() -> None:
    assert issubclass(MemoryEngineError, KnowledgeEngineError)


def test_memory_not_found_is_store_error_and_memory_error() -> None:
    assert issubclass(MemoryNotFoundError, MemoryEngineError)
    assert issubclass(MemoryNotFoundError, StoreError)


def test_memory_extraction_is_ingestion_error_and_memory_error() -> None:
    assert issubclass(MemoryExtractionError, MemoryEngineError)
    assert issubclass(MemoryExtractionError, IngestionError)


def test_memory_not_found_carries_id() -> None:
    exc = MemoryNotFoundError("missing", memory_row_id="abc")
    assert exc.memory_row_id == "abc"
    assert "missing" in str(exc)
