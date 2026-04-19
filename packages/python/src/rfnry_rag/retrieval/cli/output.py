from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfnry_rag.retrieval.common.models import Chunk, RetrievedChunk, Source, SourceStats
    from rfnry_rag.retrieval.modules.generation.models import QueryResult


class OutputMode(Enum):
    JSON = "json"
    PRETTY = "pretty"


def get_output_mode(explicit: str | None) -> OutputMode:
    """Determine output mode: explicit flag > TTY detection."""
    if explicit == "json":
        return OutputMode.JSON
    if explicit == "pretty":
        return OutputMode.PRETTY
    return OutputMode.PRETTY if sys.stdout.isatty() else OutputMode.JSON


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def print_json(data: Any) -> None:
    """Print data as JSON. Handles dataclasses, dicts, and lists."""
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    print(json.dumps(data, default=_json_default, indent=2))


def print_source(source: Source) -> None:
    """Human-readable source."""
    name = source.metadata.get("name") or source.source_id
    print(f"  {name}")
    print(f"    ID: {source.source_id}")
    print(f"    Chunks: {source.chunk_count}")
    if source.knowledge_id:
        print(f"    Knowledge: {source.knowledge_id}")
    if source.source_type:
        print(f"    Type: {source.source_type}")
    if source.embedding_model:
        print(f"    Model: {source.embedding_model}")
    if source.stale:
        print("    STALE — needs re-ingestion")
    print()


def print_source_list(sources: list[Source]) -> None:
    """Human-readable source list."""
    if not sources:
        print("No sources found.")
        return
    print(f"Sources ({len(sources)}):\n")
    for s in sources:
        print_source(s)


def _truncate(text: str, limit: int = 120) -> str:
    flat = text[:limit].replace("\n", " ")
    return f"{flat}..." if len(text) > limit else flat


def print_chunks(chunks: list[Chunk]) -> None:
    """Human-readable chunk list."""
    if not chunks:
        print("No chunks found.")
        return
    print(f"Chunks ({len(chunks)}):\n")
    for c in chunks:
        page = f" (p.{c.page_number})" if c.page_number else ""
        print(f"  [{c.chunk_index}]{page} {_truncate(c.content)}")
    print()


def print_retrieved_chunks(chunks: list[RetrievedChunk]) -> None:
    """Human-readable retrieved chunk list."""
    if not chunks:
        print("No results found.")
        return
    print(f"Results ({len(chunks)}):\n")
    for c in chunks:
        page = f" (p.{c.page_number})" if c.page_number else ""
        print(f"  [{c.score:.2f}]{page} {_truncate(c.content)}")
        print(f"    source: {c.source_id}")
    print()


def print_query_result(result: QueryResult) -> None:
    """Human-readable query result."""
    if result.answer:
        print(f"\n{result.answer}\n")
    if result.sources:
        print("Sources:")
        for s in result.sources:
            page = f" (p.{s.page_number})" if s.page_number else ""
            name = s.name or s.source_id
            print(f"  {name}{page} — {s.score:.2f}")
        print()
    if result.grounded:
        print(f"Grounded ({result.confidence:.0%} confidence)")
    if result.clarification:
        print(f"\nClarification needed: {result.clarification.question}")
        if result.clarification.options:
            for opt in result.clarification.options:
                print(f"  - {opt}")


def print_stats(stats: SourceStats) -> None:
    """Human-readable source stats."""
    print(f"  Source: {stats.source_id}")
    print(f"  Chunks: {stats.total_chunks}")
    print(f"  Pages: {stats.total_pages}")
    print(f"  Avg chunk size: {stats.avg_chunk_size} chars")
    print(f"  Processing time: {stats.processing_time:.1f}s")
    print(f"  Hits: {stats.total_hits} (grounded: {stats.grounded_hits}, ungrounded: {stats.ungrounded_hits})")


def print_error(message: str, mode: OutputMode) -> None:
    """Print error in the appropriate format."""
    if mode == OutputMode.JSON:
        print(json.dumps({"error": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)


def print_success(message: str, data: Any, mode: OutputMode) -> None:
    """Print success: JSON data or human message."""
    if mode == OutputMode.JSON:
        print_json(data)
    else:
        print(message)
