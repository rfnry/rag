from rfnry_rag.retrieval.modules.ingestion.models import ChunkedContent


def build_context(
    source_name: str,
    source_type: str | None,
    page_number: int | None,
    section: str | None,
) -> str:
    parts = []
    if source_name:
        parts.append(f"Document: {source_name}")
    if source_type:
        parts.append(f"Type: {source_type}")
    if page_number is not None:
        parts.append(f"Page: {page_number}")
    if section:
        parts.append(f"Section: {section}")
    return " | ".join(parts)


def contextualize_chunks(
    chunks: list[ChunkedContent],
    source_name: str,
    source_type: str | None,
) -> list[ChunkedContent]:
    result = []
    for chunk in chunks:
        ctx = build_context(
            source_name=source_name,
            source_type=source_type,
            page_number=chunk.page_number,
            section=chunk.section,
        )
        contextualized = f"{ctx}\n\n{chunk.content}" if ctx else chunk.content
        result.append(
            ChunkedContent(
                content=chunk.content,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                context=ctx,
                contextualized=contextualized,
                parent_id=chunk.parent_id,
                chunk_type=chunk.chunk_type,
            )
        )
    return result
