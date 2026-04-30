from rfnry_rag.logging import get_logger
from rfnry_rag.stores.metadata.base import BaseMetadataStore

logger = get_logger("knowledge/migration")


async def check_embedding_migration(
    metadata_store: BaseMetadataStore | None,
    embedding_model_name: str,
) -> int:
    """Check if the configured embedding model differs from stored sources.

    Flags affected sources as stale. Returns count of stale sources.
    Only works with metadata store. Returns 0 without one.
    """
    if metadata_store is None:
        return 0
    # No embedding model configured (retrieval-only / document-only setups)
    # means no staleness check applies — otherwise every prior-ingest source
    # would be marked stale against embedding_model=="", flooding logs.
    if not embedding_model_name:
        return 0

    sources = await metadata_store.list_sources()
    stale_count = 0

    # SERIAL: update_source calls are independent in principle, but the
    # metadata store (SQLite default) serialises writes via a single WAL
    # writer; gathering here would not improve throughput and adds complexity.
    # This path runs once at startup against a typically small source count.
    for source in sources:
        if source.embedding_model != embedding_model_name and not source.stale:
            await metadata_store.update_source(source.source_id, stale=True)
            stale_count += 1
            logger.warning(
                "source '%s' marked stale: embedded with '%s', current is '%s'",
                source.metadata.get("name", source.source_id),
                source.embedding_model,
                embedding_model_name,
            )

    if stale_count:
        logger.warning(
            "%d sources need re-ingestion due to embedding model change (%s)",
            stale_count,
            embedding_model_name,
        )

    return stale_count
