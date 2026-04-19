"""Modular pipeline — direct access to individual retrieval and ingestion methods.

Shows how to:
- Access retrieval methods individually via rag.retrieval namespace
- Run fallback search strategies
- Check which methods are configured
- Iterate all configured methods
"""

import asyncio

from rfnry_rag.retrieval import (
    Embeddings,
    GenerationConfig,
    IngestionConfig,
    LanguageModelClient,
    LanguageModelProvider,
    PersistenceConfig,
    PostgresDocumentStore,
    QdrantVectorStore,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
)

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
        document_store=PostgresDocumentStore(url="postgresql://user:pass@localhost:5432/rag"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(
            LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
        ),
    ),
    retrieval=RetrievalConfig(top_k=10),
    generation=GenerationConfig(
        lm_client=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic", model="claude-sonnet-4-20250514", api_key="your_api_key"
            ),
        ),
    ),
)


async def main():
    async with RagEngine(config) as rag:
        # Ingest a document
        source = await rag.ingest_text(
            content=(
                "The XR-500 compressor requires SAE 30 non-detergent oil, changed every "
                "500 hours. Maximum discharge pressure is 175 PSI at 3450 RPM. "
                "Part number for the reed valve kit is RV-2201."
            ),
            knowledge_id="equipment",
            metadata={"name": "XR-500 Specs"},
        )
        print(f"Ingested: {source.source_id} ({source.chunk_count} chunks)")

        # --- Check what retrieval methods are configured ---
        print("\nConfigured retrieval methods:")
        for method in rag.retrieval:
            print(f"  - {method.name} (weight={method.weight})")

        print(f"\nHas vector search: {'vector' in rag.retrieval}")
        print(f"Has document search: {'document' in rag.retrieval}")
        print(f"Has graph search: {'graph' in rag.retrieval}")

        # --- Use individual methods directly ---
        print("\n--- Vector search ---")
        vector_chunks = await rag.retrieval.vector.search("oil specifications", top_k=5)
        for chunk in vector_chunks:
            print(f"  [{chunk.score:.2f}] {chunk.content[:80]}...")

        print("\n--- Document search ---")
        doc_chunks = await rag.retrieval.document.search("RV-2201", top_k=5, knowledge_id="equipment")
        for chunk in doc_chunks:
            print(f"  [{chunk.score:.2f}] {chunk.content[:80]}...")

        # --- Fallback strategy ---
        print("\n--- Fallback: vector first, document if insufficient ---")
        chunks = await rag.retrieval.vector.search("reed valve", top_k=10)
        if len(chunks) < 3 and "document" in rag.retrieval:
            extra = await rag.retrieval.document.search("reed valve", top_k=5, knowledge_id="equipment")
            chunks.extend(extra)
            print(f"  Added {len(extra)} document results as fallback")
        print(f"  Total: {len(chunks)} chunks")

        # --- Standard pipeline still works ---
        print("\n--- Standard query (all methods + generation) ---")
        result = await rag.query("What oil does the XR-500 need?", knowledge_id="equipment")
        print(f"Answer: {result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
