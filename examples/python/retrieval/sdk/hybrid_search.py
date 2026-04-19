import asyncio

from rfnry_rag.retrieval import (
    Embeddings,
    FastEmbedSparseEmbeddings,
    GenerationConfig,
    HyDeRewriting,
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
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="hybrid-demo"),
        document_store=PostgresDocumentStore(url="postgresql://user:pass@localhost:5432/rag"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(
            LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
        ),
        sparse_embeddings=FastEmbedSparseEmbeddings(),
    ),
    retrieval=RetrievalConfig(
        query_rewriter=HyDeRewriting(
            lm_client=LanguageModelClient(
                provider=LanguageModelProvider(
                    provider="anthropic",
                    model="claude-haiku-4-5-20251001",
                    api_key="your_api_key",
                )
            )
        ),
        top_k=5,
    ),
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
        await rag.ingest_text(
            content=(
                "The XR-500 compressor operates at 3450 RPM with a maximum discharge pressure "
                "of 175 PSI. The unit requires SAE 30 non-detergent oil, changed every 500 hours "
                "of operation. The thermal overload protector trips at 285F."
            ),
            knowledge_id="equipment",
            metadata={"name": "XR-500 Compressor Specs"},
        )
        await rag.ingest_text(
            content=(
                "Troubleshooting: If the XR-500 compressor fails to start, check the thermal "
                "overload reset button on the motor housing. If the unit runs but pressure is low, "
                "inspect the reed valves in the cylinder head for carbon buildup. Part number "
                "RV-2201 is the reed valve replacement kit."
            ),
            knowledge_id="equipment",
            metadata={"name": "XR-500 Troubleshooting Guide"},
        )
        await rag.ingest_text(
            content=(
                "Maintenance log 2024-03-15: Replaced reed valves (RV-2201) on unit #7. "
                "Previous valves showed excessive carbon deposits after 2,100 hours. "
                "Oil changed to Mobil DTE 24, SAE 30 equivalent. Next service due at 2,600 hours."
            ),
            knowledge_id="equipment",
            metadata={"name": "Maintenance Log"},
        )
        print("Ingested 3 documents\n")

        print("=== Semantic query (HyDE rewrites for better embedding match) ===")
        result = await rag.query("compressor won't build pressure", knowledge_id="equipment")
        print(f"Answer: {result.answer}\n")

        print("=== Exact term lookup (sparse embeddings match part numbers) ===")
        chunks = await rag.retrieve("RV-2201", knowledge_id="equipment")
        for chunk in chunks:
            print(f"  [{chunk.score:.2f}] {chunk.content[:120]}...")
        print()

        print("=== Full-text search via document store ===")
        matches = await config.persistence.document_store.search_content("reed valve", knowledge_id="equipment")
        for match in matches:
            print(f"  [{match.score:.2f}] {match.title}: {match.excerpt[:100]}...")

        # --- Direct method access (new modular API) ---
        print("\n=== Direct method access ===")
        print(f"Configured methods: {', '.join(m.name for m in rag.retrieval)}")

        # Vector search only (dense + SPLADE hybrid)
        vector_only = await rag.retrieval.vector.search("compressor pressure", top_k=3)
        print(f"\nVector-only results: {len(vector_only)}")
        for chunk in vector_only:
            print(f"  [{chunk.score:.2f}] {chunk.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
