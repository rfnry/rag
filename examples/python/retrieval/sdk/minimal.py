import asyncio

from rfnry_rag.retrieval import (
    Embeddings,
    GenerationConfig,
    IngestionConfig,
    LanguageModelClient,
    LanguageModelProvider,
    PersistenceConfig,
    QdrantVectorStore,
    RagEngine,
    RagServerConfig,
)

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(
            LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
        ),
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
        source = await rag.ingest("manual.pdf", metadata={"name": "Equipment Manual"})
        print(f"Ingested: {source.source_id} ({source.chunk_count} chunks)")

        result = await rag.query("How do I replace the filter?")
        print(f"Answer: {result.answer}")
        print(f"Sources: {len(result.sources)}")


if __name__ == "__main__":
    asyncio.run(main())
