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
    Reranking,
    RetrievalConfig,
    SQLAlchemyMetadataStore,
    Vision,
)

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
        metadata_store=SQLAlchemyMetadataStore(url="sqlite+aiosqlite:///rag.db"),
        document_store=PostgresDocumentStore(url="postgresql://user:pass@localhost:5432/rag"),
    ),
    ingestion=IngestionConfig(
        embeddings=Embeddings(
            LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
        ),
        vision=Vision(
            LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="your_api_key")
        ),
        sparse_embeddings=FastEmbedSparseEmbeddings(),
        chunk_size=500,
        chunk_overlap=50,
        parent_chunk_size=1500,
        contextual_chunking=True,
    ),
    retrieval=RetrievalConfig(
        reranker=Reranking(LanguageModelProvider(provider="cohere", model="rerank-v3.5", api_key="your_api_key")),
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
        source_type_weights={"manual": 1.0, "drawing": 0.9, "transcript": 0.5, "community": 0.8},
    ),
    generation=GenerationConfig(
        lm_client=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic", model="claude-sonnet-4-20250514", api_key="your_api_key"
            ),
        ),
        system_prompt=(
            "You are a manufacturing equipment assistant. "
            "Use only the provided context to answer. "
            "Cite page numbers. If unsure, say so."
        ),
        grounding_enabled=True,
        grounding_threshold=0.5,
        relevance_gate_enabled=True,
        relevance_gate_model=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key="your_api_key",
            )
        ),
        guiding_enabled=True,
    ),
)


async def main():
    async with RagEngine(config) as rag:
        pdf_source = await rag.ingest(
            "helios_max_manual.pdf",
            knowledge_id="helios-max",
            source_type="manual",
            metadata={"name": "Helios Max Manual", "file_url": "s3://docs/helios_max.pdf"},
        )
        print(f"Ingested PDF: {pdf_source.source_id} ({pdf_source.chunk_count} chunks)")

        await rag.ingest_text(
            content="Q: What oil viscosity for Helios Max?\nA: SAE 30 at 72F operating temp.",
            knowledge_id="helios-max",
            source_type="community",
            metadata={"name": "Community Q&A"},
        )

        await rag.ingest(
            "wiring_diagram.png",
            knowledge_id="helios-max",
            source_type="drawing",
            metadata={"name": "Wiring Diagram"},
        )

        result = await rag.query("How do I replace the air filter?", knowledge_id="helios-max")
        print(f"\nAnswer: {result.answer}")
        print(f"Grounded: {result.grounded}")
        print(f"Confidence: {result.confidence}")
        for src in result.sources:
            print(f"  - {src.name} (page {src.page_number}, score {src.score:.2f})")
        if result.clarification:
            print(f"\nClarification needed: {result.clarification.question}")
            print(f"Options: {result.clarification.options}")

        print("\nStreaming response:")
        async for event in rag.query_stream("What oil should I use?", knowledge_id="helios-max"):
            if event.type == "chunk":
                print(event.content, end="", flush=True)
            elif event.type == "sources":
                for ref in event.sources:
                    print(f"\n  - {ref.name} (page {ref.page_number})")
            elif event.type == "done":
                print(f"\n[grounded={event.grounded}, confidence={event.confidence:.2f}]")

        chunks = await rag.retrieve("part number 8842-A", knowledge_id="helios-max")
        print("\nRetrieval results:")
        for chunk in chunks:
            print(f"  [{chunk.score:.2f}] {chunk.content[:100]}...")

        history = [
            ("How do I replace the air filter?", result.answer),
        ]
        followup = await rag.query("How often should I do that?", knowledge_id="helios-max", history=history)
        print(f"\nFollow-up answer: {followup.answer}")

        sources = await rag.knowledge.list(knowledge_id="helios-max")
        print("\nIngested sources:")
        for s in sources:
            print(f"  {s.source_id}: {s.metadata.get('name', 'unnamed')} ({s.chunk_count} chunks)")

        source_chunks = await rag.knowledge.get_chunks(pdf_source.source_id)
        print(f"\nChunks in source: {len(source_chunks)}")

        deleted = await rag.knowledge.remove(pdf_source.source_id)
        print(f"Deleted {deleted} vectors")


if __name__ == "__main__":
    asyncio.run(main())
