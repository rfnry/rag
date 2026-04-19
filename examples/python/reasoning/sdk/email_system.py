"""Email response system — Reasoning classifies and evaluates, Retrieval generates, Reasoning checks compliance."""

import asyncio

from rfnry_rag.reasoning import (
    CategoryDefinition,
    ClassificationService,
    ComplianceConfig,
    ComplianceDimensionDefinition,
    ComplianceService,
    EvaluationConfig,
    EvaluationDimensionDefinition,
    EvaluationPair,
    EvaluationService,
    LanguageModelClient,
    LanguageModelProvider,
)
from rfnry_rag.retrieval import (
    Embeddings,
    GenerationConfig,
    IngestionConfig,
    PersistenceConfig,
    QdrantVectorStore,
    RagEngine,
    RagServerConfig,
)

embeddings = Embeddings(
    LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
)
vector_store = QdrantVectorStore(url="http://localhost:6333", collection="email-knowledge")
lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="your_api_key")
)

classifier = ClassificationService(lm_client=lm_client)
evaluator = EvaluationService(embeddings=embeddings, lm_client=lm_client)
compliance = ComplianceService(lm_client=lm_client)

CATEGORIES = [
    CategoryDefinition(name="refund", description="Customer wants money back"),
    CategoryDefinition(name="shipping", description="Delivery tracking or shipping questions"),
    CategoryDefinition(name="cancellation", description="Customer wants to cancel"),
    CategoryDefinition(name="product_inquiry", description="Product sizing, compatibility questions"),
]

REFERENCE_RESPONSES = {
    "refund": "We've initiated your refund. You'll see the credit within 5-7 business days.",
    "shipping": "Your shipment is in transit. Track it via the link in your confirmation email.",
    "cancellation": "Your order has been cancelled. If charged, a full refund will be issued.",
    "product_inquiry": "Check the dimensions on your current filter or use our sizing guide.",
}

COMMUNICATION_POLICY = (
    "All email responses must: address the customer's specific concern, "
    "include a timeline or next step, maintain professional empathetic tone, "
    "reference company policies when applicable."
)

rag_config = RagServerConfig(
    persistence=PersistenceConfig(vector_store=vector_store),
    ingestion=IngestionConfig(embeddings=embeddings),
    generation=GenerationConfig(
        lm_client=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key="your_api_key",
            ),
        ),
        system_prompt="You are a customer service agent. Write helpful, professional email responses.",
        grounding_enabled=True,
        grounding_threshold=0.4,
    ),
)


async def process_email(rag: RagEngine, email_text: str):
    print(f"\nIncoming: {email_text}")

    classification = await classifier.classify(email_text, CATEGORIES)
    print(f"  Category: {classification.category} ({classification.confidence:.0%})")

    result = await rag.query(f"Customer email ({classification.category}): {email_text}", knowledge_id="policies")
    print(f"  Response: {result.answer}")

    if result.answer:
        reference = REFERENCE_RESPONSES.get(classification.category, "")
        if reference:
            evaluation = await evaluator.evaluate(
                EvaluationPair(generated=result.answer, reference=reference, context=email_text),
                config=EvaluationConfig(
                    strategy="combined",
                    dimensions=[
                        EvaluationDimensionDefinition("accuracy", "Factual correctness"),
                        EvaluationDimensionDefinition("completeness", "Covers all key points"),
                    ],
                ),
            )
            print(f"  Quality: similarity={evaluation.similarity:.2f}, judge={evaluation.judge_score:.2f}")

        check = await compliance.check(
            result.answer,
            COMMUNICATION_POLICY,
            ComplianceConfig(
                dimensions=[
                    ComplianceDimensionDefinition("completeness", "All required elements present"),
                    ComplianceDimensionDefinition("tone", "Professional and empathetic"),
                ]
            ),
        )
        print(f"  Compliant: {check.compliant} ({check.score:.0%})")
        for v in check.violations:
            print(f"    [{v.severity}] {v.description}")


async def main():
    async with RagEngine(rag_config) as rag:
        policies = [
            ("Refund Policy", "Full refunds within 30 days. Processing: 5-7 business days."),
            ("Shipping Policy", "Standard 3-5 days. Express 2-day available. Tracking included."),
            ("Cancellation Policy", "Cancel within 24 hours for full refund. Subscriptions anytime."),
        ]
        for name, content in policies:
            await rag.ingest_text(content=content, knowledge_id="policies", metadata={"name": name})

        await process_email(rag, "I received a damaged filter and I want my money back ASAP.")
        await process_email(rag, "It's been a week and I still haven't received my order.")
        await process_email(rag, "I placed an order 10 minutes ago but I need to cancel it.")


if __name__ == "__main__":
    asyncio.run(main())
