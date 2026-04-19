"""Live chat support — Reasoning analyzes messages, Retrieval generates responses, Reasoning checks compliance."""

import asyncio

from rfnry_rag.reasoning import (
    AnalysisConfig,
    AnalysisService,
    AnalyzeStep,
    CategoryDefinition,
    ClassificationService,
    ClassificationSetDefinition,
    ClassifyStep,
    ComplianceConfig,
    ComplianceDimensionDefinition,
    ComplianceService,
    DimensionDefinition,
    EntityTypeDefinition,
    EvaluationService,
    LanguageModelClient,
    LanguageModelProvider,
    Pipeline,
    PipelineServices,
)
from rfnry_rag.retrieval import (
    Embeddings,
    GenerationConfig,
    IngestionConfig,
    PersistenceConfig,
    QdrantVectorStore,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
)

embeddings = Embeddings(
    LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
)
vector_store = QdrantVectorStore(url="http://localhost:6333", collection="product-knowledge")
lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="your_api_key")
)

intake_pipeline = Pipeline(
    services=PipelineServices(
        analysis=AnalysisService(lm_client=lm_client),
        classification=ClassificationService(lm_client=lm_client),
    )
)

compliance = ComplianceService(lm_client=lm_client)
evaluator = EvaluationService(embeddings=embeddings, lm_client=lm_client)

ROUTING_CATEGORIES = [
    CategoryDefinition(name="SILENT", description="No action needed, normal conversation"),
    CategoryDefinition(name="DELEGATE", description="Route to specialist agent"),
    CategoryDefinition(name="INTERVENE", description="Orchestrator should act directly"),
]

TOPIC_CATEGORIES = [
    CategoryDefinition(name="product_sizing", description="Filter size and compatibility questions"),
    CategoryDefinition(name="order_status", description="Order tracking and delivery questions"),
    CategoryDefinition(name="returns", description="Return or exchange requests"),
    CategoryDefinition(name="subscription", description="Subscription management"),
]

RESPONSE_POLICY = (
    "All responses must: address the customer's specific concern, "
    "include a timeline or next step, maintain a professional empathetic tone, "
    "never promise specific delivery dates, never disclose internal pricing."
)

rag_config = RagServerConfig(
    persistence=PersistenceConfig(vector_store=vector_store),
    ingestion=IngestionConfig(embeddings=embeddings),
    retrieval=RetrievalConfig(top_k=5),
    generation=GenerationConfig(
        lm_client=LanguageModelClient(
            provider=LanguageModelProvider(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key="your_api_key",
            ),
        ),
        system_prompt=(
            "You are a customer support agent. "
            "Use the provided context to give accurate, helpful answers. Keep responses concise."
        ),
        grounding_enabled=True,
        grounding_threshold=0.3,
    ),
)


async def handle_message(rag: RagEngine, message: str) -> str:
    intake = await intake_pipeline.run(
        message,
        steps=[
            AnalyzeStep(
                config=AnalysisConfig(
                    dimensions=[
                        DimensionDefinition("urgency", "Time sensitivity", "0.0-1.0"),
                        DimensionDefinition("sentiment", "Customer emotion", "frustrated/neutral/satisfied"),
                    ],
                    entity_types=[EntityTypeDefinition("order_id", "Order identifier like FB-XXXXX")],
                    summarize=True,
                )
            ),
            ClassifyStep(
                sets=[
                    ClassificationSetDefinition("routing", ROUTING_CATEGORIES),
                    ClassificationSetDefinition("topic", TOPIC_CATEGORIES),
                ]
            ),
        ],
    )

    routing = intake.classification.classifications["routing"]
    topic = intake.classification.classifications["topic"]

    print(f"  Routing: {routing.category} ({routing.confidence:.0%})")
    print(f"  Topic: {topic.category}")
    if intake.analysis.summary:
        print(f"  Summary: {intake.analysis.summary}")

    if routing.category == "SILENT":
        return ""

    result = await rag.query(message, knowledge_id="product-knowledge")
    response = result.answer or "Let me connect you with a specialist."

    check = await compliance.check(
        response,
        RESPONSE_POLICY,
        ComplianceConfig(
            dimensions=[
                ComplianceDimensionDefinition("accuracy", "Reflects actual company policy"),
                ComplianceDimensionDefinition("tone", "Professional and empathetic"),
            ],
        ),
    )

    if not check.compliant:
        for v in check.violations:
            print(f"  Violation: [{v.severity}] {v.description}")

    return response


async def main():
    async with RagEngine(rag_config) as rag:
        docs = [
            ("Filter Sizing Guide", "Common sizes: 16x25x1, 20x20x1, 20x25x1. Check current filter."),
            ("MERV Rating Guide", "MERV 8: basic. MERV 11: pets. MERV 13: allergies."),
            ("Subscription Plans", "Save 5%. Monthly, bimonthly, or quarterly. Cancel anytime."),
            ("Return Policy", "30-day returns. Original packaging required. Custom non-returnable."),
        ]
        for name, content in docs:
            await rag.ingest_text(content=content, knowledge_id="product-knowledge", metadata={"name": name})

        print("=== Chat Session ===\n")

        for message in [
            "Hi, I need a new air filter but I'm not sure what size.",
            "I have two dogs — what MERV rating should I use?",
            "Do you have a subscription so I don't have to remember to reorder?",
        ]:
            print(f"Customer: {message}")
            response = await handle_message(rag, message)
            if response:
                print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
