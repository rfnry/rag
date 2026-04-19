"""All six modules — analysis, classification, multi-set classification, compliance, evaluation, pipeline."""

import asyncio

from rfnry_rag.reasoning import (
    AnalysisConfig,
    AnalysisService,
    AnalyzeStep,
    CategoryDefinition,
    ClassificationService,
    ClassificationSetDefinition,
    ClassifyStep,
    ClusteringConfig,
    ClusteringService,
    ComplianceConfig,
    ComplianceDimensionDefinition,
    ComplianceService,
    DimensionDefinition,
    EntityTypeDefinition,
    EvaluationConfig,
    EvaluationDimensionDefinition,
    EvaluationPair,
    EvaluationService,
    LanguageModelClient,
    LanguageModelProvider,
    Pipeline,
    PipelineServices,
)
from rfnry_rag.retrieval import Embeddings

embeddings = Embeddings(
    LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="your_api_key")
)
lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="your_api_key")
)


async def main():
    # 1. Clustering — discover categories from unlabeled data

    emails = [
        "Where is my order? It's been 5 days.",
        "I need to track my shipment.",
        "My package hasn't arrived yet.",
        "I want a full refund for this order.",
        "Please refund my credit card.",
        "The product was damaged, I want my money back.",
        "How do I cancel my subscription?",
        "Please cancel my upcoming order.",
        "I don't want this anymore, cancel it.",
    ]

    clustering = ClusteringService(embeddings=embeddings, lm_client=lm_client)
    clustering_result = await clustering.cluster_texts(
        texts=emails,
        config=ClusteringConfig(algorithm="kmeans", n_clusters=3, generate_labels=True),
    )

    print("=== Clustering ===")
    for cluster in clustering_result.clusters:
        print(f"  {cluster.label}: {cluster.size} emails ({cluster.percentage:.0f}%)")

    # 2. Analysis — extract structured insights with consumer-defined dimensions

    analyzer = AnalysisService(lm_client=lm_client)
    analysis = await analyzer.analyze(
        "My order FB-12345 hasn't arrived and I need it by Friday. This is the second time.",
        config=AnalysisConfig(
            dimensions=[
                DimensionDefinition("urgency", "How time-sensitive is this", "0.0-1.0"),
                DimensionDefinition("sentiment", "Customer emotional state", "frustrated/neutral/satisfied"),
            ],
            entity_types=[
                EntityTypeDefinition("order_id", "Order identifier like FB-XXXXX"),
                EntityTypeDefinition("deadline", "Any date or deadline mentioned"),
            ],
            summarize=True,
            generate_retrieval_hints=True,
            retrieval_hint_scopes=["policies", "customer-history"],
        ),
    )

    print("\n=== Analysis ===")
    print(f"  Intent: {analysis.primary_intent} ({analysis.confidence:.0%})")
    for name, dim in analysis.dimensions.items():
        print(f"  {name}: {dim.value}")
    for entity in analysis.entities:
        print(f"  Entity: {entity.type}={entity.value}")
    print(f"  Summary: {analysis.summary}")
    for hint in analysis.retrieval_hints:
        print(f"  Hint: {hint.query} → {hint.knowledge_scope}")

    # 3. Classification — single-set and multi-set

    classifier = ClassificationService(lm_client=lm_client)

    single = await classifier.classify(
        "I want to return this product for a refund",
        categories=[
            CategoryDefinition(name="refund", description="Customer wants money back"),
            CategoryDefinition(name="shipping", description="Delivery questions"),
            CategoryDefinition(name="cancellation", description="Customer wants to cancel"),
        ],
    )
    print("\n=== Classification ===")
    print(f"  {single.category} ({single.confidence:.0%})")

    multi = await classifier.classify_sets(
        "My order is late and I'm frustrated",
        sets=[
            ClassificationSetDefinition(
                "routing",
                [
                    CategoryDefinition(name="SILENT", description="No action needed"),
                    CategoryDefinition(name="DELEGATE", description="Route to specialist"),
                ],
            ),
            ClassificationSetDefinition(
                "topic",
                [
                    CategoryDefinition(name="billing", description="Payment issues"),
                    CategoryDefinition(name="shipping", description="Delivery issues"),
                ],
            ),
        ],
    )
    print(f"  Routing: {multi.classifications['routing'].category}")
    print(f"  Topic: {multi.classifications['topic'].category}")

    # 4. Compliance — check response against policy

    compliance = ComplianceService(lm_client=lm_client)
    check = await compliance.check(
        text="We've processed your full refund. It will appear in 3-5 business days.",
        reference=(
            "All customer responses must: include the customer's name, "
            "reference the order number, and provide a specific timeline."
        ),
        config=ComplianceConfig(
            dimensions=[
                ComplianceDimensionDefinition("completeness", "All required elements present"),
                ComplianceDimensionDefinition("tone", "Professional and empathetic"),
            ],
        ),
    )

    print("\n=== Compliance ===")
    print(f"  Compliant: {check.compliant} (score: {check.score:.0%})")
    for v in check.violations:
        print(f"  [{v.severity}] {v.dimension}: {v.description}")

    # 5. Evaluation — score generated output quality

    evaluator = EvaluationService(embeddings=embeddings, lm_client=lm_client)
    evaluation = await evaluator.evaluate(
        EvaluationPair(
            generated="We've processed your full refund. It will appear in 3-5 business days.",
            reference="Your refund has been issued and will be reflected within 5 business days.",
            context="Customer requested a refund for a damaged air filter.",
        ),
        config=EvaluationConfig(
            strategy="combined",
            dimensions=[
                EvaluationDimensionDefinition("accuracy", "Factual correctness"),
                EvaluationDimensionDefinition("completeness", "Covers all key points"),
            ],
        ),
    )

    print("\n=== Evaluation ===")
    print(f"  Similarity: {evaluation.similarity:.2f}, Judge: {evaluation.judge_score:.2f} ({evaluation.quality_band})")
    if evaluation.dimension_scores:
        for dim, score in evaluation.dimension_scores.items():
            print(f"  {dim}: {score:.2f}")

    # 6. Pipeline — compose analysis + classification in one call

    pipeline = Pipeline(
        services=PipelineServices(
            analysis=analyzer,
            classification=classifier,
        )
    )

    pipeline_result = await pipeline.run(
        "My subscription filter hasn't arrived and I want to cancel",
        steps=[
            AnalyzeStep(
                config=AnalysisConfig(
                    dimensions=[DimensionDefinition("urgency", "Time sensitivity", "0.0-1.0")],
                    summarize=True,
                )
            ),
            ClassifyStep(
                sets=[
                    ClassificationSetDefinition(
                        "routing",
                        [
                            CategoryDefinition(name="SILENT", description="No action needed"),
                            CategoryDefinition(name="DELEGATE", description="Route to specialist"),
                        ],
                    ),
                    ClassificationSetDefinition(
                        "topic",
                        [
                            CategoryDefinition(name="shipping", description="Delivery issues"),
                            CategoryDefinition(name="subscription", description="Subscription management"),
                        ],
                    ),
                ]
            ),
        ],
    )

    print("\n=== Pipeline ===")
    print(f"  Intent: {pipeline_result.analysis.primary_intent}")
    print(f"  Routing: {pipeline_result.classification.classifications['routing'].category}")
    print(f"  Topic: {pipeline_result.classification.classifications['topic'].category}")
    print(f"  Duration: {pipeline_result.duration_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
