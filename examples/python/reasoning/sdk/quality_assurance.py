"""Manufacturing QA — Reasoning clusters defect patterns, classifies reports, Retrieval generates corrective actions."""

import asyncio

from rfnry_rag.reasoning import (
    CategoryDefinition,
    ClassificationService,
    ClusteringConfig,
    ClusteringService,
    ComplianceConfig,
    ComplianceDimensionDefinition,
    ComplianceService,
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
vector_store = QdrantVectorStore(url="http://localhost:6333", collection="manufacturing-qa")
lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="your_api_key")
)

clustering = ClusteringService(embeddings=embeddings, lm_client=lm_client)
classifier = ClassificationService(lm_client=lm_client)
evaluator = EvaluationService(embeddings=embeddings, lm_client=lm_client)
compliance = ComplianceService(lm_client=lm_client)

DEFECT_REPORTS = [
    "Pleats not evenly spaced on 20x25x1 filters. Gap exceeds 2mm tolerance.",
    "Media tearing during pleating on line 3. Multiple units rejected.",
    "Adhesive bond failure on frame corners. Filters separating during shipping.",
    "Pleat depth inconsistent across batch. Some pleats 15mm instead of 20mm.",
    "Cardboard frame warping after humidity exposure in warehouse.",
    "Media discoloration on MERV 13 filters. Yellow spots on white media.",
    "Hot melt adhesive stringing between pleats. Cosmetic defect only.",
    "Frame dimensions out of spec. 20x25x1 measuring 20.3x25.1.",
    "Media not fully bonded to frame on one side. Air bypass gap.",
    "Pleat count incorrect. 20x25x1 has 28 pleats instead of 32.",
    "Electrostatic charge not holding on MERV 11 media after storage.",
    "Adhesive contamination on filter media face. Glue spots visible.",
    "Frame corner joints not square. Diagonal off by 4mm.",
    "Media moisture content 12% vs spec max 8%.",
    "Pleat spacing irregular near frame edges. First/last pleats compressed.",
]

DEFECT_CATEGORIES = [
    CategoryDefinition(name="pleating", description="Pleat spacing, count, depth, or uniformity defects"),
    CategoryDefinition(name="adhesive", description="Glue bonds, adhesive application, or frame sealing defects"),
    CategoryDefinition(name="dimensional", description="Filter dimensions, frame squareness, or tolerance defects"),
    CategoryDefinition(name="media", description="Filter media condition, contamination, or property defects"),
]

REFERENCE_ACTIONS = {
    "pleating": "Recalibrate pleat comb assembly per SOP-PLT-003. Verify count and spacing with gauge.",
    "adhesive": "Verify adhesive temp at 350F per SOP-ADH-002. Check nozzle pressure and pattern.",
    "dimensional": "Inspect cutting dies for wear per SOP-CUT-001. Realign frame assembly fixtures.",
    "media": "Quarantine lot. Test media per QC-MED-005. Contact supplier if out of spec.",
}

QA_POLICY = (
    "All corrective actions must: identify a specific root cause, "
    "reference the relevant SOP number, include corrective and preventive measures, "
    "be actionable by a line operator without engineering support."
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
        system_prompt=(
            "You are a manufacturing quality engineer. Generate corrective action "
            "recommendations with root cause, corrective steps, and preventive measures."
        ),
        grounding_enabled=True,
    ),
)


async def main():
    print("=== Discover Defect Patterns ===\n")

    result = await clustering.cluster_texts(
        texts=DEFECT_REPORTS,
        config=ClusteringConfig(algorithm="kmeans", n_clusters=4, generate_labels=True),
    )

    for cluster in result.clusters:
        print(f"  {cluster.label}: {cluster.size} reports ({cluster.percentage:.0f}%)")

    print("\n=== Classify New Reports ===\n")

    new_reports = [
        "Line 2 producing filters with only 26 pleats instead of 32.",
        "Glue gun on station 4 leaving excess adhesive on media surface.",
        "20x20x1 frames measuring 20.5 inches on one side.",
    ]

    classifications = await classifier.classify_batch(texts=new_reports, categories=DEFECT_CATEGORIES)

    print("\n=== Generate Corrective Actions ===\n")

    async with RagEngine(rag_config) as rag:
        for spec in [
            ("SOP-PLT-003", "Pleat comb spacing verified every shift. 20x25x1 = 32 pleats. Tolerance +/- 1mm."),
            ("SOP-ADH-002", "Hot melt temp 340-360F. Nozzle pressure 45-55 PSI. Replace nozzle every 50k cycles."),
            ("SOP-CUT-001", "Cutting die inspection every 10k cuts. Tolerance +/- 0.1 inches."),
        ]:
            await rag.ingest_text(content=spec[1], knowledge_id="specs", metadata={"name": spec[0]})

        for report, cls in zip(new_reports, classifications, strict=True):
            query = f"Defect ({cls.category}): {report}. Generate corrective action."
            ca_result = await rag.query(query, knowledge_id="specs")

            print(f"  Report: {report}")
            print(f"  Category: {cls.category} ({cls.confidence:.0%})")
            print(f"  Action: {ca_result.answer}")

            reference = REFERENCE_ACTIONS.get(cls.category, "")
            if reference and ca_result.answer:
                evaluation = await evaluator.evaluate(
                    EvaluationPair(generated=ca_result.answer, reference=reference, context=report)
                )
                print(f"  Quality: {evaluation.similarity:.2f}")

            if ca_result.answer:
                check = await compliance.check(
                    ca_result.answer,
                    QA_POLICY,
                    ComplianceConfig(
                        dimensions=[
                            ComplianceDimensionDefinition("completeness", "Root cause + corrective + preventive"),
                            ComplianceDimensionDefinition("actionability", "Operator can execute without engineering"),
                        ],
                    ),
                )
                print(f"  Compliant: {check.compliant} ({check.score:.0%})")
                for v in check.violations:
                    print(f"    [{v.severity}] {v.description}")

            print()


if __name__ == "__main__":
    asyncio.run(main())
