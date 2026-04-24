# rfnry-rag — Reasoning SDK

Analysis, classification, compliance, evaluation, clustering, and pipeline composition. Analyze text, classify intent, check compliance, evaluate outputs — independently or composed through pipelines.

For setup, environment variables, and observability see the [main README](../../README.md). For concepts and architecture see the [reasoning documentation](../../docs/reasoning.md).

---

## Modules

### Analysis

Extract structured insights from text — intent, dimensions, entities, summaries, and retrieval hints. You define what to extract.

```python
from rfnry_rag.reasoning import (
    AnalysisService, AnalysisConfig,
    DimensionDefinition, EntityTypeDefinition,
    LanguageModelClient, LanguageModelProvider,
)

lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="...")
)

analyzer = AnalysisService(lm_client=lm_client)

result = await analyzer.analyze(
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

print(f"Intent: {result.primary_intent} ({result.confidence:.0%})")
print(f"Urgency: {result.dimensions['urgency'].value}")
for entity in result.entities:
    print(f"Entity: {entity.type}={entity.value}")
```

Context analysis with escalation detection and intent shift tracking:

```python
from rfnry_rag.reasoning import Message, ContextTrackingConfig

result = await analyzer.analyze_context(
    messages=[
        Message(text="Where is my order #12345?", role="customer"),
        Message(text="Let me check that for you.", role="agent"),
        Message(text="It's been a week! I want to speak to a manager.", role="customer"),
    ],
    config=AnalysisConfig(
        context_tracking=ContextTrackingConfig(
            track_intent_shifts=True,
            detect_escalation=True,
            track_resolution=True,
        ),
    ),
)

print(f"Escalation: {result.escalation_detected}")
print(f"Resolution: {result.resolution_status}")
for shift in result.intent_shifts:
    print(f"Shift: {shift.from_intent} → {shift.to_intent} at message {shift.at_message}")
```

### Classification

Classify text using LLM reasoning, or a hybrid strategy that tries kNN first and escalates low-confidence results to an LLM.

```python
from rfnry_rag.reasoning import ClassificationService, ClassificationConfig, CategoryDefinition

classifier = ClassificationService(lm_client=lm_client)

categories = [
    CategoryDefinition(name="refund", description="Customer wants money back", examples=["I want a refund"]),
    CategoryDefinition(name="shipping", description="Delivery questions", examples=["Where is my order?"]),
    CategoryDefinition(name="cancellation", description="Customer wants to cancel"),
]

result = await classifier.classify("I want my money back", categories)
print(f"{result.category} ({result.confidence:.0%}): {result.reasoning}")
```

Multi-set classification — classify against multiple category sets in one call:

```python
from rfnry_rag.reasoning import ClassificationSetDefinition

result = await classifier.classify_sets(
    "My order is late and I'm frustrated",
    sets=[
        ClassificationSetDefinition("routing", routing_categories),
        ClassificationSetDefinition("topic", topic_categories),
    ],
)

print(f"Routing: {result.classifications['routing'].category}")
print(f"Topic: {result.classifications['topic'].category}")
```

Confidence flagging — flag low-confidence results for human review:

```python
result = await classifier.classify(
    "maybe check my account?",
    categories,
    config=ClassificationConfig(low_confidence_threshold=0.5),
)
if result.needs_review:
    print("Low confidence — route to human review")
```

Hybrid strategy (kNN → LLM escalation):

```python
classifier = ClassificationService(
    embeddings=embeddings,
    lm_client=lm_client,
    vector_store=vector_store,
)

result = await classifier.classify(
    "I want my money back",
    categories,
    config=ClassificationConfig(
        strategy="hybrid",
        escalation_threshold=0.7,
        knn_knowledge_id="labeled_examples",
    ),
)
```

### Clustering

Discover latent categories in a text corpus using embedding-based clustering. Supports K-Means (fixed cluster count) and HDBSCAN (automatic cluster count). Optional LLM labeling generates human-readable names.

```python
from rfnry_rag.reasoning import ClusteringService, ClusteringConfig

clustering = ClusteringService(embeddings=embeddings, lm_client=lm_client)

result = await clustering.cluster_texts(
    texts=["..."] * 1000,
    config=ClusteringConfig(algorithm="kmeans", n_clusters=20, generate_labels=True),
)

for cluster in result.clusters:
    print(f"{cluster.label}: {cluster.size} docs ({cluster.percentage}%)")
```

### Compliance

Check text against reference policies for violations. Returns a compliant/non-compliant gate signal with per-violation details.

```python
from rfnry_rag.reasoning import ComplianceService, ComplianceConfig, ComplianceDimensionDefinition

compliance = ComplianceService(lm_client=lm_client)

result = await compliance.check(
    text="We'll give you a 150% refund plus free shipping forever!",
    reference="Refund Policy: Maximum refund is original order total. No extras.",
    config=ComplianceConfig(
        dimensions=[
            ComplianceDimensionDefinition("authorization", "Must not exceed authorized limits"),
            ComplianceDimensionDefinition("accuracy", "Must reflect actual company policy"),
            ComplianceDimensionDefinition("tone", "Must maintain professional tone"),
        ],
    ),
)

print(f"Compliant: {result.compliant}")
print(f"Score: {result.score:.0%}")
for violation in result.violations:
    print(f"[{violation.severity}] {violation.dimension}: {violation.description}")
```

Batch compliance checking:

```python
results = await compliance.check_batch([
    ("response text 1", "policy document"),
    ("response text 2", "policy document"),
])
```

### Evaluation

Score generated outputs against reference texts using embedding similarity and/or LLM-as-judge.

```python
from rfnry_rag.reasoning import EvaluationService, EvaluationConfig, EvaluationPair, EvaluationDimensionDefinition

evaluator = EvaluationService(embeddings=embeddings, lm_client=lm_client)

pair = EvaluationPair(
    generated="We've processed your refund. You'll see the credit in 3-5 business days.",
    reference="Your refund has been issued and will appear on your statement within 5 business days.",
    context="Customer requested refund for damaged filters",
)

result = await evaluator.evaluate(
    pair,
    config=EvaluationConfig(
        strategy="combined",
        dimensions=[
            EvaluationDimensionDefinition("accuracy", "Factual correctness"),
            EvaluationDimensionDefinition("completeness", "Covers all key points"),
            EvaluationDimensionDefinition("tone", "Professional and empathetic"),
        ],
    ),
)

print(f"Similarity: {result.similarity:.2f}")
print(f"Judge: {result.judge_score:.2f} ({result.quality_band})")
for dim, score in result.dimension_scores.items():
    print(f"  {dim}: {score:.2f}")
```

Batch evaluation with aggregate report:

```python
report = await evaluator.evaluate_batch([
    EvaluationPair(generated="...", reference="..."),
    EvaluationPair(generated="...", reference="...", context="..."),
])
print(f"Mean similarity: {report.mean_similarity:.2f}")
print(f"Distribution: {report.distribution}")  # {"high": N, "medium": N, "low": N}
```

### Pipeline

Compose analysis, classification, compliance, and evaluation into sequential pipelines.

> **Note:** `ClusteringService` is intentionally excluded from `Pipeline`. Clustering operates over a corpus (`list[str]`) and produces labeled groups, which does not fit the single-text step chain. Use `ClusteringService` directly; compose its output into a Pipeline by selecting representative texts per cluster.

```python
from rfnry_rag.reasoning import (
    Pipeline, PipelineServices,
    AnalyzeStep, ClassifyStep, ComplianceStep,
    AnalysisService, ClassificationService, ComplianceService,
)

pipeline = Pipeline(
    services=PipelineServices(
        analysis=AnalysisService(lm_client=lm_client),
        classification=ClassificationService(lm_client=lm_client),
        compliance=ComplianceService(lm_client=lm_client),
    )
)

result = await pipeline.run(
    "My order FB-12345 is late and I need it by Friday",
    steps=[
        AnalyzeStep(config=AnalysisConfig(
            dimensions=[DimensionDefinition("urgency", "Time sensitivity", "0.0-1.0")],
            entity_types=[EntityTypeDefinition("order_id", "Order identifier")],
            summarize=True,
        )),
        ClassifyStep(sets=[
            ClassificationSetDefinition("routing", routing_categories),
            ClassificationSetDefinition("topic", topic_categories),
        ]),
    ],
)

print(f"Intent: {result.analysis.primary_intent}")
print(f"Routing: {result.classification.classifications['routing'].category}")
print(f"Duration: {result.duration_ms:.0f}ms")
```

---

## CLI

Install with `uv add "rfnry-rag[cli]"`.

### Setup

```bash
rfnry-rag reasoning init      # creates ~/.config/rfnry_rag/config.toml + .env
rfnry-rag reasoning status    # validates config and tests LLM connection
```

Minimal `config.toml` (3 lines):

```toml
[language_model]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
```

### Commands

```bash
# Analyze text
rfnry-rag reasoning analyze "My order FB-12345 hasn't arrived and I need it by Friday"
rfnry-rag reasoning analyze --file input.txt --summarize
echo "some text" | rfnry-rag reasoning analyze

# Classify text (categories from JSON file)
rfnry-rag reasoning classify "I want my money back" --categories cats.json

# Check compliance against reference document(s)
rfnry-rag reasoning compliance "We'll give you 150% refund" --references policy.txt
rfnry-rag reasoning compliance "text" --references policy.md style-guide.md
rfnry-rag reasoning compliance "text" --references reference-docs/

# Custom compliance threshold
rfnry-rag reasoning compliance --file response.md --references policy.md --threshold 0.7

# Analyze a multi-turn conversation or document sequence
rfnry-rag reasoning analyze-context --file conversation.json --summarize --dimensions dims.json

# Evaluate generated text against a reference
rfnry-rag reasoning evaluate --generated output.txt --reference expected.txt --strategy judge
```

### Output

Auto-detects TTY: terminal gets human-readable output, pipes get JSON. Override with `--json` or `--pretty`.

```bash
rfnry-rag reasoning analyze "text"                  # pretty in terminal
rfnry-rag reasoning analyze "text" --json           # force JSON
rfnry-rag reasoning analyze "text" | jq .           # auto JSON when piped
```

### Environment Variables

Override config from environment (useful for sandboxed environments):

```bash
export ANTHROPIC_API_KEY=sk-...
rfnry-rag reasoning analyze "text"
```

---

## API Reference

### `AnalysisService`

Constructor: `AnalysisService(lm_client)`

| Method | Returns | Description |
|--------|---------|-------------|
| `analyze(text, config?)` | `AnalysisResult` | Analyze a single text |
| `analyze_context(messages, config?)` | `AnalysisResult` | Analyze a conversation/context |
| `analyze_batch(texts, config?)` | `list[AnalysisResult]` | Analyze multiple texts concurrently |

### `ClassificationService`

Constructor: `ClassificationService(embeddings?, lm_client?, vector_store?)`

| Method | Returns | Description |
|--------|---------|-------------|
| `classify(text, categories, config?, metadata?)` | `Classification` | Classify a single text |
| `classify_batch(texts, categories, config?, metadata?)` | `list[Classification]` | Classify multiple texts concurrently |
| `classify_sets(text, sets, config?, metadata?)` | `ClassificationSetResult` | Classify against multiple category sets |
| `classify_sets_batch(texts, sets, config?)` | `list[ClassificationSetResult]` | Multi-set batch classification |

### `ClusteringService`

Constructor: `ClusteringService(embeddings, lm_client?)`

| Method | Returns | Description |
|--------|---------|-------------|
| `cluster_texts(texts, config?)` | `ClusteringResult` | Cluster raw texts by embedding |
| `cluster_knowledge(vector_store, knowledge_id, config?)` | `ClusteringResult` | Cluster from vector store partition |

### `ComplianceService`

Constructor: `ComplianceService(lm_client)`

| Method | Returns | Description |
|--------|---------|-------------|
| `check(text, reference, config?)` | `ComplianceResult` | Check text against a reference policy |
| `check_batch(items, config?)` | `list[ComplianceResult]` | Check multiple (text, reference) pairs |

### `EvaluationService`

Constructor: `EvaluationService(embeddings?, lm_client?)`

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(pair, config?)` | `EvaluationResult` | Evaluate a single generated-vs-reference pair |
| `evaluate_batch(pairs, config?)` | `EvaluationReport` | Evaluate multiple pairs with aggregate report |

### `Pipeline`

Constructor: `Pipeline(services)` where `services` is `PipelineServices(analysis?, classification?, evaluation?, compliance?)`

| Method | Returns | Description |
|--------|---------|-------------|
| `run(text, steps)` | `PipelineResult` | Execute steps sequentially on the input text |

Steps: `AnalyzeStep`, `ClassifyStep`, `EvaluateStep`, `ComplianceStep`

> **Clustering is intentionally excluded.** `ClusteringService` operates over a corpus (`list[str]`) and does not fit the single-text step chain. Use it directly and feed representative texts into a Pipeline if needed.

### Config Reference

#### `AnalysisConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dimensions` | `list[DimensionDefinition]` | `None` | Scoring dimensions |
| `entity_types` | `list[EntityTypeDefinition]` | `None` | Entity types to extract |
| `summarize` | `bool` | `False` | Generate a 1-2 sentence summary |
| `generate_retrieval_hints` | `bool` | `False` | Suggest what to fetch from knowledge |
| `retrieval_hint_scopes` | `list[str]` | `None` | Available knowledge scopes for hints |
| `context_tracking` | `ContextTrackingConfig` | `None` | Intent shifts, escalation, resolution tracking |
| `max_text_length` | `int` | `3000` | Truncate input text |
| `concurrency` | `int` | `10` | Batch concurrency |

#### `ClassificationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `"llm" \| "hybrid"` | `"llm"` | Classification strategy |
| `low_confidence_threshold` | `float` | `None` | Below this, `needs_review=True` |
| `escalation_threshold` | `float` | `0.7` | Hybrid: escalate to LLM below this |
| `top_k` | `int` | `10` | kNN retrieval count for hybrid |
| `concurrency` | `int` | `10` | Batch concurrency |
| `knn_knowledge_id` | `str` | `None` | Vector store partition (required for hybrid) |
| `knn_label_field` | `str` | `"category"` | Payload field with category label |
| `max_text_length` | `int` | `2000` | Truncate input text |

#### `ClusteringConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `algorithm` | `"kmeans" \| "hdbscan"` | `"kmeans"` | Clustering algorithm |
| `n_clusters` | `int` | `10` | Number of clusters (kmeans only) |
| `min_cluster_size` | `int` | `10` | Minimum cluster size (hdbscan only) |
| `samples_per_cluster` | `int` | `5` | Sample texts per cluster |
| `generate_labels` | `bool` | `False` | Generate LLM labels (requires `lm_client`) |
| `random_state` | `int` | `42` | Random seed |

#### `ComplianceConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dimensions` | `list[ComplianceDimensionDefinition]` | `None` | Compliance dimensions |
| `threshold` | `float` | `None` | Score threshold for compliant gate (0.0-1.0) |
| `max_text_length` | `int` | `3000` | Truncate input text |
| `max_reference_length` | `int` | `5000` | Truncate reference text |
| `concurrency` | `int` | `10` | Batch concurrency |

#### `EvaluationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `"similarity" \| "judge" \| "combined"` | `"similarity"` | Evaluation strategy |
| `dimensions` | `list[EvaluationDimensionDefinition]` | `[]` | Judge dimensions |
| `concurrency` | `int` | `10` | Batch concurrency |
| `high_threshold` | `float` | `0.8` | Score >= this is "high" |
| `medium_threshold` | `float` | `0.5` | Score >= this is "medium" |
| `max_text_length` | `int` | `3000` | Truncate input text |

### `LanguageModelClient`

Used by all services to configure BAML-based LLM calls with retry and fallback support.

```python
from rfnry_rag.common.language_model import LanguageModelClient, LanguageModelProvider

config = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="..."),
)

config = LanguageModelClient(
    provider=LanguageModelProvider(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="..."),
    fallback=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="..."),
    strategy="fallback",
    max_retries=3,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `LanguageModelProvider` | required | Primary LLM provider |
| `fallback` | `LanguageModelProvider` | `None` | Fallback LLM provider |
| `max_retries` | `int` | `3` | Retry attempts per provider (0-5) |
| `strategy` | `"primary_only" \| "fallback"` | `"primary_only"` | Routing strategy |
| `max_tokens` | `int` | `4096` | Max output tokens |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `boundary_api_key` | `str` | `None` | BAML observability API key |

| Provider Field | Type | Default | Description |
|------|------|---------|-------------|
| `provider` | `str` | required | Provider name (`"openai"`, `"anthropic"`, etc.) |
| `model` | `str` | required | Model identifier |
| `api_key` | `str` | `None` | API key (falls back to environment variable) |

