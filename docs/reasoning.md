# Reasoning-Augmented Generation

When you deploy an AI system — a chatbot, a document search tool, a support assistant — you quickly run into questions that the system itself can't answer about itself:

- *What are customers actually asking about?*
- *Is this new message a refund request or a shipping complaint?*
- *Is the response we generated actually any good?*
- *Does this response comply with our company policies?*
- *What entities, urgency, and sentiment are in this message — before we even start thinking?*

These are questions **about your data and your outputs**, not questions you can answer by searching a knowledge base. RAG answers questions. ACE reasons about the data flowing through your system — it analyzes patterns, categorizes incoming content, measures output quality, and gates responses against policies. They solve different problems, and together they cover the full lifecycle of an AI-powered system.

The rfnry-rag SDK has five capabilities, each one independent:

| Capability | The question it answers |
|------------|------------------------|
| **Analysis** | What's in this message — intent, urgency, sentiment, entities, retrieval hints? |
| **Classification** | Which category does this text belong to? Multiple categories? |
| **Compliance** | Does this response violate any policies? |
| **Evaluation** | How good was the output we generated? |
| **Clustering** | What natural groups exist in my data? |

And a composition layer:

| Capability | What it does |
|------------|-------------|
| **Pipeline** | Chain analysis → classification → compliance → evaluation in sequence |

---

# Part I: Principles and Modern Approaches

## Text Analysis — Understanding Before Acting

### The idea

Most AI systems receive a message and immediately start generating a response. The agent reads the message, figures out what's going on, decides who should handle it, and then starts looking things up — all in the same expensive LLM call.

Text analysis separates understanding from acting. A fast, cheap model reads the message first and extracts everything the downstream system needs to know: what the user wants, how urgent it is, how they're feeling, what entities they mentioned, and what knowledge should be fetched. By the time the expensive model starts working, it already has a complete picture.

### What modern analysis extracts

A good analysis layer doesn't have a fixed schema — different domains need different insights. A customer support system cares about urgency and sentiment. A manufacturing system cares about defect severity and affected equipment. A content moderation system cares about toxicity and target.

The modern approach is **consumer-defined extraction**: you tell the analysis layer what dimensions to score, what entity types to look for, and what knowledge scopes exist for retrieval. The analysis layer provides the machinery — structured LLM prompts, typed outputs, confidence scoring — and you define the axes.

| What | Example | Why it matters |
|------|---------|----------------|
| **Intent** | "delivery inquiry" (confidence: 92%) | What is the user trying to accomplish? |
| **Dimensions** | urgency: 0.85, sentiment: "frustrated" | Scored along axes you define |
| **Entities** | order_id: FB-12345, deadline: Friday | Structured data extracted from unstructured text |
| **Summary** | "Late delivery with deadline pressure" | Quick context for downstream agents |
| **Retrieval hints** | "shipping delay policy", "FB-12345 status" | What knowledge should be pre-fetched? |

### Retrieval hints — bridging analysis and knowledge

The most powerful output of analysis is retrieval hints. The analysis layer reads the message and suggests what knowledge the system should fetch — without knowing anything about where that knowledge lives or how to fetch it.

This is a clean architectural boundary. The analysis layer says "you should look up the shipping delay policy and order FB-12345's status." The consumer maps those hints to actual RAG calls, database queries, or API lookups. The analysis layer is domain-agnostic; the consumer wires it to domain-specific knowledge sources.

### Thread analysis — conversations are not single messages

Real conversations shift intent mid-stream. A customer starts asking about delivery, then shifts to requesting a refund, then escalates to a manager. Single-message analysis misses this context.

Thread analysis reads the full conversation and tracks how intent evolves across messages, detects escalation signals, and assesses whether the issue was resolved. This is what separates a reactive system from one that truly understands the interaction.

---

## Classification — Sorting at Scale

### The idea

After you know your categories — whether you discovered them through clustering or defined them yourself — you need to sort new incoming text into them, one by one, in real time. A customer sends an email: is it a refund request, a shipping question, or a product inquiry?

Classification answers this, and it goes further than just picking a category. It tells you *how confident* it is, *why* it made that call, and whether a human should review the decision.

### Three strategies

**LLM-based:** The model reads the text, reads the category definitions, and reasons through which fits best. It returns a category, confidence, and explanation. Slower but highly accurate for ambiguous cases.

**kNN (k-Nearest Neighbors):** Search a database of already-labeled examples, find the most similar ones, take a vote. Very fast, no LLM call needed.

**Hybrid:** Start with kNN for speed. If the vote isn't decisive, escalate to the LLM. You use the cheap path for easy cases and only pay for the expensive path when it's genuinely needed.

### Multi-set classification

Real systems need more than one classification per message. A support orchestrator needs to know both the *routing decision* (should I act, delegate, or stay silent?) and the *topic* (shipping, billing, subscription?). Running two separate classification calls doubles cost and latency.

Multi-set classification solves this: define multiple category sets and classify against all of them in a single LLM call. The model reasons about all sets simultaneously, producing one result per set.

### Confidence and human-in-the-loop

Not every classification is confident. A configurable threshold flags low-confidence results with a `needs_review` signal, enabling downstream systems to route uncertain decisions to human review instead of acting on them blindly.

This is the difference between an AI system that occasionally makes quiet mistakes and one that knows when it's uncertain and asks for help.

---

## Compliance — Gating Responses Against Policy

### The idea

Evaluation scores quality. Compliance enforces policy. The distinction matters: a response can be high-quality (well-written, accurate, helpful) but still violate a company policy (offering a refund amount that exceeds authorized limits, disclosing internal pricing, or using language that doesn't meet tone guidelines).

Compliance checking takes a generated response and a reference policy, then returns a binary gate signal (compliant/non-compliant) along with specific violations — each with a dimension, severity, description, and fix suggestion.

### Why compliance is separate from evaluation

Evaluation asks "how good is this?" and returns a score. Compliance asks "does this break any rules?" and returns a pass/fail. A response that scores 0.95 on evaluation can still fail compliance if it promises something the agent isn't authorized to promise.

In production, evaluation monitors quality trends over time. Compliance gates individual responses before they reach the customer. They serve different purposes in the pipeline.

### Consumer-defined compliance dimensions

Like analysis dimensions, compliance dimensions are defined by the consumer. A customer support system might check authorization, accuracy, and tone. A medical system might check clinical accuracy, contraindication safety, and informed consent language. The compliance layer provides the judgment machinery; you define the rules.

### Compliance gate with auto-retry

The most effective pattern for compliance is not just detection but correction. When a response fails compliance, the violations are fed back to the agent as context, and it regenerates. If the regenerated response still fails after a configurable number of retries, the violations are surfaced to a human reviewer. The system self-corrects when it can and escalates when it can't.

---

## Evaluation — Measuring Output Quality

### The idea

Your AI system generates responses — answers, email replies, recommendations, summaries. How do you know if they're any good?

You can't just check if the system ran without errors. You need to know if the *content* is correct, helpful, and well-written. Evaluation does this by comparing generated outputs against reference examples and returning quality scores.

### Two signals

**Semantic similarity:** Both texts are turned into vectors. The cosine similarity measures how close they are in meaning — not just whether they share words, but whether they're saying the same thing. Fast, no LLM call needed. Good for automated regression testing.

**LLM judge:** An LLM reads both texts and scores the generated output on dimensions you define. Returns per-dimension scores and written reasoning. Slower but captures nuances that vector similarity misses.

**Combined:** Run both and get the most complete picture. Use similarity for fast filtering and the judge for detailed assessment.

### Quality bands and distribution

Individual scores are useful, but the real value is in aggregation. Batch evaluation produces a distribution — how many responses fell into "high", "medium", and "low" quality bands. This turns evaluation from a per-response tool into a system health metric.

Track the distribution over time: did last week's prompt change improve the "high" percentage? Did a new knowledge base cause the "low" count to spike?

---

## Clustering — Discovering What's in Your Data

### The idea

You have ten thousand customer support emails and you've never seen them before. You need to understand what people are writing about — but you can't read all ten thousand, and you don't know the categories ahead of time.

Clustering solves this. You give ACE the raw texts and it groups similar ones together automatically — no labels, no predefined categories. It finds the structure that's already there in the data.

### Two algorithms

**K-Means** — You decide how many groups you want. The algorithm finds the best grouping. Good when you have a rough idea of how many categories exist.

**HDBSCAN** — The algorithm decides how many groups exist on its own, based on natural density in the data. Good when you have no idea what to expect.

### Why clustering comes first

You can't set up a good support system without knowing what support requests look like. You can't train a classifier without knowing the categories. You can't evaluate response quality without knowing what response types exist. Clustering is the **discovery step** that makes everything else possible.

It's also invaluable for ongoing monitoring: run clustering on last month's data, compare to this month's — did a new complaint category emerge? Did a problem spike?

---

## Pipelines — Composing Capabilities

### The idea

In production, you rarely use a single capability in isolation. A message arrives and you need to analyze it, classify it, generate a response, check compliance, and evaluate quality. Running these as independent calls works but creates boilerplate — error handling, timing, result passing.

A pipeline chains capabilities into a sequential flow. You define the steps, the pipeline executes them in order, passes results between steps, times each one, and returns a single structured result.

### The intake pattern

The most common pipeline is the **intake pipeline** — the first-touch analysis that happens before any expensive work:

```
Message → Analyze (intent, urgency, entities, hints) → Classify (routing + topic)
```

This runs on a fast, cheap model (GPT-4o-mini) and produces everything the orchestrator needs to decide what happens next. If the routing is "SILENT" (most common), the expensive agent call never happens. If it's "DELEGATE", the agent starts with the full analysis already loaded — entities extracted, sentiment scored, policies pre-fetched.

---

## Why These Five Work Together

The five capabilities form a natural pipeline for understanding and improving any AI system:

```
Raw data → Clustering (discover categories)
               ↓
New data → Analysis (extract intent, dimensions, entities, hints)
               ↓
           Classification (routing + topic + confidence gate)
               ↓
           [Agent generates response using RAG]
               ↓
           Compliance (gate against policies)
               ↓
           Evaluation (score quality for monitoring)
               ↓
           Feedback into improvements
```

In practice: you cluster your historical data to discover categories, you deploy an intake pipeline to analyze and classify new messages in real time, you gate agent responses through compliance, and you evaluate outputs to continuously monitor quality.

---

# Part II: The rfnry-rag SDK

## What It Solves

Most AI systems bolt intelligence onto their pipeline piece by piece — a classification call here, a quality check there. Each piece uses different prompting patterns, different output formats, and different error handling. The result is fragile glue code that's hard to test and harder to extend.

ACE is a structured reasoning layer that provides all five capabilities through a consistent, typed interface. Every capability uses BAML for type-safe structured LLM output, follows the same constructor/config/result pattern, and composes through pipelines.

The SDK is **domain-agnostic**. You don't get a customer-support classifier or a manufacturing evaluator — you get machinery for defining your own dimensions, categories, entity types, and compliance rules. The SDK handles the LLM prompting, output parsing, confidence scoring, and result aggregation. You bring the domain knowledge.

## How Analysis Works

The `AnalysisService` takes a text (or a thread of messages) and extracts structured insights based on configuration you provide.

You define what to extract:
- **Dimensions** — scored axes with names, descriptions, and scales you specify
- **Entity types** — what structured data to pull from unstructured text
- **Retrieval hints** — suggestions for what to fetch from external knowledge, scoped to available sources
- **Summary** — a 1-2 sentence synthesis
- **Thread tracking** — intent shifts, escalation detection, resolution status (for conversations)

The service builds dynamic BAML prompts from your config, calls the LLM, and returns typed results. If you define an "urgency" dimension with scale "0.0-1.0", the result contains `dimensions["urgency"].value` as a float.

Thread analysis supports multi-message conversations through `analyze_thread()`. It reads the full message history and produces the same base result plus conversation-specific insights: where the customer's intent shifted, whether escalation was detected, and whether the issue was resolved.

## How Classification Works

The `ClassificationService` supports four modes:

**Single-set classification** (`classify`) — The standard mode. One text, one set of categories, one result with category, confidence, reasoning, and optional runner-up.

**Multi-set classification** (`classify_sets`) — Classify against multiple category sets in a single LLM call. Returns a `ClassificationSetResult` with one `Classification` per set, keyed by set name. This is how the intake pipeline classifies routing and topic simultaneously.

**Batch classification** (`classify_batch`, `classify_sets_batch`) — Process multiple texts concurrently with bounded concurrency.

**Hybrid strategy** — When configured with `strategy="hybrid"`, classification tries kNN first using the vector store. If the kNN vote confidence falls below `escalation_threshold`, the classification escalates to an LLM call. The result reports which strategy was actually used: `"knn"`, `"hybrid_knn"` (kNN was confident), or `"hybrid_llm_escalation"` (LLM was needed).

**Confidence flagging** — Set `low_confidence_threshold` on the config. Any classification below this threshold gets `needs_review=True`, enabling downstream routing to human review.

## How Compliance Works

The `ComplianceService` takes a text and a reference policy, then returns a `ComplianceResult`:
- `compliant: bool` — the gate signal
- `score: float` — overall compliance score (0.0-1.0)
- `violations: list[Violation]` — each with dimension, description, severity (high/medium/low), and an optional fix suggestion
- `dimension_scores: dict[str, float]` — per-dimension compliance scores

Dimensions are consumer-defined through `ComplianceDimensionDefinition(name, description)`. You define what compliance means for your domain.

Batch checking is available through `check_batch()` for processing multiple (text, reference) pairs concurrently.

## How Evaluation Works

The `EvaluationService` scores generated outputs against reference texts using one of three strategies:

**Similarity** — Embedding-based cosine similarity. Fast, no LLM call. Requires an embeddings provider.

**Judge** — LLM reads both texts and scores on consumer-defined dimensions. Returns per-dimension scores and reasoning. Requires `lm_config`.

**Combined** — Runs both similarity and judge. Returns the most complete picture.

Dimensions are defined through `EvaluationDimensionDefinition(name, description)` — not bare strings.

Batch evaluation produces an `EvaluationReport` with aggregate statistics: mean similarity, mean judge score, and a quality band distribution (high/medium/low counts).

## How Clustering Works

The `ClusteringService` groups texts by embedding similarity:

**K-Means** — Fixed cluster count. You specify `n_clusters` and the algorithm finds the optimal grouping.

**HDBSCAN** — Automatic cluster count. You specify `min_cluster_size` and the algorithm discovers however many natural clusters exist.

Both algorithms use the embeddings provider to convert texts to vectors before clustering. When `generate_labels=True`, an LLM generates a human-readable label for each cluster based on its sample texts.

`cluster_knowledge()` clusters directly from a vector store partition, avoiding re-embedding when vectors already exist.

## How Pipelines Work

The `Pipeline` chains steps sequentially. You provide a `PipelineServices` container with the services you need, then call `run()` with a text and a list of steps.

Available steps:
- `AnalyzeStep(config?)` — run analysis
- `ClassifyStep(categories?, sets?, config?)` — run single-set or multi-set classification
- `EvaluateStep(reference, config?)` — evaluate against a reference
- `ComplianceStep(reference, config?)` — check compliance against a reference

The result is a `PipelineResult` with one field per step type, plus total `duration_ms`.

Steps execute in order. Each step is independent — it operates on the original input text, not on previous step outputs. The pipeline provides composition, timing, and a unified result object.

## Protocol-Based Extensibility

ACE uses Python `Protocol` types for external dependencies:

- `BaseEmbeddings` — any object with an `async embed(texts) -> list[list[float]]` method
- `BaseVectorStore` — any object with `scroll()` and `search()` methods (read-only)

Any rfnry-rag SDK provider satisfies these protocols automatically via structural typing. No inheritance needed — if the signatures match, it works.

## ACE + RAG — Better Together

ACE and RAG are sibling SDKs designed to share infrastructure. You create your providers once and hand them to both:

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key="...")
vector_store = QdrantVectorStore(url="http://localhost:6333", collection="docs")

lm_config = LanguageModelConfig(
    client=LanguageModelClientConfig(provider="openai", model="gpt-4o-mini", api_key="...")
)

rag = RagEngine(RagServerConfig(
    persistence=PersistenceConfig(vector_store=vector_store),
    ingestion=IngestionConfig(embeddings=embeddings),
    augmentation=AugmentationConfig(provider=generation),
))

analyzer = AnalysisService(lm_config=lm_config)
classifier = ClassificationService(lm_config=lm_config)
compliance = ComplianceService(lm_config=lm_config)
evaluator = EvaluationService(embeddings=embeddings, lm_config=lm_config)
```

No duplication. The same embedding model and vector store serve both pipelines.

### How they divide the work

| Task | RAG | ACE |
|------|-----|-----|
| Answer a user's question from documents | yes | |
| Understand what the user is asking (intent, urgency, entities) | | yes |
| Suggest what knowledge to fetch before the agent starts | | yes |
| Retrieve relevant context | yes | |
| Score how good the answer was | | yes |
| Gate the answer against company policy | | yes |
| Discover patterns across a corpus | | yes |
| Ingest and search a knowledge base | yes | |

### A concrete end-to-end flow

```
Customer message arrives
        |
        v
  ACE intake pipeline (fast, cheap model)
  → Analysis: urgency=0.85, sentiment=frustrated, order_id=FB-12345
  → Classification: routing=DELEGATE, topic=shipping
  → Retrieval hints: "shipping delay policy", "FB-12345 status"
        |
        v
  RAG retrieves shipping policy + order status
  (guided by ACE's retrieval hints)
        |
        v
  Agent generates response
  (with full context: policy, order data, urgency, sentiment)
        |
        v
  ACE compliance check
  → compliant: true (no violations)
        |
        v
  Response delivered to customer
        |
        v
  ACE evaluation (async, for monitoring)
  → similarity: 0.91, judge: 0.87, quality_band: high
```

The cheap model (GPT-4o-mini) handles analysis and classification for every message. The expensive model (GPT-4) only runs for messages that need agent action. ACE's classification acts as a cost gate: if the message is SILENT (most common), the expensive call never happens.

---

## Real Applications

### Customer Support Automation

A company receives thousands of support messages per day. Before ACE:

- A human reads each message and routes it to the right team
- Agents start every interaction blind — no context, no pre-fetched knowledge
- Response quality varies and policy violations go undetected until escalation

With ACE:

1. **Analyze** every message: extract intent, urgency, sentiment, entities, and retrieval hints
2. **Classify** routing (silent/delegate/intervene) and topic (shipping/billing/subscription) in one call
3. Pre-fetch relevant policies and order data using retrieval hints
4. Agent starts with full context — entities extracted, sentiment scored, policies loaded
5. **Compliance** gates every response against company policies before delivery
6. **Evaluate** responses for quality monitoring and agent training

The human reviewer's job shifts from *reading every message* to *reviewing flagged cases* — low-confidence classifications and compliance failures.

### Manufacturing Quality Assurance

A factory has years of defect reports written in free text. Engineers need to understand failure patterns and act on new reports quickly.

1. **Cluster** historical defect reports → discover defect categories (pleating, adhesive, dimensional, media)
2. **Classify** new defect reports as they come in → trigger the right corrective action workflow
3. **Compliance** check corrective action recommendations against quality SOPs
4. **Evaluate** generated recommendations against standard procedures

### Content Moderation

A platform receives user-generated text that needs to be sorted and reviewed. Clustering reveals the landscape of what's being submitted. Classification sorts each submission. Compliance checks whether moderation decisions are consistent with written policy.

### AI System Monitoring

Any system using RAG to generate answers can use Evaluation as a continuous quality monitor:

- Run evaluation on a sample of real queries and answers each day
- Alert when mean score drops below a threshold
- Drill into low-scoring answers to understand what changed
- Track quality band distribution over time per category

---

## Summary

ACE gives you the reasoning layer that every serious AI deployment needs:

- **Analysis** to understand incoming data before acting — extracting dimensions, entities, and retrieval hints along axes you define
- **Classification** to sort and route content automatically with explainable reasoning, multi-set support, and confidence gating
- **Compliance** to gate every response against policy before it reaches the user, with auto-retry self-correction
- **Evaluation** to continuously measure output quality and track system health over time
- **Clustering** to discover patterns in your data before you build anything
- **Pipeline** to compose these capabilities into cohesive flows without glue code

These aren't features you add later. They're what separate a prototype from a production system — the difference between *it sometimes works* and *we know exactly how well it works, where it fails, and whether it stays within policy*.

---

See also: [Retrieval SDK documentation](retrieval.md) · [SDK reference](../src/rfnry-rag/reasoning/README.md) · [Main README](../README.md)
