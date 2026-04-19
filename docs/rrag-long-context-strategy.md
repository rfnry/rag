# rfnry-rag in the Long-Context Era: Strategic Feature Roadmap

## The State of Play (April 2026)

Context windows have exploded. Claude Opus 4.6 and Sonnet 4.6 process 1M tokens. Gemini 3.1 Pro handles similar scales. GPT 5.4 works at hundreds of thousands. Token prices have collapsed — a 200k-token query on Sonnet 4.6 costs $0.63; with prompt caching, subsequent queries on the same context drop to ~$0.10.

The question every RAG tool faces: **when the LLM can just read everything, why bother retrieving?**

The research is clear: **neither RAG nor Long Context (LC) wins universally.** The answer depends on corpus size, task type, model strength, query complexity, and cost tolerance. The tools that survive are the ones that let users navigate this decision space intelligently — not the ones that force a single paradigm.

This document synthesizes findings from seven research sources and maps them against rfnry-rag's current capabilities to identify what we need to build.

---

## What the Research Says

### 1. Long Context Is Not a Silver Bullet

**LaRA (ICML 2025)** tested 2,326 cases across 11 LLMs at 32K and 128K contexts:

- At 32K tokens, LC wins by 2.4% on average
- At 128K tokens, **RAG wins by 3.68%** — even strong models degrade as context grows
- Weaker models benefit enormously from RAG: Mistral-Nemo-12B gets +38% accuracy with RAG at 128K
- Strong models (GPT-4o, Claude-3.5-Sonnet) favor LC at 32K but the gap narrows at 128K

**Key insight:** The LC advantage is model-dependent and context-length-dependent. It's not a general truth — it's a parameter.

### 2. RAG Has Unique Strengths LC Cannot Replace

**Hallucination detection:** RAG dominates by 10-22% across all context lengths (LaRA). LC feeds the entire text and becomes susceptible to noise — even GPT-4o only manages 56% accuracy on hallucination detection with full context. RAG's selective retrieval helps models identify when they lack information.

**Lost in the Middle (Liu et al., TACL 2024):** Models exhibit U-shaped attention — strong at beginning and end, weak in the middle. Even 2026 models haven't fully solved this. RAG bypasses the problem entirely by surfacing relevant content to privileged positions.

**Cost at scale:** SELF-ROUTE (Google DeepMind) achieves near-LC accuracy at 35-61% of the token cost by routing only "unanswerable" queries to full-context LC. The rest are answered from retrieved chunks.

**Out-of-domain robustness:** BEIR (NeurIPS 2021) showed that dense retrieval alone is fragile across domains. But BM25 + cross-encoder reranking generalizes to 16/18 heterogeneous datasets. RAG with hybrid retrieval is more robust than any single method.

### 3. The Retrieval Method Matters More Than RAG vs LC

**Li et al. (2025)** found that most studies comparing RAG vs LC used only basic chunk-based retrieval, which severely underrepresented RAG:

| Retrieval Method | Accuracy |
|---|---|
| BM25 (chunk-based) | 20.4% |
| Contriever (chunk-based) | 20.1% |
| OpenAI Embeddings (chunk-based) | 21.6% |
| Tree Index | 30.1% |
| Sentence Window | 35.5% |
| **RAPTOR (summarization-based)** | **38.5%** |

RAPTOR (hierarchical clustering + recursive summarization) nearly **doubles** chunk-based RAG performance. The lesson: when someone says "LC beats RAG," they usually mean "LC beats bad RAG."

### 4. BM25 Is Not Optional

Every source confirms this:

- **Anthropic Contextual Retrieval:** Adding contextual BM25 to contextual embeddings reduces retrieval failure by 49%. Adding reranking reaches 67%.
- **BEIR:** BM25 remains the strongest zero-shot baseline. Dense retrievers only win when fine-tuned on domain data.
- **Akita (2026):** Claude Code's memory system uses no vector DB — grep + markdown index. "Start with keyword, add vectors only if you feel the gap."

BM25 catches what embeddings miss (exact codes, identifiers, technical terms) and embeddings catch what BM25 misses (paraphrased queries, synonyms). Neither alone is sufficient.

### 5. The Future Is Hybrid Routing, Not One-or-the-Other

**SELF-ROUTE (Google DeepMind):** RAG first, then route "unanswerable" queries to full-context LC. 76.8% of queries resolved by RAG alone on Gemini-1.5-Pro.

**LaRA's conclusion:** The optimal choice depends on model strength x context length x task type x context type. No static rule works. The system must adapt.

**Akita's "Lazy Retrieval":** For small/medium corpora, skip the vector DB. Lexical filter → generous context window → LLM reasoning. Add embeddings only where lexical search demonstrably fails.

---

## What rfnry-rag Already Does Well

Mapping current capabilities against the research:

| Research Recommendation | rfnry-rag Status |
|---|---|
| Multi-path retrieval with fusion | **Strong** — Vector + Document + Graph + Tree with RRF |
| BM25 / keyword search | **Present** — BM25 via vector store + full-text via document store |
| Hybrid dense + sparse retrieval | **Present** — Dense + SPLADE sparse embeddings |
| Contextual chunking (LLM-enriched) | **Present** — `contextual_chunking` option in IngestionConfig |
| Cross-encoder reranking | **Present** — Cohere + Voyage reranking |
| Query rewriting | **Present** — HyDE, multi-query, step-back |
| Hierarchical/tree-based retrieval | **Present** — TOC detection, LLM-navigated tree search |
| Graph-based entity retrieval | **Present** — Neo4j entity extraction + multi-hop traversal |
| Grounding / confidence gates | **Present** — Score gate + LLM relevance gate |
| Chunk refinement | **Present** — Abstractive (LLM compression) + extractive (embedding similarity) |
| Protocol-based extensibility | **Strong** — Pluggable methods, stores, embeddings, rerankers |
| No mandatory vector DB | **Strong** — Run with document-only, graph-only, or any combination |

rfnry-rag's multi-path architecture with optional components is well-positioned. The research validates this design — no single retrieval method wins everywhere.

---

## What We Need to Build

### Priority 1: Context-Aware Routing (RAG vs Direct Context)

**The gap:** rfnry-rag always retrieves. It has no concept of "this corpus is small enough to just pass directly." Every paper we reviewed — Anthropic, DeepMind, LaRA, Akita — agrees: for small knowledge bases (under ~200K tokens), direct context often outperforms retrieval.

**What to build:**

**A. Corpus-Aware Mode Selection**

At `RagEngine.initialize()` or query time, the system should be able to assess corpus size per `knowledge_id` and route accordingly:

- **Direct Context mode:** When total corpus tokens < configurable threshold (e.g., 150K), skip retrieval entirely. Load all documents into the prompt. Leverage provider prompt caching for repeated queries on the same corpus.
- **Retrieval mode:** Current behavior. Multi-path search + fusion + reranking.
- **Hybrid mode (SELF-ROUTE):** Attempt retrieval first. If the model judges the retrieved chunks as insufficient ("unanswerable"), escalate to full-context mode for that query.

This is the single most important feature to prevent rfnry-rag from becoming obsolete. Users with 50-page knowledge bases shouldn't need a vector DB, chunking pipeline, and reranking — they should get the best answer at the lowest cost, which in 2026 is often just "read the whole thing."

**B. Prompt Caching Integration**

Direct Context mode only makes economic sense with prompt caching. When the same corpus is loaded repeatedly (which is the common case — many queries against one knowledge base), cached tokens cost ~90% less. rfnry-rag should:

- Detect when the LLM provider supports prompt caching (Anthropic, OpenAI, Google)
- Structure prompts to maximize cache hit rates (stable corpus prefix, variable query suffix)
- Track cache hit rates in observability/logging
- Expose caching stats through the knowledge manager

### Priority 2: RAPTOR-Style Summarization Retrieval

**The gap:** rfnry-rag has tree-based retrieval (TOC navigation), but this is structurally different from RAPTOR. Tree search navigates *existing* document structure (headings, sections). RAPTOR creates *new* hierarchical structure by recursively clustering and summarizing chunks — it works on any corpus, even unstructured ones.

Li et al. (2025) showed RAPTOR achieves 38.5% accuracy vs 20-21% for chunk-based methods — nearly doubling performance. This is the single biggest retrieval quality improvement available.

**What to build:**

A new retrieval method implementing `BaseRetrievalMethod`:

1. **Index time:** Cluster leaf chunks by semantic similarity (embeddings → K-Means or HDBSCAN). Summarize each cluster via LLM. Cluster the summaries. Summarize again. Build a tree of increasingly abstract summaries.
2. **Query time:** Search across all levels of the tree (leaf chunks + summaries at each level). Higher-level summaries capture thematic/conceptual content that leaf chunks miss.
3. **Fusion:** RAPTOR results participate in RRF alongside vector, document, and graph results.

rfnry-rag already has the building blocks — the clustering service in the Reasoning SDK, the tree infrastructure, and the embedding providers. This is an assembly task more than a ground-up build.

### Priority 3: Document Expansion at Index Time

**The gap:** BEIR showed that `docT5query`-style document expansion (generating synthetic queries for each document/chunk at index time) outperforms BM25 on 11/18 datasets and generalizes well across domains. Anthropic's Contextual Retrieval is a variant of this — enriching chunks with LLM-generated context before indexing.

rfnry-rag has `contextual_chunking` (LLM-generated context per chunk), which addresses the Anthropic approach. But it doesn't generate **synthetic queries** per chunk.

**What to build:**

- **Query generation at index time:** For each chunk, generate 3-5 synthetic questions that the chunk answers. Append these to the chunk's searchable content (both for BM25 and embedding indexing).
- This bridges the vocabulary gap between how users ask questions and how documents describe answers ("how to change oil" vs "lubricant replacement procedure").
- Configurable per ingestion config. Uses the same LLM infrastructure as contextual chunking.
- Store synthetic queries as metadata for transparency/debugging.

### Priority 4: Chunk Position Optimization

**The gap:** "Lost in the Middle" demonstrated that LLMs perform best when relevant information is at the beginning or end of the context window. rfnry-rag currently orders chunks by fusion score (highest first), which is reasonable but not optimized for this effect.

**What to build:**

- **Position-aware context assembly:** When building the generation prompt, place the highest-confidence chunks at the beginning AND end, with lower-confidence chunks in the middle.
- **Configurable strategies:** "score-descending" (current), "primacy-recency" (optimize for Lost-in-the-Middle), "interleaved" (alternate high/low).
- This is a low-effort, high-impact change in the generation service's `chunks_to_context()` method.

### Priority 5: Adaptive Retrieval Parameters

**The gap:** LaRA showed that optimal chunk count and retrieval strategy depend on model strength, context length, and task type. rfnry-rag uses static `top_k` and fixed method weights. The system should adapt.

**What to build:**

**A. Dynamic Top-K**

- Query complexity estimation (via the existing Analysis service or a lightweight LLM call)
- Simple queries (factual, specific): lower top_k (3-5)
- Complex queries (multi-hop, comparative): higher top_k (15-20)
- Configurable bounds and default

**B. Task-Aware Method Weighting**

LaRA found that comparison tasks favor full context while hallucination detection favors retrieval. The Analysis service already detects intent — use this to dynamically adjust method weights:

- Factual/location queries: boost vector + BM25 weight
- Comparative queries: boost tree + document retrieval (broader context)
- Entity-relationship queries: boost graph weight

**C. Confidence-Based Expansion**

If initial retrieval returns low-confidence results (below grounding threshold), automatically:
1. Increase top_k and retry
2. Try alternative query rewriting strategies
3. Expand to additional retrieval methods
4. Ultimately escalate to full-context mode (ties back to Priority 1)

### Priority 6: Multi-Hop Iterative Retrieval

**The gap:** SELF-ROUTE's failure analysis identified multi-hop reasoning as RAG's biggest weakness — queries like "What nationality is the performer of song X?" require chaining retrieval (find song → find performer → find nationality). rfnry-rag does single-pass retrieval.

**What to build:**

- **Iterative retrieval loop:** The LLM identifies sub-questions from the original query. Each sub-question triggers a retrieval pass. Results from earlier passes inform later queries.
- **Chain-of-retrieval:** Similar to chain-of-thought but for the search phase. The model generates intermediate retrieval queries, accumulates context, then generates the final answer.
- This can build on the existing query rewriting infrastructure — multi-query rewriting already generates variants, but they all execute in parallel against the same corpus. Iterative retrieval executes sequentially with accumulated context.

### Priority 7: Cross-Encoder Local Reranking

**The gap:** BEIR showed cross-encoder reranking is the best-generalizing approach (16/18 datasets). rfnry-rag supports Cohere and Voyage reranking (API-based). But API reranking adds latency and cost. Local cross-encoder models (e.g., ColBERT, BGE-reranker, MiniLM cross-encoders) provide the same benefit without external API calls.

**What to build:**

- Add a `LocalCrossEncoderReranking` implementation of `BaseReranking`
- Support sentence-transformers cross-encoder models
- Run locally via the same protocol interface
- Useful for latency-sensitive or air-gapped deployments

### Priority 8: Retrieval Evaluation & Diagnostics

**The gap:** Akita's strongest argument against vector DBs is debuggability — "when BM25 misses, you know why; when a vector DB returns garbage, you get zero diagnostic signal." rfnry-rag has evaluation metrics (EM, F1, LLMJudge, recall, precision) but lacks **per-query retrieval diagnostics** that help users understand why retrieval succeeded or failed.

**What to build:**

- **Retrieval trace per query:** Which methods returned what, at what scores, before and after fusion, before and after reranking. Which chunks were promoted/demoted. What the fusion weights were.
- **Retrieval failure classification:** When grounding fails, categorize why (vocabulary mismatch, chunk boundary problem, entity not indexed, wrong knowledge scope).
- **A/B comparison mode:** Run the same query with two different configurations and compare results side-by-side.
- **Benchmark harness:** Run a set of (query, expected_answer) pairs against a knowledge base and report retrieval recall, precision, and generation accuracy. This already partially exists in evaluation metrics but needs a user-facing workflow.

---

## Feature Priority Matrix

| Priority | Feature | Impact on Relevance | Effort | Research Backing |
|---|---|---|---|---|
| P1 | Context-Aware Routing (RAG/Direct/Hybrid) | **Critical** — without this, rfnry-rag is irrelevant for small corpora | Medium | Anthropic, DeepMind, LaRA, Akita |
| P2 | RAPTOR-Style Summarization Retrieval | **High** — nearly doubles retrieval accuracy | Medium | Li et al. 2025 |
| P3 | Document Expansion at Index Time | **High** — bridges query-document vocabulary gap | Low-Medium | BEIR, Anthropic Contextual |
| P4 | Chunk Position Optimization | **Medium** — easy win for generation quality | Low | Lost in the Middle |
| P5 | Adaptive Retrieval Parameters | **Medium** — right-sizes retrieval per query | Medium | LaRA, SELF-ROUTE |
| P6 | Multi-Hop Iterative Retrieval | **Medium** — solves RAG's biggest failure mode | High | SELF-ROUTE failure analysis |
| P7 | Local Cross-Encoder Reranking | **Medium** — better generalization, lower latency | Low-Medium | BEIR |
| P8 | Retrieval Evaluation & Diagnostics | **Medium** — debuggability, user trust | Medium | Akita, BEIR |

---

## What NOT to Build

The research also tells us what would be wasted effort:

1. **More embedding providers** — The gap between embedding models is small (BEIR, LaRA). Adding 10 providers doesn't move the needle. Focus on how chunks are constructed and retrieved, not which model embeds them.

2. **Agent/chain orchestration** — LangChain territory. rfnry-rag's value is being the retrieval engine, not the workflow orchestrator. The research shows the retrieval quality is what matters.

3. **Fine-tuned dense retrievers** — BEIR showed fine-tuned retrievers only win in-domain and collapse OOD. Hybrid retrieval with BM25 + general embeddings + reranking is more robust.

4. **Longer context handling via chunking tricks** — The trend is clear: context windows are growing faster than corpora. Spending effort on clever chunking to fit more into smaller windows is swimming against the current.

---

## The Strategic Position

rfnry-rag's architecture — modular, protocol-based, no mandatory components — is the right foundation for the long-context era. The research unanimously shows that no single approach wins. The tool that thrives is the one that lets users:

1. **Skip retrieval entirely** when the corpus is small enough (Direct Context mode)
2. **Use smart retrieval** when the corpus is large (multi-path with RAPTOR, fusion, reranking)
3. **Route intelligently** between the two based on corpus size, query type, and model capability
4. **Understand what happened** through diagnostics and evaluation

The danger is not that RAG becomes obsolete — it's that **rigid RAG tools** become obsolete. rfnry-rag's flexibility is its moat, but only if we add the routing intelligence to exploit it.

---

## References

1. Akita, F. (2026). "RAG Est Morto? Contexto Longo, Grep e o Fim do Vector DB Obrigatorio"
2. Anthropic (2024). "Contextual Retrieval" — anthropic.com/news/contextual-retrieval
3. Li, X. et al. (2025). "Long Context vs. RAG for LLMs: An Evaluation and Revisits" — arXiv:2501.01880
4. Li, K. et al. (2025). "LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs" — ICML 2025
5. Li, Z. et al. (2024). "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach" — arXiv:2407.16833
6. Liu, N. et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts" — TACL
7. Thakur, N. et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" — NeurIPS 2021
