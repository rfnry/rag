# Retrieval-Augmented Generation

Large Language Models are impressive, but they only know what they were trained on. Ask one about your company's refund policy, your product specifications, or last week's incident report, and it will either hallucinate an answer or refuse. The model has no access to your private data.

**Retrieval-Augmented Generation** solves this by retrieving relevant information from your documents and feeding it to the model as context at the moment it needs to answer a question. You don't retrain the model. You give it the right documents at the right time.

| Without Retrieval-Augmented Generation | With Retrieval-Augmented Generation |
|-------------|----------|
| Student taking a closed-book exam | Student taking an open-book exam |
| Must rely on memory alone | Can look up the right reference material |
| Makes up answers when unsure | Cites specific sources |
| Knowledge frozen at training time | Always has access to current information |

## Why RAG Over the Alternatives

### vs. Fine-Tuning

Fine-tuning means retraining the model on your data. It's expensive, slow, and the model can still hallucinate. RAG is cheaper, updates instantly when your data changes, and provides source citations.

| | Fine-Tuning | RAG |
|---|---|---|
| Cost | High (GPU hours) | Low (API calls) |
| Update speed | Days to retrain | Instant (re-ingest docs) |
| Hallucination risk | Still present | Reduced (grounded in sources) |
| Source citations | Not possible | Built-in |
| Data freshness | Frozen at training time | Always current |

### vs. Stuffing Everything in the Prompt

You could paste your entire knowledge base into the prompt — but LLMs have context limits (even large ones). A 500-page manual won't fit. And even when it does, models perform worse with too much context: they lose focus, miss details, and cost more per query. RAG retrieves only the most relevant pieces, keeping the context focused and the costs low.

### vs. Traditional Search

Traditional keyword search finds exact word matches. RAG's semantic search understands meaning. "I want my money back" matches "Refund Policy" even though they share no words. This is the difference between a search engine and a system that actually understands what you're asking.

---

# Part I: RAG Principles and Modern Approaches

## The RAG Pipeline

Every RAG system has two phases: **ingestion** (teaching the system your data) and **retrieval + generation** (answering questions). The quality of each phase directly determines the quality of the final answer.

### Ingestion: From Documents to Searchable Knowledge

Before the system can answer questions, it needs to process and store your data. This happens once per document (and again whenever documents change).

```
Your Documents                    What the System Stores
─────────────                    ──────────────────────
Refund Policy.pdf        →       1. Parsed text
Product Specs.pdf        →       2. Semantic chunks + embeddings (vector store)
Installation Manuals     →       3. Full document text (document store)
Historical Emails        →       4. Metadata (source tracking, stats)
```

**Parsing** is the first and most underrated step. PDFs need text extraction (with OCR fallback for scanned pages). Images need vision-model descriptions. Technical drawings need structured analysis. A RAG system is only as good as what it can read.

**Chunking** splits long documents into smaller pieces — typically 300-800 tokens each. This matters because retrieval works by finding the most relevant pieces, and relevance is more precise at the chunk level than the document level.

The quality of chunking goes beyond just splitting text. Three techniques make a significant difference:

- **Contextual chunking** — Each chunk is blind to its surroundings by default. A chunk from page 12 of a manual doesn't know it's from a manual, what section it's in, or what the document is about. By prepending a brief document-level context to each chunk before embedding — "This chunk is from the Pump Model X Installation Manual, Section 3: Specifications" — the embedding captures both the chunk's content AND its position in the knowledge base. This dramatically improves retrieval precision for knowledge bases with many similar documents.

- **Parent-child retrieval** — Small chunks enable precise matching, but they often lack enough context for the LLM to generate a good answer. The solution: embed small chunks for search precision, but when a chunk matches, return its parent — a larger window of surrounding text. You search narrow but retrieve wide. The LLM gets the focused match plus the context it needs to understand and answer accurately.

- **Sparse vectors** — Dense embeddings capture meaning but lose exact terms. Sparse vector models (like SPLADE) produce a complementary representation: a high-dimensional sparse vector where each dimension corresponds to a vocabulary term, weighted by learned importance. This is essentially a smarter version of BM25 — it captures keyword importance through learning rather than raw term frequency. When stored alongside dense vectors in the same vector database, sparse and dense search can run as a single hybrid query, combining semantic understanding with keyword precision natively.

**Full-document preservation** stores the complete, unbroken text of every document alongside the chunks. Chunking enables precise semantic search, but it necessarily fragments the original text — a part number can be split across chunk boundaries, a spec table separated from its diagram. Keeping the full document available means the system can always fall back to exact, structural search when chunk-level search isn't enough.

**Embedding** converts each chunk into a high-dimensional vector — a mathematical representation of its meaning. Similar texts produce similar vectors, enabling similarity search. The choice of embedding model determines what "similar" means: general-purpose models capture broad semantics, while domain-specific models can better distinguish technical terminology.

**Storage** goes to two complementary stores. A **vector store** holds chunks, their embeddings, and optionally sparse vectors — optimized for fast nearest-neighbor search across millions of points. A **document store** holds the complete, unbroken text of every ingested document — optimized for full-text ranking and exact substring matching. Metadata (source IDs, page numbers, tags) is stored alongside both for filtering and attribution.

These two stores serve fundamentally different purposes:

| | Vector Store | Document Store |
|---|---|---|
| **Contains** | Chunks (~500 tokens) + embedding vectors | Complete, unbroken documents |
| **Search method** | Nearest-neighbor similarity in vector space | Full-text ranking + substring matching |
| **Result shape** | Fixed-size fragments | Variable-length excerpts extracted at query time |
| **Best for** | "What filters reduce allergens?" (conceptual) | "FBD-20254-MERV13 pressure drop" (exact) |
| **Context** | Only what fits in the chunk window | Full document structure, cross-references, tables |

They're not competing — they're complementary. The vector store excels at understanding what you mean. The document store excels at finding exactly what you said. A production pipeline searches both in parallel and fuses the results.

### The Retrieval Problem: Finding the Right Information

Retrieval is where RAG systems succeed or fail. A perfect generation model with bad retrieval will produce confident, well-written wrong answers. A mediocre generation model with excellent retrieval will produce grounded, useful responses.

The challenge: given a natural language query, find the most relevant information across your entire knowledge base. There are fundamentally different approaches to search, each with distinct strengths. Understanding what each one is good at — and bad at — is essential for building a system that handles diverse query types.

#### Semantic Search (Vector Similarity)

The query is embedded into the same vector space as the stored chunks. The system finds chunks whose vectors are closest, meaning they're about similar topics.

```
Query: "Can I get my money back?"
                │
                ▼ embed
         [0.12, -0.45, 0.78, ...]
                │
                ▼ nearest-neighbor search
         Matches: "Refund Policy: full refunds within 30 days..."
```

**Strengths:** Understands meaning regardless of wording. "Can I get my money back?" matches "Refund Policy" even though they share no words. Handles synonyms, paraphrasing, and conceptual similarity naturally.

**Weaknesses:** Poor at exact terms. Searching for part number "FBD-20254-MERV13" via embeddings may return chunks about filters in general rather than that specific model. Embeddings capture semantic neighborhoods, not literal strings.

#### Keyword Search (BM25)

BM25 is a term-frequency algorithm. It tokenizes both the query and the stored text, then ranks documents by how often query terms appear (weighted by rarity). It's the algorithm behind most traditional search engines.

**Strengths:** Respects exact terms. If you search for "FBD-20254", documents containing that exact string score highest. Fast, deterministic, and well-understood.

**Weaknesses:** No understanding of meaning. "Money back" does not match "refund." Sensitive to morphology — "running" may not match "ran" without stemming. Searches over tokenized words, so hyphenated identifiers like "FBD-20254-MERV13" may be broken into separate tokens.

#### Full-Document Search (Text Search + Substring Matching)

Full-document search operates on complete, unbroken document text rather than chunks. Two techniques work together:

- **Ranked full-text search** uses the same term-frequency principles as BM25 but operates on the entire document. The system tokenizes the full text and maintains an inverted index for fast ranked retrieval.
- **Substring matching** finds exact character sequences anywhere in the document — including within hyphenated identifiers, model numbers, and multi-word technical terms that tokenization would break apart.

When a match is found in a large document, the system extracts a **contextual excerpt** — a window of text centered on the match, expanded to natural paragraph boundaries. This gives the LLM enough surrounding context without overwhelming it with the full document.

**Strengths:** Finds exact terms that both semantic search and BM25 can miss. Preserves full document context around matches. Handles part numbers, model codes, and technical identifiers reliably.

**Weaknesses:** No semantic understanding — the query must contain terms that literally appear in the document. Lower recall for conceptual queries.

#### The Tokenization Problem

There's a subtle but critical distinction between BM25, full-text ranking, and substring matching that's easy to miss: **BM25 and full-text ranking both tokenize**.

Tokenization splits text into individual words. The identifier "FBD-20254-MERV13" gets broken into `["fbd", "20254", "merv13"]` — three separate tokens. A search for the original string finds documents containing those individual tokens, but doesn't require them to be adjacent or hyphenated. This means:

- BM25 over chunks: finds chunks containing "fbd" and "20254" and "merv13" as separate words. May match chunks that mention these terms in unrelated contexts.
- Full-text ranking over documents (e.g., PostgreSQL `ts_rank`): same tokenization behavior, but operates on full documents so there's more context around matches.
- **Substring matching** (e.g., `ILIKE`, `str.find()`): treats "FBD-20254-MERV13" as a single character sequence. The only technique that finds the literal string exactly as written.

This is why a production system needs all three layers. Each one catches what the others miss:

| Search Technique | Operates On | Tokenizes? | "FBD-20254-MERV13" |
|---|---|---|---|
| Semantic (dense vectors) | Chunks | No (embeds meaning) | Finds "filter" chunks, not this specific model |
| Sparse vectors / BM25 | Chunks | Yes (into words) | Finds chunks with "fbd" + "20254" + "merv13" separately |
| Full-text ranking | Complete documents | Yes (into words) | Same tokenization, but more surrounding context |
| Substring matching | Complete documents | No (raw characters) | Exact match — finds the literal string |

#### Why Hybrid Matters

No single search technique handles all query types well:

| Query Type | Semantic | Sparse / BM25 | Full-Text Ranking | Substring |
|-----------|----------|------|------|------|
| "What filters work for allergies?" | Best | Poor | Poor | Poor |
| "filter specifications" | Good | Good | Good | Good |
| "FBD-20254-MERV13" | Poor | Partial (tokenized) | Partial (tokenized) | Best |
| "20x25x1 MERV 13 pressure drop" | Good | Good | Good | Best |
| "0.25 inches WG" | Poor | Partial | Partial | Best |

The best approach is to run **multiple search paths in parallel** and merge the results. This is called multi-path retrieval, and it produces consistently better results across diverse query types than any single search strategy alone.

#### Knowledge Graphs (Relational Search)

The search techniques above — semantic, keyword, full-text, substring — all treat documents as independent pieces of text. But knowledge bases have structure. A motor connects to a breaker. A breaker is controlled by a PLC. A PLC references a wiring diagram on page 14.

A knowledge graph stores these relationships explicitly as entities (nodes) and connections (edges). When a query asks "what connects to Motor M1?", the graph doesn't search for documents containing the words "Motor M1" — it traverses the actual connections: Motor M1 → powered by Breaker CB3 → controlled by PLC DR340.

```
Query: "What connects to Motor M1?"

        Motor M1
        ├── powered_by → Breaker CB3
        │                └── controlled_by → PLC DR340
        ├── described_in → Page 12, Electrical Panel Drawing
        └── spec_sheet → Page 4, Motor Specifications
```

**Strengths:** Answers relational questions ("what feeds this panel?", "which components share this circuit?") that no amount of text search can answer reliably. Enables multi-hop reasoning — finding entities two or three connections away from the starting point.

**Weaknesses:** Requires structured entity extraction during ingestion. Only as complete as the entities and relationships that were extracted. Adds infrastructure (a graph database) and ingestion complexity.

Knowledge graphs complement text search. Text search finds relevant documents. Graph search finds relevant relationships. Together, they handle both "what does the manual say about Motor M1?" and "what other components depend on Motor M1?"

### Query Rewriting: Searching Smarter

All search techniques above operate on the query as the user wrote it. But users don't always phrase queries the way documents phrase answers. A technician asking "why is it overheating?" may need the document that says "thermal protection activates when ambient temperature exceeds 40°C." The words are completely different.

Query rewriting reformulates the query before search to bridge this gap. It adds one LLM call before retrieval — the model reads the original query and produces better search queries. Three strategies are common:

**Hypothetical Document Embeddings (HyDE)** — Instead of embedding the question, generate a hypothetical answer and embed that. The embedding of "Motor M1 typically overheats when ambient temperature exceeds 40°C, the cooling fan fails, or the VFD drive frequency drops below rated speed" is much closer to the actual document chunk than the embedding of "why is it overheating?" This is the most effective strategy when the language of questions and answers differs significantly — which is common in technical domains.

**Multi-query expansion** — Generate 2-3 alternative phrasings of the query. "Why is it overheating?" becomes ["motor overheating causes", "thermal protection activation conditions", "cooling system failure modes"]. All variants are searched, and results are fused. This catches documents that match different phrasings of the same intent.

**Step-back prompting** — Generate a broader version of the query. "Why is Motor M1 overheating in Panel 3?" becomes "Motor M1 operating specifications and thermal limits." The broader query retrieves background context that the specific query might miss.

Query rewriting is **proactive** — it improves what retrieval finds. This is complementary to the relevance gate, which is **reactive** — it catches cases where retrieval failed and asks the user to be more specific. Good query rewriting reduces how often the relevance gate needs to intervene.

### Multi-Path Retrieval and Fusion

Running multiple search strategies produces multiple ranked lists of results. These need to be merged into a single ranked list. **Reciprocal Rank Fusion (RRF)** is the standard approach: each result receives a score based on its rank position across all lists, and results that appear in multiple lists get boosted.

```
Query: "FBD-20254 pressure specifications"

Path 1 (Semantic):  [Chunk A (0.82), Chunk C (0.71), Chunk D (0.65)]
Path 2 (BM25):      [Chunk B (4.2), Chunk A (3.8), Chunk E (2.1)]
Path 3 (Full-text): [Doc excerpt F (0.9), Doc excerpt B (0.7)]

                    │
                    ▼ Reciprocal Rank Fusion
                    │
Fused:  [A, B, F, C, D, E]
        (A boosted: appeared in semantic + BM25)
        (B boosted: appeared in BM25 + full-text)
```

After fusion, an optional **reranking** step can further improve precision. A cross-encoder model takes each (query, result) pair and produces a relevance score that's more accurate than any single retrieval method — but slower, which is why it runs only on the already-filtered top results.

### Quality Assurance: Grounding and Confidence

Retrieval finds candidate information. But not every query has a clear answer in your data. A production RAG system needs multiple layers of quality assurance to avoid confidently presenting bad answers.

**Score gating** sets a minimum relevance threshold. If the best retrieval results score below the threshold, the system knows it hasn't found good context and should not attempt to generate an answer.

**Relevance gating** uses an LLM to judge whether the retrieved context actually answers the query. This catches cases where retrieval finds topically related but not actually useful content — a common failure mode that score thresholds alone miss. The relevance gate can also identify when a query is ambiguous and generate clarifying questions instead of guessing.

**Confidence scoring** combines multiple signals — retrieval scores, relevance judgment, source coverage, answer consistency — into a composite confidence score. This enables downstream systems to route high-confidence answers to automation and low-confidence answers to human review.

```
"Can I get a refund?"           → High confidence → Auto-respond
"What's your CEO's birthday?"   → Low confidence  → Escalate to human
"What size filter for my unit?" → Medium + ambiguous → Ask clarifying question
```

### What Defines a Modern RAG System

The field has matured. Here are the approaches that define production-grade RAG:

**Multi-path retrieval** — Running semantic, keyword, and full-document search in parallel and fusing results. Single-path retrieval leaves too many query types poorly served.

**Context-aware chunking** — Chunks that know where they came from. Contextual chunking prepends document-level context before embedding. Parent-child retrieval searches narrow but retrieves wide. These techniques close the gap between precise matching and useful context.

**Hybrid dense + sparse search** — Dense embeddings capture meaning. Sparse vectors capture keyword importance. Running both in a single query combines semantic understanding with term-level precision, replacing separate BM25 indexes with a native, scalable solution.

**Dual-store architecture** — Vector stores for chunks and embeddings (semantic search). Document stores for complete text (exact search). Both are searched in parallel and fused. This ensures that no query type — conceptual, keyword, or exact — is left underserved.

**Query rewriting** — Reformulating queries before search to bridge the gap between how users ask questions and how documents phrase answers. HyDE, multi-query expansion, and step-back prompting each address a different failure mode. The best systems make this opt-in — it adds an LLM call per query, so consumers control the cost-quality tradeoff.

**Knowledge graphs** — Storing entity-relationship structures extracted from documents and traversing them during retrieval. When the knowledge base describes interconnected systems (electrical panels, mechanical assemblies, software architectures), graph traversal answers relational questions that text search cannot.

**Structured analysis** — Using vision models to understand technical drawings, diagrams, and scanned documents. Pages are classified by type (schematic, spec sheet, wiring diagram) and entities are extracted (component names, ratings, connections). This makes visual content searchable alongside text — and feeds the knowledge graph with structured relationships.

**Grounding and verification** — Not trusting retrieval results blindly. Score gates, relevance gates, and confidence scoring create multiple checkpoints that prevent hallucination and enable graceful degradation when the knowledge base doesn't contain the answer.

**Source attribution** — Every answer traces back to specific sources, pages, and sections. This makes answers verifiable and auditable — a requirement in any professional context.

**Knowledge lifecycle management** — Tracking which documents have been ingested, detecting when embedding models change (making old vectors stale), and providing tools to update, re-ingest, and remove sources. A RAG system is only useful if its knowledge base is current.

---

# Part II: The rfnry-rag SDK

## What It Solves

Most RAG implementations start simple — embed some chunks, search by similarity, feed results to an LLM. This works for demos and simple Q&A, but breaks down when the knowledge base contains technical content: manuals with part numbers, specifications with exact values, drawings with cross-referenced components.

rfnry-rag is built for this reality. It provides a complete RAG pipeline that handles both conceptual questions ("what filters reduce allergens?") and precise lookups ("what's the pressure drop for FBD-20254-MERV13?") through the same interface. You ingest your documents once, and the SDK builds multiple searchable representations automatically — so you don't have to choose between semantic understanding and exact matching.

The SDK is composable. Every integration point — embedding providers, generation providers, vector stores, document stores, reranking strategies — is defined as a protocol interface. You bring your own providers, configure the pipeline depth you need (from retrieval-only to full generation with quality gates), and the SDK orchestrates it.

## How Retrieval Works

The SDK searches across two stores in parallel, each providing search techniques the other cannot:

### Vector Store (chunks)

The vector store holds chunks and their embeddings. Two search techniques run as a single hybrid query:

- **Dense vector search (semantic)** — Finds chunks whose meaning is similar to the query. Handles paraphrasing, synonyms, and conceptual questions. Does not tokenize — operates on continuous vector similarity.
- **Sparse vector search (learned keyword)** — Finds chunks containing terms that match the query, weighted by learned importance. Replaces traditional BM25 with a smarter, model-based approach that runs natively inside the vector database. Tokenizes — so hyphenated identifiers like "FBD-20254-MERV13" are broken into separate terms.

The vector database fuses dense and sparse scores internally via reciprocal rank fusion, returning a single ranked list. When a matching chunk has a parent reference, the larger parent context is retrieved alongside it — precise matching with rich surrounding context.

### Document Store (full documents)

The document store holds the complete, unbroken text of every ingested document. Two search techniques run per query:

- **Ranked full-text search** — Scores documents by term frequency and relevance. In the PostgreSQL backend this uses `ts_rank` with `tsvector/tsquery` — PostgreSQL's own ranking algorithm (similar in purpose to BM25 but a different formula). In the filesystem backend this uses BM25 over full document content. Both tokenize, so they share the same hyphenated-identifier limitation as sparse vectors.
- **Substring matching** — Finds exact character sequences anywhere in the document text. In PostgreSQL this uses `ILIKE` with trigram indexing. In the filesystem backend this uses direct string matching. **Does not tokenize** — this is the only search technique in the entire pipeline that finds "FBD-20254-MERV13" as a single unbroken string.

When a match is found in a large document, the SDK extracts a contextual excerpt centered on the match rather than returning the full text.

### How They Fit Together

| Technique | Where | Tokenizes? | Best for |
|---|---|---|---|
| Dense vector (semantic) | Vector store | No | Conceptual questions, synonyms |
| Sparse vector (learned keyword) | Vector store | Yes | Known terms, product names |
| Full-text ranking (`ts_rank` / BM25) | Document store | Yes | Keyword matching with full document context |
| Substring matching (`ILIKE` / string search) | Document store | No | Exact identifiers, part numbers, precise values |

### Query Rewriting (pre-retrieval)

Before any search runs, the SDK can optionally rewrite the query to improve retrieval quality. This adds one LLM call per query — an opt-in cost-quality tradeoff.

Three strategies are available:

- **HyDE** generates a hypothetical answer and searches using that embedding instead of the question's. Most effective when question language and document language differ (common in technical domains).
- **Multi-query** generates 2-3 query variants that capture different phrasings of the same intent. All are searched, results fused.
- **Step-back** generates a broader version of the query to retrieve background context the specific query would miss.

When enabled, rewriting runs before all search paths. The rewritten queries feed into the same pipeline — vector store, document store, and graph store all benefit from the improved queries.

### Knowledge Graph (relational search)

For structured technical content — electrical schematics, mechanical assemblies, PLC programs — the SDK can maintain a knowledge graph alongside the vector and document stores. Entities and relationships extracted during structured analysis (components, connections, specifications) are stored as graph nodes and edges.

During retrieval, the graph store runs as a fourth concurrent search path. Entity references are extracted from the query, matching nodes are found, and connected entities within N hops are returned. Graph results are formatted as entity cards (name, type, properties, connections) and fused into the same reciprocal rank fusion pipeline as all other results.

This answers relational questions — "what connects to Motor M1?", "which breakers feed this panel?" — that no amount of text search can answer reliably.

### Tree Search (reasoning-based retrieval)

All the search techniques above work by matching content — finding text that's similar to the query (semantic), contains the right terms (keyword), or matches exactly (substring). But they share a limitation: they treat documents as flat collections of chunks or text. A 200-page annual report gets split into hundreds of independent fragments, and the system has no concept of the document's natural hierarchy — chapters, sections, subsections.

**Tree search** takes a fundamentally different approach. During ingestion, the SDK builds a hierarchical tree index that represents the document's natural structure — detecting or extracting its table of contents, mapping sections to page ranges, and generating summaries at each level. At query time, instead of similarity search, an LLM reads the tree structure and *reasons* about which sections are relevant — the same way a human would scan a table of contents before flipping to the right chapter.

The retrieval uses a BAML tool-use loop where the LLM can:
- **Fetch pages** — read specific pages to gather evidence
- **Drill down** — zoom into a subtree for deeper inspection
- **Resolve** — declare the final set of relevant pages

Simple queries resolve in a single step. Complex documents get multi-step traversal. Cost is bounded by a configurable maximum step count.

**Why this matters:** Vector search finds content that *looks similar* to the query. Tree search finds content that *is relevant*. For long, structured documents — financial reports, legal contracts, technical manuals, compliance filings — tree search consistently finds the right sections, even when the query uses completely different language than the document.

| Scenario | Vector Search | Tree Search | Both |
|----------|--------------|-------------|------|
| Short documents, simple questions | Best | Overkill | — |
| Long structured docs (reports, filings, contracts) | Weak | Best | Ideal |
| High accuracy matters, cost secondary | — | Best | Ideal |
| Low latency, high throughput | Best | Slow | — |
| Explainable retrieval needed | Weak | Best | — |

Tree search is opt-in at both ingestion time (`tree_index=True`) and query time (via `TreeSearchConfig`). When enabled alongside vector search, tree results are merged with all other search paths via reciprocal rank fusion — the generation layer gets the best of both worlds.

### How They All Fit Together

| Technique | Where | Tokenizes? | Best for |
|---|---|---|---|
| Dense vector (semantic) | Vector store | No | Conceptual questions, synonyms |
| Sparse vector (learned keyword) | Vector store | Yes | Known terms, product names |
| Full-text ranking (`ts_rank` / BM25) | Document store | Yes | Keyword matching with full document context |
| Substring matching (`ILIKE` / string search) | Document store | No | Exact identifiers, part numbers, precise values |
| Graph traversal (Cypher) | Graph store | No | Relational queries, connected components |
| Tree search (LLM reasoning) | Metadata store | No | Long structured documents, section-level relevance |

Results from all active stores are merged using reciprocal rank fusion. An optional reranking pass then re-scores the top results for additional precision. Source-type weighting lets you tune the balance — boosting manuals over transcripts, for example — so the pipeline reflects your domain's priorities.

```
                              Query Rewriting (optional)
                                          │
            ┌──────────────┬──────────────┼──────────────────────┬──────────────┐
            │              │              │                      │              │
      Vector Store   Document Store  Graph Store          Tree Search    Enrich Store
    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐  ┌──────────────┐ ┌────────────┐
    │ Dense        │ │ Full-text   │ │ Entity match │  │ LLM reasons  │ │ Structured │
    │ Sparse       │ │ Substring   │ │ N-hop        │  │ over tree    │ │ field      │
    │ + Parent exp │ │ + Excerpts  │ │ + Cards      │  │ + Tool loop  │ │ filtering  │
    └──────┬───────┘ └──────┬──────┘ └──────┬───────┘  └──────┬───────┘ └─────┬──────┘
            │               │               │                  │               │
            └───────────────┴───────────────┼──────────────────┴───────────────┘
                                            │
                                            ▼
                                Reciprocal Rank Fusion
                                            │
                                            ▼
                                    Reranking (optional)
                                            │
                                            ▼
                                Grounding Gates → Generation → Answer with sources
```

## How Ingestion Works

Documents go through a multi-track pipeline from a single parse step:

**Chunk track** — Parsed pages are split into semantic chunks. Each chunk is enriched with document-level context (source name, section, document type) before embedding, so the resulting vectors capture both the chunk's content and its position in the knowledge base. Dense embeddings go to the vector store alongside sparse vectors for hybrid search. Parent chunk references are preserved, enabling narrow-search-wide-retrieve at query time.

**Document track** — Parsed pages are concatenated into a full rendered document and stored in the document store. This feeds full-document search. The full text is preserved before chunking, so no information is lost.

**Tree track (opt-in)** — When `tree_index=True` is passed to ingestion, the SDK builds a hierarchical tree index of the document's natural structure. It detects the table of contents (or extracts one via LLM if none exists), maps sections to page ranges, verifies section positions, splits oversized sections, and optionally generates section summaries. The tree index and page content are stored in the metadata store for query-time tree search. This adds multiple LLM calls during ingestion — a deliberate cost-quality tradeoff for documents where section-level retrieval matters.

```
File → Parse (PDF/text/image/vision) → ┬─→ Chunk → Contextualize → Embed (dense + sparse) → Vector Store
                                       ├─→ Full text → Document Store
                                       └─→ Tree index (opt-in) → TOC → Structure → Summaries → Metadata Store
```

For structured content — electrical schematics, mechanical drawings, wiring diagrams — a specialized ingestion path uses a vision model to analyze each page: classifying page types, extracting entities (component names, ratings, connections), detecting tables, and discovering cross-page relationships. The full analyzed content is stored in the vector store (as embeddings), the document store (as searchable text), and optionally the graph store (as entity nodes and relationship edges for relational queries).

The SDK supports two document store backends:

**Database backend** — Stores full document text in PostgreSQL with automatic full-text indexing and trigram-indexed substring matching. This is the simplest option: no new infrastructure needed, it reuses the database connection you already provide for metadata.

**Filesystem backend** — Stores documents as markdown files organized in a directory tree by knowledge scope and document type. The knowledge base becomes human-browsable — you can navigate it, grep it, and read files directly from the command line. Useful when you want visibility into exactly what the system has ingested.

## How Quality Gates Work

Not every query has a clear answer in your data, and a system that confidently gives wrong answers is worse than one that says "I don't know." The SDK provides three layers of quality assurance:

**Score gate** filters retrieval results below a configurable similarity threshold. If nothing passes, the system reports low confidence rather than generating from weak context.

**Relevance gate** uses an LLM to judge whether the retrieved context actually answers the query — not just whether it's topically related. When the query is ambiguous, the relevance gate generates clarifying questions with suggested options instead of guessing.

**Confidence scoring** combines retrieval quality, relevance judgment, source coverage, and answer consistency into a composite score. Downstream systems use this to decide: automate, escalate to a human, or ask the user to be more specific.

The gates are composable. A retrieval-only deployment skips generation entirely. A high-throughput deployment can disable the relevance gate and rely on score thresholds alone. The defaults are conservative — the system prefers silence over hallucination.

## Knowledge Lifecycle

A RAG system is only useful if its knowledge base is current. The SDK tracks the full lifecycle of ingested content:

- **Duplicate detection** — File hashes prevent re-ingestion of identical documents.
- **Embedding model tracking** — When the configured embedding model changes, existing vectors are flagged as stale. The system warns about stale sources and supports selective re-ingestion.
- **Source management** — Sources can be listed, inspected, updated, and removed. Removal cascades to all stores (vector, document, and graph).
- **Hit tracking** — Retrieval hits are recorded per source, distinguishing grounded from ungrounded hits. This enables data-driven decisions about which sources are actually useful.

## Composability

The SDK is configured through composable dataclass objects. Every component is optional — you compose exactly the pipeline your use case requires:

```
Server Config
├── Persistence
│   ├── Vector Store          (chunks, dense + sparse embeddings)
│   ├── Metadata Store        (source tracking, stats)
│   ├── Document Store        (full document text — PostgreSQL or filesystem)
│   └── Graph Store           (entity-relationship graph — Neo4j)
├── Ingestion
│   ├── Dense Embeddings      (semantic embedding provider)
│   ├── Sparse Embeddings     (keyword embedding provider — enables hybrid search)
│   ├── Vision                (vision provider for images/PDFs)
│   ├── Contextual chunking   (prepend document context before embedding)
│   └── Chunk settings        (child size, parent size, overlap)
├── Retrieval
│   ├── Query Rewriter        (optional pre-retrieval query expansion)
│   ├── Reranker              (optional cross-encoder)
│   ├── Parent expansion      (return parent chunk for wider context)
│   ├── Source weights         (boost manuals over transcripts)
│   └── Top-k                 (how many results to return)
├── Tree Indexing (optional)
│   ├── LLM Model             (model for TOC detection, structure extraction, summaries)
│   ├── TOC scan pages         (how many pages to scan for table of contents)
│   ├── Max pages/tokens per node (triggers recursive splitting)
│   └── Generate summaries/description (opt-in per feature)
├── Tree Search (optional)
│   ├── LLM Model             (model for reasoning-based tree traversal)
│   ├── Max steps              (cap on tool-use loop iterations)
│   └── Max context tokens     (budget for accumulated page content)
└── Generation
    ├── LLM Provider          (generation model)
    ├── System prompt          (instructions for the model)
    ├── Grounding gate         (score threshold)
    └── Relevance gate         (LLM-based judgment)
```

A minimal deployment needs only a vector store and embeddings. A full deployment uses all three retrieval paths, reranking, relevance gating, and confidence scoring. The SDK handles wiring, lifecycle, and orchestration — you focus on choosing the right components for your domain.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Embedding** | A high-dimensional vector representing the meaning of a text. Similar texts produce similar vectors. |
| **Dense Vector** | A fixed-length vector where every dimension carries a value. Captures semantic meaning. |
| **Sparse Vector** | A high-dimensional vector where most values are zero. Non-zero values correspond to specific vocabulary terms weighted by learned importance. |
| **Vector Store** | A database optimized for fast nearest-neighbor search across embedding vectors, supporting both dense and sparse representations. |
| **Document Store** | Storage for full, unbroken document text, enabling exact search and structural queries. |
| **Chunk** | A fragment of a larger document (~300-800 tokens), the unit of semantic search. |
| **Contextual Chunking** | Prepending document-level context (source name, section, type) to each chunk before embedding, so the vector captures both content and position. |
| **Parent-Child Retrieval** | Embedding small chunks for precise matching, but returning the larger parent chunk for richer context. Search narrow, retrieve wide. |
| **BM25** | A term-frequency ranking algorithm. Scores documents by how often query terms appear, weighted by rarity. |
| **Hybrid Search** | Combining dense (semantic) and sparse (keyword) vector search in a single query for broader coverage. |
| **Semantic Search** | Finding results by meaning similarity, not keyword matching. "Money back" finds "Refund Policy." |
| **Full-Text Search** | Searching complete document text for ranked term matches and exact substrings. |
| **Reciprocal Rank Fusion** | A technique for merging multiple ranked result lists into a single ranking. |
| **Reranking** | A second-pass scoring using a cross-encoder model to improve retrieval precision. |
| **Grounding** | Ensuring the AI's answer is supported by retrieved documents, not fabricated. |
| **Relevance Gate** | An LLM-based check that verifies retrieved context actually answers the query. |
| **Confidence Score** | A composite metric combining retrieval quality, relevance, and coverage signals. |
| **Query Rewriting** | Reformulating a query before search to improve retrieval. Strategies include HyDE, multi-query expansion, and step-back prompting. |
| **HyDE** | Hypothetical Document Embeddings — generating a hypothetical answer and embedding that instead of the question, bridging the question-answer language gap. |
| **Knowledge Graph** | A database of entities (nodes) and relationships (edges) that enables traversal-based retrieval for relational queries. |
| **Graph Traversal** | Following connections between entities in a knowledge graph. N-hop traversal finds entities within N relationship steps of a starting point. |
| **Entity Deduplication** | Recognizing that the same entity mentioned across multiple documents is one node in the graph, not many. |
| **Multi-Path Retrieval** | Running multiple search strategies in parallel and fusing results for broader coverage. |
| **Tree Index** | A hierarchical structure representing a document's sections, page ranges, and summaries — built from the table of contents or extracted via LLM. |
| **Tree Search** | Reasoning-based retrieval where an LLM navigates a document's tree structure to find relevant sections, instead of using similarity search. |
| **Hallucination** | When an AI confidently states something unsupported by the provided context. |
| **Source Attribution** | Tracing every answer back to specific documents, pages, and sections. |

---

See also: [Reasoning SDK documentation](reasoning.md) · [SDK reference](../src/rfnry-rag/retrieval/README.md) · [Main README](../README.md)
