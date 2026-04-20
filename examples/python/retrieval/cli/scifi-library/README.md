# CLI Example: Sci-Fi Library

A walkthrough showing how to use the `rag` CLI to build a searchable knowledge base from a series of science fiction novels — then use it to research and write new content in the same universe.

The `scifi-library/` folder contains 8 books from the "Sands of Araxis" saga. Each is a markdown file with several chapters (~800-1200 words each).

## 1. Install and Configure

```bash
uv add "rfnry-rag[cli]"
rfnry-rag retrieval init
```

Start Qdrant (if not already running):

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Edit `~/.config/rfnry_rag/.env`:

```bash
VOYAGE_API_KEY=pa-...
ANTHROPIC_API_KEY=sk-ant-...
```

Edit `~/.config/rfnry_rag/config.toml`:

```toml
[persistence]
vector_store = "qdrant"
url = "http://localhost:6333"
collection = "scifi-library"

[ingestion]
embeddings = "voyage"
model = "voyage-3"

[retrieval]
top_k = 5
reranker = "voyage"
reranker_model = "rerank-2.5-lite"
rewriter = "hyde"
rewriter_provider = "anthropic"
rewriter_model = "claude-haiku-4-5-20251001"

[generation]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
system_prompt = "You are a writing assistant for science fiction. Use only the provided context from the Araxis saga to answer questions. Cite book titles and chapter numbers when referencing specific events."
grounding_enabled = true
grounding_threshold = 0.5
relevance_gate_enabled = true
relevance_gate_provider = "anthropic"
relevance_gate_model = "claude-haiku-4-5-20251001"
```

**What this enables:**
- **Reranking** — a cross-encoder re-scores retrieval results for better semantic matching
- **Query rewriting (HyDE)** — rewrites vague queries ("that thing about the caves glowing") into hypothetical document passages before retrieval, bridging the gap between how you remember a detail and how it was actually written
- **Grounding gates** — prevents the LLM from inventing lore that doesn't exist in the books

## 2. Verify Setup

```bash
$ rfnry-rag retrieval --pretty status

Config: valid
Qdrant: connected
Sources: 0

Ready.
```

Global flags (`--pretty`, `--json`, `--config`) go before the command name.

## 3. Ingest the Books

Ingest all 8 books with a knowledge namespace and source type:

```bash
$ rfnry-rag retrieval ingest scifi-library/book-01-sands-of-araxis.md -k araxis --source-type book
{
  "source_id": "a76d8591-a640-4180-a27c-c7c98585716f",
  "chunk_count": 12,
  "status": "completed"
}

$ rfnry-rag retrieval ingest scifi-library/book-02-the-chronodust-war.md -k araxis --source-type book
{
  "source_id": "5ebc84a5-2ae1-44ac-a731-a2fc985264d3",
  "chunk_count": 9,
  "status": "completed"
}
```

Or ingest all at once in a loop:

```bash
for f in scifi-library/book-*.md; do
  rfnry-rag retrieval ingest "$f" -k araxis --source-type book
done
```

Total: 90 chunks across 8 books.

You can also ingest raw text directly — useful for adding notes, outlines, or world-building fragments:

```bash
$ rfnry-rag retrieval ingest --text "The Araxeen calendar divides the year into 13 months of 28 days each, named after the 13 known sandweaver migration routes. The first month, Deepcurrent, marks the beginning of the breeding season when juvenile weavers emerge from the polar caves." -k araxis --source-type notes
{
  "source_id": "b7e41f93-...",
  "chunk_count": 1,
  "status": "completed"
}
```

## 4. Retrieve Raw Chunks

`rfnry-rag retrieval retrieve` returns source chunks without LLM generation. With reranking enabled, results are re-scored by a cross-encoder — the top result scores 0.79 (vs ~0.52 without reranking):

```bash
$ rfnry-rag retrieval retrieve "What is chronodust and where does it come from?" -k araxis --min-score 0.4
{
  "chunks": [
    {
      "content": "\"The chronodust is not merely a fuel,\" Dhamira told him as they sat in the cool darkness of a cave system beneath the Shield Wall. \"It is the waste product of the sandweavers' metabolism. The sandweavers themselves are the navigators — they fold the substrate of space as they move through the deep sand. The Hegemony harvests the residue and calls it power. They do not understand what they are handling.\"",
      "score": 0.79,
      "source_type": "book"
    },
    {
      "content": "Araxis. The desert world. The only known source of chronodust — the substance that made interstellar navigation possible. Whoever controlled Araxis controlled the flow of commerce across twelve star systems.",
      "score": 0.74,
      "source_type": "book"
    },
    {
      "content": "\"They did not merely navigate through folds,\" Tallis explained to the assembled council of House Vandris and the Keepers. \"They created the folds. The chronodust is not a byproduct of their movement. It is the material they use to repair tears in space-time — the same tears that destroyed their origin system. They are not miners. They are healers.\"",
      "score": 0.71,
      "source_type": "book"
    },
    {
      "content": "It was in the Endless that the sandweavers lived — creatures sixty meters long that burrowed through the dunes, their passage generating the very chronodust that powered civilization. To harvest the dust, you had to enter the domain of the sandweavers. Many who entered did not return.",
      "score": 0.70,
      "source_type": "book"
    },
    {
      "content": "This revelation reframed everything. The Hegemony had been harvesting chronodust — the sandweavers' healing substrate — and burning it for fuel. Every fold jump made by a navigator consumed material that the sandweavers had produced to maintain the structural integrity of local space-time.\n\n\"We are burning the bandages,\" Dhamira said quietly. \"And the wound is still open.\"",
      "score": 0.67,
      "source_type": "book"
    }
  ]
}
```

The `--min-score 0.4` flag filters out low-relevance results. Without it, you'd also get chunks that mention chronodust tangentially.

Force `--json` explicitly (useful when scripting in a TTY):

```bash
$ rfnry-rag retrieval --json retrieve "sandweaver biology" -k araxis | jq '.chunks[0].content'
"It was in the Endless that the sandweavers lived — creatures sixty meters long..."
```

## 5. Vague Queries (HyDE Query Rewriting)

Writers rarely remember exact quotes. With HyDE enabled, you can search the way you think — the CLI rewrites your vague recollection into a hypothetical passage before retrieval:

```bash
$ rfnry-rag retrieval retrieve "that thing about the caves glowing" -k araxis --min-score 0.3
{
  "chunks": [
    {
      "content": "The cave systems were not mere tunnels. They were a living network, maintained by juvenile sandweavers too small to surface. The walls pulsed with bioluminescence — chronodust in its raw, unprocessed form, glowing with a faint amber light that shifted and rippled like liquid...",
      "score": 0.69,
      "source_type": "book"
    }
  ]
}
```

Without HyDE, "that thing about the caves glowing" would return weak semantic matches. With HyDE, the query is rewritten into something like "bioluminescent cave systems with glowing walls" before searching, finding the exact passage about chronodust bioluminescence.

## 6. Query with AI Generation

`rfnry-rag retrieval query` retrieves relevant chunks and generates a synthesized answer via LLM. With grounding gates enabled, the answer shows grounding confidence:

```bash
$ rfnry-rag retrieval --pretty query "Describe the three pillars of the Cognitive Order and how they compare to the Araxeen Resonance practice" -k araxis

Based on the provided context from the Araxis saga, the Cognitive Order teaches
three pillars of mental discipline:

1. **Pattern** - The ability to perceive systems within systems and see hidden
   structures that govern markets, ecosystems, and human behavior. A trained
   Pattern reader can quickly identify power dynamics, fear, and potential
   betrayal in social situations (Keepers of the Fold, Chapter 1).

2. **Control** - Mastery over biological processes including heart rate,
   adrenaline, pain perception, and body temperature through breathing and
   neural feedback loops.

3. **Projection** - Influencing others through voice modulation,
   micro-expression, and linguistic patterns. It amplifies existing
   inclinations rather than overriding will.

The Araxeen Keepers reject this division entirely. Their single discipline,
Resonance, aligns consciousness with the vibrational frequency of the
environment. As Dhamira explains: "Your Order separates the mind into
functions — Pattern, Control, Projection — as if consciousness is a machine
with interchangeable parts. The Keepers do not divide. We resonate."

Sources:
  43c756e0-... (p.1) — 0.52
  43c756e0-... (p.1) — 0.50
  43c756e0-... (p.1) — 0.49

Grounded (100% confidence)
```

## 7. Research Sessions

The `--session` flag enables multi-turn conversations. Each follow-up query includes the previous exchange as context, so the AI retrieves results relevant to the ongoing research thread:

```bash
$ rfnry-rag retrieval --pretty query "What happened to Dhamira throughout the saga?" \
    -k araxis --session araxis-research

Based on the provided context, here is what happened to Dhamira throughout
the Araxis saga:

**Her Role and Identity:**
Dhamira was revealed to be far more than just an instructor at the Resonance
Academy. She was a **Keeper** — one of the Araxeen elders who maintained
ancient knowledge predating the Hegemony by ten thousand years.

**Her Longevity:**
Dhamira **outlived them all**, benefiting from the Araxeen lifespan that
extended well past two centuries due to low-level chronodust exposure in
the caves.

**Her Activities:**
- She was present at Lady Senna's death, receiving Senna's final words
- She monitored the sandweaver consciousness through the cave network
  during the space-time crisis
- She pushed Kael past every boundary in his Resonance practice during
  his exile
- She explained crucial revelations about the sandweavers' memory

**Her Final Fate:**
The provided context does not specify what ultimately happened to Dhamira
or whether she was still alive at the saga's conclusion.

Sources:
  acc6ec92-... (p.1) — 0.78
  5ebc84a5-... (p.1) — 0.59
  acc6ec92-... (p.1) — 0.55

Grounded (100% confidence)
```

Follow up in the same session — the context helps the AI understand what you're researching:

```bash
$ rfnry-rag retrieval --pretty query "What was Dhamira's relationship with the Resonance Academy?" \
    -k araxis --session araxis-research

Dhamira had a foundational and enduring relationship with the Resonance
Academy:

**As a Teacher:**
Dhamira taught at the Resonance Academy, continuing her work there with
"unchanged methods" and "infinite patience" throughout her exceptionally
long life (Chapter 1: The Resonance Academy).

**Her Teaching Philosophy:**
She taught a fundamentally different approach than the Cognitive Order.
While the Order divided consciousness into three pillars (Pattern, Control,
Projection), Dhamira and the other Keepers taught **Resonance** as a
unified practice.

**Her Influence on Curriculum:**
Dhamira supported Kael's hybrid approach that combined the Cognitive Order's
Pattern discipline with Resonance techniques for interpreting the ancient
codex, even though this made other Keepers uncomfortable.

The Academy itself was established five years after the moratorium and
represented a collaboration between Hegemony and Araxeen students, with
Dhamira as one of its key instructors.

Sources:
  acc6ec92-... (p.1) — 0.80
  5ebc84a5-... (p.1) — 0.59
  43c756e0-... (p.1) — 0.57

Grounded (100% confidence)
```

### Session management

```bash
$ rfnry-rag retrieval session list
  araxis-research (4 turns)

$ rfnry-rag retrieval session clear araxis-research
Session 'araxis-research' cleared.
```

## 8. Grounding Protection

With grounding gates enabled, the system refuses to generate answers when the books don't contain the information. This prevents the LLM from inventing lore:

```bash
$ rfnry-rag retrieval query "What is the name of the planet where the sandweavers originated?" -k araxis
{
  "answer": "I couldn't find specific information about this in the available knowledge sources...",
  "sources": [],
  "grounded": false,
  "confidence": 0.0
}
```

The books describe the sandweavers' origin as "a system so distant that its light had not yet reached the Hegemony's most powerful telescopes" — but they never name it. Without grounding gates, the LLM would invent a planet name and present it as canon. With them, it correctly refuses. For a writer maintaining consistency across 8 novels, this prevents hallucinated lore from contaminating the knowledge base.

## 9. Iterative Drafting

The real power of the rfnry-rag CLI for writing is the feedback loop: ingest a draft, check it against the existing canon, revise, re-ingest.

The `scifi-library/` folder includes `draft-book-09-chapter-01.md` — a draft chapter set 40 years after the moratorium. Ingest it alongside the published books:

```bash
$ rfnry-rag retrieval ingest scifi-library/draft-book-09-chapter-01.md -k araxis --source-type draft
{
  "source_id": "54793450-bef0-430a-ac37-6a7c42bbd6c2",
  "chunk_count": 7,
  "status": "completed"
}
```

Now check the draft for consistency against the established canon:

```bash
$ rfnry-rag retrieval --pretty query "The draft has Dhamira using Projection. Is that consistent \
    with what we know about Keeper abilities?" -k araxis --min-score 0.3

Based on the provided context, there appears to be an **inconsistency**.

The context establishes that:

1. **Keepers practice Resonance, not the three pillars**: In Chapter 3,
   Dhamira explicitly tells Kael: "Your Order separates the mind into
   functions... Pattern, Control, Projection — as if consciousness is a
   machine with interchangeable parts. The Keepers do not divide. We resonate."

2. **Projection is described as a human/Cognitive Order discipline**: The
   mysterious fold signature is notable specifically because "Projection was a
   human discipline" and someone appears to have merged it with Resonance in
   an unprecedented way.

**Conclusion**: If your draft has Dhamira using Projection as a technique,
this would be inconsistent with established lore. As a Keeper elder, Dhamira
would use Resonance, not Projection.

Sources:
  54793450-... (p.1) — 0.73
  02413b9e-... (p.1) — 0.60
  02413b9e-... (p.1) — 0.58

Grounded (100% confidence)
```

The rfnry-rag system cross-references the draft against all 8 published books and catches the inconsistency. After revising the draft, remove the old version and re-ingest:

```bash
$ rfnry-rag retrieval knowledge remove 54793450-bef0-430a-ac37-6a7c42bbd6c2
$ rfnry-rag retrieval ingest scifi-library/draft-book-09-chapter-01.md -k araxis --source-type draft
```

Use research sessions to explore the established world while drafting:

```bash
$ rfnry-rag retrieval --pretty query "I'm writing book 9. What was the state of the world \
    at the end of book 8?" -k araxis --session book9-draft

The Hegemony had evolved into "the Accord" — a loose federation of twelve
systems connected by symbiotic fold routes. Each fold route was maintained
through partnerships between human navigators and local sandweaver populations.
The chronodust trade had ceased entirely — the sandweavers produced what was
needed, where it was needed. The moratorium had been successfully maintained
for 547 days despite food riots, and the substrate fractures were healed.

Sources:
  c0820f59-... (p.1) — 0.65
  2f808645-... (p.1) — 0.62
  c0820f59-... (p.1) — 0.60

Grounded (100% confidence)
```

## 10. Character Research

```bash
$ rfnry-rag retrieval --pretty query "What is the relationship between Kael and Dhamira throughout the saga?" -k araxis

Kael and Dhamira share a multifaceted relationship that evolves throughout
the saga:

**Teacher and Student**
Dhamira initially serves as Kael's instructor in the water discipline. Later,
during his exile, she continues to mentor him in Resonance practice, pushing
him past every boundary he had previously encountered.

**Keeper and Initiate**
Dhamira is revealed to be far more than a simple teacher — she is a Keeper,
one of the Araxeen elders who maintained the old knowledge, the knowledge
that predated the Hegemony by ten thousand years.

**Collaborator**
When Kael proposes a hybrid approach combining the Cognitive Order's Pattern
discipline with Resonance to interpret the codex — a method that makes other
Keepers uncomfortable — Dhamira supports his innovative thinking.

**Guide to Ancient Knowledge**
Dhamira serves as Kael's guide to understanding the sandweavers' legacy,
explaining the crystallized library and how "the network has memory."

Sources:
  acc6ec92-... (p.1) — 0.78
  5ebc84a5-... (p.1) — 0.59
  43c756e0-... (p.1) — 0.55
```

## 11. World-Building Research

An AI agent writing a new story in this universe can pull specific details:

```bash
$ rfnry-rag retrieval retrieve "stillsuit water discipline araxeen survival" -k araxis
{
  "chunks": [
    {
      "content": "The Araxeen stillsuit was a marvel of biological engineering. It recovered moisture from breath, sweat, and waste, recycling it into drinkable water through a series of micro-filters woven from sandweaver silk. A properly fitted stillsuit could reduce water loss to less than a thimbleful per day, even in the deep desert.",
      "score": 0.55,
      "source_type": "book"
    },
    {
      "content": "The water discipline extended beyond the stillsuit. Every action was evaluated against its water cost. Speaking consumed moisture. Crying was considered obscene — a waste of the body's most precious resource. The Araxeen greeting was not a handshake or a bow, but the offering of a small measure of water from one's own supply. To refuse the offering was a declaration of enmity.",
      "score": 0.51,
      "source_type": "book"
    }
  ]
}
```

## 12. Knowledge Management

List all ingested sources:

```bash
$ rfnry-rag retrieval --pretty knowledge list -k araxis

Sources (8):

  a76d8591-a640-4180-a27c-c7c98585716f
    ID: a76d8591-a640-4180-a27c-c7c98585716f
    Chunks: 12
    Knowledge: araxis
    Type: book
  ...
```

Get details for a specific source:

```bash
$ rfnry-rag retrieval knowledge get a76d8591-a640-4180-a27c-c7c98585716f
{
  "source_id": "a76d8591-a640-4180-a27c-c7c98585716f",
  "status": "completed",
  "chunk_count": 12,
  "knowledge_id": "araxis",
  "source_type": "book"
}
```

Inspect chunks from a specific book:

```bash
$ rfnry-rag retrieval --pretty knowledge chunks a76d8591-a640-4180-a27c-c7c98585716f

Chunks (12):

  [0] The day Kael Vandris learned he would govern Araxis, the most dangerous planet in the Hege...
  [1] Araxis was worse than the briefings suggested. The surface temperature exceeded 70 degrees ...
  [2] The native Araxeen had adapted over millennia. Their skin was dark and leathery, their eyes...
  [3] It was in the Endless that the sandweavers lived — creatures sixty meters long that burrowed...
  ...
```

Remove a source and all its chunks:

```bash
$ rfnry-rag retrieval knowledge remove a76d8591-a640-4180-a27c-c7c98585716f
{
  "source_id": "a76d8591-a640-4180-a27c-c7c98585716f",
  "deleted_vectors": 12
}
```

## AI Agent Usage

JSON output (the default when piped or when no TTY is detected) makes the CLI ideal for AI agents. An AI writing assistant can:

1. **Research lore**: `rfnry-rag retrieval retrieve "sandweaver biology" -k araxis --min-score 0.4`
2. **Multi-turn research**: `rfnry-rag retrieval query "What happened to Dhamira?" -k araxis --session book5-rewrite`
3. **Follow up**: `rfnry-rag retrieval query "What was her role in the moratorium?" -k araxis --session book5-rewrite`
4. **Ask synthesized questions**: `rfnry-rag retrieval query "What happened to Navigator Primus Leth?" -k araxis`
5. **Verify consistency**: `rfnry-rag retrieval retrieve "Kael's age timeline" -k araxis`
6. **Add notes**: `rfnry-rag retrieval ingest --text "New timeline note: ..." -k araxis --source-type notes`
7. **Manage sessions**: `rfnry-rag retrieval session list` / `rfnry-rag retrieval session clear book5-rewrite`
8. **Manage sources**: `rfnry-rag retrieval knowledge list -k araxis` / `rfnry-rag retrieval knowledge remove <id>`

Grounding gates ensure the AI never invents lore — if the books don't contain the answer, it says so instead of hallucinating.

All output is JSON by default, parseable by any AI framework. Use `--pretty` when reading in a terminal, or `--json` to force JSON in a TTY.
