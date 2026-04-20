# CLI Example: Sci-Fi Library

A walkthrough showing how to use the `rfnry-rag` CLI to analyze, classify, check consistency, and evaluate a series of science fiction novels — helping you research and write in a shared universe.

The `scifi-library/` folder contains 8 books from the "Sands of Araxis" saga. Each is a markdown file with several chapters (~800-1200 words each).

## 1. Install and Configure

```bash
uv add "rfnry-rag[cli]"
rfnry-rag reasoning init
```

Edit `~/.config/rfnry_rag/.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Edit `~/.config/rfnry_rag/config.toml`:

```toml
[language_model]
provider = "anthropic"
model = "claude-sonnet-4-5-20250929"
```

Suppress verbose BAML logging (optional):

```bash
export BAML_LOG=error
```

Verify:

```bash
rfnry-rag reasoning status
```

```
Config: /home/you/.config/rfnry_rag/config.toml
.env: /home/you/.config/rfnry_rag/.env
Provider: anthropic
Model: claude-sonnet-4-5-20250929
LLM: connected

Ready.
```

## 2. Analyze a Book

Analyze Book 1 to understand its structure, themes, and narrative qualities:

```bash
rfnry-rag reasoning analyze --file scifi-library/book-01-sands-of-araxis.md --summarize \
  --dimensions dimensions.json
```

```
Intent: To introduce a science fiction narrative establishing a protagonist's dangerous
  political assignment to control a desert planet that is the sole source of a critical
  resource, while building a complex world with ecological, political, and cultural
  systems. (95% confidence)

Summary: Seventeen-year-old Kael Vandris learns his family has been granted stewardship
  of Araxis, a lethal desert planet that is the only source of chronodust — a substance
  essential for interstellar navigation. The assignment appears to be a political death
  sentence, as the previous ruling house was completely destroyed, and survival requires
  mastering the harsh environment and its native inhabitants' water-conservation techniques.

Dimensions:
  tension: 0.75 (90%)
    High political stakes established immediately (previous rulers extinguished, potential
    trap by Emperor), combined with environmental dangers (extreme heat, deadly creatures,
    resource scarcity).
  pacing: moderate (85%)
    The narrative balances plot advancement (inheritance announcement, journey to Araxis,
    beginning training) with substantial worldbuilding exposition.
  worldbuilding: 0.9 (95%)
    Extremely dense with specific details: political structure (Hegemony, Emperor, House
    system), unique resource economics (chronodust for navigation, water as currency),
    environmental specifications (70°C, 3% humidity, 4000km desert).
  character_depth: 0.5 (75%)
    Moderate depth shown primarily through dialogue and reactions. Kael demonstrates
    analytical thinking, his mother shows political sophistication and emotional control.
    However, internal emotional states are minimally explored.
```

The `dimensions.json` file defines what to score:

```json
[
  {"name": "tension", "description": "Level of narrative tension and conflict", "scale": "0.0-1.0"},
  {"name": "pacing", "description": "Speed of plot progression", "scale": "slow/moderate/fast"},
  {"name": "worldbuilding", "description": "Density of world details introduced", "scale": "0.0-1.0"},
  {"name": "character_depth", "description": "Emotional and psychological depth of characters", "scale": "0.0-1.0"}
]
```

## 3. Classify Books by Theme

Create categories for the types of content in your series:

```json
[
  {"name": "action", "description": "Combat, chase, or physical conflict scenes"},
  {"name": "political_intrigue", "description": "Power plays, alliances, betrayals, and court machinations"},
  {"name": "exploration", "description": "Discovery of new places, technologies, or knowledge"},
  {"name": "character_drama", "description": "Interpersonal conflict, relationships, and emotional arcs"},
  {"name": "worldbuilding_exposition", "description": "Establishing setting, history, culture, or technology"},
  {"name": "crisis_and_survival", "description": "Characters facing existential threats or impossible choices"}
]
```

Classify each book:

```bash
rfnry-rag reasoning classify --file scifi-library/book-01-sands-of-araxis.md --categories categories.json
```

```
Category: worldbuilding_exposition (95%)
Strategy: llm
Reasoning: The text primarily establishes the setting, culture, and key elements of the
  world including the planet Araxis, its extreme environment, the native Araxeen people,
  and the political/economic importance of chronodust. While there is a political element
  in the inheritance announcement, the bulk of the content focuses on describing the
  world's physical characteristics, history, and inhabitants.
Runner-up: political_intrigue (40%)
```

```bash
rfnry-rag reasoning classify --file scifi-library/book-05-the-hegemony-falls.md --categories categories.json
```

```
Category: political_intrigue (85%)
Strategy: llm
Reasoning: The text centers on the Emperor's decision regarding fold travel and the
  resulting political split of the Hegemony into factions, with advisors debating the
  validity of information and different systems forming alliances based on their
  interests. While there are elements of crisis and worldbuilding, the primary focus is
  on power dynamics, alliances, and the political consequences of the Emperor's choice.
Runner-up: crisis_and_survival (65%)
```

You can batch-classify all books to see the arc of your series:

```bash
for book in scifi-library/book-*.md; do
  name=$(basename "$book" .md)
  category=$(rfnry-rag reasoning --json classify --file "$book" --categories categories.json | jq -r '.category')
  echo "$name: $category"
done
```

```
book-01-sands-of-araxis: worldbuilding_exposition
book-02-the-chronodust-war: action
book-03-keepers-of-the-fold: exploration
book-04-the-sandweaver-codex: exploration
book-05-the-hegemony-falls: political_intrigue
book-06-exile-of-house-vandris: character_drama
book-07-the-substrate-war: crisis_and_survival
book-08-the-new-navigators: worldbuilding_exposition
```

This tells you: Books 1 and 8 bookend the series with worldbuilding, the middle is action/exploration/politics, and Book 6 is the emotional core. If you're planning Book 9, you can see what's missing.

## 4. Check Consistency

The compliance command checks whether new writing contradicts established canon. Use an earlier book as the reference document:

```bash
rfnry-rag reasoning compliance --file scifi-library/book-08-the-new-navigators.md \
  --references scifi-library/book-01-sands-of-araxis.md
```

```
Compliance: FAIL (0.30)

Violations (8):
  [high] narrative_continuity — The text begins 'Five years after the moratorium' but
    no moratorium is mentioned or established in the reference document.
    Suggestion: Provide context that bridges from the reference material, or ensure the
    reference document includes this event.
  [high] technological_consistency — The text introduces 'Resonance discipline',
    'Keepers', and 'symbiotic fold' technology that has no basis in the reference
    material. The reference establishes chronodust navigation but provides no foundation
    for these alternative methods.
    Suggestion: Show how these technologies evolved from the chronodust navigation system
    described in the reference.
  [high] world_building_consistency — The 'Resonance Academy' and the 'Keepers'
    organization are introduced without any precedent in the reference material.
    Suggestion: Add foundation for the Keepers in earlier books, or explain their origin
    and relationship to established institutions like the Cognitive Order.
  [high] world_building_consistency — The reference establishes sandweavers as dangerous
    creatures. The evaluated text presents them as cooperative beings that can be
    'acclimated' to human presence, which contradicts the established characterization.
    Suggestion: Add foundation in earlier books for sandweaver intelligence and potential
    cooperation.
  [medium] narrative_continuity — The text references 'the Consortium way' without any
    mention of a Consortium in the reference material.
  [medium] technological_consistency — The reference describes chronodust as making
    'interstellar navigation possible' but doesn't explain the mechanism. The evaluated
    text introduces 'fold navigation' and 'substrate topology' as established concepts.
  [medium] narrative_continuity — The text mentions 'the Cognitive Order's three pillars'
    (Pattern, Control, and Projection) which are not established in the reference material.
  [low] character_consistency — Kael is 17 in the reference and has just arrived on
    Araxis. The evaluated text shows him as an established leader, which is plausible but
    represents a significant time jump.

Reasoning: The text introduces significant narrative elements, technologies, and plot
  developments that have no foundation in the reference material. The text jumps forward
  five years and references events, institutions, and concepts that are completely absent
  from the reference chapters.
```

This is exactly the kind of feedback a writer needs when Book 8 contradicts Book 1. You can now go back and plant the seeds for "Keepers", "the moratorium", and sandweaver cooperation in the earlier books.

But checking against Book 1 alone is unfair — Book 8 builds on 7 books of established canon. Use `--references` to check against multiple books at once:

```bash
rfnry-rag reasoning compliance --file scifi-library/book-08-the-new-navigators.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-07-the-substrate-war.md
```

```
Compliance: FAIL (0.85)

Violations (4):
  [medium] technological_consistency — The reference states chronodust is 'generated by the
    passage' of sandweavers through dunes, implying a byproduct of their movement. The
    evaluated text describes 'chronodust crystal' chambers, which seems inconsistent with
    the dust/powder nature implied in the reference.
    Suggestion: Clarify that the chamber is carved from compressed or crystallized chronodust
    deposits, or use terminology like 'chronodust-infused stone'.
  [medium] world_building_consistency — The text mentions 'Keepers' as a distinct tradition
    with their own 'Resonance discipline' spanning generations, but this group is not
    introduced in the reference material.
    Suggestion: Either introduce the Keepers in earlier reference material, or provide
    context explaining who they are.
  [low] terminology_consistency — The text introduces 'three pillars' of the Cognitive Order
    (Pattern, Control, and Projection) without these being established in the reference.
    Suggestion: Either remove the specific pillar names or add a brief explanation that
    these pillars were formalized after the events of the earlier books.
  [low] narrative_logic — The text states the Academy opened 'five years after the
    moratorium' but doesn't clarify how the transition period unfolded.
    Suggestion: Add a brief reference to the moratorium's completion to provide clearer
    timeline continuity.

Dimension Scores:
  world_building_consistency: 0.90
  character_continuity: 0.95
  terminology_consistency: 0.75
  narrative_logic: 0.85
  thematic_alignment: 0.90
```

Score jumped from 0.30 (Book 1 only) to 0.85 (Books 1 + 7) — with Book 7 as context, concepts like "the moratorium", "Consortium", and "fold navigation" are now established canon, not violations. The remaining issues are minor terminology gaps.

## 5. Analyze Chapter Progression

Use `analyze-context` to track how narrative dimensions evolve across chapters within a book. Each chapter is a segment in a JSON array:

```bash
rfnry-rag reasoning analyze-context --file chapters-book-08.json --summarize --dimensions dimensions.json
```

```
Intent: Narrative conclusion depicting the successful transformation of a society from
  exploitative resource extraction to symbiotic cooperation between species (95% confidence)

Summary: This epilogue chronicles the establishment of the Resonance Academy on Araxis,
  where humans learn to cooperatively navigate space with sandweavers rather than exploiting
  them, culminating in a peaceful federation that replaces extraction with partnership.

Dimensions:
  tension: 0.2 (90%)
    Minimal conflict present; the text focuses on resolution and peaceful outcomes. Early
    mentions of integration challenges provide slight tension, but overall tone is
    harmonious and reflective.
  pacing: slow (85%)
    The narrative spans decades (40+ years) with contemplative, reflective passages. Time
    moves in large jumps with detailed descriptions of processes and philosophical insights.
  worldbuilding: 0.85 (90%)
    Dense with specific details about the Resonance Academy curriculum, symbiotic fold
    navigation mechanics, political structures (Hegemony to Accord), and the twelve-system
    federation.
  character_depth: 0.7 (80%)
    Characters are presented through their life choices, final words, and philosophical
    reflections. Kael's self-assessment as 'adequate translator,' Senna's trust despite
    lack of understanding, and Dhamira's patient longevity reveal meaningful depth.

Intent Shifts:
  [1] Establishing educational framework → Demonstrating practical application
    Chapter 1 focuses on the Academy's founding, while Chapter 2 shifts to showing the
    first successful symbiotic fold navigation as proof of concept.
  [2] Demonstrating practical application → Providing long-term resolution and legacy
    Chapter 2 demonstrates immediate success, while Chapter 3 jumps decades forward to
    show lasting societal transformation and philosophical summation.

Escalation: no
Resolution: resolved
```

The `chapters-book-08.json` represents each chapter as a `{role, text}` segment:

```json
[
  {"role": "chapter_1", "text": "Five years after the moratorium..."},
  {"role": "chapter_2", "text": "The first symbiotic fold..."},
  {"role": "chapter_3", "text": "Kael Vandris governed Araxis for forty more years..."}
]
```

This tells the writer: tension drops from moderate to low across the final book, pacing slows dramatically, but worldbuilding density stays high. If Book 8 needs more tension, the data shows exactly where to add it.

## 6. Evaluate Revisions

After rewriting a chapter, compare the revision against the original:

```bash
rfnry-rag reasoning evaluate \
  --generated scifi-library/book-08-the-new-navigators.md \
  --reference scifi-library/book-01-sands-of-araxis.md \
  --strategy judge
```

```
Score: 0.35 (low)
Judge: 0.35
  The generated output appears to be from a later section of the same story, showing
  narrative progression and thematic consistency with the reference material. However,
  it suffers from significant context issues — it references events, characters, and
  concepts that are not established in the provided text. The writing quality is competent
  with clear prose and coherent world-building, but it reads like a sequel chapter without
  the necessary foundation.
```

This is more useful when comparing two versions of the same chapter — the original draft vs a revision — to see if the rewrite actually improved.

## 7. Piped Input for AI Assistants

The CLI is designed for AI models working alongside you. When piped, output is automatically JSON:

```bash
echo "The sandweavers emerged from the dunes at sunset, their crystalline bodies \
refracting the last light into prismatic arcs across the desert floor. Kael watched \
from the observation platform, knowing that these creatures held the key to everything." \
  | rfnry-rag reasoning analyze --summarize
```

```json
{
  "primary_intent": "Narrative storytelling — establishing a scene with world-building elements and tension",
  "confidence": 0.95,
  "summary": "Kael observes mysterious crystalline creatures called sandweavers at sunset, harboring dangerous knowledge about chronodust and the fold that could destabilize peace if discovered.",
  "dimensions": {},
  "entities": [],
  "retrieval_hints": []
}
```

An AI assistant can parse this and act on it — checking every paragraph you write for consistency, classifying scenes as you draft them, or analyzing pacing across chapters.

## 8. A Writing Research Session

Here's a realistic workflow — you're writing the final chapter of Book 8 and need help:

```bash
# What's the narrative arc of Book 7 going into the finale?
rfnry-rag reasoning analyze --file scifi-library/book-07-the-substrate-war.md --summarize \
  --dimensions dimensions.json

# What category should the final chapter be? Check what's missing.
for book in scifi-library/book-*.md; do
  echo "$(basename $book): $(rfnry-rag reasoning --json classify --file $book --categories categories.json | jq -r .category)"
done

# Write your chapter, then check it against the full series canon
rfnry-rag reasoning compliance --file final-chapter-draft.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-07-the-substrate-war.md \
  --references scifi-library/book-08-the-new-navigators.md

# Revise and evaluate improvement
rfnry-rag reasoning evaluate --generated final-chapter-v2.md --reference final-chapter-draft.md \
  --strategy judge
```

The CLI is the simple, global interface — one command per operation, no code required. For batching, custom pipelines, or embedding in applications, use the Python SDK directly.

## 9. Deep Research with Retrieval

The reasoning CLI works on the files it can see — great for quick checks. But when your series has hundreds of pages across 8 books, you need the retrieval CLI to search the full corpus semantically. This section shows how to combine both tools for deep research sessions.

### Set up retrieval

Install and configure the retrieval pipeline alongside reasoning:

```bash
rfnry-rag retrieval init
docker run -d -p 6333:6333 qdrant/qdrant
```

Edit `~/.config/rfnry_rag/config.toml` to add the retrieval sections:

```toml
[persistence]
vector_store = "qdrant"
url = "http://localhost:6333"
collection = "knowledge"

[ingestion]
embeddings = "voyage"
model = "voyage-4"

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

Add the Voyage API key to `~/.config/rfnry_rag/.env`:

```bash
VOYAGE_API_KEY=pa-...
```

### Ingest the full series

```bash
for f in scifi-library/book-*.md; do
  rfnry-rag retrieval ingest "$f" -k araxis --source-type book
done
```

```
{
  "source_id": "9db640aa-886f-44ca-9c8d-acfce9384b04",
  "chunk_count": 12,
  "status": "completed"
}
{
  "source_id": "1f524293-8012-432d-8210-f112fc8dc032",
  "chunk_count": 9,
  "status": "completed"
}
...
```

Total: 90 chunks across 8 books, fully searchable.

### Cross-book research sessions

With hundreds of pages ingested, you can now ask questions that span the entire saga. The `--session` flag keeps multi-turn context so follow-up questions are informed by what you already asked:

```bash
rfnry-rag retrieval --pretty query "What are all the major political factions and how do they evolve across the saga?" \
    -k araxis --session deep-research
```

```
# Major Political Factions in the Araxis Saga

## Initial Factions

**The Hegemony** - The central governing authority that includes the Emperor
  and controls the core worlds (Thessaly, Caldera, Meridian). Supports the
  fold travel moratorium to address the existential risk revealed by the
  sandweavers' warning.

**House Vandris** - Led by Lady Senna, serves as stewards and the Keepers'
  only ally within the Hegemony. Supports the moratorium despite economic
  pressure.

**The Spacing Consortium** - Initially under embargo by Araxis, this faction
  represents the outer systems' economic interests. Opposes the moratorium,
  viewing it as a power grab by core worlds to control chronodust reserves.

**The Keepers** - Guardians of the sandweavers' warning; their only Hegemony
  ally is House Vandris.

## Evolution

**The Coalition** - Proposed by Kael as a temporary governing body to manage
  the crisis, offering representation to all factions including former enemies.

**The Accord** - The ultimate evolution of governance: a loose federation of
  twelve systems connected by symbiotic fold routes, replacing the Hegemony.

Sources:
  7ba41a18-... (p.1) — 0.45
  5b027ab0-... (p.1) — 0.45
  f0a64f56-... (p.1) — 0.44

Grounded (100% confidence)
```

Follow up in the same session — the context from the previous answer helps the retrieval find related passages:

```bash
rfnry-rag retrieval --pretty query "How does the sandweavers' role change from being harvested to being partners?" \
    -k araxis --session deep-research
```

```
# The Sandweavers' Arc: From Resource to Partners

## Phase 1: Exploitation and Misunderstanding
The Hegemony treats sandweavers as animals or resources to be harvested.
Navigators "burn chronodust in sealed chambers until body breaks down,"
consuming the material without understanding its true nature.

## Phase 2: Discovery and Recognition
Tallis reveals: "They did not merely navigate through folds. They created
the folds. The chronodust is not a byproduct of their movement. It is the
material they use to repair tears in space-time."

## Phase 3: The Transition Solution
Kael teaches navigators "a new way — one that works with the sandweavers,
not against them." An 18-month moratorium on fold travel allows sandweavers
to repair existing fractures.

## Phase 4: Partnership and Integration
The Hegemony evolves into the Accord — "a loose federation of twelve systems
connected by symbiotic fold routes." The chronodust trade ceases entirely:
"sandweavers produced what was needed, where it was needed, in cooperation
with navigators who asked rather than took."

Sources:
  45291a8d-... (p.1) — 0.75
  45291a8d-... (p.1) — 0.57
  38c1c6d8-... (p.1) — 0.55

Grounded (100% confidence)
```

### Combining retrieval + reasoning for analysis

The reasoning CLI analyzes structure. The retrieval CLI searches content. Together they give you a complete research workflow:

```bash
# Step 1: Reasoning — analyze Book 7 to understand what leads into the finale
rfnry-rag reasoning analyze --file scifi-library/book-07-the-substrate-war.md --summarize \
  --dimensions dimensions.json
```

```
Intent: To narrate a science fiction story about political coalition-building
  and sacrifice in the face of an existential crisis involving space-time
  degradation (95% confidence)

Summary: Kael proposes a Coalition to enforce an 18-month moratorium on fold
  travel to repair substrate fractures, but faces military opposition from
  Admiral Voss and the Consortium. After a decisive battle involving the
  sacrifice of Fold Navigators, the Coalition defeats Voss and begins
  implementing the moratorium to save the universe.

Dimensions:
  tension: 0.75 (90%)
    High stakes conflict with existential threat, military confrontation,
    political opposition, and life-or-death sacrifices
  pacing: fast (85%)
    Moves rapidly from political proposal to military conflict to battle
    resolution within three chapters
  worldbuilding: 0.8 (90%)
    Dense introduction of sci-fi concepts: sandweaver network, fold travel,
    substrate fractures, chronodust vapor
  character_depth: 0.55 (75%)
    Characters serve primarily functional roles; some emotional moments
    but limited psychological exploration
```

```bash
# Step 2: Reasoning — batch-classify to see what's missing
for book in scifi-library/book-*.md; do
  echo "$(basename $book): $(rfnry-rag reasoning --json classify --file $book --categories categories.json | jq -r .category)"
done
```

```
book-01-sands-of-araxis: worldbuilding_exposition
book-02-the-chronodust-war: political_intrigue
book-03-keepers-of-the-fold: worldbuilding_exposition
book-04-the-sandweaver-codex: exploration
book-05-the-hegemony-falls: political_intrigue
book-06-exile-of-house-vandris: crisis_and_survival
book-07-the-substrate-war: political_intrigue
book-08-the-new-navigators: worldbuilding_exposition
```

Missing from the series: `action` and `character_drama`. Book 9 could fill one of these gaps.

```bash
# Step 3: Retrieval — pull specific lore details for the sequel
rfnry-rag retrieval retrieve "sandweaver origin system distant light catastrophe substrate failure" \
    -k araxis --min-score 0.3
```

```json
{
  "chunks": [
    {
      "content": "## Chapter 3: The Revelation\n\nThe sandweavers had not originated on Araxis. They had arrived — folded in from a system so distant that its light had not yet reached the Hegemony's most powerful telesc...",
      "score": 0.88,
      "source_type": "book"
    },
    {
      "content": "\"It is not angry,\" Kael told his mother afterward, still trembling. \"It is not vengeful. It is afraid. It has seen this before — in its home system, twelve thousand years ago...\"",
      "score": 0.58,
      "source_type": "book"
    }
  ]
}
```

```bash
# Step 4: Retrieval — use sessions to investigate specific characters across the full canon
rfnry-rag retrieval --pretty query "What happens at the end of book 8? What is the final state of the world?" \
    -k araxis --session book9-planning
```

```
The Hegemony had evolved into "the Accord" — a loose federation of twelve
systems connected by symbiotic fold routes. Each fold route was maintained
through partnerships between human navigators and local sandweaver populations.
The chronodust trade had ceased entirely. Kael Vandris governed Araxis for
forty more years, refusing the title of Emperor twice.

Grounded (100% confidence)
```

```bash
# Step 5: Retrieval — explore unanswered mysteries for the sequel
rfnry-rag retrieval --pretty query "What mysteries or unanswered questions remain?" \
    -k araxis --session book9-planning
```

```
## The Sandweavers' Origin System
The sandweavers fled "a cascading failure in the space-time substrate of
their home system" twelve thousand years ago. The specific cause is never
explained. Was it natural, or caused by another civilization's technology?

## Unanswered Questions
- How did the sandweavers travel from their distant origin system to Araxis?
- When and how did they develop their collective planetary consciousness?
- Are there sandweaver populations beyond the twelve systems of the Accord?

Grounded (100% confidence)
```

This workflow — reasoning for structural analysis, retrieval for deep lore searches — is the combination you need when working with content that's too large to fit in a single LLM context window.

## 10. Automated Book Writing

This section shows the full loop: using an AI model to research, plan, draft, verify, and revise new content — turning both CLIs into a writing pipeline. The goal is an automated system that can generate a new book in a shared universe without inventing lore or contradicting canon.

### The writing pipeline

The process has five stages, each using a different combination of reasoning and retrieval tools:

1. **Research** — retrieval queries to understand the world state
2. **Plan** — reasoning analysis + classification to identify gaps
3. **Draft** — AI writes content using retrieved lore as source material
4. **Verify** — compliance checking against established canon + retrieval consistency checks
5. **Revise** — evaluate the draft, fix violations, re-check

### Stage 1: Research the world

Start a research session to build context for the new book:

```bash
rfnry-rag retrieval --pretty query "What happens at the end of book 8? What is the final state of the world, characters, and political structures?" \
    -k araxis --session book9-planning
```

```
The Hegemony had evolved into "the Accord" — a loose federation of twelve
systems connected by symbiotic fold routes. Kael Vandris governed Araxis for
forty more years. He never took the title of Emperor, though it was offered
twice. He spent his final decades deepening his Resonance with the sandweaver
consciousness.

Grounded (100% confidence)
```

```bash
rfnry-rag retrieval --pretty query "What mysteries or unanswered questions remain? The sandweavers' origin, any prophecies?" \
    -k araxis --session book9-planning
```

```
## The Sandweavers' Origin System
The specific cause of the catastrophe in their origin system is never explained.
What triggered the initial substrate failure? Was it natural, or caused by
another civilization's technology?

## Unanswered Questions
- How did the sandweavers travel from their distant origin system to Araxis?
- Are there sandweaver populations on other worlds beyond the twelve systems?
- Did the sandweavers possess collective consciousness before the catastrophe?

Grounded (100% confidence)
```

Pull raw lore chunks that will inform the draft:

```bash
rfnry-rag retrieval retrieve "Dhamira Keeper abilities Resonance teaching philosophy caves" \
    -k araxis --min-score 0.3
```

```json
{
  "chunks": [
    {
      "content": "## Chapter 3: Dhamira's Teaching\n\nOn Araxis, Kael discovered that the Cognitive Order's techniques were a pale imitation of what the Araxeen Keepers practiced...",
      "score": 0.79,
      "source_type": "book"
    },
    {
      "content": "Resonance, as Dhamira taught it, was the practice of aligning one's consciousness with the vibrational frequency of the environment...",
      "score": 0.75,
      "source_type": "book"
    }
  ]
}
```

### Stage 2: Plan with reasoning analysis

Use classification to see what the series is missing:

```bash
for book in scifi-library/book-*.md; do
  echo "$(basename $book): $(rfnry-rag reasoning --json classify --file $book --categories categories.json | jq -r .category)"
done
```

```
book-01: worldbuilding_exposition
book-02: political_intrigue
book-03: worldbuilding_exposition
book-04: exploration
book-05: political_intrigue
book-06: crisis_and_survival
book-07: political_intrigue
book-08: worldbuilding_exposition
```

The series has never had a book classified as pure `action` or `character_drama`. The research session revealed the sandweavers' origin is the biggest unresolved mystery. This gives us a plan: **Book 9 should be an `exploration` story about a signal from the sandweavers' origin system, centering the character relationship between aged Dhamira and the new generation.**

### Stage 3: Draft with retrieved lore

Write the chapter using the retrieved context as source material. An AI writing assistant with access to the CLI tools would:

1. Retrieve specific lore for each scene (`rfnry-rag retrieval retrieve "..." -k araxis`)
2. Write the chapter grounded in those details
3. Use `rfnry-rag retrieval query` to ask clarifying questions mid-draft

Here's a chapter produced this way — `draft-book-09-chapter-02.md`:

```markdown
# Book 9: The Resonance Wars — Chapter 2: The Signal

The deep monitoring station occupied the lowest level of the Academy, carved
into the living rock where juvenile sandweavers still maintained their ancient
tunnels. The walls pulsed with bioluminescence — chronodust in its raw form,
glowing amber and shifting like breath.

Navigator Primus Veris stood before the fold resonance array, her hands moving
across holographic displays that tracked substrate harmonics in real-time.

"Show me," Dhamira said.

Veris expanded the display. A three-dimensional map of known fold space
materialized — the twelve systems of the Accord connected by luminous threads.
Beyond the mapped network, darkness. And in that darkness, a single point of
light pulsing at irregular intervals.

"The signature contains two components layered together," Veris explained.
"The first matches our symbiotic fold patterns. Someone learned to fold space
the way we do. But the second component is Projection. Pure Cognitive Order
Projection, woven into the fold itself."

"That should be impossible," Dhamira said. "Projection manipulates perception.
It cannot manipulate space-time."

"Unless someone found a way to merge it with Resonance at a fundamental level."

Dhamira opened her eyes. "The codex spoke of the sandweavers' origin system.
A civilization that understood the substrate so deeply they could reshape it
at will. We assumed that civilization was the sandweavers themselves. What if
it wasn't? What if the sandweavers were the tools, and their makers are still
out there?"
```

### Stage 4: Verify against canon

Run compliance to catch contradictions before they become permanent:

```bash
rfnry-rag reasoning compliance --file draft-book-09-chapter-02.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-04-the-sandweaver-codex.md \
  --references scifi-library/book-08-the-new-navigators.md
```

```
Compliance: FAIL (0.85)

Violations (5):
  [medium] technological_consistency — The text introduces 'fold space' and
    'folding space' as the method of interstellar travel, but the reference
    material establishes that chronodust makes 'interstellar navigation possible'
    without specifying a folding mechanism.
    Suggestion: Use more ambiguous language like 'interstellar transit' that
    aligns with the reference's vaguer description.
  [medium] world_building_consistency — The text refers to 'the Accord' and
    'Accord Council' as the governing body. The reference material uses 'the
    Hegemony.' This transition should be explained.
    Suggestion: Include a brief reference explaining the transition from
    Hegemony to Accord.
  [low] technological_consistency — 'Projection' as a specific Cognitive Order
    technique is introduced without foundation in the earlier references.
  [low] terminology_consistency — 'Substrate harmonics' as a term has no
    precedent in the earlier references.
  [low] character_continuity — 'Kael combined Pattern with Keeper teachings'
    references a specific discipline not named in Book 1.

Dimension Scores:
  world_building_consistency: 0.80
  character_continuity: 0.95
  technological_consistency: 0.75
  terminology_consistency: 0.85
  thematic_coherence: 0.90
```

Score of 0.85 — the medium violations are terminology issues (Books 3-7 establish these terms but they aren't in the subset of references we checked). The compliance tool catches exactly where the seams are.

Now cross-reference the new lore using retrieval:

```bash
rfnry-rag retrieval ingest draft-book-09-chapter-02.md -k araxis --source-type draft
```

```json
{
  "source_id": "19ad92b2-c119-4945-a458-a0ee621c64ec",
  "chunk_count": 10,
  "status": "completed"
}
```

```bash
rfnry-rag retrieval --pretty query "The draft introduces 'makers' of the sandweavers. Does this contradict anything established about sandweaver origins?" \
    -k araxis --session book9-planning
```

```
This does not appear to be a direct contradiction, but rather an expansion
of an existing ambiguity.

The codex apparently depicted "a civilization" but didn't explicitly identify
whether that civilization was the sandweavers or another species. The crucial
phrase "We assumed that civilization was the sandweavers themselves" implies
this was always an assumption, not confirmed fact.

Potential issue: the sandweavers are described as repairing "the same tears
that destroyed their origin system," suggesting they existed before the
catastrophe. The "makers" concept should be developed carefully to align with
the sandweavers' established role as refugees.

Grounded (100% confidence)
```

The retrieval system cross-references the draft against all 90 ingested chunks and provides nuanced guidance — not just "yes/no" but how to develop the new lore safely.

### Stage 5: Evaluate and classify

Run the reasoning tools on the AI-generated content to see how it fits the series:

```bash
rfnry-rag reasoning classify --file draft-book-09-chapter-02.md --categories categories.json
```

```
Category: exploration (85%)
Strategy: llm
Reasoning: The text focuses on the discovery of an unknown signal from beyond
  mapped space, with characters investigating mysterious fold space technology.
Runner-up: worldbuilding_exposition (60%)
```

Good — the plan was `exploration`, and that's what the classifier found.

```bash
rfnry-rag reasoning analyze --file draft-book-09-chapter-02.md --summarize \
  --dimensions dimensions.json
```

```
Intent: To introduce a major plot development by revealing an unexplained
  signal from beyond known space (95% confidence)

Summary: Navigator Primus Veris detects an unprecedented fold space signal
  that combines symbiotic technology with Cognitive Order Projection. Elder
  Dhamira theorizes this may indicate first contact with the original
  civilization that created the sandweavers.

Dimensions:
  tension: 0.75 (90%)
  pacing: moderate (85%)
  worldbuilding: 0.85 (90%)
  character_depth: 0.55 (75%)
```

```bash
rfnry-rag reasoning evaluate --generated draft-book-09-chapter-02.md \
  --reference scifi-library/book-08-the-new-navigators.md --strategy judge
```

```
Score: 0.72
Judge: 0.72
  The generated output demonstrates strong worldbuilding and compelling
  narrative tension, introducing intriguing concepts like the mysterious fold
  signature and potential sandweaver creators. The writing quality is solid.
  However, it diverges significantly from the reference's focus on the
  Academy's founding, representing a different narrative direction.

Dimension Scores:
  narrative_coherence: 0.65
  worldbuilding_consistency: 0.78
  character_development: 0.68
  prose_quality: 0.75
  thematic_alignment: 0.70
  continuity_with_reference: 0.60
```

### Clean up after drafting

Remove the draft from the knowledge base to avoid contaminating future searches:

```bash
rfnry-rag retrieval knowledge remove 19ad92b2-c119-4945-a458-a0ee621c64ec
```

```json
{
  "source_id": "19ad92b2-c119-4945-a458-a0ee621c64ec",
  "deleted_vectors": 10
}
```

### The full automated loop

Here's the complete script an AI writing agent would run to generate a chapter:

```bash
#!/bin/bash
# Automated chapter writer using rfnry-rag reasoning + retrieval

KNOWLEDGE="araxis"
SESSION="book9-writer"

# 1. Research — pull the world state
world_state=$(rfnry-rag retrieval --json query \
  "What is the final state at the end of book 8?" \
  -k $KNOWLEDGE --session $SESSION)

mysteries=$(rfnry-rag retrieval --json query \
  "What unanswered questions remain about the sandweavers and their origins?" \
  -k $KNOWLEDGE --session $SESSION)

# 2. Research — pull specific lore for the scene
lore=$(rfnry-rag retrieval --json retrieve \
  "Dhamira Keeper abilities Resonance Academy teaching" \
  -k $KNOWLEDGE --min-score 0.3)

# 3. Classify the series arc to find gaps
for book in scifi-library/book-*.md; do
  rfnry-rag reasoning --json classify --file "$book" --categories categories.json
done

# 4. Write (AI generates the chapter using the retrieved context)
# ... your LLM writes draft-chapter.md using $world_state, $mysteries, $lore ...

# 5. Verify — compliance check against canon
rfnry-rag reasoning compliance --file draft-chapter.md \
  --references scifi-library/book-01-sands-of-araxis.md \
  --references scifi-library/book-04-the-sandweaver-codex.md \
  --references scifi-library/book-08-the-new-navigators.md

# 6. Verify — ingest draft and cross-reference
source_id=$(rfnry-rag retrieval --json ingest draft-chapter.md \
  -k $KNOWLEDGE --source-type draft | jq -r .source_id)

rfnry-rag retrieval --json query \
  "Does the draft contradict any established canon?" \
  -k $KNOWLEDGE --session $SESSION

# 7. Evaluate quality
rfnry-rag reasoning evaluate --generated draft-chapter.md \
  --reference scifi-library/book-08-the-new-navigators.md --strategy judge

# 8. Analyze the result
rfnry-rag reasoning analyze --file draft-chapter.md --summarize \
  --dimensions dimensions.json

# 9. Clean up draft from knowledge base
rfnry-rag retrieval knowledge remove $source_id

# 10. Clear the session
rfnry-rag retrieval session clear $SESSION
```

Each step produces structured JSON output, so the AI can parse results, decide whether to revise, and loop until the compliance score and judge score meet thresholds. The grounding gates prevent the AI from inventing lore, the compliance checker catches contradictions, and the evaluate command measures prose quality — turning the rfnry-rag CLI into a complete automated writing assistant.
