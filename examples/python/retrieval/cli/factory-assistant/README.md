# CLI Example: Factory Equipment Manuals

A walkthrough showing how to use the `rag` CLI to build a searchable knowledge base from factory equipment manuals — enabling maintenance technicians and AI assistants to find troubleshooting steps, part numbers, and service procedures instantly.

The `factory-assistant/` folder contains 4 equipment manuals:

| Manual | Equipment | Chunks |
|--------|-----------|--------|
| `manual-01-xr500-compressor.md` | XR-500 Industrial Air Compressor | 19 |
| `manual-02-ht200-heat-sealer.md` | HT-200 Continuous Heat Sealer | 20 |
| `manual-03-pl1000-palletizer.md` | PL-1000 Automatic Palletizer | 20 |
| `manual-04-cw750-stretch-wrapper.md` | CW-750 Automatic Stretch Wrapper | 24 |

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
collection = "factory-assistant"

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
system_prompt = "You are a factory maintenance assistant. Use only the provided equipment manuals to answer questions. Always include part numbers and specific measurements when referencing maintenance procedures."
grounding_enabled = true
grounding_threshold = 0.5
relevance_gate_enabled = true
relevance_gate_provider = "anthropic"
relevance_gate_model = "claude-haiku-4-5-20251001"
```

**What this enables:**
- **Reranking** — a cross-encoder re-scores retrieval results, pushing the best matches to the top
- **Query rewriting (HyDE)** — rewrites vague queries ("thing keeps stopping") into hypothetical document passages before retrieval, closing the vocabulary gap between how technicians describe problems and how manuals are written
- **Grounding gates** — prevents the LLM from generating answers when retrieval quality is too low (critical for safety-related maintenance procedures)

## 2. Ingest the Manuals

```bash
$ for f in factory-assistant/manual-*.md; do rfnry-rag retrieval ingest "$f" -k factory --source-type manual; done

{
  "source_id": "71657b57-b626-4813-ae79-e62bf42ce8bf",
  "chunk_count": 19,
  "status": "completed"
}
{
  "source_id": "8cdaa46e-ce41-4f66-a700-c8a9cd583661",
  "chunk_count": 20,
  "status": "completed"
}
{
  "source_id": "266faaf7-c56f-42b9-b1d0-866ee74b65b9",
  "chunk_count": 20,
  "status": "completed"
}
{
  "source_id": "7ad9237b-0e61-420d-833e-61dae4044569",
  "chunk_count": 24,
  "status": "completed"
}
```

83 chunks across 4 manuals.

## 3. Troubleshooting Queries

### Retrieve raw context (with reranking)

Find the relevant manual sections for a symptom. With reranking enabled, the cross-encoder re-scores results — the troubleshooting section scores 0.71 (vs 0.45 without reranking):

```bash
$ rfnry-rag retrieval retrieve "compressor won't build pressure" -k factory
{
  "chunks": [
    {
      "content": "### Compressor runs but does not build pressure\n1. Check for air leaks at all fittings and connections — use soapy water spray\n2. Inspect reed valves in cylinder head for carbon buildup or damage. Remove the four hex bolts on the cylinder head cover. Reed valves should lay flat against the valve seat with no visible gaps. Replace if warped, cracked, or carboned. Part number RV-2201.\n3. Check the unloader valve — if stuck open, air escapes during compression. Part number UV-100.",
      "score": 0.71,
      "source_type": "manual"
    },
    {
      "content": "## 5. Troubleshooting\n\n### Compressor will not start\n1. Check power supply — verify voltage at disconnect\n2. Check ON/OFF switch — replace if damaged\n3. Check thermal overload — press reset button on motor housing\n4. Check pressure switch — if tank is already at cut-out pressure, the compressor will not start. Drain some air to reduce pressure below cut-in.\n5. Check motor capacitor (single-phase models only) — a failed capacitor prevents starting. Part number MC-230.",
      "score": 0.68,
      "source_type": "manual"
    }
  ]
}
```

### Filter low-quality results with `--min-score`

Without `--min-score`, the query returns 6 chunks — including marginally relevant ones about thermal overload and pressure switch restart. With `--min-score 0.5`, only the 3 most relevant chunks are returned:

```bash
$ rfnry-rag retrieval retrieve "compressor won't build pressure" -k factory --min-score 0.5
{
  "chunks": [
    {
      "content": "### Compressor runs but does not build pressure\n1. Check for air leaks...",
      "score": 0.71,
      "source_type": "manual"
    },
    {
      "content": "## 5. Troubleshooting\n\n### Compressor will not start...",
      "score": 0.68,
      "source_type": "manual"
    },
    {
      "content": "1. Verify oil level is between MIN and MAX on sight glass\n2. Open the tank drain valve...",
      "score": 0.57,
      "source_type": "manual"
    }
  ]
}
```

### AI-generated answer with part numbers

```bash
$ rfnry-rag retrieval --pretty query "The heat sealer is making wrinkled seals on our PP bags. What should we check?" -k factory --min-score 0.4

Based on the troubleshooting section for wrinkled, distorted seals on your
heat sealer, here's what you should check:

## Immediate Adjustments:

1. **Decrease temperature by 10-20F increments** from your current setting
2. **Increase conveyor speed** to reduce dwell time
3. **Decrease jaw pressure** using the two spring-loaded pressure screws on
   top of the upper jaw assembly
4. **Check for hotspots on the heating element**

## Additional Considerations for PP Material:

Since you're sealing PP (polypropylene) bags, verify your seal quality using
the peel test:
- Cut a 1-inch wide strip across the sealed area
- Pull apart at a 180-degree angle
- For PP, you need **minimum 4 lbs/inch of seal width** for acceptable seal
  strength
- The material should tear before the seal separates

## Important Note:

When adjusting jaw pressure, ensure **equal pressure on both sides** - uneven
pressure causes weak seals on one side and material distortion on the other.

Sources:
  8cdaa46e-... (p.1) — 0.81
  8cdaa46e-... (p.1) — 0.69
  8cdaa46e-... (p.1) — 0.61

Grounded (100% confidence)
```

## 4. Vague Queries (HyDE Query Rewriting)

Technicians don't always use precise terminology. With HyDE (Hypothetical Document Embeddings) enabled, the CLI rewrites vague queries into hypothetical manual passages before retrieval — bridging the gap between how people describe problems and how manuals are written:

```bash
$ rfnry-rag retrieval --pretty query "stretch wrapper film breaks during wrapping cycle" \
    -k factory --min-score 0.3

# Film Breaks During Wrapping - Troubleshooting Steps

Based on the equipment manual, follow these steps in order:

## 1. Reduce Film Tension
- Decrease tension by **10% increments**
- Test after each adjustment

## 2. Inspect Pre-Stretch Rollers
- This is identified as the **primary cause of film breaks**
- Check for scoring (scratches/grooves) and debris buildup
- Clean or replace rollers as needed

## 3. Verify Film Gauge and Pre-Stretch Compatibility
- **Do NOT use 60 gauge film with 200% pre-stretch on heavy loads**
- If using 250% pre-stretch (part number **PG-250**):
  - Only use with 60-80 gauge films

## 4. Check Load for Sharp Edges
- Apply corner protectors if sharp edges are present

## After Film Break Occurs:
The **FBS-100** ultrasonic sensor will detect the break and stop the machine.
Rethread film per section 2.1, attach to load, press START to resume.

Sources:
  9aeef2ec-... (p.1) — 0.82
  9aeef2ec-... (p.1) — 0.64
  9aeef2ec-... (p.1) — 0.54

Grounded (100% confidence)
```

Without HyDE, a vague query like "the plastic wrap keeps tearing" might miss the relevant troubleshooting section entirely because the manual uses "film break" terminology, not "plastic wrap tearing."

## 5. Multi-Turn Troubleshooting (Sessions)

The `--session` flag enables multi-turn conversations. Each follow-up query includes the previous exchange as context, so the technician can drill down into a problem:

```bash
$ rfnry-rag retrieval --pretty query "compressor won't build pressure, what should I check first?" \
    -k factory --session troubleshoot

Based on the manual for "Compressor runs but does not build pressure", check
these items in order:

## First Check: Air Leaks
1. **Check for air leaks at all fittings and connections** — use soapy water
   spray to identify leaks

## Second Check: Reed Valves
2. **Inspect reed valves in cylinder head** for carbon buildup or damage:
   - Remove the four hex bolts on the cylinder head cover
   - Reed valves should lay flat against the valve seat with no visible gaps
   - Replace if warped, cracked, or carboned
   - **Part number: RV-2201**

## Third Check: Unloader Valve
3. **Check the unloader valve** — if stuck open, air escapes during compression
   - **Part number: UV-100**

Sources:
  71657b57-... (p.1) — 0.85
  71657b57-... (p.1) — 0.74
  71657b57-... (p.1) — 0.61

Grounded (100% confidence)
```

Now the follow-up — the session context helps the retrieval find the right answer about oil for the compressor, even though the question doesn't mention "compressor":

```bash
$ rfnry-rag retrieval --pretty query "I checked and the reed valves look carboned up. What oil should I be using to prevent this?" \
    -k factory --session troubleshoot

Based on the manual, you should be using **SAE 30 non-detergent compressor oil**
to prevent reed valve carbon buildup.

## Oil Specifications:
- **Type:** SAE 30 non-detergent compressor oil
- **Quantity:** 1.5 quarts
- **Fill location:** Oil fill cap on the crankcase (left side, yellow cap)

## Important - DO NOT USE:
- Automotive motor oil
- Synthetic oil
- Multi-viscosity oil

**The manual specifically warns that using these incorrect oil types will cause
premature reed valve failure**, which is exactly what you're experiencing with
the carbon buildup.

After replacing your carboned reed valves (**Part number: RV-2201**), make sure to:
1. Drain the old oil completely
2. Refill with the correct SAE 30 non-detergent compressor oil
3. Check the sight glass to ensure oil level is between MIN and MAX marks

Sources:
  71657b57-... (p.1) — 0.74
  71657b57-... (p.1) — 0.68
  71657b57-... (p.1) — 0.56

Grounded (100% confidence)
```

### Session management

```bash
$ rfnry-rag retrieval session list
  troubleshoot (2 turns)

$ rfnry-rag retrieval session clear troubleshoot
Session 'troubleshoot' cleared.

$ rfnry-rag retrieval session list
No active sessions.
```

## 6. Grounding Gates

With grounding gates enabled, the system refuses to answer questions that aren't covered by the manuals. This is critical for a factory assistant — a wrong answer about electrical procedures or chemical handling could be dangerous.

```bash
$ rfnry-rag retrieval query "What is the recommended paint color for the factory floor?" -k factory
{
  "answer": "I couldn't find specific information about this in the available knowledge sources...",
  "sources": [],
  "grounded": false,
  "confidence": 0.0
}
```

Without grounding gates, the LLM would generate a plausible-sounding answer. With them, it correctly refuses — the equipment manuals don't cover floor paint.

## 7. Parts Lookup

```bash
$ rfnry-rag retrieval --pretty query "What part numbers do I need for a 1000-hour service on the XR-500 compressor?" \
    -k factory --min-score 0.4

Based on the maintenance schedule for the XR-500 Industrial Air Compressor,
the **1,000-hour service** requires inspection of the following components
with these part numbers:

1. **Reed valves** - Part Number: **RV-2201** (replacement kit)
2. **Belt** - Part Number: **BT-150** (replacement belt)

At 1,000 hours, you need to:
- Inspect the reed valves for carbon buildup, warping, or cracks
- Inspect belt tension and condition

**Note:** These are inspection intervals. You would only need to order the
replacement parts (RV-2201 or BT-150) if the inspection reveals wear or
damage requiring replacement.

Sources:
  71657b57-... (p.1) — 0.79
  71657b57-... (p.1) — 0.59
  71657b57-... (p.1) — 0.56

Grounded (100% confidence)
```

## 8. Cross-Equipment Search

Search across all manuals at once:

```bash
$ rfnry-rag retrieval retrieve "palletizer vacuum head drops cases" -k factory --min-score 0.4
{
  "chunks": [
    {
      "content": "### Vacuum head drops cases during transfer\n1. Check vacuum level on the HMI diagnostic page — should read 18-22 inHg during pickup\n2. Inspect suction cups for damage — even one failed cup can cause uneven vacuum distribution\n3. Check the Venturi generator for blockage — disassemble and clean with compressed air\n4. Verify air supply pressure is 80 PSI at the machine regulator\n5. Check vacuum hoses for kinks, cracks, or loose fittings",
      "score": 0.72,
      "source_type": "manual"
    }
  ]
}
```

## 9. Add Field Notes

Technicians can add field observations that become searchable alongside the official manuals:

```bash
$ rfnry-rag retrieval ingest --text "2024-03-15: Replaced reed valves (RV-2201) on XR-500 unit #7. Previous valves showed excessive carbon deposits after 2,100 hours. Root cause: operator was using automotive 10W-30 oil instead of SAE 30 non-detergent. Switched to correct oil type." -k factory --source-type field-note
{
  "source_id": "a1b2c3d4-...",
  "chunk_count": 1,
  "status": "completed"
}
```

Now a query about reed valve issues returns both the official troubleshooting guide and the field experience:

```bash
$ rfnry-rag retrieval retrieve "reed valve carbon buildup cause" -k factory
```

## 10. Knowledge Management

```bash
# List all sources
$ rfnry-rag retrieval knowledge list -k factory

# Get details for a specific manual
$ rfnry-rag retrieval knowledge get 71657b57-b626-4813-ae79-e62bf42ce8bf

# Inspect how a manual was chunked
$ rfnry-rag retrieval --pretty knowledge chunks 71657b57-b626-4813-ae79-e62bf42ce8bf

# Remove and re-ingest an updated manual
$ rfnry-rag retrieval knowledge remove 71657b57-b626-4813-ae79-e62bf42ce8bf
$ rfnry-rag retrieval ingest manual-01-xr500-compressor-v2.md -k factory --source-type manual
```

## AI Agent Usage

The JSON output makes this ideal for AI-powered maintenance systems:

```bash
# A maintenance chatbot can pipe queries and parse JSON responses
rfnry-rag retrieval retrieve "stretch wrapper film keeps breaking" -k factory --min-score 0.4

# Multi-turn troubleshooting with session context
rfnry-rag retrieval query "motor keeps tripping" -k factory --session ticket-4521
rfnry-rag retrieval query "checked the thermal overload, it resets but trips again" -k factory --session ticket-4521

# An inventory system can extract part numbers from answers
rfnry-rag retrieval query "What parts are needed for quarterly maintenance on all machines?" -k factory

# Grounding gates ensure the AI never invents procedures
# If the answer isn't in the manuals, it says so instead of guessing
```

All output is JSON by default, parseable by any AI framework. Use `--pretty` when reading in a terminal, or `--json` to force JSON in a TTY.
