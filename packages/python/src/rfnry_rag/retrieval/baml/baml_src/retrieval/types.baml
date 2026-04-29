class QueryAnalysis {
  keywords string[]
  entity_references string[]
  intent string
}

class RankedChunk {
  index int
  score float
  reasoning string?
}

// R5.1: query-classifier output. The strict 4-label QueryType allowlist
// (FACTUAL/COMPARATIVE/ENTITY_RELATIONSHIP/PROCEDURAL) describes query
// SHAPE, not domain — adding a fifth label requires explicit redesign.
enum QueryComplexity {
  SIMPLE
  MODERATE
  COMPLEX
}

enum QueryType {
  FACTUAL
  COMPARATIVE
  ENTITY_RELATIONSHIP
  PROCEDURAL
}

class QueryClassification {
  complexity QueryComplexity
  query_type QueryType
  reasoning string
}

// R6.1: multi-hop iterative retrieval decomposer output. The loop terminates
// on `done=true`; otherwise `next_sub_question` carries the next single
// retrieval target. `findings_from_last_hop` is a 1-3 sentence restatement
// of accumulated context (no invention).
class DecomposeResult {
  done bool
  next_sub_question string?
  findings_from_last_hop string
  reasoning string
}

// R2.1: RAPTOR cluster summarisation output. The summary is the public payload
// (embedded + persisted as a summary node); `reasoning` is trace-only and
// records what theme the cluster cohered around.
class SummarizeClusterResult {
  summary string @description("Domain-neutral summary of the thematic content shared across the cluster member texts. 1-3 paragraphs, at most max_summary_tokens. Does not invent content beyond what the members say.")
  reasoning string @description("Brief WHY: what theme the cluster is organized around. Trace-only.")
}
