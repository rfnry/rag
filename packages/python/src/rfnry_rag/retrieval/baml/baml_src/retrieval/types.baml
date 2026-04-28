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
