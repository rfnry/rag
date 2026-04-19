class QueryAnalysis {
  keywords string[]
  entity_references string[]
  intent string
  domain_hint string?
}

class RankedChunk {
  index int
  score float
  reasoning string?
}
