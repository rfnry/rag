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
