class RetrievalNecessityJudgment {
  should_retrieve bool @description("True if the query requires domain-specific knowledge retrieval")
  confidence float @description("Confidence in the judgment from 0.0 to 1.0")
  reasoning string @description("Brief explanation of why retrieval is or is not needed")
}
