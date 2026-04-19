class RelevanceJudgment {
  relevant bool
  reasoning string
  needs_clarification bool
  clarifying_question string?
  clarifying_options string[]?
}

class DocumentSynthesis {
  cross_references SynthesisCrossReference[]
  page_clusters SynthesisPageCluster[]
  document_summary string
}

class SynthesisCrossReference {
  source_page int
  target_page int
  relationship string
  shared_entities string[]
}

class SynthesisPageCluster {
  pages int[]
  reason string
}
