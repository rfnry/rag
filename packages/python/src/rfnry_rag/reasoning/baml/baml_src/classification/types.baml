class ClassificationResult {
  category string
  confidence float @description("0.0-1.0 certainty score")
  reasoning string @description("1-2 sentence explanation")
  runner_up string? @description("second best category or null")
  runner_up_confidence float? @description("0.0-1.0 or null")
}

class SetClassificationBAML {
  set_name string @description("name of the category set this classification belongs to")
  category string @description("selected category from this set")
  confidence float @description("0.0-1.0 certainty score")
  reasoning string @description("1-2 sentence explanation")
  runner_up string? @description("second best category or null")
  runner_up_confidence float? @description("0.0-1.0 or null")
}

class ClassificationSetResultBAML {
  classifications SetClassificationBAML[] @description("one classification per category set")
}
