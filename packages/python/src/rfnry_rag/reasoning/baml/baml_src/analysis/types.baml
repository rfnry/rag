class DimensionResultBAML {
  name string @description("matches the consumer-defined dimension name")
  value string @description("score value matching the consumer-defined scale")
  confidence float @description("0.0-1.0 confidence in this assessment")
  reasoning string @description("brief explanation for this score")
}

class EntityBAML {
  type string @description("matches the consumer-defined entity type name")
  value string @description("the extracted entity value")
  context string @description("surrounding text where the entity was found")
}

class RetrievalHintBAML {
  query string @description("suggested search query for retrieval")
  knowledge_scope string? @description("which knowledge domain to search, or null")
  reasoning string @description("why this retrieval would be helpful")
  priority float @description("0.0-1.0 importance of this retrieval")
}

class AnalysisTextResult {
  primary_intent string @description("the main purpose or goal of this text")
  confidence float @description("0.0-1.0 confidence in the primary intent")
  summary string? @description("1-2 sentence summary, or null if not requested")
  dimensions DimensionResultBAML[] @description("scored dimensions, empty if none requested")
  entities EntityBAML[] @description("extracted entities, empty if none requested")
  retrieval_hints RetrievalHintBAML[] @description("suggested retrieval queries, empty if none requested")
}

class IntentShiftResult {
  from_intent string @description("the previous intent before the shift")
  to_intent string @description("the new intent after the shift")
  at_message int @description("0-based index in the message list where the shift occurred")
  reasoning string @description("brief explanation of why this is an intent shift")
}

class AnalysisContextResult {
  primary_intent string @description("the main purpose or goal of this context")
  confidence float @description("0.0-1.0 confidence in the primary intent")
  summary string? @description("1-2 sentence summary, or null if not requested")
  dimensions DimensionResultBAML[] @description("scored dimensions, empty if none requested")
  entities EntityBAML[] @description("extracted entities, empty if none requested")
  retrieval_hints RetrievalHintBAML[] @description("suggested retrieval queries, empty if none requested")
  intent_shifts IntentShiftResult[] @description("intent changes detected, empty if none or not tracked")
  escalation_detected bool @description("whether escalation signals were detected")
  escalation_reasoning string? @description("explanation of escalation signals, null if none")
  resolution_status string @description("resolved, pending, or escalated")
}
