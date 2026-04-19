class HypotheticalDocument {
  passage string @description("A hypothetical document passage answering the question")
}

class QueryVariants {
  variants string[] @description("Alternative query formulations")
}

class StepBackQuery {
  broader_query string @description("A broader version of the original question")
}
