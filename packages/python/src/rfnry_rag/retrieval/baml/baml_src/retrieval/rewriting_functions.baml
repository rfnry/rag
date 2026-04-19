function GenerateHypotheticalDocument(query: string) -> HypotheticalDocument {
  client Default
  prompt #"
    Given the following question, write a short passage (2-3 sentences) that would
    appear in a technical document and directly answer the question. Write it as
    factual content, not as a response to a question. Use specific technical
    terminology that would appear in real documentation.

    Question: {{ query }}

    {{ ctx.output_format }}
  "#
}

function GenerateQueryVariants(query: string, num_variants: int) -> QueryVariants {
  client Default
  prompt #"
    Generate {{ num_variants }} alternative search queries for the following question.
    Each variant should approach the topic from a different angle, use different
    keywords, or rephrase the question to capture different aspects. Keep each
    variant concise (1-2 sentences).

    Original query: {{ query }}

    {{ ctx.output_format }}
  "#
}

function GenerateStepBackQuery(query: string) -> StepBackQuery {
  client Default
  prompt #"
    Given the following specific question, generate a broader, more general version
    of the question that would retrieve background context and overview information.
    The step-back question should be about the general topic, not the specific detail.

    Specific question: {{ query }}

    {{ ctx.output_format }}
  "#
}
