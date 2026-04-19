function JudgeRetrievalNecessity(query: string, knowledge_description: string) -> RetrievalNecessityJudgment {
  client Default
  prompt #"
    You are a retrieval necessity classifier. Decide whether the following query
    requires searching a domain-specific knowledge base, or whether it can be
    answered from general knowledge alone.

    ======== USER QUERY ========
    {{ query }}
    ======== END QUERY ========

    ======== KNOWLEDGE BASE DESCRIPTION ========
    {{ knowledge_description }}
    ======== END DESCRIPTION ========

    Decision criteria:
    - should_retrieve=true when the query asks about specific products, part numbers,
      procedures, specifications, or domain-specific information that would be found
      in the described knowledge base.
    - should_retrieve=false when the query is about general knowledge, common facts,
      conversational pleasantries, or topics clearly outside the knowledge base scope.
    - When in doubt, prefer should_retrieve=true (it is safer to search and find
      nothing than to skip search and miss relevant information).

    {{ ctx.output_format }}
  "#
}
