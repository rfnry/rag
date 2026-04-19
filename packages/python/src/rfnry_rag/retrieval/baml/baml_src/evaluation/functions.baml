function JudgeAnswerQuality(query: string, prediction: string, reference: string) -> AnswerQualityJudgment {
  client Default
  prompt #"
    You are an impartial answer quality judge. Compare the predicted answer against
    a reference answer for the given question, and rate the quality.

    ======== QUESTION ========
    {{ query }}
    ======== END QUESTION ========

    ======== REFERENCE ANSWER ========
    {{ reference }}
    ======== END REFERENCE ========

    ======== PREDICTED ANSWER ========
    {{ prediction }}
    ======== END PREDICTED ========

    Scoring guidelines:
    - 1.0: The prediction is correct and complete — it answers the question as well as or better than the reference.
    - 0.7-0.9: The prediction is mostly correct but missing minor details or has slight inaccuracies.
    - 0.4-0.6: The prediction is partially correct — it addresses the question but misses key information.
    - 0.1-0.3: The prediction is mostly wrong but contains a small amount of relevant information.
    - 0.0: The prediction is completely wrong, irrelevant, or contradicts the reference.

    Focus on factual correctness, not style or formatting. A prediction that conveys
    the same information in different words should score high.

    {{ ctx.output_format }}
  "#
}
