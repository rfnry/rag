class AnswerQualityJudgment {
  score float @description("Answer quality score from 0.0 (completely wrong) to 1.0 (perfect)")
  reasoning string @description("Brief explanation of the score")
}
