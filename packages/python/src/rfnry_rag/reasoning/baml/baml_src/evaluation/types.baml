class JudgeResult {
  overall_score float @description("0.0-1.0 holistic quality score")
  reasoning string @description("2-3 sentences on strengths and weaknesses")
  dimension_scores map<string, float> @description("score per evaluation dimension")
}
