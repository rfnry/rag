class ComplianceViolationResult {
  dimension string @description("which compliance dimension was violated")
  description string @description("specific description of the breach")
  severity string @description("high, medium, or low")
  suggestion string? @description("how to fix the violation, or null")
}

class ComplianceJudgeResult {
  overall_score float @description("0.0-1.0 overall compliance score")
  reasoning string @description("2-3 sentences explaining the evaluation")
  dimension_scores map<string, float> @description("compliance score per dimension")
  violations ComplianceViolationResult[] @description("specific breaches found, empty if fully compliant")
}
