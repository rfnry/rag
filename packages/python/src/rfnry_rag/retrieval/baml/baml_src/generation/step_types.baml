class ReasoningStep {
  text string @description("The generated reasoning step or final answer")
  is_final bool @description("True if this is a final answer, false if more reasoning is needed")
}
