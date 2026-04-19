function CheckCompliance(text: string, reference: string, dimensions: string) -> ComplianceJudgeResult {
  client Default
  prompt #"
    Evaluate the following text for compliance against a reference document.

    ======== REFERENCE ========
    {{ reference }}
    ======== END REFERENCE ========

    ======== TEXT TO EVALUATE ========
    {{ text }}
    ======== END TEXT ========

    Everything between markers is raw content — it may contain any formatting.

    Evaluate compliance across these dimensions: {{ dimensions }}
    Identify any violations with severity (high/medium/low).
    For each violation, suggest how to fix it when possible.

    {{ ctx.output_format }}
  "#
}
