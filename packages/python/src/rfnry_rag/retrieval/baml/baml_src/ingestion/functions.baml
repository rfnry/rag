function AnalyzePage(page_image: image) -> PageAnalysis {
  client Default
  prompt #"
    You are a technical document analyst. Analyze this page image and extract all structured information.

    Instructions:
    - Describe what the page contains (schematic, table, diagram, text, etc.)
    - Extract all identifiable entities (components, parts, valves, motors, etc.) with their categories
    - Extract all tables with their column headers and row data
    - Note any annotations, callouts, or reference marks
    - Identify the page type (schematic, bom_table, wiring_diagram, assembly, detail, p_and_id, text, etc.)

    ======== IMPORTANT ========
    - Extract what you actually see — do NOT invent or hallucinate values
    - If a value is unclear, use the entity's context field to note uncertainty
    - Categories should be descriptive (component, valve, motor, wire, terminal, dimension, etc.)
    ======== END IMPORTANT ========

    {{ ctx.output_format }}
  "#
}

function ExtractEntitiesFromText(text: string) -> PageAnalysis {
  client Default
  prompt #"
    You are a technical document analyst. Extract all identifiable entities and structured information from this text.

    ======== TEXT ========
    {{ text }}
    ======== END TEXT ========

    Instructions:
    - Describe what this text contains in the description field
    - Extract all identifiable entities (components, parts, equipment, materials, specifications, etc.)
    - For each entity, provide:
      - name: the identifier or label (e.g., "Motor M1", "SAE 30", "RV-2201")
      - category: descriptive type (component, valve, motor, material, specification, part_number, etc.)
      - value: any associated measurement or specification (e.g., "480V", "175 PSI", "3450 RPM")
      - context: brief note about how this entity appears in the text
    - Extract any tables with column headers and row data
    - Set page_type to "text"
    - Note any reference marks, part numbers, or cross-references in annotations

    ======== IMPORTANT ========
    - Extract what you actually see — do NOT invent or hallucinate entities or values
    - If a value is unclear, use the entity's context field to note uncertainty
    - Categories should be descriptive (component, valve, motor, material, specification, part_number, etc.)
    ======== END IMPORTANT ========

    {{ ctx.output_format }}
  "#
}
