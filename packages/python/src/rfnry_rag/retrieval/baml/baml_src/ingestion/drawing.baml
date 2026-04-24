// Drawing-aware ingestion functions.
// Consumers can override symbol_library + off_page_patterns via DrawingIngestionConfig.

class Port {
  port_id string                    @description("label or index of the port")
  position int[]?                   @description("[x, y] in page pixel coords, optional")
}

class DetectedComponent {
  component_id string               @description("stable within-page ID: 'R1', 'V-101', or synthesised")
  symbol_class string               @description("resistor | capacitor | valve | pump | junction | off_page_connector | ...")
  label string?                     @description("visible printed text near the symbol")
  bbox int[]                        @description("[x, y, w, h] page pixel coords")
  ports Port[]
  properties map<string, string>?
}

class DetectedConnection {
  from_component string
  from_port string?
  to_component string
  to_port string?
  net_label string?                 @description("'N5', '+24V', or null for unnamed")
  wire_style string?                @description("solid | dashed | pneumatic | hydraulic | signal")
}

class OffPageConnector {
  tag string                        @description("'/A2', 'OPC-1', 'to sheet 3 zone B2'")
  bound_component string            @description("on-page component_id it terminates at")
  target_hint string?               @description("parsed 'sheet 3, zone B2' hint if present")
}

class DrawingPageAnalysis {
  page_number int
  sheet_number string?
  zone_grid string?                 @description("'A-H horizontal, 1-8 vertical' or similar grid spec")
  domain string                     @description("electrical | p_and_id | mechanical | mixed")
  components DetectedComponent[]
  connections DetectedConnection[]
  off_page_connectors OffPageConnector[]
  title_block map<string, string>?
  notes string[]
  page_type string                  @description("drawing | title_page | bom | text | mixed")
}

function AnalyzeDrawingPage(
  page_image: image,
  domain_hint: string,
  symbol_library: string,
  off_page_patterns: string,
) -> DrawingPageAnalysis {
  client "openai/gpt-4o"
  prompt #"
    You are a technical-drawing analyst. Extract the structural content of this sheet.

    DOMAIN HINT (electrical, p_and_id, mechanical, mixed, or auto — guide detection but
    override if the visible content disagrees):
    {{ domain_hint }}

    SYMBOL LIBRARY — the consumer-supplied component classes you should prefer when
    labelling symbols. You MAY add classes not listed here if the drawing clearly shows
    them, but prefer the listed classes verbatim when the match is clear.
    ======== SYMBOL LIBRARY START ========
    {{ symbol_library }}
    ======== SYMBOL LIBRARY END ========

    OFF-PAGE CONNECTOR PATTERNS — regex/glob patterns indicating cross-sheet references
    (e.g. "/A2", "OPC-1", "to sheet 3 zone B2"). Use these to populate
    off_page_connectors[] precisely.
    ======== OFF PAGE CONNECTOR PATTERNS START ========
    {{ off_page_patterns }}
    ======== OFF PAGE CONNECTOR PATTERNS END ========

    OUTPUT RULES
    - components[]: one entry per recognised symbol on the sheet. bbox in page pixel
      coordinates (x, y, w, h), origin top-left. ports[] if visibly distinguishable.
    - connections[]: one entry per visible wire/line between components. If a wire
      terminates at an off-page connector, list the off-page connector as the to_component
      (or from_component) and ensure that same component_id appears in off_page_connectors[].
    - Do NOT interpret or simulate the circuit. Only describe what is visible.
    - Output JSON matching the DrawingPageAnalysis schema exactly.

    {{ ctx.output_format }}
  "#
}

class DrawingSetSynthesis {
  ambiguous_component_merges Merge[]
  narrative_cross_references NarrativeXref[]
  document_summary string
}

class Merge {
  page_a int
  component_a string
  page_b int
  component_b string
  confidence float
  rationale string
}

class NarrativeXref {
  source_page int
  target_page int
  description string
}

function SynthesizeDrawingSet(
  per_page_digest: string,
  already_linked: string,
) -> DrawingSetSynthesis {
  client "openai/gpt-4o"
  prompt #"
    A drawing set has been analysed page by page. Deterministic cross-sheet linking has
    already resolved the unambiguous off-page-connector tags and exact-match component IDs.
    Your job is to surface the RESIDUE: ambiguous component merges, narrative relationships
    between sheets, and any obvious links the deterministic pass missed.

    PER-PAGE DIGEST (trusted):
    ======== DIGEST START ========
    {{ per_page_digest }}
    ======== DIGEST END ========

    ALREADY LINKED (do not duplicate these):
    ======== LINKED START ========
    {{ already_linked }}
    ======== LINKED END ========

    Rules:
    - ambiguous_component_merges: only include when confidence > 0.5. Cite the rationale.
    - narrative_cross_references: textual relationships ("valve V-101 is documented on
      sheet 5"), not wire connections (those are deterministic).
    - Do not output a merge for two components that are already linked.

    {{ ctx.output_format }}
  "#
}
