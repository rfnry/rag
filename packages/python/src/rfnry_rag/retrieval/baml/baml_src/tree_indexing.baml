class TocDetectionResult {
  has_toc bool @description("whether the page contains a table of contents")
  has_page_numbers bool @description("whether the TOC entries include page numbers")
}

class TocEntry {
  structure string @description("hierarchical numbering, e.g. '1', '1.1', '1.1.2'")
  title string @description("section title text")
  page int? @description("page number if present in TOC, null otherwise")
}

class TocStructure {
  entries TocEntry[] @description("ordered list of TOC entries with hierarchy")
}

class ExtractedSection {
  structure string @description("hierarchical numbering")
  title string @description("section title")
  start_page int @description("page where this section begins")
}

class ExtractedStructure {
  sections ExtractedSection[] @description("document sections discovered from content")
}

function DetectTableOfContents(page_text: string) -> TocDetectionResult {
  client Default
  prompt #"
    Analyze the following page from a document. Determine if it contains
    a table of contents (TOC).

    PAGE TEXT:
    {{ page_text }}

    A table of contents typically lists section titles with optional page numbers.
    It may appear as a formatted list of chapters, sections, or headings.

    {{ ctx.output_format }}
  "#
}

function ParseTableOfContents(toc_text: string) -> TocStructure {
  client Default
  prompt #"
    Parse this table of contents into a structured list of entries.
    Assign hierarchical structure codes (e.g., "1", "1.1", "1.1.2")
    based on the indentation, numbering, or visual hierarchy.

    TABLE OF CONTENTS:
    {{ toc_text }}

    Rules:
    - Preserve the original hierarchy (chapters > sections > subsections)
    - Include page numbers if they appear in the TOC
    - Set page to null if no page number is present for an entry
    - Use dot notation for structure: "1", "1.1", "1.2", "2", "2.1", etc.

    {{ ctx.output_format }}
  "#
}

function FindSectionStart(section_title: string, pages_text: string) -> int {
  client Default
  prompt #"
    Find the page number where the following section begins.

    SECTION TITLE: {{ section_title }}

    DOCUMENT PAGES:
    {{ pages_text }}

    Return the page number (as shown in the page headers) where this section
    title first appears as a heading. Return only the integer page number.

    {{ ctx.output_format }}
  "#
}

function VerifySectionPosition(title: string, page_text: string) -> bool {
  client Default
  prompt #"
    Does the following section title appear as a heading on this page?

    SECTION TITLE: {{ title }}

    PAGE TEXT:
    {{ page_text }}

    Return true if the title (or a close match) appears as a section heading
    on this page. Return false otherwise.

    {{ ctx.output_format }}
  "#
}

function ExtractDocumentStructure(pages_text: string) -> ExtractedStructure {
  client Default
  prompt #"
    This document does not have a table of contents. Analyze the content
    and extract the document structure by identifying sections and their
    starting pages.

    DOCUMENT PAGES:
    {{ pages_text }}

    Rules:
    - Identify section headings from formatting, numbering, or content shifts
    - Assign hierarchical structure codes ("1", "1.1", "1.2", "2", etc.)
    - Record the page number where each section begins
    - Only include significant sections, not every paragraph

    {{ ctx.output_format }}
  "#
}

function ContinueDocumentStructure(existing_structure: string, pages_text: string) -> ExtractedStructure {
  client Default
  prompt #"
    Continue extracting the document structure. The following sections
    have already been identified from earlier pages:

    EXISTING STRUCTURE:
    {{ existing_structure }}

    NEW PAGES:
    {{ pages_text }}

    Continue from where the existing structure left off. Use the same
    numbering scheme and hierarchy level. Only return NEW sections found
    in the new pages.

    {{ ctx.output_format }}
  "#
}

function GenerateNodeSummary(title: string, section_text: string) -> string {
  client Default
  prompt #"
    Write a concise summary (2-3 sentences) of this document section.
    Focus on the key topics, findings, or information covered.

    SECTION: {{ title }}

    TEXT:
    {{ section_text }}

    Return only the summary text, no formatting or labels.

    {{ ctx.output_format }}
  "#
}

function GenerateDocDescription(tree_structure: string) -> string {
  client Default
  prompt #"
    Based on this document's structure, write a single sentence describing
    what this document is about.

    DOCUMENT STRUCTURE:
    {{ tree_structure }}

    Return only the one-sentence description.

    {{ ctx.output_format }}
  "#
}
