class ToolFetchPages {
  pages string @description("page ranges to retrieve, e.g. '5-7' or '3,8,12-15'")
  reasoning string @description("why these pages are likely relevant to the query")
}

class ToolDrillDown {
  node_id string @description("ID of the node to explore deeper")
  reasoning string @description("why this subtree needs closer inspection")
}

class ToolResolvedPages {
  pages string @description("final relevant page ranges that answer the query")
  reasoning string @description("summary of how these pages address the query")
}

function TreeRetrievalStep(query: string, tree_structure: string, accumulated_context: string) -> ToolFetchPages | ToolDrillDown | ToolResolvedPages {
  client Default
  prompt #"
    You are a document retrieval specialist. Given a query and a document's
    hierarchical structure, decide your next action to find the most relevant
    pages.

    QUERY: {{ query }}

    DOCUMENT STRUCTURE:
    {{ tree_structure }}

    {% if accumulated_context %}
    PAGES ALREADY RETRIEVED:
    {{ accumulated_context }}
    {% endif %}

    AVAILABLE ACTIONS:
    - ToolFetchPages: retrieve specific pages to read their content and
      gather evidence. Use this when you need to inspect page content
      before making a final decision.
    - ToolDrillDown: zoom into a specific subtree node to see its children
      in more detail. Use this when a section looks promising but you need
      to see its internal structure.
    - ToolResolvedPages: declare the final set of relevant pages. Use this
      when you have enough information to confidently identify which pages
      answer the query.

    Think step by step:
    1. Consider which sections are most likely relevant based on titles and summaries
    2. If you have already retrieved pages, assess whether they answer the query
    3. Choose the action that gets you closer to a confident answer

    {{ ctx.output_format }}
  "#
}
