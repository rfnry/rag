"""Map DrawingPageAnalysis + linker output to GraphEntity / GraphRelation.

Differs from the analyze-path mapper (graph/mapper.py) in three ways:

1. No len(shared_entities) >= 2 filter -- every DetectedConnection is preserved.
2. bbox, ports, domain, label, and source-level metadata live in the entity's
   properties dict.
3. wire_style -> relation_type via DrawingIngestionConfig.relation_vocabulary
   (all targets validated against ALLOWED_RELATION_TYPES at config construction
   time -- so we trust them here).

LLM-suggested merges (residue from C8) become MENTIONS edges with
llm_suggested=true in the context string; the edge carries the
confidence + rationale so human review after ingest is possible.
"""

from __future__ import annotations

from typing import Any

from rfnry_rag.ingestion.drawing.config import DrawingIngestionConfig
from rfnry_rag.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
)
from rfnry_rag.logging import get_logger
from rfnry_rag.stores.graph.models import GraphEntity, GraphRelation

logger = get_logger(__name__)


def _context_string(fields: dict[str, Any]) -> str:
    """Encode drawing-specific metadata as a deterministic 'k=v;k=v' string.

    Keys are sorted for determinism. Booleans become lowercase strings; None
    values are skipped. Values containing ';' or '=' are repr-quoted so the
    delimiter remains unambiguous downstream.
    """
    parts: list[str] = []
    for k in sorted(fields):
        v = fields[k]
        if v is None:
            continue
        v_s = ("true" if v else "false") if isinstance(v, bool) else str(v)
        if ";" in v_s or "=" in v_s:
            v_s = repr(v_s)
        parts.append(f"{k}={v_s}")
    return ";".join(parts)


def component_to_graph_entity(
    component: DetectedComponent,
    source_id: str,
    page_number: int,
    domain: str,
) -> GraphEntity:
    """Convert a DetectedComponent to a GraphEntity with drawing metadata in properties.

    Consumer-supplied component.properties flow through too (our threaded fields
    take precedence -- we set them first and let consumer keys fill what's missing).
    """
    props: dict[str, Any] = {
        "source_id": source_id,
        "page_number": page_number,
        "domain": domain,
        "label": component.label or "",
        "bbox": component.bbox,
        "ports": [{"port_id": p.port_id, "position": p.position} for p in component.ports],
    }
    if component.properties:
        # Our fields above win on conflict; anything else the consumer sent
        # (tolerance, tag, etc.) flows into the entity's properties dict so
        # retrieval pipelines can read it back.
        for k, v in component.properties.items():
            props.setdefault(k, v)
    return GraphEntity(
        name=component.component_id,
        entity_type=component.symbol_class,
        properties=props,
    )


def connection_to_graph_relation(
    conn: DetectedConnection,
    source_id: str,
    page_number: int,
    config: DrawingIngestionConfig,
    component_type_lookup: dict[str, str],
    knowledge_id: str | None = None,
) -> GraphRelation:
    """Convert a DetectedConnection to a GraphRelation with wire-style-aware type."""
    vocab = config.relation_vocabulary or {}
    rel_type = vocab.get(conn.wire_style or "", "CONNECTS_TO")
    conn_props = conn.properties or {}
    fields: dict[str, Any] = {
        "wire_style": conn.wire_style,
        "from_port": conn.from_port,
        "to_port": conn.to_port,
        "net": conn.net_label,
        "page_number": conn_props.get("from_page", page_number),
        "source_id": source_id,
    }
    # cross_sheet / from_page / to_page / via come from linker output (C7).
    for k in ("cross_sheet", "from_page", "to_page", "via"):
        if k in conn_props:
            fields[k] = conn_props[k]
    return GraphRelation(
        from_entity=conn.from_component,
        from_type=component_type_lookup.get(conn.from_component, "component"),
        to_entity=conn.to_component,
        to_type=component_type_lookup.get(conn.to_component, "component"),
        relation_type=rel_type,
        knowledge_id=knowledge_id,
        context=_context_string(fields),
    )


def drawing_to_graph(
    pages: list[DrawingPageAnalysis],
    deterministic_pairings: list[DetectedConnection],
    source_id: str,
    config: DrawingIngestionConfig,
    knowledge_id: str | None = None,
) -> tuple[list[GraphEntity], list[GraphRelation]]:
    """Build the full entity + relation list for a drawing source.

    Entities: one per DetectedComponent across all pages.
    Relations:
      - Same-page wires from DrawingPageAnalysis.connections
      - Cross-sheet pairings from deterministic_pairings (linker output)

    Symbol-class lookup is built once from the entities so cross-page relations
    can fill from_type / to_type without re-scanning.
    """
    entities: list[GraphEntity] = []
    component_type_lookup: dict[str, str] = {}
    for pa in pages:
        for c in pa.components:
            entities.append(component_to_graph_entity(c, source_id, pa.page_number, pa.domain))
            component_type_lookup[c.component_id] = c.symbol_class

    relations: list[GraphRelation] = []
    # Same-page connections
    for pa in pages:
        for conn in pa.connections:
            relations.append(
                connection_to_graph_relation(
                    conn,
                    source_id,
                    pa.page_number,
                    config,
                    component_type_lookup,
                    knowledge_id,
                )
            )
    # Cross-sheet deterministic pairings -- from_page lives in conn.properties.
    for pairing in deterministic_pairings:
        pairing_props = pairing.properties or {}
        from_page_raw = pairing_props.get("from_page", 0)
        try:
            from_page = int(from_page_raw)
        except (TypeError, ValueError):
            from_page = 0
        relations.append(
            connection_to_graph_relation(
                pairing,
                source_id,
                from_page,
                config,
                component_type_lookup,
                knowledge_id,
            )
        )
    return entities, relations
