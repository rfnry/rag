"""Python dataclasses mirroring the BAML DrawingPageAnalysis schema.

These types carry a drawing page's structured content in memory and serialize
to/from plain dicts for JSONB persistence in the ``knowledge_page_analyses.data``
column. The BAML-to-dataclass converters (``from_baml``/``to_baml``) are
intentionally NOT defined here — they land in C5 when the extract phase
actually calls ``AnalyzeDrawingPage``. For C3 we only need in-Python dict
round-trips so the service skeleton can persist and reload analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Port:
    port_id: str
    position: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"port_id": self.port_id, "position": self.position}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        return cls(port_id=d["port_id"], position=d.get("position"))


@dataclass
class DetectedComponent:
    component_id: str
    symbol_class: str
    bbox: list[int]
    label: str | None = None
    ports: list[Port] = field(default_factory=list)
    properties: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "symbol_class": self.symbol_class,
            "label": self.label,
            "bbox": self.bbox,
            "ports": [p.to_dict() for p in self.ports],
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectedComponent:
        return cls(
            component_id=d["component_id"],
            symbol_class=d["symbol_class"],
            label=d.get("label"),
            bbox=list(d.get("bbox", [])),
            ports=[Port.from_dict(p) for p in d.get("ports", [])],
            properties=d.get("properties"),
        )


@dataclass
class DetectedConnection:
    from_component: str
    to_component: str
    from_port: str | None = None
    to_port: str | None = None
    net_label: str | None = None
    wire_style: str | None = None
    # ``properties`` is a local extension not in BAML; the linker uses it in C7
    # to attach cross-sheet pair metadata (matched off-page-connector tag,
    # confidence, etc.) before the graph write in C10.
    properties: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_component": self.from_component,
            "from_port": self.from_port,
            "to_component": self.to_component,
            "to_port": self.to_port,
            "net_label": self.net_label,
            "wire_style": self.wire_style,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectedConnection:
        return cls(
            from_component=d["from_component"],
            to_component=d["to_component"],
            from_port=d.get("from_port"),
            to_port=d.get("to_port"),
            net_label=d.get("net_label"),
            wire_style=d.get("wire_style"),
            properties=d.get("properties"),
        )


@dataclass
class OffPageConnector:
    tag: str
    bound_component: str | None = None
    target_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "bound_component": self.bound_component,
            "target_hint": self.target_hint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OffPageConnector:
        return cls(
            tag=d["tag"],
            bound_component=d.get("bound_component"),
            target_hint=d.get("target_hint"),
        )


@dataclass
class DrawingPageAnalysis:
    page_number: int
    domain: str
    page_type: str
    components: list[DetectedComponent] = field(default_factory=list)
    connections: list[DetectedConnection] = field(default_factory=list)
    off_page_connectors: list[OffPageConnector] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sheet_number: str | None = None
    zone_grid: str | None = None
    title_block: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "sheet_number": self.sheet_number,
            "zone_grid": self.zone_grid,
            "domain": self.domain,
            "components": [c.to_dict() for c in self.components],
            "connections": [c.to_dict() for c in self.connections],
            "off_page_connectors": [o.to_dict() for o in self.off_page_connectors],
            "title_block": self.title_block,
            "notes": list(self.notes),
            "page_type": self.page_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DrawingPageAnalysis:
        return cls(
            page_number=d["page_number"],
            sheet_number=d.get("sheet_number"),
            zone_grid=d.get("zone_grid"),
            domain=d["domain"],
            components=[DetectedComponent.from_dict(c) for c in d.get("components", [])],
            connections=[DetectedConnection.from_dict(c) for c in d.get("connections", [])],
            off_page_connectors=[OffPageConnector.from_dict(o) for o in d.get("off_page_connectors", [])],
            title_block=d.get("title_block"),
            notes=list(d.get("notes", [])),
            page_type=d["page_type"],
        )
