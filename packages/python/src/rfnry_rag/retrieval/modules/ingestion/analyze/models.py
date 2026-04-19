from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveredEntity:
    name: str
    category: str
    context: str
    value: str | None = None


@dataclass
class DiscoveredTable:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    title: str | None = None


@dataclass
class PageAnalysis:
    page_number: int
    description: str
    entities: list[DiscoveredEntity] = field(default_factory=list)
    tables: list[DiscoveredTable] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    page_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossReference:
    source_page: int
    target_page: int
    relationship: str
    shared_entities: list[str] = field(default_factory=list)


@dataclass
class PageCluster:
    pages: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class DocumentSynthesis:
    cross_references: list[CrossReference] = field(default_factory=list)
    page_clusters: list[PageCluster] = field(default_factory=list)
    document_summary: str = ""
