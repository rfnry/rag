"""GraphIngestionConfig — consumer-overridable vocabularies for graph mapping.

The analyze-path graph mapper (`stores/graph/mapper.py`) used to hardcode
electrical/mechanical regex patterns and a 4-entry relationship-keyword map,
which silently dropped legitimate cross-references in any other domain.
This config lifts those choices into the consumer's hands.

Defaults are **empty** — the SDK ships no domain assumption. Consumers who
want the prior electrical/mechanical behavior can construct the config
with those patterns themselves. `unclassified_relation_default="MENTIONS"`
ensures that even when no keyword matches, edges still reach the graph as
generic references rather than silently disappearing; consumers who prefer
strict drop can pass `None`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rfnry_knowledge.exceptions import ConfigurationError


@dataclass
class GraphIngestionConfig:
    entity_type_patterns: list[tuple[str, str]] = field(default_factory=list)
    """List of (regex_string, entity_type) pairs. First match wins. When empty
    or no pattern matches, the entity's ``category`` field (lowercased) is used,
    falling back to ``"entity"`` when category is also empty."""

    relationship_keyword_map: dict[str, str] = field(default_factory=dict)
    """Maps lowercase substring keywords in a cross-reference's ``relationship``
    text to a Neo4j relation_type. All values must be in
    ``rfnry_knowledge.stores.graph.neo4j.ALLOWED_RELATION_TYPES``."""

    unclassified_relation_default: str | None = "MENTIONS"
    """When no keyword in ``relationship_keyword_map`` matches, relations are
    emitted with this relation_type. Set to ``None`` to drop unclassifiable
    cross-references entirely (the pre-Phase-D behavior). Must be in
    ``ALLOWED_RELATION_TYPES`` when not None."""

    def __post_init__(self) -> None:
        # Compile regexes eagerly so invalid patterns fail fast at config construction.
        for pattern, _type_name in self.entity_type_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigurationError(
                    f"GraphIngestionConfig.entity_type_patterns: invalid regex {pattern!r}: {exc}"
                ) from exc

        from rfnry_knowledge.stores.graph.neo4j import ALLOWED_RELATION_TYPES

        for keyword, rel in self.relationship_keyword_map.items():
            if rel not in ALLOWED_RELATION_TYPES:
                raise ConfigurationError(
                    f"GraphIngestionConfig.relationship_keyword_map: "
                    f"keyword={keyword!r} -> {rel!r} not in ALLOWED_RELATION_TYPES="
                    f"{sorted(ALLOWED_RELATION_TYPES)}"
                )
        if (
            self.unclassified_relation_default is not None
            and self.unclassified_relation_default not in ALLOWED_RELATION_TYPES
        ):
            raise ConfigurationError(
                f"GraphIngestionConfig.unclassified_relation_default="
                f"{self.unclassified_relation_default!r} not in ALLOWED_RELATION_TYPES="
                f"{sorted(ALLOWED_RELATION_TYPES)}"
            )
