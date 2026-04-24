"""Default vocabularies for DrawingIngestion.

Ships IEC 60617 (electrical) + ISA 5.1 (P&ID) + common mechanical defaults as a
baseline so the SDK works out-of-the-box. Every entry here is overridable via
``DrawingIngestionConfig.symbol_library`` (full replace) or
``DrawingIngestionConfig.symbol_library_extensions`` (additive merge).
"""

from __future__ import annotations

# IEC 60617 electrical + ISA 5.1 P&ID + common mechanical symbols.
# The lists are intentionally small and conservative; consumers extend them via
# ``symbol_library_extensions`` or replace wholesale via ``symbol_library``.
DEFAULT_SYMBOL_LIBRARY: dict[str, list[str]] = {
    "electrical": [
        "resistor",
        "capacitor",
        "inductor",
        "diode",
        "transistor",
        "ground",
        "battery",
        "switch",
        "fuse",
        "relay",
        "transformer",
        "motor",
        "generator",
        "lamp",
        "led",
        "ic_pack",
        "connector",
        "terminal",
    ],
    "p_and_id": [
        "valve_gate",
        "valve_globe",
        "valve_ball",
        "valve_check",
        "valve_butterfly",
        "valve_control",
        "pump_centrifugal",
        "pump_positive_displacement",
        "compressor",
        "heat_exchanger",
        "tank",
        "vessel",
        "reactor",
        "filter",
        "instrument_indicator",
        "instrument_transmitter",
        "instrument_controller",
        "pipe",
        "flange",
    ],
    "mechanical": [
        "bearing",
        "gear",
        "shaft",
        "coupling",
        "bolt",
        "nut",
        "washer",
        "spring",
        "pulley",
        "belt",
        "chain",
        "sprocket",
    ],
}

# Off-page-connector regex patterns. Matches idioms like:
#   "/A2"          — slash-prefixed sheet/zone reference
#   "OPC-N"        — explicit off-page-connector tag
#   "to sheet 2 zone A3"
#   "see sheet 4"
#   "cont. on 3/B2"
DEFAULT_OFF_PAGE_CONNECTOR_PATTERNS: list[str] = [
    r"/[A-Z]\d+",
    r"\bOPC-\d+\b",
    r"(?i)\bto\s+sheet\s+\d+(?:\s+zone\s+[A-Z]?\d+)?",
    r"(?i)\bsee\s+sheet\s+\d+",
    r"(?i)\bcont(?:inued|\.)\s+on\s+\d+(?:/[A-Z]?\d+)?",
    r"(?i)\bfrom\s+sheet\s+\d+",
]

# Wire-style -> Neo4j relation type.
# All targets MUST be in rfnry_rag.retrieval.stores.graph.neo4j.ALLOWED_RELATION_TYPES
# (currently: CONNECTS_TO, POWERED_BY, CONTROLLED_BY, FEEDS, FLOWS_TO, REFERENCES, MENTIONS).
DEFAULT_RELATION_VOCABULARY: dict[str, str] = {
    "solid": "CONNECTS_TO",
    "dashed": "CONNECTS_TO",
    "signal": "CONNECTS_TO",
    "pneumatic": "FLOWS_TO",
    "hydraulic": "FLOWS_TO",
    "power": "POWERED_BY",
    "control": "CONTROLLED_BY",
    "feed": "FEEDS",
}
