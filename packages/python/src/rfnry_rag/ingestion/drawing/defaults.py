"""Default vocabularies for DrawingIngestion.

Ships IEC 60617 (electrical) + ISA 5.1 (P&ID) + common mechanical defaults as a
baseline so the SDK works out-of-the-box. Every entry here is overridable via
``DrawingIngestionConfig.symbol_library`` (full replace) or
``DrawingIngestionConfig.symbol_library_extensions`` (additive merge).
"""

from __future__ import annotations

# IEC 60617 electrical drawing symbols (common subset).
_ELECTRICAL_SYMBOLS = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "led",
    "transistor_npn",
    "transistor_pnp",
    "mosfet_n",
    "mosfet_p",
    "opamp",
    "ic_dip",
    "ic_soic",
    "switch_spst",
    "switch_spdt",
    "relay",
    "battery",
    "voltage_source",
    "current_source",
    "ground",
    "junction",
    "off_page_connector",
    "fuse",
    "transformer",
    "motor_dc",
    "motor_ac",
    "lamp",
]

# ISA 5.1 P&ID symbols (common subset).
_P_AND_ID_SYMBOLS = [
    "valve_ball",
    "valve_gate",
    "valve_globe",
    "valve_check",
    "valve_control",
    "pump_centrifugal",
    "pump_positive_displacement",
    "tank",
    "vessel",
    "heat_exchanger",
    "compressor",
    "filter",
    "instrument_field",
    "instrument_panel",
    "flow_transmitter",
    "pressure_transmitter",
    "temperature_transmitter",
    "controller_local",
    "controller_distributed",
    "off_page_connector",
    "pipe_tee",
    "pipe_reducer",
]

# Mechanical drawing common call-outs.
_MECHANICAL_SYMBOLS = [
    "bolt",
    "nut",
    "washer",
    "gear",
    "bearing",
    "shaft",
    "dimension_line",
    "leader_line",
    "section_view",
    "datum_feature",
    "surface_finish_symbol",
    "gd_t_symbol",
]

DEFAULT_SYMBOL_LIBRARY: dict[str, list[str]] = {
    "electrical": _ELECTRICAL_SYMBOLS,
    "p_and_id": _P_AND_ID_SYMBOLS,
    "mechanical": _MECHANICAL_SYMBOLS,
}

# Off-page-connector regex patterns.
DEFAULT_OFF_PAGE_CONNECTOR_PATTERNS: list[str] = [
    r"/[A-Z]\d+",  # "/A2", "/B12"
    r"OPC[-_]\d+",  # "OPC-1", "OPC_12"
    r"(?i)to\s+sheet\s+\d+(\s+zone\s+[A-H]\d+)?",  # "to sheet 3", "to sheet 3 zone B2"
    r"(?i)continues\s+(on|at)\s+sheet\s+\d+",  # "continues on sheet 4"
    r"Z\d+\.[A-H]\d+",  # "Z3.B2" style zone refs
]

# Wire-style -> Neo4j relation type.
# All targets MUST be in rfnry_rag.stores.graph.neo4j.ALLOWED_RELATION_TYPES
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
