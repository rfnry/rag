"""DrawingIngestionConfig: bounds, defaults, consumer overrides."""

import pytest

from rfnry_knowledge.exceptions import ConfigurationError


def test_default_symbol_library_covers_iec_and_isa() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    cfg = DrawingIngestionConfig(enabled=True)
    lib = cfg.symbol_library
    assert lib is not None
    assert "resistor" in lib["electrical"]
    assert "valve" in str(lib["p_and_id"])  # any "valve_*" entry is fine


def test_consumer_can_fully_replace_symbol_library() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    cfg = DrawingIngestionConfig(
        enabled=True,
        symbol_library={"custom": ["widget_a", "widget_b"]},
    )
    assert cfg.symbol_library == {"custom": ["widget_a", "widget_b"]}
    assert cfg.symbol_library is not None
    assert "electrical" not in cfg.symbol_library


def test_consumer_can_extend_default_symbol_library() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    cfg = DrawingIngestionConfig(
        enabled=True,
        symbol_library_extensions={"electrical": ["custom_ic_pack"]},
    )
    assert cfg.symbol_library is not None
    assert "custom_ic_pack" in cfg.symbol_library["electrical"]
    # Extensions sit alongside defaults, not replacing them
    assert "resistor" in cfg.symbol_library["electrical"]


def test_off_page_patterns_default_non_empty() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    cfg = DrawingIngestionConfig(enabled=True)
    assert cfg.off_page_connector_patterns is not None
    assert len(cfg.off_page_connector_patterns) >= 3


def test_relation_vocabulary_default_maps_standard_wire_styles() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    cfg = DrawingIngestionConfig(enabled=True)
    assert cfg.relation_vocabulary is not None
    assert cfg.relation_vocabulary["pneumatic"] == "FLOWS_TO"
    assert cfg.relation_vocabulary["signal"] == "CONNECTS_TO"


def test_dpi_bounds() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, dpi=100)  # below 150
    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, dpi=700)  # above 600


def test_analyze_concurrency_bounds() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, analyze_concurrency=0)
    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, analyze_concurrency=101)


def test_graph_write_batch_size_bounds() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, graph_write_batch_size=0)
    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, graph_write_batch_size=10_001)


def test_default_domain_enum_validation() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(enabled=True, default_domain="bogus")  # type: ignore[arg-type]


def test_relation_vocabulary_only_allows_graph_store_allowlisted_types() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig
    from rfnry_knowledge.stores.graph.neo4j import ALLOWED_RELATION_TYPES

    cfg = DrawingIngestionConfig(enabled=True)
    assert cfg.relation_vocabulary is not None
    for wire_style, rel in cfg.relation_vocabulary.items():
        assert rel in ALLOWED_RELATION_TYPES, f"wire_style={wire_style!r} -> {rel!r} not in allowlist"


def test_relation_vocabulary_rejects_invalid_relation_type() -> None:
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    with pytest.raises(ConfigurationError):
        DrawingIngestionConfig(
            enabled=True,
            relation_vocabulary={"weird_style": "NOT_A_REAL_RELATION"},
        )


def test_disabled_config_skips_validation() -> None:
    """enabled=False should skip bounds checks; defaults-only is always safe."""
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig

    # enabled=False with a pathological value should not raise (config is inert)
    cfg = DrawingIngestionConfig(enabled=False, dpi=9999)
    assert cfg.enabled is False


def test_drawing_method_carries_config_into_ingestion_config() -> None:
    from unittest.mock import MagicMock

    from rfnry_knowledge.config import IngestionConfig
    from rfnry_knowledge.config.drawing import DrawingIngestionConfig
    from rfnry_knowledge.ingestion.methods.drawing import DrawingIngestion

    method = DrawingIngestion(
        config=DrawingIngestionConfig(enabled=True),
        store=MagicMock(),
        embeddings=MagicMock(),
    )
    cfg = IngestionConfig(methods=[method])
    drawing_methods = [m for m in cfg.methods if isinstance(m, DrawingIngestion)]
    assert len(drawing_methods) == 1
    assert drawing_methods[0]._config.enabled is True
    assert drawing_methods[0]._config.dpi == 400


def test_ingestion_config_methods_empty_by_default() -> None:
    from rfnry_knowledge.config import IngestionConfig
    from rfnry_knowledge.ingestion.methods.drawing import DrawingIngestion

    cfg = IngestionConfig()
    assert [m for m in cfg.methods if isinstance(m, DrawingIngestion)] == []
