"""GraphIngestionConfig: bounds, defaults, consumer overrides, allowlist validation."""
import pytest

from rfnry_rag.common.errors import ConfigurationError
from rfnry_rag.retrieval.modules.ingestion.graph.config import GraphIngestionConfig


def test_default_config_has_empty_vocabularies() -> None:
    """Agnostic by default — no electrical/mechanical assumption."""
    cfg = GraphIngestionConfig()
    assert cfg.entity_type_patterns == []
    assert cfg.relationship_keyword_map == {}
    # Fallback is MENTIONS so edges aren't silently dropped on vocab miss
    assert cfg.unclassified_relation_default == "MENTIONS"


def test_consumer_can_supply_entity_type_patterns() -> None:
    cfg = GraphIngestionConfig(
        entity_type_patterns=[
            (r"\bmotor\b", "motor"),
            (r"\bvalve\b", "valve"),
        ],
    )
    assert len(cfg.entity_type_patterns) == 2


def test_consumer_can_supply_relationship_keyword_map() -> None:
    cfg = GraphIngestionConfig(
        relationship_keyword_map={"feed": "POWERED_BY", "control": "CONTROLLED_BY"},
    )
    assert cfg.relationship_keyword_map == {"feed": "POWERED_BY", "control": "CONTROLLED_BY"}


def test_relationship_keyword_map_values_must_be_in_allowlist() -> None:
    """Consumer cannot smuggle an arbitrary relation_type string into the graph."""
    with pytest.raises(ConfigurationError, match="(?i)not in.*ALLOWED_RELATION_TYPES"):
        GraphIngestionConfig(relationship_keyword_map={"bogus": "NOT_A_REAL_REL"})


def test_unclassified_relation_default_must_be_in_allowlist_or_none() -> None:
    with pytest.raises(ConfigurationError, match="(?i)not in.*ALLOWED_RELATION_TYPES"):
        GraphIngestionConfig(unclassified_relation_default="NOT_A_REAL_REL")
    # None is explicitly allowed — it means "drop on miss"
    cfg = GraphIngestionConfig(unclassified_relation_default=None)
    assert cfg.unclassified_relation_default is None


def test_entity_type_patterns_must_be_compilable_regex() -> None:
    with pytest.raises(ConfigurationError, match="(?i)invalid regex"):
        GraphIngestionConfig(entity_type_patterns=[(r"[invalid(regex", "type")])


def test_nested_into_ingestion_config() -> None:
    from rfnry_rag.retrieval.server import IngestionConfig
    cfg = IngestionConfig(graph=GraphIngestionConfig(
        entity_type_patterns=[(r"\bmotor\b", "motor")],
    ))
    assert cfg.graph is not None
    assert len(cfg.graph.entity_type_patterns) == 1


def test_graph_none_by_default() -> None:
    from rfnry_rag.retrieval.server import IngestionConfig
    cfg = IngestionConfig()
    assert cfg.graph is None   # opt-in


def test_registered_in_config_bounds_contract() -> None:
    """If the config has any int/float field we add later, the contract must catch it."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "_retrieval_config_bounds_contract",
        Path(__file__).parent / "test_config_bounds_contract.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert GraphIngestionConfig in mod._CONFIGS_TO_AUDIT
