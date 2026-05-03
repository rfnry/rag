from unittest.mock import MagicMock

from rfnry_knowledge.knowledge.engine import KnowledgeEngine


def test_enabled_flows_uses_snake_case_only() -> None:
    engine = KnowledgeEngine.__new__(KnowledgeEngine)
    engine._retrieval_namespace = None
    engine._structured_ingestion = None
    engine._generation_service = MagicMock()
    flows = engine._enabled_flows()
    for f in flows:
        assert "-" not in f, f"kebab case in flows: {f}"
