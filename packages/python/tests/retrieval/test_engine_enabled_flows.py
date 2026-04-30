from unittest.mock import MagicMock

from rfnry_rag.server import RagEngine


def test_enabled_flows_uses_snake_case_only() -> None:
    rag = RagEngine.__new__(RagEngine)
    rag._retrieval_namespace = None
    rag._structured_ingestion = None
    rag._generation_service = MagicMock()
    flows = rag._enabled_flows()
    for f in flows:
        assert "-" not in f, f"kebab case in flows: {f}"
