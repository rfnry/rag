from unittest.mock import MagicMock

from rfnry_rag.retrieval.modules.namespace import MethodNamespace
from rfnry_rag.retrieval.modules.retrieval.base import BaseRetrievalMethod


def test_method_namespace_iter_returns_expected_elements() -> None:
    method = MagicMock(spec=BaseRetrievalMethod)
    method.name = "vector"
    ns: MethodNamespace[BaseRetrievalMethod] = MethodNamespace([method])
    items = list(ns)
    assert len(items) == 1
    assert items[0] is method
