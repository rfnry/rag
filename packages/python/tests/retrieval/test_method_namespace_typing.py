from unittest.mock import MagicMock

from rfnry_knowledge.retrieval.base import BaseRetrievalMethod
from rfnry_knowledge.retrieval.namespace import MethodNamespace


def test_method_namespace_iter_returns_expected_elements() -> None:
    method = MagicMock(spec=BaseRetrievalMethod)
    method.name = "vector"
    ns: MethodNamespace[BaseRetrievalMethod] = MethodNamespace([method])
    items = list(ns)
    assert len(items) == 1
    assert items[0] is method
