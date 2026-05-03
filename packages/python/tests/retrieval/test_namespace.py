from types import SimpleNamespace

import pytest

from rfnry_knowledge.retrieval.namespace import MethodNamespace


def _method(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_attribute_access():
    ns = MethodNamespace([_method("vector"), _method("document")])
    assert ns.vector.name == "vector"
    assert ns.document.name == "document"


def test_attribute_access_missing_raises():
    ns = MethodNamespace([_method("vector")])
    with pytest.raises(AttributeError, match="No method 'graph' configured"):
        _ = ns.graph


def test_iteration():
    methods = [_method("vector"), _method("document")]
    ns = MethodNamespace(methods)
    names = [m.name for m in ns]
    assert names == ["vector", "document"]


def test_len():
    ns = MethodNamespace([_method("a"), _method("b"), _method("c")])
    assert len(ns) == 3


def test_contains():
    ns = MethodNamespace([_method("vector")])
    assert "vector" in ns
    assert "graph" not in ns


def test_empty():
    ns = MethodNamespace([])
    assert len(ns) == 0
    assert list(ns) == []
