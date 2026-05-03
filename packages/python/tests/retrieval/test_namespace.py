from types import SimpleNamespace

import pytest

from rfnry_knowledge.retrieval.namespace import MethodNamespace


def _method(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_attribute_access():
    ns = MethodNamespace([_method("semantic"), _method("keyword")])
    assert ns.semantic.name == "semantic"
    assert ns.keyword.name == "keyword"


def test_attribute_access_missing_raises():
    ns = MethodNamespace([_method("semantic")])
    with pytest.raises(AttributeError, match="No method 'entity' configured"):
        _ = ns.entity


def test_iteration():
    methods = [_method("semantic"), _method("keyword")]
    ns = MethodNamespace(methods)
    names = [m.name for m in ns]
    assert names == ["semantic", "keyword"]


def test_len():
    ns = MethodNamespace([_method("a"), _method("b"), _method("c")])
    assert len(ns) == 3


def test_contains():
    ns = MethodNamespace([_method("semantic")])
    assert "semantic" in ns
    assert "entity" not in ns


def test_empty():
    ns = MethodNamespace([])
    assert len(ns) == 0
    assert list(ns) == []
