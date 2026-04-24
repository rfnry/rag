import pytest

from rfnry_rag.retrieval.stores.graph.neo4j import ALLOWED_RELATION_TYPES, _validate_relation_type


@pytest.mark.parametrize(
    "injection",
    [
        "DROP_ALL",
        "CONNECTS_TO]->()-[r] MATCH (n) DELETE n RETURN [(a)-[r:x",
        "CONNECTS_TO; MATCH (n) DETACH DELETE n",
        "",
        "../../CONNECTS_TO",
    ],
)
def test_validate_relation_type_rejects_non_allowlisted(injection: str) -> None:
    with pytest.raises(ValueError):
        _validate_relation_type(injection)


def test_validate_relation_type_accepts_allowlist() -> None:
    for rel in ALLOWED_RELATION_TYPES:
        assert _validate_relation_type(rel) == rel
