from pathlib import Path

import pytest

from rfnry_rag.retrieval.stores.document.postgres import PostgresDocumentStore
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore

TARGETS = [
    Path(__file__).parents[1] / "stores" / "metadata" / "sqlalchemy.py",
    Path(__file__).parents[1] / "stores" / "document" / "postgres.py",
]


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: p.name)
def test_stores_do_not_call_hide_password_false(path: Path) -> None:
    src = path.read_text()
    assert "hide_password=False" not in src, (
        f"{path} must not render URLs with hide_password=False — "
        "pass the URL object directly to create_async_engine instead"
    )


@pytest.mark.parametrize(
    "store_cls, url",
    [
        (SQLAlchemyMetadataStore, "sqlite:///:memory:"),
        (PostgresDocumentStore, "sqlite:///:memory:"),
    ],
)
def test_store_engine_url_hides_password_by_default(store_cls, url) -> None:
    store = store_cls(url.replace("sqlite:///:memory:", "postgresql://alice:s3cr3t@localhost:5432/db"))
    rendered = store._engine.url.render_as_string()
    assert "s3cr3t" not in rendered
    assert "s3cr3t" not in repr(store._engine)
    assert store._engine.url.password == "s3cr3t"
