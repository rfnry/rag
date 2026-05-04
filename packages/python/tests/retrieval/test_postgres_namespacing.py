from rfnry_knowledge.stores.document.postgres import PostgresDocumentStore


def test_default_table_name_unchanged() -> None:
    store = PostgresDocumentStore(url="sqlite:///:memory:")
    assert store.table_name == "knowledge_source_content"


def test_custom_table_name_applied() -> None:
    store = PostgresDocumentStore(url="sqlite:///:memory:", table_name="memory_source_content")
    assert store.table_name == "memory_source_content"


def test_two_instances_have_distinct_tables() -> None:
    a = PostgresDocumentStore(url="sqlite:///:memory:", table_name="ta")
    b = PostgresDocumentStore(url="sqlite:///:memory:", table_name="tb")
    assert a._row_cls.__tablename__ != b._row_cls.__tablename__
