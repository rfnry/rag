from unittest.mock import MagicMock

from rfnry_rag.ingestion.methods.analyzed import AnalyzedIngestion


def test_analyzed_method_exposes_protocol_attrs():
    method = AnalyzedIngestion(
        store=MagicMock(),
        embeddings=MagicMock(name="emb"),
        vision=MagicMock(),
        lm_client=MagicMock(),
        embedding_model_name="openai:text-embedding-3-large",
    )
    assert method.name == "analyzed"
    assert method.required is True
    assert hasattr(method, "analyze")
    assert hasattr(method, "synthesize")
    assert hasattr(method, "ingest")


def test_analyzed_method_clone_for_store_returns_new_instance():
    emb = MagicMock(name="emb")
    method = AnalyzedIngestion(
        store=MagicMock(),
        embeddings=emb,
        vision=MagicMock(),
        embedding_model_name="x",
    )
    new_store = MagicMock()
    cloned = method.clone_for_store(new_store)
    assert cloned is not method
    assert cloned._store is new_store
    assert cloned._embeddings is emb
