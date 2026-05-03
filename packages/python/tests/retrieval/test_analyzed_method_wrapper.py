from pathlib import Path
from unittest.mock import MagicMock

from rfnry_knowledge.ingestion.methods.analyzed import AnalyzedIngestion


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


def test_analyzed_method_accepts_structured_extensions():
    method = AnalyzedIngestion(store=MagicMock(), embeddings=MagicMock())
    assert method.accepts(Path("doc.pdf"), None) is True
    assert method.accepts(Path("doc.xml"), None) is True
    assert method.accepts(Path("doc.l5x"), None) is True
    assert method.accepts(Path("doc.PDF"), None) is True
    assert method.accepts(Path("doc.txt"), None) is False
    assert method.accepts(Path("doc.dxf"), None) is False


def test_analyzed_method_clone_for_store_round_trips_full_state():
    async def _on_complete(_knowledge_id: str | None) -> None:
        return None

    delegate = [MagicMock(name="delegate")]
    source_type_weights = {"drawing": 1.5}
    sentinels = {
        "embeddings": MagicMock(name="emb"),
        "vision": MagicMock(name="vision"),
        "lm_client": MagicMock(name="lm"),
        "graph_store": MagicMock(name="graph"),
        "metadata_store": MagicMock(name="metadata"),
        "graph_config": MagicMock(name="gcfg"),
    }
    method = AnalyzedIngestion(
        store=MagicMock(),
        embeddings=sentinels["embeddings"],
        vision=sentinels["vision"],
        lm_client=sentinels["lm_client"],
        graph_store=sentinels["graph_store"],
        metadata_store=sentinels["metadata_store"],
        embedding_model_name="x:y",
        dpi=400,
        analyze_text_skip_threshold_chars=42,
        analyze_concurrency=7,
        graph_config=sentinels["graph_config"],
        source_type_weights=source_type_weights,
        on_ingestion_complete=_on_complete,
        delegate_methods=delegate,
    )
    new_store = MagicMock(name="new_store")
    cloned = method.clone_for_store(new_store)
    assert cloned is not method
    assert cloned._store is new_store
    assert cloned._embeddings is sentinels["embeddings"]
    assert cloned._vision is sentinels["vision"]
    assert cloned._lm_client is sentinels["lm_client"]
    assert cloned._graph_store is sentinels["graph_store"]
    assert cloned._metadata_store is sentinels["metadata_store"]
    assert cloned._embedding_model_name == "x:y"
    assert cloned._dpi == 400
    assert cloned._analyze_text_skip_threshold_chars == 42
    assert cloned._analyze_concurrency == 7
    assert cloned._graph_config is sentinels["graph_config"]
    assert cloned._source_type_weights == source_type_weights
    assert cloned._on_ingestion_complete is _on_complete
    assert cloned._delegate_methods == delegate
