from pathlib import Path
from unittest.mock import MagicMock

from rfnry_rag.config.drawing import DrawingIngestionConfig
from rfnry_rag.ingestion.methods.drawing import DrawingIngestion


def test_drawing_method_exposes_protocol_attrs():
    cfg = DrawingIngestionConfig(enabled=True)
    method = DrawingIngestion(
        config=cfg,
        store=MagicMock(),
        embeddings=MagicMock(name="emb"),
        vision=MagicMock(),
        lm_client=MagicMock(),
        embedding_model_name="openai:text-embedding-3-large",
    )
    assert method.name == "drawing"
    assert method.required is True
    assert hasattr(method, "render")
    assert hasattr(method, "extract")
    assert hasattr(method, "link")
    assert hasattr(method, "ingest")


def test_drawing_method_accepts_drawing_route():
    cfg = DrawingIngestionConfig(enabled=True)
    method = DrawingIngestion(config=cfg, store=MagicMock(), embeddings=MagicMock())
    assert method.accepts(Path("schematic.dxf"), None) is True
    assert method.accepts(Path("schematic.dxf"), "drawing") is True
    assert method.accepts(Path("schematic.dxf"), "document") is True
    assert method.accepts(Path("SCHEMATIC.DXF"), None) is True
    assert method.accepts(Path("schematic.pdf"), "drawing") is True
    assert method.accepts(Path("schematic.pdf"), None) is False
    assert method.accepts(Path("schematic.pdf"), "document") is False
    assert method.accepts(Path("doc.xml"), None) is False
    assert method.accepts(Path("doc.txt"), None) is False


def test_drawing_method_clone_for_store_round_trips_full_state():
    cfg = DrawingIngestionConfig(enabled=True)
    delegate = [MagicMock(name="delegate")]
    sentinels = {
        "embeddings": MagicMock(name="emb"),
        "vision": MagicMock(name="vision"),
        "lm_client": MagicMock(name="lm"),
        "graph_store": MagicMock(name="graph"),
        "metadata_store": MagicMock(name="metadata"),
    }
    method = DrawingIngestion(
        config=cfg,
        store=MagicMock(),
        embeddings=sentinels["embeddings"],
        vision=sentinels["vision"],
        lm_client=sentinels["lm_client"],
        graph_store=sentinels["graph_store"],
        metadata_store=sentinels["metadata_store"],
        embedding_model_name="x:y",
        delegate_methods=delegate,
    )
    new_store = MagicMock(name="new_store")
    cloned = method.clone_for_store(new_store)
    assert cloned is not method
    assert cloned._store is new_store
    assert cloned._config is method._config
    assert cloned._embeddings is sentinels["embeddings"]
    assert cloned._vision is sentinels["vision"]
    assert cloned._lm_client is sentinels["lm_client"]
    assert cloned._graph_store is sentinels["graph_store"]
    assert cloned._metadata_store is sentinels["metadata_store"]
    assert cloned._embedding_model_name == "x:y"
    assert cloned._delegate_methods == delegate
