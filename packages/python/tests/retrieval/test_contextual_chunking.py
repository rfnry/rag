from rfnry_knowledge.ingestion.chunk.context import build_context, contextualize_chunks
from rfnry_knowledge.ingestion.models import ChunkedContent


def test_build_context_full():
    ctx = build_context(source_name="Pump Manual", source_type="manuals", page_number=12, section="Specifications")
    assert "Document: Pump Manual" in ctx
    assert "Type: manuals" in ctx
    assert "Page: 12" in ctx
    assert "Section: Specifications" in ctx


def test_build_context_minimal():
    ctx = build_context(source_name="Manual", source_type=None, page_number=None, section=None)
    assert ctx == "Document: Manual"


def test_build_context_empty():
    ctx = build_context(source_name="", source_type=None, page_number=None, section=None)
    assert ctx == ""


def test_contextualize_chunks():
    chunks = [
        ChunkedContent(content="Pressure drop is 0.25 inches.", page_number=3, section=None, chunk_index=0),
        ChunkedContent(content="Temperature range: -20F to 200F.", page_number=4, section=None, chunk_index=1),
    ]
    result = contextualize_chunks(chunks, source_name="Pump Manual", source_type="manuals")
    assert len(result) == 2
    assert result[0].context.startswith("Document: Pump Manual")
    assert result[0].contextualized.endswith("Pressure drop is 0.25 inches.")
    assert result[0].content == "Pressure drop is 0.25 inches."
    assert "Page: 3" in result[0].context


def test_contextualize_chunks_preserves_original():
    chunks = [ChunkedContent(content="Original text.", page_number=1, section=None, chunk_index=0)]
    result = contextualize_chunks(chunks, source_name="Doc", source_type=None)
    assert result[0].content == "Original text."
