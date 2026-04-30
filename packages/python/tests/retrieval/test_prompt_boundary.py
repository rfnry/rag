from rfnry_rag.providers.text_generation import assemble_user_message


def test_native_user_message_has_content_boundary() -> None:
    """Plain-text generation must fence both query and context with the
    contract markers (untrusted-data treatment) before the LLM call."""
    message = assemble_user_message(query="user query", context="ingested content")
    assert "======== QUERY START ========" in message
    assert "======== QUERY END ========" in message
    assert "======== CONTEXT START ========" in message
    assert "======== CONTEXT END ========" in message
    # Query fence must appear BEFORE the context fence.
    query_end = message.index("======== QUERY END ========")
    context_start = message.index("======== CONTEXT START ========")
    assert context_start > query_end


def test_native_user_message_marks_query_as_untrusted() -> None:
    message = assemble_user_message(query="q", context="c")
    assert "untrusted user text" in message
    assert "untrusted data" in message
