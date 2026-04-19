"""Startup checks for retrieval SDK."""

from rfnry_rag.common.startup import check_baml as _check_baml


def check_baml() -> None:
    _check_baml("retrieval", "rfnry_rag.retrieval.baml.baml_client")
