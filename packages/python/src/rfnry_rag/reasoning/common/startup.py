"""Startup checks for reasoning SDK."""

from rfnry_rag.common.startup import check_baml as _check_baml


def check_baml() -> None:
    _check_baml("reasoning", "rfnry_rag.reasoning.baml.baml_client")
