"""Tests for TreeIndexingConfig and TreeSearchConfig dataclasses."""

import pytest

from rfnry_rag.retrieval.common.errors import ConfigurationError
from rfnry_rag.retrieval.server import TreeIndexingConfig, TreeSearchConfig


class TestTreeIndexingConfig:
    def test_defaults(self) -> None:
        cfg = TreeIndexingConfig()
        assert cfg.enabled is False
        assert cfg.model is None
        assert cfg.toc_scan_pages == 20
        assert cfg.max_pages_per_node == 10
        assert cfg.max_tokens_per_node == 20_000
        assert cfg.generate_summaries is True
        assert cfg.generate_description is True

    def test_toc_scan_pages_zero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="toc_scan_pages must be positive"):
            TreeIndexingConfig(toc_scan_pages=0)

    def test_toc_scan_pages_negative_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="toc_scan_pages must be positive"):
            TreeIndexingConfig(toc_scan_pages=-1)

    def test_max_pages_per_node_zero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_pages_per_node must be positive"):
            TreeIndexingConfig(max_pages_per_node=0)

    def test_max_pages_per_node_negative_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_pages_per_node must be positive"):
            TreeIndexingConfig(max_pages_per_node=-1)

    def test_max_tokens_per_node_zero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_tokens_per_node must be positive"):
            TreeIndexingConfig(max_tokens_per_node=0)

    def test_max_tokens_per_node_negative_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_tokens_per_node must be positive"):
            TreeIndexingConfig(max_tokens_per_node=-1)


class TestTreeSearchConfig:
    def test_defaults(self) -> None:
        cfg = TreeSearchConfig()
        assert cfg.enabled is False
        assert cfg.model is None
        assert cfg.max_steps == 5
        assert cfg.max_context_tokens == 50_000

    def test_max_steps_zero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_steps must be positive"):
            TreeSearchConfig(max_steps=0)

    def test_max_steps_negative_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_steps must be positive"):
            TreeSearchConfig(max_steps=-1)

    def test_max_context_tokens_zero_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_context_tokens must be positive"):
            TreeSearchConfig(max_context_tokens=0)

    def test_max_context_tokens_negative_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_context_tokens must be positive"):
            TreeSearchConfig(max_context_tokens=-1)
