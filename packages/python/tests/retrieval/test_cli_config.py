import os
from unittest.mock import patch

import pytest

from rfnry_rag.cli.config import ConfigError, load_config
from rfnry_rag.cli.constants import load_dotenv

_BASE_CONFIG = """\
[persistence]
vector_store = "qdrant"
url = "http://localhost:6333"
collection = "test"

[ingestion]
embeddings = "openai"
model = "text-embedding-3-small"
"""


def _write_config(tmp_path, extra_toml="", env_keys=None):
    """Write a config.toml + .env, return the config path string."""
    config = tmp_path / "config.toml"
    config.write_text(_BASE_CONFIG + extra_toml)
    env = tmp_path / ".env"
    keys = {"OPENAI_API_KEY": "sk-test"}
    if env_keys:
        keys.update(env_keys)
    env.write_text("\n".join(f"{k}={v}" for k, v in keys.items()) + "\n")
    return str(config)


class TestLoadDotenv:
    def test_loads_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO_KEY=bar123\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOO_KEY", None)
            load_dotenv(env_file)
            assert os.environ["FOO_KEY"] == "bar123"

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEY", None)
            load_dotenv(env_file)
            assert os.environ["KEY"] == "val"

    def test_env_var_takes_precedence(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=from_file\n")
        with patch.dict(os.environ, {"MY_KEY": "from_env"}, clear=False):
            load_dotenv(env_file)
            assert os.environ["MY_KEY"] == "from_env"

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hello world"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUOTED", None)
            load_dotenv(env_file)
            assert os.environ["QUOTED"] == "hello world"

    def test_missing_file_is_noop(self, tmp_path):
        load_dotenv(tmp_path / "nonexistent")


class TestLoadConfig:
    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Config not found"):
            load_config(str(tmp_path / "nope.toml"))

    def test_minimal_config(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("""\
[persistence]
vector_store = "qdrant"
url = "http://localhost:6333"
collection = "test"

[ingestion]
embeddings = "openai"
model = "text-embedding-3-small"
""")
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-test\n")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            server_config = load_config(str(config))

        assert server_config is not None

    def test_missing_persistence_raises(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("[ingestion]\nembeddings = 'openai'\n")
        env = tmp_path / ".env"
        env.write_text("")

        with pytest.raises(ConfigError, match="persistence"):
            load_config(str(config))

    def test_unknown_embeddings_raises(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("""\
[persistence]
vector_store = "qdrant"
[ingestion]
embeddings = "unknown"
""")
        env = tmp_path / ".env"
        env.write_text("")

        with pytest.raises(ConfigError, match="Unknown embeddings"):
            load_config(str(config))


class TestLoadConfigProviders:
    def test_vision_anthropic(self, tmp_path):
        """Vision TOML knob is accepted and validated; load_config() returns
        a usable RagEngineConfig (no exception)."""
        path = _write_config(
            tmp_path,
            '\nvision = "anthropic"\n',
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            cfg = load_config(path)
        assert cfg is not None

    def test_vision_unknown_raises(self, tmp_path):
        path = _write_config(tmp_path, '\nvision = "bad"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigError, match="Unknown vision"):
                load_config(path)

    def test_reranker_voyage(self, tmp_path):
        path = _write_config(
            tmp_path,
            '\n[retrieval]\nreranker = "voyage"\n',
            {"VOYAGE_API_KEY": "pa-test"},
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("VOYAGE_API_KEY", None)
            cfg = load_config(path)
        assert cfg.retrieval.reranker is not None

    def test_reranker_unknown_raises(self, tmp_path):
        path = _write_config(tmp_path, '\n[retrieval]\nreranker = "bad"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigError, match="Unknown reranker"):
                load_config(path)

    def test_rewriter_multi_query(self, tmp_path):
        rewriter_cfg = (
            '\n[retrieval]\nrewriter = "multi_query"\nrewriter_provider = "anthropic"\n'
            'rewriter_model = "claude-haiku-4-5-20251001"\n'
        )
        path = _write_config(
            tmp_path,
            rewriter_cfg,
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            cfg = load_config(path)
        from rfnry_rag.retrieval.search.rewriting.multi_query import MultiQueryRewriting

        assert isinstance(cfg.retrieval.query_rewriter, MultiQueryRewriting)

    def test_rewriter_missing_provider_raises(self, tmp_path):
        path = _write_config(tmp_path, '\n[retrieval]\nrewriter = "multi_query"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ConfigError, match="rewriter_provider"):
                load_config(path)

    def test_rewriter_unknown_type_raises(self, tmp_path):
        path = _write_config(
            tmp_path,
            '\n[retrieval]\nrewriter = "bad"\nrewriter_provider = "anthropic"\nrewriter_model = "m"\n',
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ConfigError, match="Unknown rewriter"):
                load_config(path)

    def test_sparse_embeddings_enabled(self, tmp_path):
        from rfnry_rag.ingestion.methods.vector import VectorIngestion

        path = _write_config(tmp_path, "sparse_embeddings = true\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            cfg = load_config(path)
        vector = next(m for m in cfg.ingestion.methods if isinstance(m, VectorIngestion))
        assert vector._sparse is not None

    def test_sparse_embeddings_disabled_by_default(self, tmp_path):
        from rfnry_rag.ingestion.methods.vector import VectorIngestion

        path = _write_config(tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            cfg = load_config(path)
        vector = next(m for m in cfg.ingestion.methods if isinstance(m, VectorIngestion))
        assert vector._sparse is None

    def test_chunking_options(self, tmp_path):
        path = _write_config(
            tmp_path,
            "chunk_size = 300\nchunk_overlap = 30\nparent_chunk_size = 1500\nparent_chunk_overlap = 100\n",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            cfg = load_config(path)
        assert cfg.ingestion.chunk_size == 300
        assert cfg.ingestion.chunk_overlap == 30
        assert cfg.ingestion.parent_chunk_size == 1500
        assert cfg.ingestion.parent_chunk_overlap == 100

    def test_grounding_gates(self, tmp_path):
        path = _write_config(
            tmp_path,
            '\n[generation]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\n'
            "grounding_enabled = true\ngrounding_threshold = 0.7\n"
            "relevance_gate_enabled = true\n"
            'relevance_gate_provider = "anthropic"\n'
            'relevance_gate_model = "claude-haiku-4-5-20251001"\n'
            "guiding_enabled = true\n",
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            cfg = load_config(path)
        assert cfg.generation.grounding_enabled is True
        assert cfg.generation.grounding_threshold == 0.7
        assert cfg.generation.relevance_gate_enabled is True
        assert cfg.generation.relevance_gate_model is not None
        assert cfg.generation.guiding_enabled is True
