"""R5.1 — Query classifier (heuristic + LLM, plumbing for R5.2/R5.3/R6).

Pure async `classify_query(text, lm_client=None)` returns a
`QueryClassification(complexity, query_type, signals, source)`. Heuristic
path runs by default (free, deterministic regex); LLM path opts in via
`lm_client` and falls back to heuristic on any exception so a classifier
failure never blocks retrieval.

Bias-term hygiene: fixtures use neutral identifiers (`R-101`, `V-203` are
entity-shape but not domain words; `controller`, `pump`, `procedure` are
generic). No `valve|motor|wire|terminal|480V|PSI|RPM|SAE|RV-2201|electrical|
mechanical`.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.retrieval import (
    AdaptiveRetrievalConfig,
    IngestionConfig,
    PersistenceConfig,
    QueryClassification,
    QueryComplexity,
    QueryType,
    RagEngine,
    RagServerConfig,
    RetrievalConfig,
    classify_query,
)
from rfnry_rag.retrieval.common.errors import ConfigurationError


async def test_classify_query_factual_simple_default() -> None:
    """Short factual question with no entity tokens -> SIMPLE / FACTUAL."""
    result = await classify_query("Who is the project manager?")

    assert isinstance(result, QueryClassification)
    assert result.complexity is QueryComplexity.SIMPLE
    assert result.query_type is QueryType.FACTUAL
    assert result.source == "heuristic"
    assert result.signals["query_length"] == len("Who is the project manager?")
    assert result.signals["entity_count"] == 0


async def test_classify_query_comparative_complex() -> None:
    """`compare ... and ...` -> COMPARATIVE; comparative auto-promotes to COMPLEX."""
    result = await classify_query(
        "Compare procedure A and procedure B for the controller reset workflow."
    )

    assert result.query_type is QueryType.COMPARATIVE
    assert result.complexity is QueryComplexity.COMPLEX


async def test_classify_query_entity_relationship_priority_over_factual() -> None:
    """Two entity tokens + relational verb -> ENTITY_RELATIONSHIP, even when
    the query also matches the PROCEDURAL pattern (`how does`). Priority order
    asserts ENTITY_RELATIONSHIP > COMPARATIVE > PROCEDURAL > FACTUAL.
    """
    result = await classify_query("How does R-101 connect to V-203?")

    assert result.query_type is QueryType.ENTITY_RELATIONSHIP
    assert result.complexity is QueryComplexity.COMPLEX  # entity-rel auto-promotes
    assert result.signals["entity_count"] == 2


async def test_classify_query_procedural_question() -> None:
    """`How to ...` with no entity-shaped tokens -> PROCEDURAL, not ENTITY_RELATIONSHIP."""
    result = await classify_query("How to reset the controller?")

    assert result.query_type is QueryType.PROCEDURAL
    assert result.query_type is not QueryType.ENTITY_RELATIONSHIP


async def test_classify_query_complexity_by_length_and_entity_count() -> None:
    """Three independent triggers map to COMPLEX / COMPLEX / MODERATE."""
    long_query = "x " * 130  # 260 chars, 0 entity tokens, falls to FACTUAL
    long_result = await classify_query(long_query)
    assert long_result.complexity is QueryComplexity.COMPLEX

    multi_entity = "Look up R-101 V-203 P-404 settings"  # 3 entity tokens, factual
    multi_result = await classify_query(multi_entity)
    assert multi_result.complexity is QueryComplexity.COMPLEX

    moderate = "Describe the standard procedure for the R-101 controller setup."
    moderate_result = await classify_query(moderate)
    assert moderate_result.complexity is QueryComplexity.MODERATE


async def test_classify_query_signals_include_query_length_and_entity_count() -> None:
    """Signals dict carries the heuristic indicators that drove the verdict."""
    text = "Look up R-101 status"
    result = await classify_query(text)

    assert result.signals["query_length"] == len(text)
    assert result.signals["entity_count"] == 1
    assert result.source == "heuristic"


async def test_classify_query_llm_path_calls_baml_when_lm_client_provided() -> None:
    """With a non-None `lm_client`, BAML is called once and source == 'llm'."""
    fake_verdict = SimpleNamespace(
        complexity=SimpleNamespace(value="SIMPLE"),
        query_type=SimpleNamespace(value="FACTUAL"),
        reasoning="single fact lookup",
    )
    fake_client = MagicMock()

    with (
        patch(
            "rfnry_rag.retrieval.modules.retrieval.search.classification.build_registry",
            return_value=MagicMock(),
        ),
        patch(
            "rfnry_rag.retrieval.modules.retrieval.search.classification.b.ClassifyQueryComplexity",
            new=AsyncMock(return_value=fake_verdict),
        ) as mock_baml,
    ):
        result = await classify_query("Who is the project manager?", lm_client=fake_client)

    mock_baml.assert_awaited_once()
    assert result.source == "llm"
    assert result.complexity is QueryComplexity.SIMPLE
    assert result.query_type is QueryType.FACTUAL
    assert result.signals["llm_reasoning"] == "single fact lookup"


async def test_classify_query_llm_path_falls_back_to_heuristic_on_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """BAML raising must NOT escape — fall back to heuristic and log a warning."""
    import logging

    fake_client = MagicMock()
    caplog.set_level(
        logging.WARNING, logger="rfnry_rag.retrieval.search.classification"
    )

    with (
        patch(
            "rfnry_rag.retrieval.modules.retrieval.search.classification.build_registry",
            return_value=MagicMock(),
        ),
        patch(
            "rfnry_rag.retrieval.modules.retrieval.search.classification.b.ClassifyQueryComplexity",
            new=AsyncMock(side_effect=RuntimeError("rate limit")),
        ),
    ):
        result = await classify_query("Who is the project manager?", lm_client=fake_client)

    assert result.source == "heuristic"
    assert result.complexity is QueryComplexity.SIMPLE
    assert result.query_type is QueryType.FACTUAL
    assert any("query classification" in record.message.lower() for record in caplog.records)


def _mock_embeddings() -> MagicMock:
    m = MagicMock()
    m.model = "test"
    return m


def test_adaptive_llm_classification_requires_enrich_lm_client() -> None:
    """`adaptive.enabled=True AND use_llm_classification=True` requires
    `RetrievalConfig.enrich_lm_client` — enforced in `_validate_config`.

    Regression guard for the cross-config rule introduced in R5.1.
    """
    base_persistence = PersistenceConfig(vector_store=MagicMock())
    base_ingestion = IngestionConfig(embeddings=_mock_embeddings())

    # Missing enrich_lm_client -> ConfigurationError
    bad_config = RagServerConfig(
        persistence=base_persistence,
        ingestion=base_ingestion,
        retrieval=RetrievalConfig(
            adaptive=AdaptiveRetrievalConfig(enabled=True, use_llm_classification=True),
            enrich_lm_client=None,
        ),
    )
    with pytest.raises(ConfigurationError, match="use_llm_classification.*enrich_lm_client"):
        RagEngine(bad_config)._validate_config()

    # Inverse: providing a non-None enrich_lm_client does NOT raise.
    good_config = RagServerConfig(
        persistence=PersistenceConfig(vector_store=MagicMock()),
        ingestion=IngestionConfig(embeddings=_mock_embeddings()),
        retrieval=RetrievalConfig(
            adaptive=AdaptiveRetrievalConfig(enabled=True, use_llm_classification=True),
            enrich_lm_client=MagicMock(),
        ),
    )
    RagEngine(good_config)._validate_config()


def test_adaptive_confidence_expansion_requires_enabled() -> None:
    """`confidence_expansion=True AND enabled=False` is a silent no-op
    footgun — `_query_via_retrieval` requires BOTH flags before firing
    the retry loop. Surface the misconfig at construction time so it
    can't ship to production.

    Regression guard mirroring the `use_llm_classification` precedent.
    Lives in `__post_init__` (not `_validate_config`) because both
    fields are on the same dataclass — single-config invariant, not
    cross-config.
    """
    # confidence_expansion=True without enabled=True -> ConfigurationError.
    with pytest.raises(
        ConfigurationError, match="confidence_expansion.*enabled"
    ):
        AdaptiveRetrievalConfig(enabled=False, confidence_expansion=True)

    # enabled=True + confidence_expansion=True succeeds (the live path).
    AdaptiveRetrievalConfig(enabled=True, confidence_expansion=True)

    # Both False (the default) succeeds — back-compat for existing consumers.
    AdaptiveRetrievalConfig(enabled=False, confidence_expansion=False)
