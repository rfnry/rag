"""R6.1 — IterativeRetrievalConfig dataclass invariants.

R6.1 lands the compile-time scaffold for multi-hop iterative retrieval.
The hop loop, engine integration, and trace surface are deferred to R6.2;
post-loop DIRECT escalation lands in R6.3. These tests pin the dataclass
defaults, bounds, allowlist, and the cross-field rule (`gate_mode='llm'`
requires `decomposition_model` when `enabled=True`).
"""

from __future__ import annotations

import pytest

from rfnry_rag.retrieval.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.modules.retrieval.iterative.config import IterativeRetrievalConfig


def _lm_client() -> LanguageModelClient:
    return LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-test", api_key="x"),
    )


def test_iterative_config_defaults_are_off() -> None:
    cfg = IterativeRetrievalConfig()
    assert cfg.enabled is False
    assert cfg.max_hops == 3
    assert cfg.gate_mode == "type"
    assert cfg.escalate_to_direct is True
    assert cfg.grounding_threshold is None
    assert cfg.decomposition_model is None


def test_iterative_config_max_hops_lower_bound_raises() -> None:
    with pytest.raises(ValueError, match="max_hops"):
        IterativeRetrievalConfig(max_hops=0)


def test_iterative_config_max_hops_upper_bound_raises() -> None:
    with pytest.raises(ValueError, match="max_hops"):
        IterativeRetrievalConfig(max_hops=11)


def test_iterative_config_gate_mode_allowlist() -> None:
    with pytest.raises(ValueError, match="gate_mode"):
        IterativeRetrievalConfig(gate_mode="bogus")
    # Valid values accepted
    assert IterativeRetrievalConfig(gate_mode="type").gate_mode == "type"
    assert IterativeRetrievalConfig(gate_mode="llm", decomposition_model=_lm_client()).gate_mode == "llm"


def test_iterative_config_grounding_threshold_bounds() -> None:
    with pytest.raises(ValueError, match="grounding_threshold"):
        IterativeRetrievalConfig(grounding_threshold=-0.1)
    with pytest.raises(ValueError, match="grounding_threshold"):
        IterativeRetrievalConfig(grounding_threshold=1.1)
    # Boundary + None accepted
    assert IterativeRetrievalConfig(grounding_threshold=None).grounding_threshold is None
    assert IterativeRetrievalConfig(grounding_threshold=0.0).grounding_threshold == 0.0
    assert IterativeRetrievalConfig(grounding_threshold=1.0).grounding_threshold == 1.0


def test_iterative_config_llm_gate_requires_model() -> None:
    # Cross-field invariant: LLM-gated decomposition needs a model. Without it
    # the loop would have no way to ask the gate question, so reject at
    # construction rather than at first call.
    with pytest.raises(ValueError, match="decomposition_model"):
        IterativeRetrievalConfig(enabled=True, gate_mode="llm", decomposition_model=None)
