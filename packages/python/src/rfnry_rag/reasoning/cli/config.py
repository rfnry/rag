from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rfnry_rag.common.cli import get_api_key as _get_api_key
from rfnry_rag.reasoning.cli.constants import CONFIG_FILE, ENV_FILE, ConfigError, load_dotenv
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider

if TYPE_CHECKING:
    from rfnry_rag.reasoning.modules.analysis.service import AnalysisService
    from rfnry_rag.reasoning.modules.classification.service import ClassificationService
    from rfnry_rag.reasoning.modules.compliance.service import ComplianceService
    from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService


_LM_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_LM_DEFAULTS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _build_lm_provider(cfg: dict[str, Any]) -> LanguageModelProvider:
    provider = cfg.get("provider")
    if not provider:
        raise ConfigError("[language_model] requires 'provider' (anthropic or openai)")

    env_var = _LM_API_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown language model provider: {provider!r}. Supported: {', '.join(_LM_API_KEYS)}")

    api_key = _get_api_key(env_var, provider)
    model = cfg.get("model", _LM_DEFAULTS[provider])

    return LanguageModelProvider(
        provider=provider,
        model=model,
        api_key=api_key,
    )


def build_lm_client(toml: dict[str, Any]) -> LanguageModelClient:
    """Build LanguageModelClient from [language_model] section."""
    lm_cfg = toml.get("language_model", {})
    if not lm_cfg:
        raise ConfigError("[language_model] section required in config.toml")

    primary_provider = _build_lm_provider(lm_cfg)

    fallback_provider = None
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    fallback_cfg = lm_cfg.get("fallback")
    if fallback_cfg:
        for forbidden_key in ("max_tokens", "temperature"):
            if forbidden_key in fallback_cfg:
                raise ConfigError(
                    f"[language_model.fallback] cannot set {forbidden_key!r}. "
                    f"Generation parameters (max_tokens, temperature) now apply to both "
                    f"primary and fallback — set them at the top-level [language_model] section."
                )
        fallback_provider = _build_lm_provider(fallback_cfg)
        strategy = "fallback"

    return LanguageModelClient(
        provider=primary_provider,
        fallback=fallback_provider,
        max_retries=lm_cfg.get("max_retries", 3),
        strategy=strategy,
        max_tokens=lm_cfg.get("max_tokens", 4096),
        temperature=lm_cfg.get("temperature", 0.0),
    )


def build_analysis_service(toml: dict[str, Any]) -> AnalysisService:
    from rfnry_rag.reasoning.modules.analysis.service import AnalysisService as _AnalysisService

    return _AnalysisService(lm_client=build_lm_client(toml))


def build_classification_service(toml: dict[str, Any]) -> ClassificationService:
    from rfnry_rag.reasoning.modules.classification.service import ClassificationService as _ClassificationService

    return _ClassificationService(lm_client=build_lm_client(toml))


def build_compliance_service(toml: dict[str, Any]) -> ComplianceService:
    from rfnry_rag.reasoning.modules.compliance.service import ComplianceService as _ComplianceService

    return _ComplianceService(lm_client=build_lm_client(toml))


def build_evaluation_service(toml: dict[str, Any]) -> EvaluationService:
    from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService as _EvaluationService

    return _EvaluationService(lm_client=build_lm_client(toml))


_ALLOWED_TOP_KEYS = {"language_model", "analysis", "classification", "compliance", "evaluation"}


def _validate_toml_keys(toml: dict[str, Any]) -> None:
    """Reject unknown top-level keys in config.toml to surface typos early."""
    unknown = set(toml.keys()) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown top-level key(s) in config.toml: {sorted(unknown)}. "
            f"Allowed keys: {sorted(_ALLOWED_TOP_KEYS)}"
        )


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load TOML config + .env, return raw TOML dict."""
    path = Path(config_path) if config_path else CONFIG_FILE
    if not path.exists():
        raise ConfigError(f"Config not found: {path}\nRun 'rfnry-rag reasoning init' to create it.")

    env_path = path.parent / ".env" if config_path else ENV_FILE
    load_dotenv(env_path)

    with open(path, "rb") as f:
        toml = tomllib.load(f)
    _validate_toml_keys(toml)
    return toml
