from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from rfnry_rag.exceptions import ConfigurationError


class _BaseModelProvider(BaseModel):
    api_key: SecretStr
    model: str
    context_size: int | None = None

    model_config = {"frozen": True}

    @field_validator("context_size")
    @classmethod
    def _validate_context_size(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ConfigurationError(f"context_size={v} must be a positive integer or None")
        return v

    @property
    def name(self) -> str:
        kind = getattr(self, "kind", "")
        return f"{kind}:{self.model}"


class AnthropicModelProvider(_BaseModelProvider):
    kind: Literal["anthropic"] = "anthropic"
    base_url: str | None = None


class OpenAIModelProvider(_BaseModelProvider):
    kind: Literal["openai"] = "openai"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None


class GoogleModelProvider(_BaseModelProvider):
    kind: Literal["google"] = "google"


class VoyageModelProvider(_BaseModelProvider):
    kind: Literal["voyage"] = "voyage"


class CohereModelProvider(_BaseModelProvider):
    kind: Literal["cohere"] = "cohere"


ModelProvider = Annotated[
    AnthropicModelProvider
    | OpenAIModelProvider
    | GoogleModelProvider
    | VoyageModelProvider
    | CohereModelProvider,
    Field(discriminator="kind"),
]
