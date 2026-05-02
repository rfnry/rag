from __future__ import annotations

from contextvars import ContextVar, Token

from rfnry_rag.observability.runtime import Observability

_obs_var: ContextVar[Observability | None] = ContextVar("rfnry_rag_observability", default=None)


def current_obs() -> Observability | None:
    return _obs_var.get()


def _set_obs(obs: Observability) -> Token[Observability | None]:
    return _obs_var.set(obs)


def _reset_obs(token: Token[Observability | None]) -> None:
    _obs_var.reset(token)
