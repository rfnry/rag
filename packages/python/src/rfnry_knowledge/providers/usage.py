from __future__ import annotations

from typing import TypedDict


class TokenUsage(TypedDict, total=False):
    input: int
    output: int
    cache_creation: int
    cache_read: int


def empty_usage() -> TokenUsage:
    return TokenUsage(input=0, output=0, cache_creation=0, cache_read=0)


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    if isinstance(value, float):
        return int(value)
    return 0


def merge_usage(*usages: TokenUsage | None) -> TokenUsage:
    out = empty_usage()
    for u in usages:
        if not u:
            continue
        for k in ("input", "output", "cache_creation", "cache_read"):
            out[k] = _coerce_int(out.get(k, 0)) + _coerce_int(u.get(k, 0))
    return out


def usage_to_int_dict(usage: TokenUsage | None) -> dict[str, int]:
    """Coerce a TokenUsage TypedDict into a plain ``dict[str, int]``."""
    if not usage:
        return {}
    out: dict[str, int] = {}
    for k in ("input", "output", "cache_creation", "cache_read"):
        v = usage.get(k)
        if v is not None:
            out[k] = _coerce_int(v)
    return out
