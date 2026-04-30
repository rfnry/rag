"""Token counting with tiktoken when available, falling back to word count."""

from __future__ import annotations

from typing import Any

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
    _ENC: Any = tiktoken.get_encoding("cl100k_base")  # covers GPT-3.5 / 4 / 4o
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    _ENC = None


def count_tokens(text: str) -> int:
    """Return the token count of ``text``.

    Uses tiktoken's cl100k_base encoding when available (covers OpenAI
    GPT-3.5/4/4o). Falls back to a simple ``len(text.split())`` word
    count when tiktoken is not installed — word count is a strictly
    better heuristic than ``len(text) / 4`` for English.
    """
    if _TIKTOKEN_AVAILABLE and _ENC is not None:
        return len(_ENC.encode(text))
    return len(text.split())
