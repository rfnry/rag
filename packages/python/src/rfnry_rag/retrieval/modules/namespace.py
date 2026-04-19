from __future__ import annotations

from typing import Any


class MethodNamespace[T]:
    """Exposes pipeline methods as attributes and supports iteration.

    Methods must have a ``name`` attribute used as the access key.
    """

    def __init__(self, methods: list[T]) -> None:
        self._methods: dict[str, T] = {}
        for method in methods:
            self._methods[method.name] = method  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> T:
        try:
            return self._methods[name]
        except KeyError:
            raise AttributeError(f"No method '{name}' configured") from None

    def __iter__(self) -> Any:
        return iter(self._methods.values())

    def __len__(self) -> int:
        return len(self._methods)

    def __contains__(self, name: object) -> bool:
        return name in self._methods
