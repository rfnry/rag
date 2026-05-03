from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    sparse_vector: SparseVector | None = None


@dataclass
class VectorResult:
    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
