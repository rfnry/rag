from __future__ import annotations

from dataclasses import dataclass

from rfnry_rag.exceptions import ConfigurationError


@dataclass
class BenchmarkConfig:
    """Knobs for `RagEngine.benchmark`. Both fields are bounded.

    `concurrency` defaults to 1 (serial) so a small benchmark behaves
    identically to a `for case in cases: ...` loop. Larger benchmarks
    opt into parallelism via `run_concurrent`.
    """

    concurrency: int = 1
    failure_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not (1 <= self.concurrency <= 20):
            raise ConfigurationError(f"BenchmarkConfig.concurrency={self.concurrency} out of range [1, 20]")
        if not (0.0 <= self.failure_threshold <= 1.0):
            raise ConfigurationError(
                f"BenchmarkConfig.failure_threshold={self.failure_threshold} out of range [0.0, 1.0]"
            )
