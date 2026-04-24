from rfnry_rag.common.errors import ConfigurationError as ConfigurationError
from rfnry_rag.common.errors import SdkBaseError


class ReasoningError(SdkBaseError):
    """Base exception for reasoning SDK errors."""


class ReasoningInputError(ReasoningError, ValueError):
    """Raised when a reasoning SDK config or input fails validation.

    Inherits both ``ReasoningError`` (for catching SDK-specific errors) and
    ``ValueError`` (for back-compat: existing ``except ValueError:`` clauses
    still catch this).
    """


class ClassificationError(ReasoningError):
    """Error during text classification."""


class ClusteringError(ReasoningError):
    """Error during text clustering."""


class EvaluationError(ReasoningError):
    """Error during evaluation."""


class ComplianceError(ReasoningError):
    """Error during compliance checking."""


class AnalysisError(ReasoningError):
    """Error during text analysis."""
