"""rfnry-knowledge — Provider-agnostic retrieval engine."""

from importlib.metadata import version

__version__ = version("rfnry-knowledge")

from rfnry_knowledge.baml.version_check import check_baml as _check_baml

_check_baml()

from rfnry_knowledge.config import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from rfnry_knowledge.config import ContextualChunkConfig as ContextualChunkConfig
from rfnry_knowledge.config import DocumentExpansionConfig as DocumentExpansionConfig
from rfnry_knowledge.config import DrawingIngestionConfig as DrawingIngestionConfig
from rfnry_knowledge.config import EntityIngestionConfig as EntityIngestionConfig
from rfnry_knowledge.config import GenerationConfig as GenerationConfig
from rfnry_knowledge.config import IngestionConfig as IngestionConfig
from rfnry_knowledge.config import KnowledgeEngineConfig as KnowledgeEngineConfig
from rfnry_knowledge.config import QueryMode as QueryMode
from rfnry_knowledge.config import RetrievalConfig as RetrievalConfig
from rfnry_knowledge.config import RoutingConfig as RoutingConfig
from rfnry_knowledge.exceptions import ConfigurationError as ConfigurationError
from rfnry_knowledge.exceptions import DuplicateSourceError as DuplicateSourceError
from rfnry_knowledge.exceptions import EmbeddingError as EmbeddingError
from rfnry_knowledge.exceptions import EmptyDocumentError as EmptyDocumentError
from rfnry_knowledge.exceptions import GenerationError as GenerationError
from rfnry_knowledge.exceptions import IngestionError as IngestionError
from rfnry_knowledge.exceptions import IngestionInterruptedError as IngestionInterruptedError
from rfnry_knowledge.exceptions import InputError as InputError
from rfnry_knowledge.exceptions import KnowledgeEngineError as KnowledgeEngineError
from rfnry_knowledge.exceptions import ParseError as ParseError
from rfnry_knowledge.exceptions import RetrievalError as RetrievalError
from rfnry_knowledge.exceptions import SourceNotFoundError as SourceNotFoundError
from rfnry_knowledge.exceptions import StoreError as StoreError
from rfnry_knowledge.generation.models import QueryResult as QueryResult
from rfnry_knowledge.generation.models import StreamEvent as StreamEvent
from rfnry_knowledge.ingestion.base import BaseIngestionMethod as BaseIngestionMethod
from rfnry_knowledge.ingestion.chunk.chunker import SemanticChunker as SemanticChunker
from rfnry_knowledge.ingestion.chunk.service import IngestionService as IngestionService
from rfnry_knowledge.ingestion.methods import DrawingIngestion as DrawingIngestion
from rfnry_knowledge.ingestion.methods import EntityIngestion as EntityIngestion
from rfnry_knowledge.ingestion.methods import KeywordIngestion as KeywordIngestion
from rfnry_knowledge.ingestion.methods import SemanticIngestion as SemanticIngestion
from rfnry_knowledge.ingestion.methods import StructuredIngestion as StructuredIngestion
from rfnry_knowledge.knowledge.engine import KnowledgeEngine as KnowledgeEngine
from rfnry_knowledge.models import ContentMatch as ContentMatch
from rfnry_knowledge.models import RetrievedChunk as RetrievedChunk
from rfnry_knowledge.models import Source as Source
from rfnry_knowledge.models import SparseVector as SparseVector
from rfnry_knowledge.observability import BenchmarkCase as BenchmarkCase
from rfnry_knowledge.observability import BenchmarkCaseResult as BenchmarkCaseResult
from rfnry_knowledge.observability import BenchmarkConfig as BenchmarkConfig
from rfnry_knowledge.observability import BenchmarkReport as BenchmarkReport
from rfnry_knowledge.observability.metrics import ExactMatch as ExactMatch
from rfnry_knowledge.observability.metrics import F1Score as F1Score
from rfnry_knowledge.observability.metrics import LLMJudgment as LLMJudgment
from rfnry_knowledge.observability.models import MetricResult as MetricResult
from rfnry_knowledge.observability.retrieval_metrics import RetrievalPrecision as RetrievalPrecision
from rfnry_knowledge.observability.retrieval_metrics import RetrievalRecall as RetrievalRecall
from rfnry_knowledge.observability.trace import RetrievalTrace as RetrievalTrace
from rfnry_knowledge.providers import BaseEmbeddings as BaseEmbeddings
from rfnry_knowledge.providers import BaseReranking as BaseReranking
from rfnry_knowledge.providers import BaseSparseEmbeddings as BaseSparseEmbeddings
from rfnry_knowledge.providers import EmbeddingResult as EmbeddingResult
from rfnry_knowledge.providers import ProviderClient as ProviderClient
from rfnry_knowledge.providers import RerankResult as RerankResult
from rfnry_knowledge.providers import TokenCounter as TokenCounter
from rfnry_knowledge.providers import TokenUsage as TokenUsage
from rfnry_knowledge.providers import build_registry as build_registry
from rfnry_knowledge.retrieval.base import BaseRetrievalMethod as BaseRetrievalMethod
from rfnry_knowledge.retrieval.methods.entity import EntityRetrieval as EntityRetrieval
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval as KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval as SemanticRetrieval
from rfnry_knowledge.retrieval.search.service import RetrievalService as RetrievalService
from rfnry_knowledge.stores.document.filesystem import FilesystemDocumentStore as FilesystemDocumentStore
from rfnry_knowledge.stores.document.postgres import PostgresDocumentStore as PostgresDocumentStore
from rfnry_knowledge.stores.graph.models import GraphEntity as GraphEntity
from rfnry_knowledge.stores.graph.models import GraphPath as GraphPath
from rfnry_knowledge.stores.graph.models import GraphRelation as GraphRelation
from rfnry_knowledge.stores.graph.models import GraphResult as GraphResult
from rfnry_knowledge.stores.graph.neo4j import Neo4jGraphStore as Neo4jGraphStore
from rfnry_knowledge.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore as SQLAlchemyMetadataStore
from rfnry_knowledge.stores.vector.qdrant import QdrantVectorStore as QdrantVectorStore

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "BaseEmbeddings",
    "BaseIngestionMethod",
    "BaseReranking",
    "BaseRetrievalMethod",
    "BaseSparseEmbeddings",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkConfig",
    "BenchmarkReport",
    "ConfigurationError",
    "ContentMatch",
    "ContextualChunkConfig",
    "DocumentExpansionConfig",
    "DrawingIngestion",
    "DrawingIngestionConfig",
    "DuplicateSourceError",
    "EmbeddingError",
    "EmbeddingResult",
    "EmptyDocumentError",
    "EntityIngestion",
    "EntityIngestionConfig",
    "EntityRetrieval",
    "ExactMatch",
    "F1Score",
    "FilesystemDocumentStore",
    "GenerationConfig",
    "GenerationError",
    "GraphEntity",
    "GraphPath",
    "GraphRelation",
    "GraphResult",
    "IngestionConfig",
    "IngestionError",
    "IngestionInterruptedError",
    "IngestionService",
    "InputError",
    "KeywordIngestion",
    "KeywordRetrieval",
    "KnowledgeEngine",
    "KnowledgeEngineConfig",
    "KnowledgeEngineError",
    "LLMJudgment",
    "MetricResult",
    "Neo4jGraphStore",
    "ParseError",
    "PostgresDocumentStore",
    "ProviderClient",
    "QdrantVectorStore",
    "QueryMode",
    "QueryResult",
    "RerankResult",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalPrecision",
    "RetrievalRecall",
    "RetrievalService",
    "RetrievalTrace",
    "RetrievedChunk",
    "RoutingConfig",
    "SQLAlchemyMetadataStore",
    "SemanticChunker",
    "SemanticIngestion",
    "SemanticRetrieval",
    "Source",
    "SourceNotFoundError",
    "SparseVector",
    "StoreError",
    "StreamEvent",
    "StructuredIngestion",
    "TokenCounter",
    "TokenUsage",
    "build_registry",
]
