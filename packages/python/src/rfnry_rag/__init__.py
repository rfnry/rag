"""rfnry-rag — Retrieval-Augmented Generation SDK."""

from importlib.metadata import version

__version__ = version("rfnry-rag")

from rfnry_rag.baml.version_check import check_baml as _check_baml

_check_baml()

from rfnry_rag.config import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from rfnry_rag.config import DocumentExpansionConfig as DocumentExpansionConfig
from rfnry_rag.config import DrawingIngestionConfig as DrawingIngestionConfig
from rfnry_rag.config import GenerationConfig as GenerationConfig
from rfnry_rag.config import GraphIngestionConfig as GraphIngestionConfig
from rfnry_rag.config import IngestionConfig as IngestionConfig
from rfnry_rag.config import QueryMode as QueryMode
from rfnry_rag.config import RagEngineConfig as RagEngineConfig
from rfnry_rag.config import RetrievalConfig as RetrievalConfig
from rfnry_rag.config import RoutingConfig as RoutingConfig
from rfnry_rag.exceptions import ConfigurationError as ConfigurationError
from rfnry_rag.exceptions import DuplicateSourceError as DuplicateSourceError
from rfnry_rag.exceptions import EmbeddingError as EmbeddingError
from rfnry_rag.exceptions import EmptyDocumentError as EmptyDocumentError
from rfnry_rag.exceptions import GenerationError as GenerationError
from rfnry_rag.exceptions import IngestionError as IngestionError
from rfnry_rag.exceptions import IngestionInterruptedError as IngestionInterruptedError
from rfnry_rag.exceptions import InputError as InputError
from rfnry_rag.exceptions import ParseError as ParseError
from rfnry_rag.exceptions import RagError as RagError
from rfnry_rag.exceptions import RetrievalError as RetrievalError
from rfnry_rag.exceptions import SourceNotFoundError as SourceNotFoundError
from rfnry_rag.exceptions import StoreError as StoreError
from rfnry_rag.generation.models import QueryResult as QueryResult
from rfnry_rag.generation.models import StreamEvent as StreamEvent
from rfnry_rag.ingestion.base import BaseIngestionMethod as BaseIngestionMethod
from rfnry_rag.ingestion.chunk.chunker import SemanticChunker as SemanticChunker
from rfnry_rag.ingestion.chunk.service import IngestionService as IngestionService
from rfnry_rag.ingestion.embeddings.sparse.fastembed import (
    FastEmbedSparseEmbeddings as FastEmbedSparseEmbeddings,
)
from rfnry_rag.ingestion.methods import AnalyzedIngestion as AnalyzedIngestion
from rfnry_rag.ingestion.methods import DocumentIngestion as DocumentIngestion
from rfnry_rag.ingestion.methods import DrawingIngestion as DrawingIngestion
from rfnry_rag.ingestion.methods import GraphIngestion as GraphIngestion
from rfnry_rag.ingestion.methods import VectorIngestion as VectorIngestion
from rfnry_rag.models import ContentMatch as ContentMatch
from rfnry_rag.models import RetrievedChunk as RetrievedChunk
from rfnry_rag.models import Source as Source
from rfnry_rag.models import SparseVector as SparseVector
from rfnry_rag.observability import BenchmarkCase as BenchmarkCase
from rfnry_rag.observability import BenchmarkCaseResult as BenchmarkCaseResult
from rfnry_rag.observability import BenchmarkConfig as BenchmarkConfig
from rfnry_rag.observability import BenchmarkReport as BenchmarkReport
from rfnry_rag.observability.metrics import ExactMatch as ExactMatch
from rfnry_rag.observability.metrics import F1Score as F1Score
from rfnry_rag.observability.metrics import LLMJudgment as LLMJudgment
from rfnry_rag.observability.models import MetricResult as MetricResult
from rfnry_rag.observability.retrieval_metrics import RetrievalPrecision as RetrievalPrecision
from rfnry_rag.observability.retrieval_metrics import RetrievalRecall as RetrievalRecall
from rfnry_rag.observability.trace import RetrievalTrace as RetrievalTrace
from rfnry_rag.providers import Embeddings as Embeddings
from rfnry_rag.providers import LanguageModelClient as LanguageModelClient
from rfnry_rag.providers import LanguageModelProvider as LanguageModelProvider
from rfnry_rag.providers import Reranking as Reranking
from rfnry_rag.providers import Vision as Vision
from rfnry_rag.retrieval.base import BaseRetrievalMethod as BaseRetrievalMethod
from rfnry_rag.retrieval.methods.document import DocumentRetrieval as DocumentRetrieval
from rfnry_rag.retrieval.methods.enrich import StructuredRetrieval as StructuredRetrieval
from rfnry_rag.retrieval.methods.graph import GraphRetrieval as GraphRetrieval
from rfnry_rag.retrieval.methods.vector import VectorRetrieval as VectorRetrieval
from rfnry_rag.retrieval.search.service import RetrievalService as RetrievalService
from rfnry_rag.server import RagEngine as RagEngine
from rfnry_rag.stores.document.filesystem import FilesystemDocumentStore as FilesystemDocumentStore
from rfnry_rag.stores.document.postgres import PostgresDocumentStore as PostgresDocumentStore
from rfnry_rag.stores.graph.models import GraphEntity as GraphEntity
from rfnry_rag.stores.graph.models import GraphPath as GraphPath
from rfnry_rag.stores.graph.models import GraphRelation as GraphRelation
from rfnry_rag.stores.graph.models import GraphResult as GraphResult
from rfnry_rag.stores.graph.neo4j import Neo4jGraphStore as Neo4jGraphStore
from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore as SQLAlchemyMetadataStore
from rfnry_rag.stores.vector.qdrant import QdrantVectorStore as QdrantVectorStore

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AnalyzedIngestion",
    "BaseIngestionMethod",
    "BaseRetrievalMethod",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkConfig",
    "BenchmarkReport",
    "ConfigurationError",
    "ContentMatch",
    "DocumentExpansionConfig",
    "DocumentIngestion",
    "DocumentRetrieval",
    "DrawingIngestion",
    "DrawingIngestionConfig",
    "DuplicateSourceError",
    "EmbeddingError",
    "Embeddings",
    "EmptyDocumentError",
    "ExactMatch",
    "F1Score",
    "FastEmbedSparseEmbeddings",
    "FilesystemDocumentStore",
    "GenerationConfig",
    "GenerationError",
    "GraphEntity",
    "GraphIngestion",
    "GraphIngestionConfig",
    "GraphPath",
    "GraphRelation",
    "GraphResult",
    "GraphRetrieval",
    "IngestionConfig",
    "IngestionError",
    "IngestionInterruptedError",
    "IngestionService",
    "InputError",
    "LLMJudgment",
    "LanguageModelClient",
    "LanguageModelProvider",
    "MetricResult",
    "Neo4jGraphStore",
    "ParseError",
    "PostgresDocumentStore",
    "QdrantVectorStore",
    "QueryMode",
    "QueryResult",
    "RagEngine",
    "RagEngineConfig",
    "RagError",
    "Reranking",
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
    "Source",
    "SourceNotFoundError",
    "SparseVector",
    "StoreError",
    "StreamEvent",
    "StructuredRetrieval",
    "VectorIngestion",
    "VectorRetrieval",
    "Vision",
]
