from rfnry_rag.common.errors import SdkBaseError
from rfnry_rag.retrieval.common.errors import (
    IngestionError,
    RagError,
    RetrievalError,
    TreeIndexingError,
    TreeSearchError,
)


class TestTreeIndexingError:
    def test_is_ingestion_error(self):
        err = TreeIndexingError("build failed")
        assert isinstance(err, IngestionError)

    def test_is_rag_error(self):
        err = TreeIndexingError("build failed")
        assert isinstance(err, RagError)

    def test_is_rfnry_rag_error(self):
        err = TreeIndexingError("build failed")
        assert isinstance(err, SdkBaseError)

    def test_message(self):
        err = TreeIndexingError("tree construction failed for node 5")
        assert str(err) == "tree construction failed for node 5"

    def test_empty_message(self):
        err = TreeIndexingError()
        assert str(err) == ""


class TestTreeSearchError:
    def test_is_retrieval_error(self):
        err = TreeSearchError("search failed")
        assert isinstance(err, RetrievalError)

    def test_is_rag_error(self):
        err = TreeSearchError("search failed")
        assert isinstance(err, RagError)

    def test_is_rfnry_rag_error(self):
        err = TreeSearchError("search failed")
        assert isinstance(err, SdkBaseError)

    def test_message(self):
        err = TreeSearchError("no matching nodes found")
        assert str(err) == "no matching nodes found"

    def test_empty_message(self):
        err = TreeSearchError()
        assert str(err) == ""
