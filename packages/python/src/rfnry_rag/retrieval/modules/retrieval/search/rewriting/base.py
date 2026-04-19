from typing import Protocol


class BaseQueryRewriting(Protocol):
    async def rewrite(self, query: str, conversation_context: str | None = None) -> list[str]:
        """Transform a single query into one or more retrieval queries.

        Args:
            query: The user's original query text.
            conversation_context: Optional conversation history context
                (the output of _build_retrieval_query enrichment, if any).

        Returns:
            A list of 1+ queries to search. The original query may or may
            not be included — the caller always searches the original
            alongside whatever the rewriter returns.
        """
        ...
