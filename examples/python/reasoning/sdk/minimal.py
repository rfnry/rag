"""Minimal — classify a single text with LLM reasoning."""

import asyncio

from rfnry_rag.reasoning import (
    CategoryDefinition,
    ClassificationService,
    LanguageModelClient,
    LanguageModelProvider,
)

lm_client = LanguageModelClient(
    provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="your_api_key")
)

classifier = ClassificationService(lm_client=lm_client)

categories = [
    CategoryDefinition(name="refund", description="Customer wants money back"),
    CategoryDefinition(name="shipping", description="Delivery or tracking questions"),
    CategoryDefinition(name="cancellation", description="Customer wants to cancel an order"),
]


async def main():
    result = await classifier.classify("I never received my order and I want my money back", categories)
    print(f"Category: {result.category} ({result.confidence:.0%})")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
