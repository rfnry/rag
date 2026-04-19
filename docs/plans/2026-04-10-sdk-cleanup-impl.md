# SDK Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize naming, patterns, and public API surface across both SDKs. Replace the embeddings/LM pattern asymmetry with a shared `LanguageModelProvider` primitive. Collapse provider-specific classes into dispatching facades. Gerundify agent-noun class names. Remove confusing env vars. Trim public API to strict core.

**Architecture:** One shared primitive (`LanguageModelProvider`) for provider+model+api_key. Active wrappers (`LanguageModelClient`, `Embeddings`, `Vision`, `Reranking`) consume it and dispatch to private per-provider implementations. `BaseEmbeddings` deduplicated to `x64rag/common/protocols.py`. Public `__init__.py` surfaces strict end-user API only; `Base*` protocols go private.

**Tech Stack:** Python 3.12, Protocol typing, asyncio, dataclasses, pytest + AsyncMock, BAML for LLM routing.

**Design doc:** `docs/plans/2026-04-10-sdk-cleanup-design.md`

**Pre-1.0, hard breaks allowed.** No backward-compat shims.

---

## Phase 0: Baseline

### Task 0: Establish green baseline

**Step 1:** Run the full test suite to confirm a clean starting state.

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
uv run poe check 2>&1 | tail -20
```

Expected: all tests pass, mypy clean, ruff clean. If anything is already failing, **stop** and investigate before starting the cleanup — we need a green baseline so that every later `poe test` run isolates cleanup regressions.

**Step 2:** Capture test count for later comparison.

```bash
uv run poe test -q 2>&1 | tail -5
```

Record the number of tests passing in your notes. It should match at the end.

**Step 3:** Commit any unrelated local state separately (if any) before starting.

No file changes in Task 0. No commit.

---

## Phase 1: Core primitive rename

### Task 1: `LanguageModelProvider` and `LanguageModelClient`

**Files:**
- Modify: `src/rfnry_rag/common/language_model.py`
- Modify: all files importing `LanguageModelConfig` / `LanguageModelClientConfig` (grep first)

**Rationale:** This is the foundation. All later facades use `LanguageModelProvider`. Do this rename first so nothing else builds on stale names.

**Step 1: Survey all call sites**

```bash
# Count references so you can verify completeness after.
```
Use Grep tool with pattern `LanguageModelConfig|LanguageModelClientConfig` across `src/` and `examples/` and `docs/`. Expect ~40 hits in `src/`, ~20 in `examples/`, several in READMEs.

**Step 2: Rewrite `src/rfnry_rag/common/language_model.py`**

Replace the file contents with:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from baml_py import ClientRegistry

from x64rag.common.errors import ConfigurationError

_MAX_RETRIES_LIMIT = 5

_CLIENT_DEFAULT = "Default"
_CLIENT_FALLBACK = "Fallback"
_CLIENT_ROUTER = "Router"


@dataclass
class LanguageModelProvider:
    """Identity: which API, which model, auth key.

    Used directly by Embeddings/Vision/Reranking for dedicated provider APIs,
    or wrapped by LanguageModelClient for BAML-routed LLM calls.
    """

    provider: str
    model: str
    api_key: str | None = None


@dataclass
class LanguageModelClient:
    """BAML-backed LLM client: routing, retries, fallback, generation params."""

    provider: LanguageModelProvider
    fallback: LanguageModelProvider | None = None
    max_retries: int = 3
    strategy: Literal["primary_only", "fallback"] = "primary_only"
    max_tokens: int = 4096
    temperature: float = 0.0
    boundary_api_key: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("primary_only", "fallback"):
            raise ConfigurationError(
                f"Invalid strategy {self.strategy!r}, must be 'primary_only' or 'fallback'"
            )
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES_LIMIT:
            raise ConfigurationError(
                f"max_retries must be 0-{_MAX_RETRIES_LIMIT}, got {self.max_retries}"
            )
        if self.strategy == "fallback" and self.fallback is None:
            raise ConfigurationError("strategy='fallback' requires a fallback client")


def _retry_policy_name(max_retries: int) -> str | None:
    if max_retries == 0:
        return None
    return f"Retry{max_retries}"


def _build_client_options(
    provider: LanguageModelProvider, max_tokens: int, temperature: float
) -> dict:
    options = {
        "model": provider.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider.api_key:
        options["api_key"] = provider.api_key
    return options


def build_registry(client: LanguageModelClient) -> ClientRegistry:
    registry = ClientRegistry()
    policy = _retry_policy_name(client.max_retries)

    registry.add_llm_client(
        _CLIENT_DEFAULT,
        provider=client.provider.provider,
        options=_build_client_options(client.provider, client.max_tokens, client.temperature),
        retry_policy=policy,
    )

    if client.strategy == "fallback" and client.fallback is not None:
        registry.add_llm_client(
            _CLIENT_FALLBACK,
            provider=client.fallback.provider,
            options=_build_client_options(client.fallback, client.max_tokens, client.temperature),
            retry_policy=policy,
        )
        registry.add_llm_client(
            _CLIENT_ROUTER,
            provider="fallback",
            options={"strategy": [_CLIENT_DEFAULT, _CLIENT_FALLBACK]},
        )
        registry.set_primary(_CLIENT_ROUTER)
    else:
        registry.set_primary(_CLIENT_DEFAULT)

    if client.boundary_api_key:
        os.environ["BOUNDARY_API_KEY"] = client.boundary_api_key

    return registry
```

**Key changes:**
- `LanguageModelClientConfig` → `LanguageModelProvider` (no more `max_tokens`/`temperature` — those belong on the client)
- `LanguageModelConfig` → `LanguageModelClient`
- `build_registry(config)` parameter renamed to `client` for clarity, signature now takes `LanguageModelClient`
- `_build_client_options` now takes the provider + per-call generation params (max_tokens/temperature live on the client, not the provider)

**Step 3: Update shim re-exports**

In `src/rfnry_rag/retrieval/common/language_model.py` and `src/rfnry_rag/reasoning/common/language_model.py`, change the re-exports:

```python
# src/rfnry_rag/retrieval/common/language_model.py
from x64rag.common.language_model import (
    LanguageModelClient as LanguageModelClient,
    LanguageModelProvider as LanguageModelProvider,
    build_registry as build_registry,
)
```

Same for reasoning. Delete old re-exports for `LanguageModelConfig` / `LanguageModelClientConfig`.

**Step 4: Update all call sites**

Use Grep to find every file importing or constructing the old names. Apply these substitutions (Edit tool, `replace_all`):

- `LanguageModelClientConfig` → `LanguageModelProvider`
- `LanguageModelConfig(client=...)` → `LanguageModelClient(provider=...)` (the kwarg name changes from `client` to `provider`)
- `LanguageModelConfig` (without `client=`) → `LanguageModelClient`
- Any call-site that constructed `LanguageModelClientConfig(..., max_tokens=..., temperature=...)` — those kwargs migrate to the wrapping `LanguageModelClient` call.

Files to touch (from the Grep audit):
- `src/rfnry_rag/retrieval/server.py` — `GenerationConfig.lm_config`, etc.
- `src/rfnry_rag/retrieval/modules/generation/service.py`, `step.py`, `grounding.py`, `confidence.py`
- `src/rfnry_rag/retrieval/modules/ingestion/analyze/*.py`
- `src/rfnry_rag/retrieval/modules/retrieval/search/rewriting/*.py`
- `src/rfnry_rag/retrieval/modules/retrieval/search/reranking/llm.py`
- `src/rfnry_rag/retrieval/modules/retrieval/refinement/abstractive.py`
- `src/rfnry_rag/retrieval/modules/retrieval/judging.py`
- `src/rfnry_rag/retrieval/modules/evaluation/metrics.py`
- `src/rfnry_rag/retrieval/cli/config.py`
- `src/rfnry_rag/reasoning/modules/{analysis,classification,clustering,compliance,evaluation}/service.py`
- `src/rfnry_rag/reasoning/cli/config.py`
- All `src/rfnry_rag/**/tests/test_*.py` and inline `test_*.py`

**Step 5: Update top-level `x64rag/__init__.py`**

```python
from x64rag.common.language_model import LanguageModelClient as LanguageModelClient
from x64rag.common.language_model import LanguageModelProvider as LanguageModelProvider
```

Delete old re-exports.

**Step 6: Update retrieval/reasoning `__init__.py` re-exports**

Same treatment for `src/rfnry_rag/retrieval/__init__.py` and `src/rfnry_rag/reasoning/__init__.py` — replace the `LanguageModelConfig` / `LanguageModelClientConfig` re-exports and `__all__` entries.

**Step 7: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

Expected: all tests pass (test count unchanged), mypy clean. If tests reference `LanguageModelConfig` you missed, they'll fail with `ImportError` or `NameError` — fix and re-run.

**Step 8: Verify no stragglers**

```bash
```
Use Grep with pattern `LanguageModelConfig|LanguageModelClientConfig` across `src/` — expected: zero hits. If any remain (in comments, docstrings, README fragments under `src/`), update them.

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor: rename LanguageModelConfig/LanguageModelClientConfig → LanguageModelClient/LanguageModelProvider

- Split provider identity (LanguageModelProvider) from BAML wrapper (LanguageModelClient)
- Move max_tokens/temperature from provider primitive to client (LLM-only concerns)
- Update build_registry signature to take LanguageModelClient
- Update all call sites, re-exports, and tests"
```

---

## Phase 2: Protocol dedup + AceError

### Task 2: Deduplicate `BaseEmbeddings`, rename reasoning's vector protocol

**Files:**
- Create: `src/rfnry_rag/common/protocols.py`
- Delete: `src/rfnry_rag/retrieval/modules/ingestion/embeddings/base.py` (or keep as shim?)
- Modify: `src/rfnry_rag/reasoning/protocols.py`
- Modify: all imports of `BaseEmbeddings` and (reasoning's) `BaseVectorStore`

**Step 1: Create shared protocol module**

```python
# src/rfnry_rag/common/protocols.py
from __future__ import annotations

from typing import Any, Protocol


class BaseEmbeddings(Protocol):
    """Common embeddings contract used by both retrieval and reasoning SDKs."""

    @property
    def model(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embedding_dimension(self) -> int: ...


class BaseSemanticIndex(Protocol):
    """Read-only semantic lookup used by reasoning services that need vector search
    without the full retrieval VectorStore surface (upsert/delete/hybrid_search/etc)."""

    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[Any], str | None]: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...
```

**Step 2: Delete the duplicate definitions**

- `src/rfnry_rag/retrieval/modules/ingestion/embeddings/base.py` — replace contents with a re-export from `x64rag.common.protocols`:
  ```python
  from x64rag.common.protocols import BaseEmbeddings as BaseEmbeddings
  ```
  (Keep the file so existing relative imports still resolve; this becomes a thin re-export.)
- `src/rfnry_rag/reasoning/protocols.py` — replace contents:
  ```python
  from x64rag.common.protocols import BaseEmbeddings as BaseEmbeddings
  from x64rag.common.protocols import BaseSemanticIndex as BaseSemanticIndex
  ```

**Step 3: Rename reasoning's `BaseVectorStore` → `BaseSemanticIndex`**

Grep for `BaseVectorStore` usage in the reasoning SDK only:

Use Grep with pattern `BaseVectorStore` path `src/rfnry_rag/reasoning`. Each hit gets `BaseVectorStore` → `BaseSemanticIndex`.

**Do not touch** any `BaseVectorStore` references in `src/rfnry_rag/retrieval/` — the retrieval one keeps its name (full-CRUD vector store).

**Step 4: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

Expected: green.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: dedupe BaseEmbeddings and rename reasoning vector protocol

- Move BaseEmbeddings to x64rag/common/protocols.py (single source of truth)
- Rename reasoning's BaseVectorStore → BaseSemanticIndex to disambiguate from
  retrieval's full-CRUD BaseVectorStore (same name, different contract)
- Retrieval BaseVectorStore kept as-is with full upsert/delete/hybrid_search API"
```

---

### Task 3: Rename `AceError` → `ReasoningError`

**Files:**
- Modify: `src/rfnry_rag/reasoning/common/errors.py`
- Modify: all reasoning tests and any remaining `AceError` references
- Modify: re-exports in `src/rfnry_rag/reasoning/__init__.py`, `src/rfnry_rag/reasoning/common/errors.py`
- Modify: `CLAUDE.md` (error hierarchy section)

**Step 1: Grep for every `AceError` reference**

Use Grep with pattern `AceError` — expected hits in `errors.py`, `__init__.py`, `CLAUDE.md`, possibly tests.

**Step 2: Rewrite `src/rfnry_rag/reasoning/common/errors.py`**

```python
from x64rag.common.errors import ConfigurationError as ConfigurationError
from x64rag.common.errors import X64RagError


class ReasoningError(X64RagError):
    """Base exception for reasoning SDK errors."""


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
```

**Step 3: Update `src/rfnry_rag/reasoning/__init__.py`**

Replace `AceError as AceError` re-export with `ReasoningError as ReasoningError`. Update `__all__`.

**Step 4: Update `CLAUDE.md` error hierarchy section**

Find the block starting with `X64RagError (common base)` and replace `AceError (reasoning)` with `ReasoningError (reasoning)`.

**Step 5: Update any test that imports `AceError`**

Grep for `AceError` in tests — fix each.

**Step 6: Run tests**

```bash
uv run poe test 2>&1 | tail -20
```

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename AceError → ReasoningError

Hard rename (pre-1.0, no alias). The legacy 'Ace' name predates the current
scope and didn't match the 'reasoning' module naming."
```

---

## Phase 3: Provider-class collapse

### Task 4: `Embeddings` facade

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/embeddings/openai.py` (rename class to `_OpenAIEmbeddings`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/embeddings/voyage.py` (rename class to `_VoyageEmbeddings`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/embeddings/cohere.py` (rename class to `_CohereEmbeddings`)
- Create: `src/rfnry_rag/retrieval/modules/ingestion/embeddings/facade.py`
- Create: `src/rfnry_rag/retrieval/tests/test_embeddings_facade.py`

**Step 1: Write failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_embeddings_facade.py
import pytest

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelProvider
from x64rag.retrieval.modules.ingestion.embeddings.facade import Embeddings
from x64rag.retrieval.modules.ingestion.embeddings.openai import _OpenAIEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.voyage import _VoyageEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.cohere import _CohereEmbeddings


def test_embeddings_dispatches_to_openai():
    provider = LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _OpenAIEmbeddings)
    assert embeddings._impl.model == "text-embedding-3-small"


def test_embeddings_dispatches_to_voyage():
    provider = LanguageModelProvider(provider="voyage", model="voyage-3", api_key="vo-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _VoyageEmbeddings)


def test_embeddings_dispatches_to_cohere():
    provider = LanguageModelProvider(provider="cohere", model="embed-english-v3.0", api_key="co-test")
    embeddings = Embeddings(provider)
    assert isinstance(embeddings._impl, _CohereEmbeddings)


def test_embeddings_unsupported_provider_raises():
    provider = LanguageModelProvider(provider="unknown", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported embeddings provider"):
        Embeddings(provider)


def test_embeddings_model_property_delegates():
    provider = LanguageModelProvider(provider="openai", model="text-embedding-3-large", api_key="sk-test")
    embeddings = Embeddings(provider)
    assert embeddings.model == "text-embedding-3-large"
```

**Step 2: Verify tests fail**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_embeddings_facade.py -v
```

Expected: `ImportError` or similar (facade and private classes don't exist yet).

**Step 3: Privatize concrete classes**

In `openai.py`: rename `class OpenAIEmbeddings` → `class _OpenAIEmbeddings`. Change constructor signature to accept a `LanguageModelProvider`:

```python
from openai import AsyncOpenAI

from x64rag.common.language_model import LanguageModelProvider


class _OpenAIEmbeddings:
    def __init__(self, provider: LanguageModelProvider, max_retries: int = 3) -> None:
        self._client = AsyncOpenAI(api_key=provider.api_key, max_retries=max_retries)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
```

Do the same for `voyage.py` → `_VoyageEmbeddings` and `cohere.py` → `_CohereEmbeddings`. Each takes a `LanguageModelProvider` and extracts `api_key`/`model` from it internally.

**Step 4: Create the facade**

```python
# src/rfnry_rag/retrieval/modules/ingestion/embeddings/facade.py
from __future__ import annotations

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelProvider
from x64rag.retrieval.modules.ingestion.embeddings.cohere import _CohereEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.openai import _OpenAIEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.voyage import _VoyageEmbeddings


class Embeddings:
    """Embeddings client dispatching to the correct provider implementation."""

    def __init__(self, provider: LanguageModelProvider) -> None:
        match provider.provider:
            case "openai":
                self._impl = _OpenAIEmbeddings(provider)
            case "voyage":
                self._impl = _VoyageEmbeddings(provider)
            case "cohere":
                self._impl = _CohereEmbeddings(provider)
            case _:
                raise ConfigurationError(
                    f"Unsupported embeddings provider: {provider.provider!r}. "
                    f"Supported: openai, voyage, cohere."
                )

    @property
    def model(self) -> str:
        return self._impl.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._impl.embed(texts)

    async def embedding_dimension(self) -> int:
        return await self._impl.embedding_dimension()
```

**Step 5: Run the new tests**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_embeddings_facade.py -v
```

Expected: all 5 tests pass.

**Step 6: Update every call site that constructed `OpenAIEmbeddings`/`VoyageEmbeddings`/`CohereEmbeddings`**

Grep for these three names. Each usage changes:

```python
# Before
OpenAIEmbeddings(api_key="sk-...", model="text-embedding-3-small")
# After
Embeddings(LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="sk-..."))
```

Key files:
- `src/rfnry_rag/retrieval/cli/config.py` — `_build_embeddings()` function; the if/elif chain collapses into a single `Embeddings(LanguageModelProvider(provider=provider, model=model, api_key=api_key))` call.
- `src/rfnry_rag/retrieval/__init__.py` — remove `OpenAIEmbeddings`/`VoyageEmbeddings`/`CohereEmbeddings` re-exports. Add `Embeddings`.

Leave examples and READMEs until Phase 8 — we update all of them together.

**Step 7: Update the sparse embeddings file reference (if any)**

Grep for `from x64rag.retrieval.modules.ingestion.embeddings.openai import OpenAIEmbeddings` and similar — replace with the facade import where appropriate.

**Step 8: Run all tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

Expected: green, plus 5 new tests from step 5.

**Step 9: Commit**

```bash
git add -A
git commit -m "refactor: collapse embeddings into single Embeddings(LanguageModelProvider) facade

- OpenAIEmbeddings/VoyageEmbeddings/CohereEmbeddings become private _-prefixed
- New Embeddings facade dispatches on provider.provider
- All call sites (cli/config.py, services) updated to use the facade + primitive
- Public API surface drops from 3 classes to 1"
```

---

### Task 5: `Vision` facade

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/vision/anthropic.py` → `_AnthropicVision`
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/vision/openai.py` → `_OpenAIVision`
- Create: `src/rfnry_rag/retrieval/modules/ingestion/vision/facade.py`
- Create: `src/rfnry_rag/retrieval/tests/test_vision_facade.py`

**Step 1: Failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_vision_facade.py
import pytest

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelProvider
from x64rag.retrieval.modules.ingestion.vision.facade import Vision
from x64rag.retrieval.modules.ingestion.vision.anthropic import _AnthropicVision
from x64rag.retrieval.modules.ingestion.vision.openai import _OpenAIVision


def test_vision_dispatches_to_anthropic():
    provider = LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _AnthropicVision)


def test_vision_dispatches_to_openai():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    vision = Vision(provider)
    assert isinstance(vision._impl, _OpenAIVision)


def test_vision_unsupported_raises():
    provider = LanguageModelProvider(provider="cohere", model="m", api_key="k")
    with pytest.raises(ConfigurationError, match="Unsupported vision provider"):
        Vision(provider)
```

**Step 2: Verify fail**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_vision_facade.py -v
```

**Step 3: Privatize `AnthropicVision` → `_AnthropicVision` and `OpenAIVision` → `_OpenAIVision`**

Change constructors to take `LanguageModelProvider`. Keep `max_tokens` and `max_retries` as kwargs with sensible defaults (these are vision-specific, not generic client knobs).

```python
# anthropic.py
from anthropic import AsyncAnthropic

from x64rag.common.language_model import LanguageModelProvider

class _AnthropicVision:
    def __init__(
        self,
        provider: LanguageModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        self._client = AsyncAnthropic(api_key=provider.api_key, max_retries=max_retries)
        self._model = provider.model
        self._max_tokens = max_tokens
        # ... rest unchanged
```

Same for `openai.py`.

**Step 4: Create `vision/facade.py`**

```python
from __future__ import annotations

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelProvider
from x64rag.retrieval.modules.ingestion.vision.anthropic import _AnthropicVision
from x64rag.retrieval.modules.ingestion.vision.openai import _OpenAIVision


class Vision:
    """Vision client dispatching to the correct provider implementation."""

    def __init__(
        self,
        provider: LanguageModelProvider,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> None:
        match provider.provider:
            case "anthropic":
                self._impl = _AnthropicVision(provider, max_tokens=max_tokens, max_retries=max_retries)
            case "openai":
                self._impl = _OpenAIVision(provider, max_tokens=max_tokens, max_retries=max_retries)
            case _:
                raise ConfigurationError(
                    f"Unsupported vision provider: {provider.provider!r}. Supported: anthropic, openai."
                )

    # Delegate any vision methods the facade should expose
    async def analyze(self, *args, **kwargs):
        return await self._impl.analyze(*args, **kwargs)
```

*Check `BaseVision` Protocol in `vision/base.py` for the full method surface and mirror all methods on the facade.*

**Step 5: Run facade tests**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_vision_facade.py -v
```

Expected: all 3 pass.

**Step 6: Update call sites**

Grep for `AnthropicVision` / `OpenAIVision`. Fix:
- `src/rfnry_rag/retrieval/cli/config.py` — `_build_vision()` collapses
- `src/rfnry_rag/retrieval/__init__.py` — remove `AnthropicVision`/`OpenAIVision` exports, add `Vision`
- Any service directly constructing vision

**Step 7: Run full test suite**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor: collapse vision into single Vision(LanguageModelProvider) facade

AnthropicVision/OpenAIVision become private _-prefixed. Vision facade
dispatches on provider.provider, delegates analyze() to the implementation."
```

---

### Task 6: Unified `Reranking` facade

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/reranking/cohere.py` → `_CohereReranking`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/reranking/voyage.py` → `_VoyageReranking`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/reranking/llm.py` → `_LLMReranking`
- Create: `src/rfnry_rag/retrieval/modules/retrieval/search/reranking/facade.py`
- Create: `src/rfnry_rag/retrieval/tests/test_reranking_facade.py`

**Step 1: Failing tests**

```python
# src/rfnry_rag/retrieval/tests/test_reranking_facade.py
import pytest

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelClient, LanguageModelProvider
from x64rag.retrieval.modules.retrieval.search.reranking.facade import Reranking
from x64rag.retrieval.modules.retrieval.search.reranking.cohere import _CohereReranking
from x64rag.retrieval.modules.retrieval.search.reranking.voyage import _VoyageReranking
from x64rag.retrieval.modules.retrieval.search.reranking.llm import _LLMReranking


def test_reranking_with_cohere_provider_uses_dedicated_api():
    provider = LanguageModelProvider(provider="cohere", model="rerank-v3.5", api_key="co-test")
    reranker = Reranking(provider)
    assert isinstance(reranker._impl, _CohereReranking)


def test_reranking_with_voyage_provider_uses_dedicated_api():
    provider = LanguageModelProvider(provider="voyage", model="rerank-2.5", api_key="vo-test")
    reranker = Reranking(provider)
    assert isinstance(reranker._impl, _VoyageReranking)


def test_reranking_with_client_uses_llm_path():
    client = LanguageModelClient(
        provider=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test"),
    )
    reranker = Reranking(client)
    assert isinstance(reranker._impl, _LLMReranking)


def test_reranking_with_unsupported_provider_raises():
    provider = LanguageModelProvider(provider="openai", model="gpt-4o", api_key="sk-test")
    with pytest.raises(ConfigurationError, match="no dedicated reranker API"):
        Reranking(provider)
```

**Step 2: Verify fail**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_reranking_facade.py -v
```

**Step 3: Privatize concrete classes**

- `cohere.py`: `CohereReranking` → `_CohereReranking`, constructor takes `LanguageModelProvider`
- `voyage.py`: `VoyageReranking` → `_VoyageReranking`, same
- `llm.py`: `LLMReranking` → `_LLMReranking`, constructor takes `LanguageModelClient` (already did similar with `LanguageModelConfig` in Phase 1)

```python
# cohere.py
import cohere
from x64rag.common.language_model import LanguageModelProvider

class _CohereReranking:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = cohere.AsyncClientV2(api_key=provider.api_key)
        self._model = provider.model
    # ... rerank() unchanged
```

**Step 4: Create `reranking/facade.py`**

```python
from __future__ import annotations

from x64rag.common.errors import ConfigurationError
from x64rag.common.language_model import LanguageModelClient, LanguageModelProvider
from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.search.reranking.cohere import _CohereReranking
from x64rag.retrieval.modules.retrieval.search.reranking.llm import _LLMReranking
from x64rag.retrieval.modules.retrieval.search.reranking.voyage import _VoyageReranking

_DEDICATED_RERANKER_PROVIDERS = {"cohere", "voyage"}


class Reranking:
    """Unified reranker facade.

    Accepts either a LanguageModelProvider (dispatches to a dedicated reranker API —
    Cohere or Voyage) or a LanguageModelClient (dispatches to LLM-as-reranker via BAML).
    """

    def __init__(self, config: LanguageModelProvider | LanguageModelClient) -> None:
        if isinstance(config, LanguageModelClient):
            self._impl = _LLMReranking(config)
        else:
            if config.provider not in _DEDICATED_RERANKER_PROVIDERS:
                raise ConfigurationError(
                    f"Provider {config.provider!r} has no dedicated reranker API. "
                    f"Wrap it in LanguageModelClient to use LLM-as-reranker."
                )
            self._impl = (
                _CohereReranking(config) if config.provider == "cohere" else _VoyageReranking(config)
            )

    async def rerank(
        self, query: str, results: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        return await self._impl.rerank(query, results, top_k)
```

**Step 5: Run facade tests**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_reranking_facade.py -v
```

Expected: all 4 pass.

**Step 6: Update call sites**

Grep for `CohereReranking`, `VoyageReranking`, `LLMReranking`. Fix:
- `src/rfnry_rag/retrieval/cli/config.py` — `_build_reranking()` collapses
- `src/rfnry_rag/retrieval/__init__.py` — remove `CohereReranking`/`VoyageReranking` re-exports, add `Reranking` (single)
- Any generation/pipeline service that uses rerankers

**Step 7: Run full test suite**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor: unify reranking into single Reranking facade with type dispatch

- Cohere/Voyage/LLM rerankers become private _-prefixed
- Reranking(LanguageModelProvider) → dedicated API (Cohere, Voyage only)
- Reranking(LanguageModelClient) → LLM-as-reranker via BAML
- Input type selects mechanism; unsupported provider raises ConfigurationError"
```

---

## Phase 4: Gerundification

### Task 7: Refiner → Refinement

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/refinement/extractive.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/refinement/abstractive.py`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/refinement/base.py`
- Modify: all call sites and re-exports

**Step 1: Grep all references**

Use Grep pattern `ExtractiveRefiner|AbstractiveRefiner|BaseChunkRefiner|ChunkRefiner`. Note every file.

**Step 2: Rename classes in their defining files**

- `extractive.py`: `class ExtractiveRefiner` → `class ExtractiveRefinement`
- `abstractive.py`: `class AbstractiveRefiner` → `class AbstractiveRefinement`
- `base.py`: `class BaseChunkRefiner(Protocol)` → `class BaseChunkRefinement(Protocol)`

**Step 3: Replace across the codebase**

Use Edit `replace_all` or the refactor-expert agent to do a mechanical rename:
- `ExtractiveRefiner` → `ExtractiveRefinement`
- `AbstractiveRefiner` → `AbstractiveRefinement`
- `BaseChunkRefiner` → `BaseChunkRefinement`

Key files:
- `src/rfnry_rag/retrieval/server.py` and `RetrievalConfig.refiner` field type
- `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`
- `src/rfnry_rag/retrieval/__init__.py` (re-exports + `__all__`)
- Tests for the refinement module

**Step 4: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename Refiner → Refinement (gerundify naming)

ExtractiveRefiner → ExtractiveRefinement, AbstractiveRefiner → AbstractiveRefinement,
BaseChunkRefiner → BaseChunkRefinement. Matches Embeddings/Vision/Reranking noun-form."
```

---

### Task 8: Rewriter → Rewriting

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/search/rewriting/{hyde,multi_query,step_back,base}.py`
- Modify: all call sites

**Step 1: Grep references**

Pattern: `HyDeRewriter|MultiQueryRewriter|StepBackRewriter|BaseQueryRewriter|QueryRewriter`

**Step 2: Rename classes**

- `hyde.py`: `HyDeRewriter` → `HyDeRewriting`
- `multi_query.py`: `MultiQueryRewriter` → `MultiQueryRewriting`
- `step_back.py`: `StepBackRewriter` → `StepBackRewriting`
- `base.py`: `BaseQueryRewriter` → `BaseQueryRewriting`

**Step 3: Replace across codebase**

Mechanical rename (Edit `replace_all` per file). Same token list as step 1.

Key files:
- `src/rfnry_rag/retrieval/server.py` — `RetrievalConfig.query_rewriter` field type annotation
- `src/rfnry_rag/retrieval/modules/retrieval/search/service.py`
- `src/rfnry_rag/retrieval/__init__.py`
- Tests

**Step 4: Run tests + typecheck**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename Rewriter → Rewriting

HyDeRewriter/MultiQueryRewriter/StepBackRewriter → *Rewriting.
BaseQueryRewriter → BaseQueryRewriting."
```

---

### Task 9: Judge → Judgment

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/evaluation/metrics.py` — `LLMJudge` → `LLMJudgment`
- Modify: `src/rfnry_rag/retrieval/modules/retrieval/judging.py` — `LLMRetrievalJudge` → `RetrievalJudgment`, `BaseRetrievalJudge` → `BaseRetrievalJudgment`
- Modify: all call sites and re-exports

**Step 1: Grep references**

Pattern: `LLMJudge|LLMRetrievalJudge|BaseRetrievalJudge`

**Step 2: Rename classes**

- `metrics.py`: `class LLMJudge` → `class LLMJudgment`
- `judging.py`: `class LLMRetrievalJudge` → `class RetrievalJudgment`, `class BaseRetrievalJudge` → `class BaseRetrievalJudgment`

**Step 3: Replace call sites**

- `src/rfnry_rag/retrieval/__init__.py` — update re-exports + `__all__`
- `src/rfnry_rag/retrieval/server.py` — field types referencing these
- Tests

**Step 4: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename Judge → Judgment

LLMJudge → LLMJudgment (metric; LLM prefix retained alongside ExactMatch/F1Score).
LLMRetrievalJudge → RetrievalJudgment (LLM prefix dropped, only one implementation).
BaseRetrievalJudge → BaseRetrievalJudgment."
```

---

## Phase 5: BAML cleanup

### Task 10: Remove `env.OPENAI_API_KEY` default from `clients.baml`

**Files:**
- Modify: `src/rfnry_rag/retrieval/baml/baml_src/clients.baml`
- Modify: `src/rfnry_rag/reasoning/baml/baml_src/clients.baml`
- Regenerate: `src/rfnry_rag/retrieval/baml/baml_client/**`
- Regenerate: `src/rfnry_rag/reasoning/baml/baml_client/**`

**Step 1: Inspect the current `clients.baml`**

Use Read on both files to find the `api_key env.OPENAI_API_KEY` default. Understand the surrounding client definition.

**Step 2: Remove the env-fallback line**

In both files, delete (or comment out) the `api_key env.OPENAI_API_KEY` line. The BAML client definition should not specify a default `api_key` at all — keys now arrive exclusively via `ClientRegistry.add_llm_client(..., options={"api_key": ...})` from `build_registry`.

**Step 3: Regenerate BAML clients**

```bash
uv run poe baml:generate:retrieval
uv run poe baml:generate:reasoning
```

Expected: success. If BAML complains that the client needs a default `api_key`, we'll need to restructure the client definition — see `boundary/baml` docs.

**Step 4: Run tests**

```bash
uv run poe test 2>&1 | tail -30
```

Expected: tests that exercise BAML should still pass because `build_registry` passes `api_key` explicitly.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove env.OPENAI_API_KEY fallback from clients.baml

BAML no longer silently reads vendor env vars. API keys arrive exclusively via
ClientRegistry.add_llm_client options from build_registry(LanguageModelClient).
Regenerated baml_client for both retrieval and reasoning."
```

---

## Phase 6: Env var cleanup

### Task 11: Remove `X64RAG_PROVIDER` / `X64RAG_MODEL` / `X64RAG_API_KEY`

**Files:**
- Modify: `src/rfnry_rag/reasoning/cli/config.py`
- Delete tests: `src/rfnry_rag/reasoning/tests/test_cli_config.py::test_env_var_overrides_provider`, `::test_env_var_overrides_api_key`
- Modify: `.env.example`
- Modify: `src/rfnry_rag/reasoning/README.md`
- Modify: `CLAUDE.md`

**Step 1: Edit `src/rfnry_rag/reasoning/cli/config.py`**

Delete the env-override lookups (`reasoning/cli/config.py:45,48,66`):
- Remove `api_key_override = os.environ.get("X64RAG_API_KEY")`
- Remove `model_override = os.environ.get("X64RAG_MODEL")`
- Remove `provider_override = os.environ.get("X64RAG_PROVIDER")`
- Simplify `_build_lm_client_config` and `build_lm_config` to read exclusively from the TOML dict + vendor env keys for api_key.

The resulting `_build_lm_client_config` should look like:

```python
def _build_lm_client_config(cfg: dict[str, Any]) -> LanguageModelProvider:
    provider = cfg.get("provider")
    if not provider:
        raise ConfigError("[language_model] requires 'provider' (anthropic or openai)")

    env_var = _LM_API_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown language model provider: {provider!r}. Supported: {', '.join(_LM_API_KEYS)}")

    api_key = _get_api_key(env_var, provider)
    model = cfg.get("model", _LM_DEFAULTS[provider])

    return LanguageModelProvider(provider=provider, model=model, api_key=api_key)
```

Note the return type is now `LanguageModelProvider` (the new name from Task 1).

Also update `build_lm_config()` which constructs a `LanguageModelClient` — move `max_tokens` and `temperature` from the provider cfg into the client constructor.

**Step 2: Delete removed-env-var tests**

In `src/rfnry_rag/reasoning/tests/test_cli_config.py`, delete:
- `test_env_var_overrides_provider` (around line 89)
- `test_env_var_overrides_api_key` (around line 94)

**Step 3: Update `.env.example`**

Read the file, then Edit to remove:
```
X64RAG_PROVIDER=
X64RAG_MODEL=
X64RAG_API_KEY=
```

Also update `BAML_LOG=warn` → `X64RAG_BAML_LOG=warn` (Task 12 will wire the internal mapping; we set the new name here).

**Step 4: Update `reasoning/README.md` and `CLAUDE.md`**

Find and delete the `X64RAG_PROVIDER` / `X64RAG_MODEL` / `X64RAG_API_KEY` references. Update env var blocks to show the new three-layer split from the design doc.

**Step 5: Run tests**

```bash
uv run poe test 2>&1 | tail -30
```

Expected: green (2 tests removed; the rest still pass).

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove X64RAG_PROVIDER/MODEL/API_KEY CLI env overrides

These were reasoning-CLI-only escape hatches that overrode values already in
config.toml. Remove them — users edit config.toml directly. Deletes two tests
that exercised the removed behavior. Updates .env.example and READMEs."
```

---

### Task 12: Add `X64RAG_BAML_LOG` → `BAML_LOG` internal wiring

**Files:**
- Modify: `src/rfnry_rag/common/startup.py` OR `src/rfnry_rag/common/logging.py`

**Step 1: Read current startup/logging setup**

Use Read on `src/rfnry_rag/common/startup.py` to see where BAML initialization happens.

**Step 2: Add the env var mapping**

The cleanest place is `x64rag/common/logging.py` (it already reads `X64RAG_LOG_*`). Add:

```python
# src/rfnry_rag/common/logging.py (bottom of file or in get_logger)
import os
import logging

_BAML_LOG_ENV = "BAML_LOG"
_X64RAG_BAML_LOG_ENV = "X64RAG_BAML_LOG"


def _propagate_baml_log_env() -> None:
    """Wire X64RAG_BAML_LOG to BAML_LOG (what BAML actually reads) so users have
    one namespaced env var instead of touching BAML's internal name directly."""
    user_value = os.getenv(_X64RAG_BAML_LOG_ENV)
    if user_value and not os.getenv(_BAML_LOG_ENV):
        os.environ[_BAML_LOG_ENV] = user_value


def get_logger(name: str) -> logging.Logger:
    _propagate_baml_log_env()
    # ... rest of existing function
```

This runs on the first `get_logger()` call — early enough, before BAML reads its env.

Alternative placement: in `common/startup.py:check_baml()`, which is called during SDK init. Pick whichever is simpler and documented in the design doc.

**Step 3: Add a test**

```python
# src/rfnry_rag/common/tests/test_baml_log_env.py  (or near existing logging tests)
import os

def test_x64rag_baml_log_propagates_to_baml_log(monkeypatch):
    from x64rag.common.logging import _propagate_baml_log_env
    monkeypatch.delenv("BAML_LOG", raising=False)
    monkeypatch.setenv("X64RAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "debug"


def test_existing_baml_log_is_not_overridden(monkeypatch):
    from x64rag.common.logging import _propagate_baml_log_env
    monkeypatch.setenv("BAML_LOG", "info")
    monkeypatch.setenv("X64RAG_BAML_LOG", "debug")
    _propagate_baml_log_env()
    assert os.environ["BAML_LOG"] == "info"  # explicit BAML_LOG wins
```

**Step 4: Run the new tests**

```bash
uv run pytest src/rfnry_rag/common/tests/test_baml_log_env.py -v
```

**Step 5: Run all tests**

```bash
uv run poe test 2>&1 | tail -30
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add X64RAG_BAML_LOG env var, propagate to BAML_LOG internally

Users set X64RAG_BAML_LOG under the x64rag namespace; the SDK propagates it to
BAML_LOG during logger init. Explicit BAML_LOG still wins if set."
```

---

## Phase 7: Public API trim + docstrings

### Task 13: Trim `src/rfnry_rag/retrieval/__init__.py`

**Files:**
- Modify: `src/rfnry_rag/retrieval/__init__.py`

**Step 1: Read current file to see all imports**

Use Read on the file. Note every name currently in `__all__`.

**Step 2: Rewrite per the strict public API in the design doc (§4.1)**

Open the design doc section §4.1 and use its list verbatim. Replace the existing `__init__.py` contents:

- Keep: `RagServer`, `RagServerConfig`, `PersistenceConfig`, `IngestionConfig`, `RetrievalConfig`, `GenerationConfig`, `TreeIndexingConfig`, `TreeSearchConfig`
- Keep: `Embeddings`, `Vision`, `Reranking`, `FastEmbedSparseEmbeddings`
- Keep: All stores
- Keep: All ingestion/retrieval method classes
- Keep: `HyDeRewriting`, `MultiQueryRewriting`, `StepBackRewriting`, `ExtractiveRefinement`, `AbstractiveRefinement`, `RetrievalJudgment`
- Keep: All result models (`QueryResult`, `RetrievedChunk`, etc.)
- Keep: Evaluation metrics (`ExactMatch`, `F1Score`, `LLMJudgment`, `RetrievalPrecision`, `RetrievalRecall`)
- Keep: All errors
- Keep: `LanguageModelProvider`, `LanguageModelClient` (from common)

- **Remove from `__all__` AND from imports:** `BaseEmbeddings`, `BaseSparseEmbeddings`, `BaseVectorStore`, `BaseDocumentStore`, `BaseGraphStore`, `BaseIngestionMethod`, `BaseRetrievalMethod`, `BaseChunkRefinement`, `BaseQueryRewriting`, `BaseRetrievalJudgment`
- **Remove:** `MethodNamespace`
- **Remove:** `PageAnalysis`, `DocumentSynthesis`, `DiscoveredEntity`

**Step 3: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

If tests fail because they imported a now-private name from `x64rag.retrieval` top-level, fix them by importing from the module path directly (e.g., `from x64rag.retrieval.modules.ingestion.base import BaseIngestionMethod` instead of `from x64rag.retrieval import BaseIngestionMethod`). `Base*` types are still importable via their modules — we just don't expose them at the top-level.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(retrieval): trim __init__.py to strict public API

Remove Base* protocols, MethodNamespace, and vision intermediate models from
the top-level surface. Users who need plugin-author types can still import
from module paths; they just aren't the default public API."
```

---

### Task 14: Trim `src/rfnry_rag/reasoning/__init__.py` + fix docstring

**Files:**
- Modify: `src/rfnry_rag/reasoning/__init__.py`

**Step 1: Rewrite per §4.2 of the design doc**

Replace the first-line docstring:

```python
"""Reasoning services — analysis, classification, clustering, compliance, evaluation, pipelines."""
```

Remove from `__all__` / imports:
- `BaseEmbeddings`, `BaseVectorStore` / `BaseSemanticIndex`

Keep everything else per §4.2.

**Step 2: Update `__all__` to use `ReasoningError` instead of `AceError`**

Already done in Task 3 — verify.

**Step 3: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(reasoning): trim __init__.py, fix docstring

- Drop BaseEmbeddings/BaseSemanticIndex from public API
- Docstring: 'RAC — Reasoning-Augmented Classification' → 'Reasoning services —
  analysis, classification, clustering, compliance, evaluation, pipelines'"
```

---

### Task 15: Trim top-level `x64rag/__init__.py` + fix docstring

**Files:**
- Modify: `src/rfnry_rag/__init__.py`

**Step 1: Rewrite**

```python
"""x64rag — Retrieval-Augmented Generation + Reasoning services SDK."""

from importlib.metadata import version

__version__ = version("x64rag")

from x64rag.common.errors import ConfigurationError as ConfigurationError
from x64rag.common.errors import X64RagError as X64RagError
from x64rag.common.language_model import LanguageModelClient as LanguageModelClient
from x64rag.common.language_model import LanguageModelProvider as LanguageModelProvider
from x64rag.reasoning import *  # re-exports strict reasoning public API
from x64rag.retrieval import *  # re-exports strict retrieval public API
```

**Step 2: Run tests**

```bash
uv run poe test 2>&1 | tail -30
uv run poe typecheck 2>&1 | tail -20
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: top-level x64rag __init__.py docstring + new LM primitive re-exports"
```

---

## Phase 8: Docs & examples

### Task 16: Update retrieval examples

**Files:**
- Modify: `examples/retrieval/sdk/minimal.py`
- Modify: `examples/retrieval/sdk/basic.py`
- Modify: `examples/retrieval/sdk/hybrid_search.py`
- Modify: `examples/retrieval/sdk/modular_pipeline.py`

**Step 1:** For each file:
- Replace `OpenAIEmbeddings(api_key=..., model=...)` → `Embeddings(LanguageModelProvider(provider="openai", model=..., api_key=...))`
- Replace `LanguageModelConfig(client=LanguageModelClientConfig(...))` → `LanguageModelClient(provider=LanguageModelProvider(...))`
- If the example uses `CohereReranking`/`VoyageReranking`/`AnthropicVision`/`OpenAIVision` → replace with the `Reranking`/`Vision` facade.
- Update imports at the top of each file.

**Step 2:** Run the example imports to sanity-check (no actual API calls needed):

```bash
uv run python -c "from examples.retrieval.sdk.minimal import *; print('ok')"
uv run python -c "from examples.retrieval.sdk.basic import *; print('ok')"
uv run python -c "from examples.retrieval.sdk.hybrid_search import *; print('ok')"
uv run python -c "from examples.retrieval.sdk.modular_pipeline import *; print('ok')"
```

Expected: each prints `ok` (or at least, no `ImportError`; any runtime `ConnectionError` for missing services is fine — we only care about import success).

**Step 3:** Commit.

```bash
git add examples/retrieval/sdk/
git commit -m "docs: update retrieval examples to new SDK surface"
```

---

### Task 17: Update reasoning examples

**Files:**
- Modify: `examples/reasoning/sdk/minimal.py`
- Modify: `examples/reasoning/sdk/basic.py`
- Modify: `examples/reasoning/sdk/chat_support.py`
- Modify: `examples/reasoning/sdk/email_system.py`
- Modify: `examples/reasoning/sdk/quality_assurance.py`

**Step 1:** Same substitutions as Task 16.

**Step 2:** Same import sanity check.

**Step 3:** Commit.

```bash
git add examples/reasoning/sdk/
git commit -m "docs: update reasoning examples to new SDK surface"
```

---

### Task 18: Update top-level `README.md`

**Files:**
- Modify: `README.md`

**Step 1:** Update the first code block (retrieval quickstart):
- `OpenAIEmbeddings(api_key="...", model="text-embedding-3-small")` → `Embeddings(LanguageModelProvider(provider="openai", model="text-embedding-3-small", api_key="..."))`

**Step 2:** Update the second code block (reasoning quickstart):
- `LanguageModelClientConfig` → `LanguageModelProvider`
- `LanguageModelConfig` → `LanguageModelClient`
- Note: the `client=` kwarg becomes `provider=`

**Step 3:** Rewrite the "Env Variables" section into three blocks per the design doc §2.4:

```markdown
## SDK Env Variables

```bash
X64RAG_LOG_ENABLED=false    # true / false
X64RAG_LOG_LEVEL=INFO       # DEBUG, INFO, WARNING, ERROR
X64RAG_BAML_LOG=warn        # info, warn, debug — BAML runtime log level
```

## CLI Env Variables

Used only by the `x64rag retrieval` / `x64rag reasoning` command-line tools.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=
```
```

**Step 4:** Commit.

```bash
git add README.md
git commit -m "docs: update top-level README to new SDK surface and 3-layer env vars"
```

---

### Task 19: Update `src/rfnry_rag/retrieval/README.md` + `src/rfnry_rag/reasoning/README.md`

**Files:**
- Modify: `src/rfnry_rag/retrieval/README.md`
- Modify: `src/rfnry_rag/reasoning/README.md`

**Step 1:** Replace every code example in both files with the new pattern (same substitutions as above).

**Step 2:** Verify there are no stragglers:

Use Grep with pattern `OpenAIEmbeddings|AnthropicVision|OpenAIVision|CohereReranking|VoyageReranking|LanguageModelConfig|LanguageModelClientConfig|AceError|X64RAG_PROVIDER|X64RAG_API_KEY|X64RAG_MODEL` across these two files — zero hits expected.

**Step 3:** Commit.

```bash
git add src/rfnry_rag/retrieval/README.md src/rfnry_rag/reasoning/README.md
git commit -m "docs: update retrieval and reasoning READMEs to new SDK surface"
```

---

### Task 20: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Step 1:** Update the error hierarchy section (already partially done in Task 3; verify `AceError` → `ReasoningError`).

**Step 2:** Update the "Environment Variables" section — drop `X64RAG_PROVIDER` / `X64RAG_MODEL` / `X64RAG_API_KEY` line, add `X64RAG_BAML_LOG` alongside the log vars.

**Step 3:** Update "Key Patterns" section to reference `LanguageModelProvider` / `LanguageModelClient` instead of `LanguageModelConfig` / `LanguageModelClientConfig`.

**Step 4:** Update the architecture text where it mentions embeddings/vision/reranking — if it referenced specific provider class names, replace with the facade names.

**Step 5:** Grep for stragglers:

Use Grep pattern `LanguageModelConfig|LanguageModelClientConfig|AceError|X64RAG_PROVIDER|X64RAG_API_KEY|X64RAG_MODEL|OpenAIEmbeddings|AnthropicVision|OpenAIVision|CohereReranking|VoyageReranking` in `CLAUDE.md`. Zero hits expected.

**Step 6:** Commit.

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for SDK cleanup renames and env var layering"
```

---

## Phase 9: Final verification

### Task 21: Full verification

**Step 1:** Run the complete gate.

```bash
uv run poe format
uv run poe check
uv run poe typecheck
uv run poe test
```

Expected: format clean, check clean, typecheck clean, tests all pass. Test count should equal **baseline** from Task 0 minus 2 (deleted env-override tests) plus N (new facade tests: ~5 embeddings + ~3 vision + ~4 reranking + ~2 BAML env = ~14 added), so final ≈ baseline + 12.

**Step 2:** Smoke-test the public API from a fresh Python REPL.

```bash
uv run python -c "
from x64rag import (
    RagServer, RagServerConfig, PersistenceConfig, IngestionConfig, RetrievalConfig,
    GenerationConfig,
    LanguageModelProvider, LanguageModelClient,
    Embeddings, Vision, Reranking,
    QdrantVectorStore, Neo4jGraphStore, SQLAlchemyMetadataStore,
    PostgresDocumentStore, FilesystemDocumentStore,
    HyDeRewriting, MultiQueryRewriting, StepBackRewriting,
    ExtractiveRefinement, AbstractiveRefinement,
    RetrievalJudgment,
    ExactMatch, F1Score, LLMJudgment, RetrievalPrecision, RetrievalRecall,
    QueryResult, RetrievedChunk, StepResult, StreamEvent,
    RagError, IngestionError, RetrievalError, GenerationError,
    ConfigurationError,
)
from x64rag import (
    AnalysisService, ClassificationService, ClusteringService,
    ComplianceService, EvaluationService, Pipeline,
    AnalysisConfig, ClassificationConfig, ClusteringConfig,
    ComplianceConfig, EvaluationConfig,
    AnalysisResult, Classification, Cluster, ComplianceResult, EvaluationResult,
    ReasoningError, AnalysisError, ClassificationError, ClusteringError,
    ComplianceError, EvaluationError,
)
print('All imports OK')
"
```

Expected: `All imports OK`. If anything fails, fix the missing re-export.

**Step 3:** Smoke-test that privatized types are NOT importable from the top level.

```bash
uv run python -c "
import x64rag
for name in ['OpenAIEmbeddings', 'VoyageEmbeddings', 'CohereEmbeddings',
             'AnthropicVision', 'OpenAIVision',
             'CohereReranking', 'VoyageReranking', 'LLMReranking',
             'ExtractiveRefiner', 'AbstractiveRefiner',
             'HyDeRewriter', 'MultiQueryRewriter', 'StepBackRewriter',
             'LLMJudge', 'LLMRetrievalJudge',
             'BaseIngestionMethod', 'BaseRetrievalMethod',
             'BaseEmbeddings', 'BaseVectorStore',
             'AceError', 'MethodNamespace',
             'LanguageModelConfig', 'LanguageModelClientConfig']:
    assert not hasattr(x64rag, name), f'{name} still publicly exposed'
print('Privatization OK')
"
```

Expected: `Privatization OK`.

**Step 4:** Verify `X64RAG_PROVIDER`/`MODEL`/`API_KEY` are gone.

Use Grep with pattern `X64RAG_PROVIDER|X64RAG_MODEL|X64RAG_API_KEY` across `src/`, `examples/`, `docs/`, `README.md`, `.env.example`, `CLAUDE.md` — zero hits expected.

**Step 5:** Verify `BAML_LOG` env reads now go through `X64RAG_BAML_LOG`.

Use Grep pattern `BAML_LOG` in `README.md` and `.env.example`. If present, it should only be in the "internal" comment or docs explaining the mapping — user-facing docs should reference `X64RAG_BAML_LOG`.

**Step 6:** Verify `env.OPENAI_API_KEY` is gone from BAML sources.

Use Grep pattern `env\.OPENAI_API_KEY` in `src/rfnry_rag/**/baml_src/*.baml` — zero hits expected.

**Step 7:** Clean up intermediate stashes, run `git log --oneline` to review all commits.

```bash
git log --oneline origin/main..HEAD
```

Expected: ~21 commits, one per task. Each commit message should be self-contained and explain the change.

**Step 8:** Done. No commit here — this is a verification step.

---

## Quick reference: commit messages

- `refactor: rename LanguageModelConfig/LanguageModelClientConfig → LanguageModelClient/LanguageModelProvider`
- `refactor: dedupe BaseEmbeddings and rename reasoning vector protocol`
- `refactor: rename AceError → ReasoningError`
- `refactor: collapse embeddings into single Embeddings(LanguageModelProvider) facade`
- `refactor: collapse vision into single Vision(LanguageModelProvider) facade`
- `refactor: unify reranking into single Reranking facade with type dispatch`
- `refactor: rename Refiner → Refinement (gerundify naming)`
- `refactor: rename Rewriter → Rewriting`
- `refactor: rename Judge → Judgment`
- `refactor: remove env.OPENAI_API_KEY fallback from clients.baml`
- `refactor: remove X64RAG_PROVIDER/MODEL/API_KEY CLI env overrides`
- `feat: add X64RAG_BAML_LOG env var, propagate to BAML_LOG internally`
- `refactor(retrieval): trim __init__.py to strict public API`
- `refactor(reasoning): trim __init__.py, fix docstring`
- `refactor: top-level x64rag __init__.py docstring + new LM primitive re-exports`
- `docs: update retrieval examples to new SDK surface`
- `docs: update reasoning examples to new SDK surface`
- `docs: update top-level README to new SDK surface and 3-layer env vars`
- `docs: update retrieval and reasoning READMEs to new SDK surface`
- `docs: update CLAUDE.md for SDK cleanup renames and env var layering`

---

## Notes on execution

- **Every task commits at the end.** If something goes wrong mid-task, `git status` should show a small, isolated diff.
- **Test after every task.** A single task that breaks the suite is easy to fix; five tasks of mixed changes are not.
- **Use `replace_all` on Edit for mechanical renames**, but verify with Grep afterwards. Avoid accidentally matching substrings in unrelated tokens (e.g., `LanguageModelConfig` appearing inside `OldLanguageModelConfigHelper`).
- **Do NOT skip the BAML regeneration step** in Task 10. Stale `baml_client/` files will cause confusing runtime errors.
- **If tests reference private-now types after Task 13-15**, fix by importing from the module path (`from x64rag.retrieval.modules.ingestion.base import BaseIngestionMethod`), not by re-adding to `__init__.py`.
