# Comprehensive Review — Critical Fixes (P0)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Resolve the 5 P0 findings from the 2026-04-23 comprehensive review of the `rfnry-rag` SDK. Each of these either breaks documented presets, enables exploitation by a malicious input, causes silent data loss, or prevents clean shutdown on partial init. Every task must be done before any production rollout.

**Architecture:** One task per finding. Tasks are ordered by blast radius (public-API breakage first, then security, then correctness). Each task is TDD-first, touches 1–3 files, and ships as an independent commit.

**Tech stack:** Python 3.12, pytest (asyncio_mode=auto), Ruff, MyPy. Paths are relative to `packages/python/` unless absolute.

**Preconditions:**
- Branch is clean (`git status` empty)
- `uv run poe test` is green on main
- Read `packages/python/CLAUDE.md` for commands and conventions

---

## Task 1 — Fix the grounding-threshold guard that breaks every retrieval-only preset

**Finding P0.1.** `retrieval/server.py:541` raises `ConfigurationError("generation provider required for grounding gate")` whenever `gen.grounding_threshold > 0 and not gen.lm_client`. Default `grounding_threshold=0.5`, so every retrieval-only config (`RagEngine.vector_only(...)`, `.document_only(...)`, `.hybrid(...)` without `lm_client`) raises on `initialize()`. Preset tests only call `_validate_config()` (never `initialize()`), so this is uncaught.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:541`
- Modify: `src/rfnry_rag/retrieval/tests/test_engine_presets.py`

**Step 1 — Write the failing test (red):**

```python
# src/rfnry_rag/retrieval/tests/test_engine_presets.py — append

from unittest.mock import AsyncMock, MagicMock
import pytest
from rfnry_rag.retrieval.server import RagEngine

@pytest.mark.asyncio
async def test_vector_only_preset_initializes_without_generation() -> None:
    vector_store = MagicMock()
    vector_store.initialize = AsyncMock()
    vector_store.collections = ["knowledge"]
    embeddings = MagicMock()
    embeddings.model = "test"
    embeddings.embedding_dimension = AsyncMock(return_value=1536)

    config = RagEngine.vector_only(vector_store=vector_store, embeddings=embeddings)
    engine = RagEngine(config)

    # Must NOT raise ConfigurationError — grounding default must not require an LM
    await engine.initialize()
    assert engine._initialized
```

Run: `uv run pytest src/rfnry_rag/retrieval/tests/test_engine_presets.py::test_vector_only_preset_initializes_without_generation -v`
Expected: **FAIL** with `ConfigurationError: generation provider required for grounding gate`.

**Step 2 — Fix the guard (green):**

```python
# src/rfnry_rag/retrieval/server.py:541
# BEFORE:
if gen.grounding_threshold > 0 and not gen.lm_client:
    raise ConfigurationError("generation provider required for grounding gate")

# AFTER:
if gen.grounding_enabled and not gen.lm_client:
    raise ConfigurationError("grounding_enabled requires generation.lm_client")
```

`GenerationConfig.__post_init__` at `server.py:122-130` already enforces that `relevance_gate_enabled` and `guiding_enabled` require their dependencies, so this stays internally consistent.

**Step 3 — Verify:**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_engine_presets.py -v
uv run poe check && uv run poe typecheck
```

Expected: all pass.

**Step 4 — Commit:**

```bash
git commit -m "fix: gate grounding guard on grounding_enabled, not threshold

Default grounding_threshold=0.5 meant every retrieval-only RagEngine
preset raised ConfigurationError on initialize(). Preset tests never
called initialize(), so this was uncaught. Gate the check on
grounding_enabled to match the rest of the GenerationConfig validators."
```

---

## Task 2 — Close the XXE hole in XML and L5X parsers

**Finding P0.2.** `etree.parse()` / `etree.iterparse()` in `ingestion/analyze/parsers/xml.py` and `parsers/l5x/parser.py` are called with no parser argument. lxml's default parser resolves `file://` external entities, which lets a malicious XML or L5X file exfiltrate local files (`/etc/passwd`, `/proc/self/environ` — API keys — SSH keys) into parsed output that then flows into stores and LLM prompts.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/parsers/xml.py:16,25`
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/analyze/parsers/l5x/parser.py:37`
- Test: `src/rfnry_rag/retrieval/tests/test_xml_xxe_hardening.py` (new)

**Step 1 — Write the failing test (red):**

```python
# src/rfnry_rag/retrieval/tests/test_xml_xxe_hardening.py
import textwrap
from pathlib import Path

import pytest

from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.xml import parse_xml
from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.l5x.parser import parse_l5x


def _write_xxe_file(tmp_path: Path, root: str, secret: Path) -> Path:
    payload = textwrap.dedent(
        f"""<?xml version="1.0"?>
        <!DOCTYPE r [
          <!ENTITY xxe SYSTEM "file://{secret}">
        ]>
        <{root}><val>&xxe;</val></{root}>
        """
    )
    p = tmp_path / f"{root}.xml"
    p.write_text(payload)
    return p


def test_xml_parser_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    xml_file = _write_xxe_file(tmp_path, "root", secret)

    result = parse_xml(xml_file)

    serialized = str(result)
    assert "TOP-SECRET-CONTENT" not in serialized


def test_l5x_parser_does_not_resolve_external_entities(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    l5x_file = _write_xxe_file(tmp_path, "RSLogix5000Content", secret)
    l5x_file = l5x_file.rename(l5x_file.with_suffix(".l5x"))

    # parse_l5x may raise on malformed L5X structure — that's fine.
    # What we assert is that when it does run, the entity isn't expanded.
    try:
        result = parse_l5x(l5x_file)
        assert "TOP-SECRET-CONTENT" not in str(result)
    except Exception as exc:
        assert "TOP-SECRET-CONTENT" not in str(exc)
```

Run: `uv run pytest src/rfnry_rag/retrieval/tests/test_xml_xxe_hardening.py -v`
Expected: **FAIL** — the default lxml parser expands the `file://` entity.

**Step 2 — Fix the parsers (green):**

Define a shared hardened parser and pass it to every `etree.parse` / `etree.iterparse` call.

```python
# src/rfnry_rag/retrieval/modules/ingestion/analyze/parsers/xml.py
# Add at module top, near the lxml import:

_SAFE_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    huge_tree=False,
)

# line 16 (iterparse call) — add parser=:
context = etree.iterparse(str(file_path), events=("start", "end"), parser=_SAFE_PARSER)

# line 25 (parse call) — add parser:
tree = etree.parse(str(file_path), _SAFE_PARSER)
```

```python
# src/rfnry_rag/retrieval/modules/ingestion/analyze/parsers/l5x/parser.py
# Near the top:

_SAFE_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    huge_tree=False,
)

# line 37 — add parser:
tree = etree.parse(str(file_path), _SAFE_PARSER)
```

If the L5X parser uses `etree.fromstring(...)` anywhere else, apply `parser=_SAFE_PARSER` to it too. Search: `grep -n "etree.parse\|etree.iterparse\|etree.fromstring\|etree.XML" src/rfnry_rag/retrieval/modules/ingestion/analyze/parsers/`.

**Step 3 — Verify:**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_xml_xxe_hardening.py -v
uv run pytest src/rfnry_rag/retrieval/tests/test_analyzed_ingestion.py -v  # no regressions
uv run poe check && uv run poe typecheck
```

**Step 4 — Commit:**

```bash
git commit -m "security: harden XML/L5X parsers against XXE

Bare etree.parse/iterparse resolves file:// external entities by
default. Malicious XML/L5X files could exfiltrate local files
(credentials, SSH keys) through entity expansion into parsed output.
Use an explicit parser with resolve_entities=False, no_network=True,
load_dtd=False for every XML read."
```

---

## Task 3 — Stop committing ingestion sources when required methods fail

**Finding P0.3.** `ingestion/chunk/service.py:109-125` catches all exceptions from `_dispatch_methods` as `logger.warning(...)`. Then `service.py:215` writes the source row with `chunk_count > 0` regardless. If `VectorIngestion.upsert` fails (Qdrant down, network partition), the source looks successful to the user but has **zero vectors in the index**. Silent data loss.

**Files:**
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/base.py` (add `required` flag)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/methods/vector.py` (`required = True`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/methods/document.py` (`required = True`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/methods/graph.py` (`required = False`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/methods/tree.py` (`required = False`)
- Modify: `src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py:109-125`
- Test: `src/rfnry_rag/retrieval/tests/test_ingestion_required_methods.py` (new)

**Step 1 — Write the failing test:**

```python
# src/rfnry_rag/retrieval/tests/test_ingestion_required_methods.py
from unittest.mock import AsyncMock, MagicMock
import pytest

from rfnry_rag.retrieval.common.errors import IngestionError
from rfnry_rag.retrieval.modules.ingestion.chunk.service import IngestionService
from rfnry_rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker


def _make_method(name: str, *, required: bool, fails: bool) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.required = required
    if fails:
        m.ingest = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        m.ingest = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_required_method_failure_aborts_and_does_not_commit_source(tmp_path):
    # Required method fails → IngestionError raised, metadata NOT written
    metadata_store = MagicMock()
    metadata_store.save_source = AsyncMock()
    svc = IngestionService(
        chunker=SemanticChunker(chunk_size=100, chunk_overlap=10),
        ingestion_methods=[
            _make_method("vector", required=True, fails=True),
            _make_method("document", required=True, fails=False),
        ],
        embedding_model_name="test:model",
        source_type_weights=None,
        metadata_store=metadata_store,
        on_ingestion_complete=None,
        vision_parser=None,
        contextual_chunking=False,
    )
    fp = tmp_path / "a.txt"
    fp.write_text("hello world " * 50)

    with pytest.raises(IngestionError):
        await svc.ingest(file_path=fp)

    metadata_store.save_source.assert_not_called()


@pytest.mark.asyncio
async def test_optional_method_failure_is_logged_and_ingest_succeeds(tmp_path, caplog):
    metadata_store = MagicMock()
    metadata_store.save_source = AsyncMock()
    svc = IngestionService(
        chunker=SemanticChunker(chunk_size=100, chunk_overlap=10),
        ingestion_methods=[
            _make_method("vector", required=True, fails=False),
            _make_method("graph", required=False, fails=True),
        ],
        embedding_model_name="test:model",
        source_type_weights=None,
        metadata_store=metadata_store,
        on_ingestion_complete=None,
        vision_parser=None,
        contextual_chunking=False,
    )
    fp = tmp_path / "a.txt"
    fp.write_text("hello world " * 50)

    await svc.ingest(file_path=fp)

    metadata_store.save_source.assert_awaited_once()
```

Run. Expected: **FAIL** — both tests.

**Step 2 — Add `required` to the protocol:**

```python
# src/rfnry_rag/retrieval/modules/ingestion/base.py
@runtime_checkable
class BaseIngestionMethod(Protocol):
    name: str
    required: bool  # if True, failures abort ingestion (raise IngestionError)
    async def ingest(self, ...): ...
```

For each concrete method, set `required: bool = True` on `vector`/`document` and `required: bool = False` on `graph`/`tree`. Add as a class attribute so existing constructors don't need to change.

**Step 3 — Fix the dispatcher:**

```python
# src/rfnry_rag/retrieval/modules/ingestion/chunk/service.py
# Around line 109-125, replace the bare try/except with:

for method in self._ingestion_methods:
    try:
        await method.ingest(...)
    except Exception as exc:
        if getattr(method, "required", True):
            logger.exception("required ingestion method '%s' failed — aborting", method.name)
            raise IngestionError(
                f"required ingestion method '{method.name}' failed: {exc}"
            ) from exc
        logger.warning("optional ingestion method '%s' failed: %s", method.name, exc)
```

**Step 4 — Verify + commit:**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_ingestion_required_methods.py src/rfnry_rag/retrieval/tests/test_ingestion_service_methods.py -v
uv run poe check && uv run poe typecheck && uv run poe test
```

```bash
git commit -m "fix: abort ingestion on required-method failure instead of silently committing

Before: a failed vector upsert was caught as a warning and the source
row was still written with chunk_count > 0 — user sees success but
Qdrant has zero vectors. Now vector/document methods are marked
required=True; their failure raises IngestionError and skips the
metadata commit. graph/tree remain optional."
```

---

## Task 4 — Rename the custom `BaseException` that shadows the Python builtin

**Finding P0.4.** `common/errors.py:4` defines `class BaseException(Exception)` and `rfnry_rag/__init__.py:7` re-exports it at the top-level. Any user doing `from rfnry_rag import *` or `from rfnry_rag import BaseException` now has `except BaseException:` catch only SDK errors instead of the Python builtin.

**Files:**
- Modify: `src/rfnry_rag/common/errors.py`
- Modify: `src/rfnry_rag/__init__.py`
- Modify: `src/rfnry_rag/retrieval/common/errors.py` (uses `BaseException` as a base class for `RagError`)
- Modify: `src/rfnry_rag/reasoning/common/errors.py` (uses it for `ReasoningError`)
- Modify any `__all__` lists that mention `BaseException`
- Test: `src/rfnry_rag/retrieval/tests/test_public_api.py` (extend)

**Step 1 — Write the failing test:**

```python
# src/rfnry_rag/retrieval/tests/test_public_api.py — append
import rfnry_rag

def test_top_level_does_not_shadow_builtin_BaseException():
    # Importing rfnry_rag must not expose a name called "BaseException"
    # that shadows the Python builtin.
    assert "BaseException" not in vars(rfnry_rag), (
        "rfnry_rag exports a symbol named 'BaseException' — this shadows "
        "the Python builtin and silently narrows user except clauses."
    )
```

Run. Expected: **FAIL**.

**Step 2 — Rename the class:**

Pick `SdkBaseError` as the new name (consistent with `RagError`, `ReasoningError`, `ConfigurationError` — all `*Error`).

```python
# src/rfnry_rag/common/errors.py
class SdkBaseError(Exception):
    """Base class for all rfnry-rag SDK errors."""

class ConfigurationError(SdkBaseError):
    ...
```

Update all subclass bases:
- `RagError(SdkBaseError)` in `retrieval/common/errors.py`
- `ReasoningError(SdkBaseError)` in `reasoning/common/errors.py`

**Step 3 — Remove from top-level `__all__`:**

```python
# src/rfnry_rag/__init__.py
# Remove:  BaseException as BaseException,
# Do NOT re-export SdkBaseError either — users should catch the specific
# RagError / ReasoningError / ConfigurationError subclasses.
```

If any internal code does `from rfnry_rag.common.errors import BaseException`, update those imports. Search: `grep -rn "from rfnry_rag.common.errors import" src/`.

**Step 4 — Verify:**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_public_api.py -v
uv run poe test  # full suite — this rename touches a lot
uv run poe check && uv run poe typecheck
```

**Step 5 — Commit:**

```bash
git commit -m "refactor!: rename common BaseException to SdkBaseError

The custom class shadowed the Python builtin. Any user doing
'except BaseException:' after 'from rfnry_rag import BaseException'
silently caught only SDK errors. Renamed to SdkBaseError; kept as an
internal base class (not re-exported at top-level) since users
should catch specific RagError/ReasoningError/ConfigurationError."
```

**Note:** This is a breaking change. Add a note to CHANGELOG / release notes.

---

## Task 5 — Give `RagEngine.initialize()` a rollback path on partial failure

**Finding P0.5.** `retrieval/server.py:347-559` initializes stores sequentially with no try/except. If `graph_store.initialize()` raises after `metadata_store` and `document_store` already opened connections, those connections leak. `__aexit__` does not fire when `__aenter__` raises, so the context manager does not save us either.

**Files:**
- Modify: `src/rfnry_rag/retrieval/server.py:347-559`
- Test: `src/rfnry_rag/retrieval/tests/test_engine_init_rollback.py` (new)

**Step 1 — Write the failing test:**

```python
# src/rfnry_rag/retrieval/tests/test_engine_init_rollback.py
from unittest.mock import AsyncMock, MagicMock
import pytest

from rfnry_rag.retrieval.server import (
    IngestionConfig, PersistenceConfig, RagEngine, RagServerConfig,
)


@pytest.mark.asyncio
async def test_initialize_rolls_back_already_opened_stores_on_failure():
    metadata_store = MagicMock()
    metadata_store.initialize = AsyncMock()
    metadata_store.shutdown = AsyncMock()

    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    document_store.shutdown = AsyncMock()

    graph_store = MagicMock()
    graph_store.initialize = AsyncMock(side_effect=RuntimeError("graph failed"))
    graph_store.shutdown = AsyncMock()

    cfg = RagServerConfig(
        persistence=PersistenceConfig(
            metadata_store=metadata_store,
            document_store=document_store,
            graph_store=graph_store,
        ),
        ingestion=IngestionConfig(),
    )
    engine = RagEngine(cfg)

    with pytest.raises(RuntimeError, match="graph failed"):
        await engine.initialize()

    # Prior stores that opened must be shut down
    metadata_store.shutdown.assert_awaited_once()
    document_store.shutdown.assert_awaited_once()
    # graph_store.shutdown should NOT be called — it never opened
    graph_store.shutdown.assert_not_called()
    assert engine._initialized is False
```

Run. Expected: **FAIL**.

**Step 2 — Wrap `initialize()` body in try/except:**

```python
# src/rfnry_rag/retrieval/server.py
async def initialize(self) -> None:
    self._validate_config()
    try:
        await self._initialize_impl()
    except Exception:
        logger.exception("ragengine init failed — rolling back")
        await self.shutdown()
        raise

async def _initialize_impl(self) -> None:
    # ... existing body of initialize() from line 351 onward ...
```

`shutdown()` already individually guards each store teardown with try/except (server.py:568-586), so calling it on a partially-initialized engine is safe — it will no-op for stores that never opened, assuming their `shutdown()` is idempotent. Verify this assumption for each store; if any store raises when shut down before initialized, adjust by tracking which stores opened successfully.

**Step 3 — Verify + commit:**

```bash
uv run pytest src/rfnry_rag/retrieval/tests/test_engine_init_rollback.py -v
uv run poe test && uv run poe check && uv run poe typecheck
```

```bash
git commit -m "fix: roll back opened stores when RagEngine.initialize() fails partway

Before: if graph_store.initialize() raised, metadata_store and
document_store stayed open because __aexit__ does not fire when
__aenter__ raises. Wrap initialize() in try/except that calls
shutdown() on failure before re-raising."
```

---

## Verification — after all 5 tasks

```bash
uv run poe test          # 548+ tests — all green
uv run poe check         # ruff
uv run poe typecheck     # mypy
git log --oneline -5     # confirm 5 focused commits
```

Expected: clean. Create PR titled `fix: resolve P0 findings from 2026-04-23 review`.
