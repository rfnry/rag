from __future__ import annotations

from typing import Any, Literal

from rfnry_rag.observability.context import current_obs

Level = Literal["debug", "info", "warn", "error"]


async def record_skip(
    notes: list[str] | None,
    *,
    step: str,
    level: Level,
    reason: str,
    page_number: int | None = None,
) -> None:
    note = f"{step}:{level}:{reason}"
    if notes is not None:
        notes.append(note)
    obs = current_obs()
    if obs is None:
        return
    kind = "vision.page.skipped" if page_number is not None else "enrichment.skipped"
    ctx: dict[str, Any] = {"step": step, "reason": reason}
    if page_number is not None:
        ctx["page_number"] = page_number
    await obs.emit(kind, note, level=level, context=ctx)
