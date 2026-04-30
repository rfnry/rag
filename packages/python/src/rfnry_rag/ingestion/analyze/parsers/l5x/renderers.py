from rfnry_rag.ingestion.analyze.parsers.l5x.models import (
    AoiEntity,
    ControllerEntity,
    ModuleEntity,
    RoutineEntity,
    TagEntity,
    UdtEntity,
)


def render_controller(ctrl: ControllerEntity) -> str:
    lines = [
        f"CONTROLLER: {ctrl.name}",
        f"Processor: {ctrl.processor_type}",
        f"Software Revision: {ctrl.software_revision}",
    ]
    if ctrl.description:
        lines.append(f"Description: {ctrl.description}")
    lines.append(f"Programs: {ctrl.program_count}, Tags: {ctrl.tag_count}, Modules: {ctrl.module_count}")
    if ctrl.tasks:
        lines.append(f"Tasks: {', '.join(ctrl.tasks)}")
    return "\n".join(lines)


def render_tag_group(scope: str, tags: list[TagEntity]) -> str:
    lines = [f"TAG GROUP: [{scope}] ({len(tags)} tags)"]
    for tag in tags:
        parts = [f"  {tag.name} : {tag.data_type}"]
        if tag.tag_type != "Base":
            parts.append(f"(type={tag.tag_type})")
        if tag.alias_for:
            parts.append(f"(alias for {tag.alias_for})")
        if tag.description:
            parts.append(f"-- {tag.description}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def render_routine(routine: RoutineEntity) -> str:
    lines = [f"ROUTINE: {routine.program}/{routine.name} (type={routine.routine_type})"]
    if routine.description:
        lines.append(f"Description: {routine.description}")

    if routine.rungs:
        lines.append(f"Rungs: {len(routine.rungs)}")
        for rung in routine.rungs:
            fault_marker = " [FAULT/ALARM]" if rung.is_fault_related else ""
            lines.append(f"  Rung {rung.number}{fault_marker}:")
            if rung.comment:
                lines.append(f"    Comment: {rung.comment}")
            lines.append(f"    Logic: {rung.logic_text}")
            if rung.references:
                lines.append(f"    References: {', '.join(rung.references[:20])}")

    if routine.st_lines:
        lines.append(f"Structured Text: {len(routine.st_lines)} lines")
        for line in routine.st_lines[:50]:
            lines.append(f"  {line}")

    return "\n".join(lines)


def render_udt(udt: UdtEntity) -> str:
    lines = [f"USER DEFINED TYPE: {udt.name}"]
    if udt.description:
        lines.append(f"Description: {udt.description}")
    if udt.members:
        lines.append(f"Members ({len(udt.members)}):")
        for member in udt.members:
            desc = f" -- {member.description}" if member.description else ""
            lines.append(f"  {member.name} : {member.data_type}{desc}")
    return "\n".join(lines)


def render_aoi(aoi: AoiEntity) -> str:
    lines = [f"ADD-ON INSTRUCTION: {aoi.name}"]
    if aoi.description:
        lines.append(f"Description: {aoi.description}")
    if aoi.revision:
        lines.append(f"Revision: {aoi.revision}")

    if aoi.parameters:
        lines.append(f"Parameters ({len(aoi.parameters)}):")
        for param in aoi.parameters:
            vis = "" if param.visible else " [hidden]"
            desc = f" -- {param.description}" if param.description else ""
            lines.append(f"  {param.name} : {param.data_type} ({param.usage}){vis}{desc}")

    if aoi.routines:
        lines.append(f"Routines ({len(aoi.routines)}):")
        for routine in aoi.routines:
            lines.append(f"  {routine.name} (type={routine.routine_type}, rungs={len(routine.rungs)})")

    return "\n".join(lines)


def render_module_group(parent: str, modules: list[ModuleEntity]) -> str:
    lines = [f"MODULE GROUP: [{parent}] ({len(modules)} modules)"]
    for mod in modules:
        lines.append(f"  {mod.name}: {mod.catalog_number}")
        for port in mod.ports:
            addr = f" address={port.address}" if port.address else ""
            lines.append(f"    Port {port.port_id}: {port.port_type}{addr}")
    return "\n".join(lines)
