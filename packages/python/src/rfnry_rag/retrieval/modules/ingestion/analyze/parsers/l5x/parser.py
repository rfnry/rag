import re
from pathlib import Path

from lxml import etree

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.l5x.models import (
    AoiEntity,
    AoiParameter,
    ControllerEntity,
    ModuleEntity,
    ModulePort,
    PlcDocument,
    RoutineEntity,
    RungEntity,
    TagEntity,
    UdtEntity,
    UdtMember,
)
from rfnry_rag.retrieval.modules.ingestion.analyze.parsers.l5x.renderers import (
    render_aoi,
    render_controller,
    render_module_group,
    render_routine,
    render_tag_group,
    render_udt,
)

logger = get_logger("analyze/ingestion/analyze/l5x")

_FAULT_KEYWORDS = re.compile(r"fault|alarm|error|trip|estop|e_stop|emergency", re.IGNORECASE)

# Hardened parser: never resolve external entities, no DTD loading, no network fetch,
# no huge-tree expansion. Defends against XXE and billion-laughs regardless of lxml defaults.
_SAFE_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    huge_tree=False,
)


def parse_l5x(file_path: str | Path) -> list[PlcDocument]:
    """Parse an L5X file and return a list of PlcDocument entities."""
    file_path = Path(file_path)
    tree = etree.parse(str(file_path), _SAFE_PARSER)
    root = tree.getroot()

    docs: list[PlcDocument] = []

    controller = extract_controller(root)
    if controller:
        docs.append(
            PlcDocument(
                doc_type="controller",
                name=controller.name,
                path=f"/Controller/{controller.name}",
                content=render_controller(controller),
            )
        )

    tags = extract_tags(root)
    tag_groups: dict[str, list[TagEntity]] = {}
    for tag in tags:
        tag_groups.setdefault(tag.scope, []).append(tag)

    for scope, group_tags in tag_groups.items():
        docs.append(
            PlcDocument(
                doc_type="tag_group",
                name=f"Tags [{scope}]",
                path=f"/Tags/{scope}",
                content=render_tag_group(scope, group_tags),
            )
        )

    routines = extract_routines(root)
    for routine in routines:
        docs.append(
            PlcDocument(
                doc_type="routine",
                name=f"{routine.program}/{routine.name}",
                path=f"/Programs/{routine.program}/Routines/{routine.name}",
                content=render_routine(routine),
            )
        )

    udts = extract_udts(root)
    for udt in udts:
        docs.append(
            PlcDocument(
                doc_type="udt",
                name=udt.name,
                path=f"/DataTypes/{udt.name}",
                content=render_udt(udt),
            )
        )

    aois = extract_aois(root)
    for aoi in aois:
        docs.append(
            PlcDocument(
                doc_type="aoi",
                name=aoi.name,
                path=f"/AddOnInstructions/{aoi.name}",
                content=render_aoi(aoi),
            )
        )

    modules = extract_modules(root)
    if modules:
        module_groups: dict[str, list[ModuleEntity]] = {}
        for mod in modules:
            parent = mod.parent_module or "Local"
            module_groups.setdefault(parent, []).append(mod)
        for parent, group_mods in module_groups.items():
            docs.append(
                PlcDocument(
                    doc_type="module_group",
                    name=f"Modules [{parent}]",
                    path=f"/Modules/{parent}",
                    content=render_module_group(parent, group_mods),
                )
            )

    logger.info(
        "extracted %d tags, %d routines, %d udts, %d aois, %d modules",
        len(tags),
        len(routines),
        len(udts),
        len(aois),
        len(modules),
    )

    return docs


def extract_controller(root: etree._Element) -> ControllerEntity | None:
    ctrl = root.find(".//Controller")
    if ctrl is None:
        return None

    desc_elem = ctrl.find("Description")
    tasks = [t.get("Name", "") for t in ctrl.findall(".//Task") if t.get("Name")]
    programs = ctrl.findall(".//Program")
    tags_elem = ctrl.findall(".//Tag")
    modules_elem = ctrl.findall(".//Module")

    return ControllerEntity(
        name=ctrl.get("Name", ""),
        processor_type=ctrl.get("ProcessorType", ""),
        software_revision=ctrl.get("SoftwareRevision", ""),
        description=_text_content(desc_elem),
        program_count=len(programs),
        tag_count=len(tags_elem),
        module_count=len(modules_elem),
        tasks=tasks,
    )


def extract_tags(root: etree._Element) -> list[TagEntity]:
    tags: list[TagEntity] = []

    for tag in root.findall(".//Controller/Tags/Tag"):
        tags.append(_parse_tag(tag, scope="Controller"))

    for program in root.findall(".//Program"):
        prog_name = program.get("Name", "")
        for tag in program.findall("Tags/Tag"):
            tags.append(_parse_tag(tag, scope=prog_name))

    return tags


def _parse_tag(elem: etree._Element, scope: str) -> TagEntity:
    desc_elem = elem.find("Description")
    return TagEntity(
        name=elem.get("Name", ""),
        data_type=elem.get("DataType", ""),
        scope=scope,
        description=_text_content(desc_elem),
        tag_type=elem.get("TagType", "Base"),
        alias_for=elem.get("AliasFor", ""),
    )


def extract_routines(root: etree._Element) -> list[RoutineEntity]:
    routines: list[RoutineEntity] = []

    for program in root.findall(".//Program"):
        prog_name = program.get("Name", "")
        for routine in program.findall("Routines/Routine"):
            routines.append(_parse_routine(routine, prog_name))

    return routines


def _parse_routine(elem: etree._Element, program: str) -> RoutineEntity:
    desc_elem = elem.find("Description")
    routine_type = elem.get("Type", "RLL")

    rungs: list[RungEntity] = []
    st_lines: list[str] = []

    if routine_type == "RLL":
        for i, rung in enumerate(elem.findall(".//Rung")):
            text_elem = rung.find("Text")
            comment_elem = rung.find("Comment")
            logic = _text_content(text_elem)
            comment = _text_content(comment_elem)
            refs = _extract_tag_references(logic)
            is_fault = bool(_FAULT_KEYWORDS.search(logic + " " + comment))
            rungs.append(
                RungEntity(
                    number=i,
                    logic_text=logic,
                    comment=comment,
                    references=refs,
                    is_fault_related=is_fault,
                )
            )
    elif routine_type == "ST":
        text_elem = elem.find(".//Text")
        if text_elem is not None and text_elem.text:
            st_lines = [line for line in text_elem.text.splitlines() if line.strip()]

    return RoutineEntity(
        name=elem.get("Name", ""),
        program=program,
        routine_type=routine_type,
        description=_text_content(desc_elem),
        rungs=rungs,
        st_lines=st_lines,
    )


def extract_udts(root: etree._Element) -> list[UdtEntity]:
    udts: list[UdtEntity] = []

    for dt in root.findall(".//DataType"):
        if dt.get("Family") == "NoFamily":
            continue
        desc_elem = dt.find("Description")
        members: list[UdtMember] = []
        for member in dt.findall("Members/Member"):
            if member.get("Hidden") == "true":
                continue
            m_desc = member.find("Description")
            members.append(
                UdtMember(
                    name=member.get("Name", ""),
                    data_type=member.get("DataType", ""),
                    description=_text_content(m_desc),
                )
            )
        udts.append(
            UdtEntity(
                name=dt.get("Name", ""),
                description=_text_content(desc_elem),
                members=members,
            )
        )

    return udts


def extract_aois(root: etree._Element) -> list[AoiEntity]:
    aois: list[AoiEntity] = []

    for aoi_elem in root.findall(".//AddOnInstructionDefinition"):
        desc_elem = aoi_elem.find("Description")
        params: list[AoiParameter] = []
        for param in aoi_elem.findall("Parameters/Parameter"):
            p_desc = param.find("Description")
            params.append(
                AoiParameter(
                    name=param.get("Name", ""),
                    data_type=param.get("DataType", ""),
                    usage=param.get("Usage", ""),
                    description=_text_content(p_desc),
                    visible=param.get("Visible", "true") != "false",
                )
            )

        aoi_routines: list[RoutineEntity] = []
        for routine in aoi_elem.findall("Routines/Routine"):
            aoi_routines.append(_parse_routine(routine, program=aoi_elem.get("Name", "")))

        aois.append(
            AoiEntity(
                name=aoi_elem.get("Name", ""),
                description=_text_content(desc_elem),
                revision=aoi_elem.get("Revision", ""),
                parameters=params,
                routines=aoi_routines,
            )
        )

    return aois


def extract_modules(root: etree._Element) -> list[ModuleEntity]:
    modules: list[ModuleEntity] = []

    for mod in root.findall(".//Module"):
        ports: list[ModulePort] = []
        for port in mod.findall("Ports/Port"):
            ports.append(
                ModulePort(
                    port_id=port.get("Id", ""),
                    port_type=port.get("Type", ""),
                    address=port.get("Address", ""),
                )
            )
        modules.append(
            ModuleEntity(
                name=mod.get("Name", ""),
                catalog_number=mod.get("CatalogNumber", ""),
                parent_module=mod.get("ParentModule", ""),
                ports=ports,
            )
        )

    return modules


def _text_content(elem: etree._Element | None) -> str:
    if elem is None:
        return ""
    text = elem.text or ""
    return text.strip()


def _extract_tag_references(logic_text: str) -> list[str]:
    """Extract tag name references from rung logic text."""
    refs = re.findall(r"[A-Za-z_]\w*(?:\.\w+)*", logic_text)
    mnemonics = {
        "XIC",
        "XIO",
        "OTE",
        "OTL",
        "OTU",
        "TON",
        "TOF",
        "RTO",
        "CTU",
        "CTD",
        "RES",
        "MOV",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "CMP",
        "EQU",
        "NEQ",
        "GRT",
        "GEQ",
        "LES",
        "LEQ",
        "JSR",
        "RET",
        "SBR",
        "AFI",
        "NOP",
        "ONS",
        "OSR",
        "OSF",
        "BST",
        "BND",
        "TND",
        "MCR",
        "MSG",
        "GSV",
        "SSV",
        "COP",
        "FLL",
    }
    return [r for r in refs if r not in mnemonics and not r.startswith("__")]
