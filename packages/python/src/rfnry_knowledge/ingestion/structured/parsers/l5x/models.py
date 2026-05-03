from dataclasses import dataclass, field


@dataclass
class PlcDocument:
    doc_type: str
    name: str
    path: str
    content: str


@dataclass
class ControllerEntity:
    name: str
    processor_type: str
    software_revision: str
    description: str = ""
    program_count: int = 0
    tag_count: int = 0
    module_count: int = 0
    tasks: list[str] = field(default_factory=list)


@dataclass
class TagEntity:
    name: str
    data_type: str
    scope: str
    description: str = ""
    tag_type: str = "Base"
    alias_for: str = ""


@dataclass
class RungEntity:
    number: int
    logic_text: str
    comment: str = ""
    references: list[str] = field(default_factory=list)
    is_fault_related: bool = False


@dataclass
class RoutineEntity:
    name: str
    program: str
    routine_type: str
    description: str = ""
    rungs: list[RungEntity] = field(default_factory=list)
    st_lines: list[str] = field(default_factory=list)


@dataclass
class UdtMember:
    name: str
    data_type: str
    description: str = ""


@dataclass
class UdtEntity:
    name: str
    description: str = ""
    members: list[UdtMember] = field(default_factory=list)


@dataclass
class AoiParameter:
    name: str
    data_type: str
    usage: str
    description: str = ""
    visible: bool = True


@dataclass
class AoiEntity:
    name: str
    description: str = ""
    revision: str = ""
    parameters: list[AoiParameter] = field(default_factory=list)
    routines: list[RoutineEntity] = field(default_factory=list)


@dataclass
class ModulePort:
    port_id: str
    port_type: str
    address: str = ""


@dataclass
class ModuleEntity:
    name: str
    catalog_number: str
    parent_module: str = ""
    ports: list[ModulePort] = field(default_factory=list)
