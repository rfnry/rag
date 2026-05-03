"""DrawingIngestionService: 4-phase stubs + async method contract."""

import inspect

from rfnry_knowledge.ingestion.drawing.service import DrawingIngestionService


def test_service_exposes_four_phase_async_methods() -> None:
    for name in ("render", "extract", "link", "ingest"):
        fn = getattr(DrawingIngestionService, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async"


def test_service_module_exports_supported_extensions() -> None:
    """SUPPORTED_DRAWING_EXTENSIONS is the public contract for extension routing."""
    from rfnry_knowledge.ingestion.drawing.service import (
        SUPPORTED_DRAWING_EXTENSIONS,
    )

    assert ".dxf" in SUPPORTED_DRAWING_EXTENSIONS
    assert ".pdf" in SUPPORTED_DRAWING_EXTENSIONS


def test_models_roundtrip_drawing_page_analysis() -> None:
    """DrawingPageAnalysis dataclass must have from_dict and to_dict converters."""
    from rfnry_knowledge.ingestion.drawing.models import (
        DetectedComponent,
        DetectedConnection,
        DrawingPageAnalysis,
        OffPageConnector,
        Port,
    )

    analysis = DrawingPageAnalysis(
        page_number=1,
        components=[
            DetectedComponent(
                component_id="R1",
                symbol_class="resistor",
                label="R1 10k",
                bbox=[10, 20, 30, 40],
                ports=[Port(port_id="a", position=[10, 25]), Port(port_id="b", position=[40, 25])],
                properties={"tolerance": "5%"},
            )
        ],
        connections=[
            DetectedConnection(
                from_component="R1",
                from_port="b",
                to_component="C1",
                to_port="a",
                net_label="N5",
                wire_style="solid",
            )
        ],
        off_page_connectors=[OffPageConnector(tag="/A2", bound_component="R1", target_hint="sheet 2")],
        domain="electrical",
        page_type="drawing",
        notes=["hello"],
    )
    round_trip = DrawingPageAnalysis.from_dict(analysis.to_dict())
    assert round_trip == analysis


def test_models_roundtrip_handles_optional_fields() -> None:
    from rfnry_knowledge.ingestion.drawing.models import (
        DetectedComponent,
        DetectedConnection,
        DrawingPageAnalysis,
        Port,
    )

    analysis = DrawingPageAnalysis(
        page_number=3,
        components=[
            DetectedComponent(
                component_id="C1",
                symbol_class="capacitor",
                label=None,
                bbox=[0, 0, 5, 5],
                ports=[Port(port_id="1", position=None)],
                properties=None,
            )
        ],
        connections=[DetectedConnection(from_component="C1", to_component="R1")],
        off_page_connectors=[],
        domain="mixed",
        page_type="drawing",
        notes=[],
        sheet_number=None,
        zone_grid=None,
        title_block=None,
    )
    assert DrawingPageAnalysis.from_dict(analysis.to_dict()) == analysis
