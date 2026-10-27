"""Check field palettes and opaque surface-current depth rendering offscreen."""
import os
from pathlib import Path
import numpy as np
import pytest
from fempost.rf_data import RFStudy
from fempost.rf_scene import RFScene, RFViewState, FIELD_COLORMAPS, colour_limits, field_values

PROJECT = Path(os.environ.get("FEMVIEW_COPPER_FIXTURE", str(Path(__file__).resolve().parent / "external/copper-cavity")))


@pytest.fixture(scope="module")
def frame():
    import os
    if not (PROJECT / "eigenmode-baseline/results/refined").is_dir():
        pytest.skip("External copper-cavity result fixture unavailable")
    with RFStudy(PROJECT / "eigenmode-baseline" / "results" / "refined") as study:
        yield study.sample(1)


def test_distinct_magnitude_palettes_and_phase_consistency(frame):
    assert len(set(FIELD_COLORMAPS[f] for f in ("E", "H", "Js"))) == 3
    for field, expected in (("E", "magma"), ("H", "viridis"), ("Js", "copper")):
        state = RFViewState(field=field)
        for phase in (0, 47, 90, 180):
            state.phase_deg = phase
            assert colour_limits(field_values(frame, state), state)[3] == expected
    state = RFViewState(field="E", component="x", representation="phase")
    assert colour_limits(field_values(frame, state), state)[3] == "twilight_shifted"
    state.representation = "real"
    assert colour_limits(field_values(frame, state), state)[3] == "coolwarm"


def test_current_actor_has_no_translucent_geometry(frame):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    try:
        scene = RFScene(plotter)
        # Even a legacy translucent state must now yield opaque metal.
        state = RFViewState(field="Js", phase_deg=90, display="surface", vectors=True, opacity=.25)
        info = scene.render(frame, state)
        actor = plotter.actors["rf-field-wall"]
        assert actor.GetProperty().GetOpacity() == 1.
        assert actor.GetForceOpaque() == 1
        assert actor.HasTranslucentPolygonalGeometry() == 0
        assert actor.GetMapper().GetLookupTable().GetNanColor()[3] == 1.
        assert info["field_opacity"] == 1. and info["cmap"] == "copper"
        assert np.isfinite(frame.surface_current["Js"]).all()
    finally:
        plotter.close()
