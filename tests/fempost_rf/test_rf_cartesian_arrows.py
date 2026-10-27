"""Arrow placement, boundary clipping and density checks; no desktop control."""
import json
import time
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets

from fempost.rf_app import RFWorkspace
from fempost.rf_display_geometry import cartesian_plane_points, VisualFieldSampler
from fempost.rf_scene import RFScene, RFViewState, field_values


PROJECT = Path(os.environ.get("FEMVIEW_COPPER_FIXTURE", str(Path(__file__).resolve().parent / "external/copper-cavity")))
GEOMETRY = dict(radius=.0525, zmin=0., zmax=.070)
BOUNDS = [-.0525, .0525, -.0525, .0525, 0., .070]


@pytest.mark.parametrize("axis", list("xyz"))
def test_square_lattice_and_exact_cylinder_clipping(axis):
    origin = [.019, -.023, .031]
    fixed = "xyz".index(axis)
    free = [i for i in range(3) if i != fixed]
    counts = []
    for density in (7, 15, 27):
        points = cartesian_plane_points(BOUNDS, axis, origin, density, GEOMETRY)
        counts.append(len(points))
        assert len(points) > 0
        np.testing.assert_allclose(points[:, fixed], origin[fixed])
        assert np.max(np.hypot(points[:, 0], points[:, 1])) <= .0525+1e-12
        assert points[:, 2].min() >= -1e-12 and points[:, 2].max() <= .070+1e-12
        spacing = .105/(density-1)
        for i in free:
            np.testing.assert_allclose(np.diff(np.unique(points[:, i])), spacing, atol=1e-12)
        # Moving the cut does not rotate or stagger its Cartesian lattice.
        centered = cartesian_plane_points(BOUNDS, axis, [0, 0, .035], density, GEOMETRY)
        for i in free:
            assert all(np.min(np.abs(centered[:, i]-v)) < 1e-12 for v in points[:, i])
    assert counts[0] < counts[1] < counts[2]
    outside = [0, 0, .035]
    outside[fixed] = 1.
    assert cartesian_plane_points(BOUNDS, axis, outside, 15, GEOMETRY).shape == (0, 3)


def constant_cylinder():
    axes = (np.linspace(-.0525, .0525, 21),)*2 + (np.linspace(0, .070, 13),)
    x, y, z = np.meshgrid(*axes, indexing="ij")
    E, H = np.zeros(x.shape+(3,), complex), np.zeros(x.shape+(3,), complex)
    E[..., 0], H[..., 1] = 1., 2j
    E[np.hypot(x, y) > .0525] = np.nan
    H[np.hypot(x, y) > .0525] = np.nan
    return SimpleNamespace(axes=axes, E=E, H=H, kind="eigenmode", normalization="test",
                           source="synthetic cylinder", index=0, port=1, frequency_hz=4.6e9,
                           label="Synthetic", fem_mesh_path=None, cylinder_geometry=GEOMETRY)


def test_render_uses_grid_independent_of_colour_mesh_field_and_phase():
    import pyvista as pv
    frame = constant_cylinder()
    before_e, before_h = frame.E.copy(), frame.H.copy()
    plotter = pv.Plotter(off_screen=True)
    scene = RFScene(plotter)
    try:
        state = RFViewState(vectors=True)
        expected = {a: len(cartesian_plane_points(BOUNDS, a, [0, 0, .035], 15, GEOMETRY))
                    for a in "xyz"}
        for smooth in (True, False):
            state.smooth_boundary = smooth
            for field, phase in (("E", 0), ("H", 90)):
                state.field, state.phase_deg = field, phase
                info = scene.render(frame, state)
                assert info["vector_candidates_by_section"] == expected
                assert info["vectors_by_section"] == expected
                assert info["vector_layout"] == "uniform Cartesian cut-plane grid"
                assert all(f"rf-vectors-{a}" in plotter.actors for a in "xyz")
            state.phase_deg = 0  # H is purely imaginary: every arrow vanishes.
            info = scene.render(frame, state)
            assert info["vectors"] == 0
            assert info["vector_candidates_by_section"] == expected
        state.field, state.phase_deg, state.vector_density = "E", 0, 25
        state.display, state.axis, state.position = "slice", "x", .7
        info = scene.render(frame, state)
        assert set(info["vectors_by_section"]) == {"x"}
        assert info["vectors"] > expected["x"]
        np.testing.assert_equal(frame.E, before_e)
        np.testing.assert_equal(frame.H, before_h)
    finally:
        plotter.close()


def test_complex_interpolation_before_phase_and_unknown_boundary_mask():
    axes = (np.linspace(-1, 1, 5),)*3
    x, y, z = np.meshgrid(*axes, indexing="ij")
    E = np.stack((1+x+1j*y, 2*z+3j, x-y+1j*z), axis=-1)
    frame = SimpleNamespace(axes=axes, E=E, H=E/2, kind="driven")
    points = np.array([[.1, .2, .3], [-.7, -.3, .4]])
    sampler = VisualFieldSampler(frame)
    samples = sampler(points)
    px, py, pz = points.T
    expected = np.stack((1+px+1j*py, 2*pz+3j, px-py+1j*pz), axis=-1)
    np.testing.assert_allclose(samples["E"], expected)
    result = field_values(frame, RFViewState(phase_deg=37), samples=samples)
    np.testing.assert_allclose(result["vectors"], np.real(expected*np.exp(1j*np.deg2rad(37))))
    frame.E[x < 0] = np.nan
    sampler = VisualFieldSampler(frame)
    assert sampler.valid_at([[.25, 0, 0], [-.25, 0, 0], [2, 0, 0]]).tolist() == [True, False, False]


def test_density_control_updates_arrows_and_round_trips_saved_view(tmp_path, monkeypatch):
    case = PROJECT / "eigenmode-baseline" / "results" / "refined"
    if not case.exists():
        pytest.skip("Local copper-cavity fixture unavailable")
    monkeypatch.setenv("FEMPOST_RF_CACHE", str(PROJECT / "viewer-cache"))
    app = QtWidgets.QApplication.instance()
    workspace = RFWorkspace(case)

    def wait_loaded():
        deadline = time.monotonic()+60
        while workspace.is_busy or not workspace._selection_valid:
            app.processEvents()
            if time.monotonic() > deadline:
                raise AssertionError(workspace.status.text())
            time.sleep(.02)

    try:
        wait_loaded()
        assert workspace.vector_density.value() == 15
        assert RFViewState.from_dict({"vectors": True}).vector_density == 15
        workspace.field_combo.setCurrentIndex(workspace.field_combo.findData("H"))
        workspace.phase_spin.setValue(90)
        workspace.vectors_check.setChecked(True)
        counts = []
        for density in (9, 25):
            workspace.vector_density.setValue(density)
            workspace.render_timer.stop()
            workspace._render()
            assert workspace.state.vector_density == density
            assert workspace._info["vector_density"] == density
            counts.append(workspace._info["vectors_by_section"].copy())
        assert all(counts[1][a] > counts[0][a] for a in "xyz")
        assert workspace.state.vector_scale == 1.
        saved = workspace.save_view(tmp_path / "cartesian-arrows.view.json")
        assert json.loads(saved.read_text())["state"]["vector_density"] == 25
        workspace.vector_density.setValue(7)
        workspace.load_view(saved)
        wait_loaded()
        assert workspace.vector_density.value() == workspace.state.vector_density == 25
        assert workspace._info["vectors_by_section"] == counts[1]
    finally:
        workspace.shutdown()
        app.processEvents()
