"""Numerical/headless checks only; no desktop control or visual inspection."""
from pathlib import Path
from types import SimpleNamespace
import os
import time
import numpy as np
import pytest
from scipy.special import jv, jvp, jnp_zeros

from fempost.rf_data import RFStudy
from fempost.rf_scene import RFScene, RFViewState, field_values, colour_limits
from fempost.rf_display_geometry import cylinder_section, VisualFieldSampler
from fempost.rf_worker import conductor_surface_fields
from fempost.rf_app import RFWorkspace
from PySide6 import QtWidgets

PROJECT = Path(os.environ.get("FEMVIEW_COPPER_FIXTURE", str(Path(__file__).resolve().parent / "external/copper-cavity")))
CASE = PROJECT / "eigenmode-baseline" / "results" / "refined"


@pytest.fixture(scope="module")
def copper():
    if not CASE.exists():
        pytest.skip("Local copper cavity fixture unavailable")
    os.environ["FEMPOST_RF_CACHE"] = str(PROJECT / "viewer-cache")
    with RFStudy(CASE) as study:
        yield study.sample(1)


@pytest.mark.parametrize("field,phase_max,phase_min", [("E", 0, 90), ("H", 90, 0), ("Js", 90, 0)])
def test_instantaneous_colours_follow_phase_with_fixed_scale(copper, field, phase_max, phase_min):
    state = RFViewState(field=field, representation="real", phase_deg=phase_max)
    maximum = field_values(copper, state)
    high_scale = colour_limits(maximum, state)[0]
    state.phase_deg = phase_min
    minimum = field_values(copper, state)
    np.testing.assert_allclose(high_scale, colour_limits(minimum, state)[0])
    assert np.nanmax(minimum["values"]) < .002*np.nanmax(maximum["values"])
    np.testing.assert_allclose(maximum["values"], np.linalg.norm(maximum["vectors"], axis=-1))
    state.representation = "magnitude"
    static = field_values(copper, state)["values"]
    state.phase_deg += 63
    np.testing.assert_allclose(field_values(copper, state)["values"], static)


def test_conductor_current_sign_and_orientation_from_ampere_boundary():
    nodes = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    class Field:
        freq = 4e9
        mesh = SimpleNamespace(nodes=nodes.T, tets=np.array([[0, 1, 2, 3]]).T,
                               tris=triangles.T, tri_to_tet=np.array([[0]*4, [-1234]*4]))
        def interpolate(self, x, y, z, mapping, usenan):
            assert np.all(mapping == 0)
            self.Hx = np.ones(4, complex)
            self.Hy = self.Hz = np.zeros(4, complex)
    wall = conductor_surface_fields(Field())
    # z=0 wall: metal is below the air. n_metal->air = +z;
    # Ampere's boundary condition requires Js = +z cross +x = +y.
    np.testing.assert_allclose(wall["normals"][0], [0, 0, -1])
    np.testing.assert_allclose(wall["Js"][0], [0, 1, 0])
    np.testing.assert_allclose(np.sum(wall["Js"]*wall["normals"], axis=1), 0, atol=1e-15)
    assert wall["areas"].sum() == pytest.approx(1.5+np.sqrt(3)/2)


def test_native_currents_match_analytical_te112_shape(copper):
    wall = copper.surface_current
    x, y, z = wall["centers"].T
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    kc, kz = jnp_zeros(1, 1)[0]/.0525, 2*np.pi/.070
    basis = []
    for angle in (0, np.pi/2):
        j, jp = jv(1, kc*r), jvp(1, kc*r)
        jr = np.divide(j, r, out=np.full_like(r, kc/2), where=r > 1e-12)
        hr = kz/kc*jp*np.cos(phi-angle)*np.cos(kz*z)
        hp = -kz/kc**2*jr*np.sin(phi-angle)*np.cos(kz*z)
        hz = j*np.cos(phi-angle)*np.sin(kz*z)
        h = np.column_stack((hr*np.cos(phi)-hp*np.sin(phi), hr*np.sin(phi)+hp*np.cos(phi), hz))
        basis.append(np.cross(h, wall["normals"]).ravel())
    design = np.column_stack(basis)
    measured = wall["Js"].ravel()
    coefficients = np.linalg.lstsq(design, measured, rcond=None)[0]
    error = np.linalg.norm(measured-design@coefficients)/np.linalg.norm(measured)
    assert error < .03
    assert len(wall["Js"]) == 2496
    assert np.max(np.abs(np.sum(wall["Js"]*wall["normals"], axis=1))) < 1e-14


def test_smooth_circle_geometry_and_original_samples_unchanged(copper):
    before_e, before_h = copper.E.copy(), copper.H.copy()
    section = cylinder_section(copper.cylinder_geometry, "z", [0, 0, .035])
    edge = section.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                          manifold_edges=False, non_manifold_edges=False)
    np.testing.assert_allclose(np.hypot(edge.points[:, 0], edge.points[:, 1]), .0525, atol=1e-12)
    assert edge.n_points == 128
    samples = VisualFieldSampler(copper)(section.points)
    assert np.isfinite(samples["E"]).all() and np.isfinite(samples["H"]).all()
    state = RFViewState(field="H", representation="real", phase_deg=90)
    phase90 = field_values(copper, state, samples=samples)
    state.phase_deg = 0
    phase0 = field_values(copper, state, samples=samples)
    assert np.max(phase0["values"]) < .002*np.max(phase90["values"])
    np.testing.assert_equal(copper.E, before_e)
    np.testing.assert_equal(copper.H, before_h)


def test_vectors_on_every_plane_and_surface_rendering(copper):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True)
    scene = RFScene(plotter)
    try:
        for field, phase, position in (("E", 0, .25), ("H", 90, .5)):
            state = RFViewState(field=field, phase_deg=phase, position=position, vectors=True)
            info = scene.render(copper, state)
            assert info["display_smoothing"]
            assert set(info["vectors_by_section"]) == set("xyz")
            assert all(n > 10 for n in info["vectors_by_section"].values())
            assert all(f"rf-vectors-{axis}" in plotter.actors for axis in "xyz")
        info = scene.render(copper, RFViewState(field="Js", phase_deg=90, display="surface", vectors=True))
        assert info["unit"] == "A/m"
        assert info["vectors_by_section"]["wall"] > 100
        assert info["unloaded_q"] == copper.unloaded_q
        assert not info["display_smoothing"]
        title = plotter.actors["rf-title"]
        strings = [title.GetText(i) or "" for i in range(4)]
        combined = " ".join(strings)
        assert combined.count("Unloaded Q0") == 1
        assert "4.599 GHz" in combined
        assert "Sampled fields" not in combined and "True min/max" not in combined
    finally:
        plotter.close()


def test_workspace_phase_play_and_current_selection(copper):
    app = QtWidgets.QApplication.instance()
    workspace = RFWorkspace(CASE)
    try:
        deadline = time.monotonic()+60
        while workspace.is_busy or not workspace._selection_valid:
            app.processEvents()
            if time.monotonic() > deadline:
                raise AssertionError(workspace.status.text())
            time.sleep(.02)
        assert not hasattr(workspace, "quality_label")
        assert "4.599 GHz" in workspace.frequency_combo.currentText()
        assert workspace.frame.frequency_hz == pytest.approx(4598743966.855053)
        workspace.representation_combo.setCurrentIndex(workspace.representation_combo.findData("magnitude"))
        workspace._play(True)
        assert workspace.state.representation == "real"
        workspace._play(False)
        workspace.field_combo.setCurrentIndex(workspace.field_combo.findData("Js"))
        assert workspace.state.display == "surface"
        workspace.phase_spin.setValue(90)
        workspace.render_timer.stop()
        workspace._render()
        assert "Display error" not in workspace.status.text()
        assert workspace._info["unit"] == "A/m"
        workspace.field_combo.setCurrentIndex(workspace.field_combo.findData("H"))
        assert workspace.state.display == "slices"
    finally:
        workspace.shutdown()
        app.processEvents()
