"""Independent physical and rendering checks; no solver invocation."""
from types import SimpleNamespace
import numpy as np
import pytest

from fempost.rf_scene import RFViewState, RFScene, average_poynting, field_values, colour_limits


def plane_wave(kind="driven"):
    axes = (np.linspace(0, .08, 5), np.linspace(0, .04, 4), np.linspace(0, .46, 11))
    shape = tuple(len(a) for a in axes) + (3,)
    E, H = np.zeros(shape, complex), np.zeros(shape, complex)
    phase = np.exp(-1j * 23 * axes[2])[None, None, :]
    E[..., 1] = 10 * phase
    H[..., 0] = -2 * phase
    return SimpleNamespace(E=E, H=H, axes=axes, kind=kind, normalization="1 W incident at port 1",
                           source="synthetic analytic wave", index=0, port=1, frequency_hz=2.45e9,
                           label="Analytic test", fem_mesh_path=None)


def test_average_power_and_shared_phase():
    f = plane_wave()
    S = average_poynting(f.E, f.H)
    np.testing.assert_allclose(S[..., :2], 0)
    np.testing.assert_allclose(S[..., 2], 10)
    phase = np.exp(1j * .731)
    np.testing.assert_allclose(average_poynting(f.E * phase, f.H * phase), S)
    np.testing.assert_allclose(average_poynting(2 * f.E, 2 * f.H), 4 * S)


def test_complex_snapshot_phase_and_limits():
    f = plane_wave()
    s = RFViewState(component="y", representation="real")
    a = field_values(f, s)
    s.phase_deg = 90
    b = field_values(f, s)
    np.testing.assert_allclose(b["values"], -f.E[..., 1].imag, atol=1e-13)
    assert colour_limits(a, s)[0] == colour_limits(b, s)[0]
    np.testing.assert_allclose(colour_limits(a, s)[0], [-10., 10.])
    s.representation = "magnitude"
    np.testing.assert_allclose(field_values(f, s)["values"], 10)
    s.representation = "phase"
    assert colour_limits(field_values(f, s), s)[0] == [-180., 180.]
    s.component = "magnitude"
    with pytest.raises(ValueError, match="no unique phase"):
        field_values(f, s)


def test_eigen_quadrature_and_no_power_renormalization():
    f = plane_wave("eigenmode")
    f.E[:] = 0
    f.H[:] = 0
    f.E[..., 1] = 2j
    f.H[..., 0] = 3
    s = RFViewState(field="E", component="y", representation="real")
    np.testing.assert_allclose(field_values(f, s)["values"], 2)
    s.field, s.component = "H", "x"
    np.testing.assert_allclose(field_values(f, s)["values"], 0, atol=1e-15)
    s.phase_deg = 90
    np.testing.assert_allclose(field_values(f, s)["values"], 3)
    s.field = "Savg"
    r = field_values(f, s)
    np.testing.assert_allclose(r["values"], 0)
    assert "numerical" in r["warnings"][0]


def test_log_guard_and_probe_coordinates():
    f = plane_wave()
    s = RFViewState(component="y", representation="real", log_scale=True)
    assert colour_limits(field_values(f, s), s)[2] is False
    s.component = "magnitude"
    first_limits = colour_limits(field_values(f, s), s)[0]
    s.phase_deg = 90
    np.testing.assert_allclose(colour_limits(field_values(f, s), s)[0], first_limits)
    scene = RFScene(None)
    scene.frame = f
    point = [.04, .04/3, .23]
    probe = scene.probe(point)
    assert probe["distance_m"] < 1e-15
    np.testing.assert_allclose(probe["Savg_W_per_m2"], [0, 0, 10], atol=1e-12)
    assert "error" in scene.probe([100, 0, 0])
    f.E[2, 1, 5] = np.nan
    assert "error" in scene.probe(point)


def test_render_geometry_and_saved_state(tmp_path):
    import pyvista as pv
    f = plane_wave()
    p = pv.Plotter(off_screen=True, window_size=(900, 650))
    scene = RFScene(p)
    try:
        s = RFViewState(component="y", representation="real", vectors=True, display="slice", axis="y")
        for mode in ("slice", "slices", "clip", "surface"):
            s.display = mode
            info = scene.render(f, s)
            assert info["coordinates"] == "m"
            assert info["vectors"] > 0
            assert info["true_range"][1] > 9
        scene.set_view("+y")
        restored = RFViewState(camera=[list(v) for v in p.camera_position], parallel_scale=.12345)
        scene.render(f, restored)
        assert p.camera.parallel_scale == pytest.approx(.12345)
        p.screenshot(str(tmp_path / "analytic-wave.png"))
        assert (tmp_path / "analytic-wave.png").stat().st_size > 10000
        assert RFViewState.from_dict(s.to_dict()) == s
    finally:
        p.close()
