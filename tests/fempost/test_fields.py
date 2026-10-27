"""Derived fields: displacement, stress invariants, contact, error, element means, auto factor."""
import numpy as np
import pytest

from fempost import fields
from fempost.bundle import RunBundle


def test_short_screw_uniaxial_bar(short_screw):
    bundle = RunBundle(short_screw)
    frame = bundle.last_frame_of_step(1)           # 100 N on a 10 x 10 mm bar: 1 MPa axial
    assert frame.time == pytest.approx(1.0)
    keys = fields.available(frame)
    assert {"u_mag", "ux", "von_mises", "tresca", "s1", "s3", "sxx", "syz", "error", "f_mag"} <= set(keys)
    assert "cpress" not in keys
    vm = fields.compute("von_mises", frame)
    assert vm.location == "point" and vm.unit == "MPa" and "nodal display" in vm.note
    assert np.nanmean(vm.values) == pytest.approx(1.0, rel=0.05)
    sxx = fields.compute("sxx", frame).values
    assert np.nanmean(sxx) == pytest.approx(1.0, rel=0.05) and fields.FIELD_SPECS["sxx"].signed
    s1, s2, s3 = (fields.compute(k, frame).values for k in ("s1", "s2", "s3"))
    assert np.all(s3 <= s2 + 1e-9) and np.all(s2 <= s1 + 1e-9)
    assert np.nanmean(s1) == pytest.approx(1.0, rel=0.05) and abs(np.nanmean(s3)) < 0.1
    assert np.allclose(fields.compute("tresca", frame).values, s1 - s3)
    assert np.nanmean(fields.compute("hydrostatic", frame).values) == pytest.approx(1.0 / 3.0, rel=0.1)
    u = fields.compute("u_mag", frame).values
    assert u.shape == (162,) and np.nanmax(u) > 0
    assert np.allclose(fields.compute("ux", frame).values, frame.fields["DISP"][:, 0])
    assert fields.compute("error", frame).values.shape == (162,)
    assert "principal" in frame.derived and "stress6" in frame.derived


def test_element_mean_from_dat(short_screw):
    bundle = RunBundle(short_screw)
    frame = bundle.last_frame_of_step(1)
    record = bundle.dat.element_mean_stress(frame.time)
    result = fields.compute_element("von_mises", bundle.mesh, record, frame.derived)
    assert result.location == "cell" and result.values.shape == (16,)
    assert np.nanmean(result.values) == pytest.approx(1.0, rel=0.05)
    assert "integration-point" in result.note
    with pytest.raises(KeyError):
        fields.compute_element("u_mag", bundle.mesh, record)


def test_contact_fields(friction):
    bundle = RunBundle(friction)
    frame = bundle.last_frame_of_step(1)
    keys = fields.available(frame)
    assert {"cpress", "gap", "cslip", "cshear"} <= set(keys)
    cpress = fields.compute("cpress", frame).values
    assert cpress.shape == (150,) and np.nanmax(cpress) > 0
    assert np.all(fields.compute("cslip", frame).values >= 0)
    gap = fields.compute("gap", frame)
    assert gap.signed and gap.unit == "mm"


def test_auto_factor_and_volume_fraction(short_screw):
    bundle = RunBundle(short_screw)
    frame = bundle.frames[-1]
    factor = fields.auto_deformation_factor(frame, bundle.mesh.diagonal)
    peak = np.nanmax(np.linalg.norm(frame.fields["DISP"], axis=1))
    assert factor * peak == pytest.approx(0.1 * bundle.mesh.diagonal)
    volumes = np.ones(16)
    cell_values = np.arange(16, dtype=float)
    out = fields.volume_fraction_above(cell_values, 11.5, volumes, bundle.mesh.connectivity, "cell")
    assert out["cells_above"] == 4 and out["fraction"] == pytest.approx(0.25)
    point_values = np.full(bundle.mesh.n_nodes, 2.0)
    out = fields.volume_fraction_above(point_values, 1.0, volumes, bundle.mesh.connectivity, "point")
    assert out["fraction"] == pytest.approx(1.0)


def test_stress_scalar_synthetic():
    s6 = np.array([[100.0, 0, 0, 0, 0, 0], [0, 0, 0, 50.0, 0, 0]])
    assert fields.von_mises6(s6) == pytest.approx([100.0, 50.0 * np.sqrt(3)])
    p = fields.principal(s6)
    assert p[0] == pytest.approx([0.0, 0.0, 100.0]) and p[1] == pytest.approx([-50.0, 0.0, 50.0])
    assert fields.stress_scalar("tresca", s6) == pytest.approx([100.0, 100.0])
