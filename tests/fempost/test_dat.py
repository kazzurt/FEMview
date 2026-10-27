"""Streaming DAT reads: table index, element-mean integration-point stress, totals."""
import numpy as np
import pytest

from fempost.dat import DatReader, scan_tables


def test_scan_and_element_mean_stress(short_screw):
    reader = DatReader(short_screw / "short_screw.dat", use_cache=False)
    kinds = {t.kind for t in reader.tables}
    assert {"stress", "displacement", "force"} <= kinds
    assert reader.stress_times() == pytest.approx(list(np.arange(1, 11) * 0.2))
    mean = reader.element_mean_stress(1.0)
    assert mean is not None
    assert mean["element_ids"].tolist() == list(range(1, 17))
    assert mean["stress"].shape == (16, 6)
    assert set(mean["points_per_element"].tolist()) == {27}        # full-integration C3D20
    assert mean["sets"] == ["BOLT"]
    # Pretension step: axial stress of a 10 x 10 bar under 100 N is 1 MPa on average.
    assert np.mean(mean["stress"][:, 0]) == pytest.approx(1.0, rel=0.05)
    assert reader.element_mean_stress(0.37) is None
    cached = DatReader(short_screw / "short_screw.dat", use_cache=True)
    again = cached.element_mean_stress(1.0)
    assert np.allclose(again["stress"], mean["stress"])
    assert np.allclose(DatReader(short_screw / "short_screw.dat").element_mean_stress(1.0)["stress"], mean["stress"])


def test_contact_statistics_blocks(friction):
    reader = DatReader(friction / "friction.dat", use_cache=False)
    stats = reader.contact_statistics(1.0)
    assert stats and stats[0]["slave"] == "SLAVE" and stats[0]["master"] == "MASTER"
    titles = {s["table"] for s in stats}
    assert any("total surface force" in t for t in titles) and any("center of gravity" in t for t in titles)
    assert reader.stress_times() == []
    contact = [t for t in reader.tables if t.kind == "contact"]
    assert contact and contact[0].time == pytest.approx(0.1)


def test_total_forces_codex(codex_run):
    reader = DatReader(codex_run / "job.dat")
    totals = reader.total_forces(1.0)
    assert totals and totals[0]["set"] == "FIX"
    assert np.allclose(totals[0]["force_N"], [1108.058, 1378.556, -48.85], atol=0.5)
    assert reader.stress_times() == [1.0]


def test_scan_tables_expanded_exponent(tmp_path):
    sample = tmp_path / "x.dat"
    sample.write_text(" stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set ALL and time  0.1000000E+01\n\n"
                      "         1   1  1.234560E+02  1.0-100  0.000000E+00  1.0E+00  2.0E+00  3.0E+00\n"
                      "         1   2  1.234560E+02  1.0-100  0.000000E+00  1.0E+00  2.0E+00  3.0E+00\n")
    tables = scan_tables(sample)
    assert len(tables) == 1 and tables[0].kind == "stress" and tables[0].rows == 2 and tables[0].time == 1.0
    mean = DatReader(sample, use_cache=False).element_mean_stress(1.0)
    assert mean["stress"][0, 0] == pytest.approx(123.456) and mean["stress"][0, 1] == pytest.approx(1e-100)
