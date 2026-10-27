"""render/export CLI: figures and VTU from the same scene code as the GUI."""
import json

import numpy as np
import pytest

from fempost.__main__ import main


def test_render_cli_with_overrides(short_screw, tmp_path, capsys):
    out = tmp_path / "fig.png"
    code = main(["render", str(short_screw), "--out", str(out), "--field", "sxx", "--deform", "10", "--view", "+y",
                 "--clip", "0,1,0@0,5,5", "--slice", "1,0,0", "--overlay", "restraints,loads", "--clim=-2,2",
                 "--size", "800,600", "--step", "1"])
    assert code == 0 and out.stat().st_size > 1000
    record = json.loads(capsys.readouterr().out)
    assert record["state"]["field"] == "sxx" and record["state"]["deform"] == 10.0
    assert record["info"]["frame"].startswith("Step 1 · inc 5")
    assert record["info"]["range"] == [-2.0, 2.0] and record["info"]["clip"]["cells"] > 0
    kinds = {o["kind"] for o in record["overlays"]}
    assert "restraint" in kinds and "load" in kinds


def test_render_from_scene_spec(friction, tmp_path, capsys):
    from fempost.scene import SceneState
    spec = tmp_path / "scene.json"
    SceneState(field="cpress", frame=9, deform=0, overlays={"contact": True}, view="-z", limit=5.0).save(spec)
    out = tmp_path / "spec.png"
    assert main(["render", str(friction), "--out", str(out), "--scene", str(spec), "--cmap", "viridis"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["state"]["cmap"] == "viridis" and record["state"]["field"] == "cpress"
    assert 0.0 <= record["info"]["limit"]["fraction"] <= 1.0
    assert record["info"]["frame"].startswith("Step 1 · inc 10")


def test_export_vtu(short_screw, tmp_path, capsys):
    import pyvista as pv
    out = tmp_path / "bar.vtu"
    assert main(["export", str(short_screw), "--vtu", str(out), "--step", "1"]) == 0
    record = json.loads(capsys.readouterr().out)
    grid = pv.read(str(out))
    assert grid.n_cells == 16 and grid.n_points == 162
    assert "von_mises_MPa_nodal_display" in grid.point_data and "FRD_STRESS" in grid.point_data
    assert "von_mises_MPa_element_mean_ip" in grid.cell_data
    assert np.nanmean(grid.cell_data["von_mises_MPa_element_mean_ip"]) == pytest.approx(1.0, rel=0.05)
    assert record["frame"].startswith("Step 1 · inc 5")


def test_render_rejects_unknown_field(short_screw, tmp_path):
    with pytest.raises(SystemExit):
        main(["render", str(short_screw), "--out", str(tmp_path / "x.png"), "--field", "bogus"])
