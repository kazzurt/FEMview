"""Phase 2: reaction overlay, limit isosurface, plot groups, A - B on a shared mesh, CLI wiring."""
import json

import numpy as np
import pytest

from fempost.__main__ import main
from fempost.bundle import RunBundle
from fempost.scene import PLOT_GROUPS, Scene, SceneState, apply_plot_group, offscreen_plotter, plot_group_names


def test_reactions_limit_plot_groups_difference(short_screw, friction):
    bundle = RunBundle(short_screw)
    scene = Scene(bundle)
    plotter = offscreen_plotter((700, 500))
    info = scene.render(plotter, SceneState(field="u_mag", frame=4, deform=0, overlays={"reactions": True, "restraints": True}))
    reactions = {o["target"]: o for o in info["overlays"] if o["kind"] == "reaction"}
    assert {"ROOT", "TIP", "NALL"} <= set(reactions)
    # 100 N pretension between the two bar halves: equal and opposite axial reactions at ROOT and TIP
    assert abs(reactions["ROOT"]["force"][0]) == pytest.approx(100.0, abs=1.0)
    assert reactions["ROOT"]["force"][0] + reactions["TIP"]["force"][0] == pytest.approx(0.0, abs=1e-3)
    assert reactions["ROOT"]["count"] == 21
    # limit: isosurface and volume fraction reported
    info = scene.render(plotter, SceneState(field="sxx", frame=4, deform=0, limit=0.5))
    assert info["limit"]["isosurface_cells"] >= 0 and info["limit"]["fraction"] == pytest.approx(1.0)
    # plot groups: filtered by available fields, applied as presets
    names = plot_group_names(scene.available_fields(bundle.frames[-1]))
    assert "Mesh quality (scaled Jacobian)" in names and "Reaction forces at restrained sets" in names
    assert not any("Contact" in n for n in names)
    assert any("Contact" in n for n in plot_group_names(Scene(RunBundle(friction)).available_fields(RunBundle(friction).frames[-1])))
    state = apply_plot_group(SceneState(limit=5.0, cmap="jet"), "Reaction forces at restrained sets")
    assert state.overlays["reactions"] and state.overlays["restraints"] and state.opacity == 0.35
    assert state.limit is None and state.cmap == "turbo" and state.deform == 0.0
    with pytest.raises(KeyError):
        apply_plot_group(SceneState(), "nope")
    assert len(PLOT_GROUPS) == 14
    for name, _ in PLOT_GROUPS:
        scene.render(plotter, apply_plot_group(SceneState(frame=-1), name)) if name in names else None
    # A - B with the same run: identically zero, symmetric colour band, header mentions B
    other = Scene(RunBundle(short_screw))
    info = scene.render(plotter, SceneState(field="von_mises", frame=-1, deform=0), other=other)
    assert "difference A - B" in info["field"] and info["range"][0] == pytest.approx(-info["range"][1])
    values = scene.field_result(SceneState(field="von_mises", frame=-1), other).values
    assert np.allclose(values, 0.0)
    with pytest.raises(ValueError):
        scene.field_result(SceneState(field="von_mises"), Scene(RunBundle(friction)))
    plotter.close()


def test_render_cli_plot_group_and_minus(short_screw, tmp_path, capsys):
    out = tmp_path / "pg.png"
    assert main(["render", str(short_screw), "--out", str(out), "--plot-group", "Reaction forces at restrained sets",
                 "--size", "640,480"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["state"]["overlays"]["reactions"] is True
    assert any(o["kind"] == "reaction" for o in record["overlays"])
    out2 = tmp_path / "minus.png"
    assert main(["render", str(short_screw), "--out", str(out2), "--minus", str(short_screw), "--field", "u_mag",
                 "--size", "640,480"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert "difference A - B" in record["info"]["field"]


def test_app_plot_groups_and_minus(short_screw):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from fempost.app import ViewerWindow
    app = QApplication.instance() or QApplication([])
    window = ViewerWindow([short_screw, short_screw], offscreen=True)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    window.render_now()
    assert window.case_combo.count() == 2 and window.minus_combo.count() == 2      # same mesh hash -> B offered
    window.minus_combo.setCurrentIndex(1)
    window.render_now()
    assert "difference A - B" in window.last_info["field"]
    window.minus_combo.setCurrentIndex(0)
    results = window.tree.topLevelItem(0).child(2)
    presets = results.child(0)
    assert presets.text(0).startswith("Plot groups") and presets.childCount() >= 8
    window.apply_plot_group("Mesh quality (scaled Jacobian)")
    window.render_now()
    assert window.state.mesh_quality and window.quality_check.isChecked()
    window.apply_plot_group("Displacement magnitude")
    window.render_now()
    assert window.state.field == "u_mag" and not window.state.mesh_quality and window.deform_spin.value() > 0
    window.close()
