"""Offscreen Qt viewer checks (no window on screen): controls edit the state and render."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _window(qapp, runs):
    from PySide6.QtCore import Qt
    from fempost.app import ViewerWindow
    window = ViewerWindow(runs, offscreen=True)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    qapp.processEvents()
    window.render_now()
    return window


def test_viewer_controls(qapp, short_screw, friction, tmp_path):
    window = _window(qapp, [short_screw, friction])
    assert window.case_combo.count() == 2
    assert window.frame_combo.count() == 60                     # friction is the current case
    assert window.state.case == "vector-mu-010"
    assert window.field_combo.findData("cpress") >= 0
    window.frame_combo.setCurrentIndex(9)
    window.field_combo.setCurrentIndex(window.field_combo.findData("gap"))
    window.render_now()
    assert window.state.frame == 9 and window.state.field == "gap"
    assert window.last_info["frame"].startswith("Step 1 · inc 10")
    # switch case: frames and fields repopulate, DAT element data becomes available
    window.case_combo.setCurrentIndex(window.case_combo.findData("explicit-bar-verification"))
    window.render_now()
    assert window.frame_combo.count() == 10 and window.location_combo.isEnabled()
    assert window.field_combo.findData("cpress") < 0
    window.location_combo.setCurrentIndex(1)
    window.render_now()
    assert window.last_info["location"] == "cell"
    # deformation factor: auto fills the spin box, typing sets the factor
    window.auto_deform()
    window.render_now()
    assert window.deform_spin.value() == pytest.approx(window.last_info["deformation_factor"])
    window.deform_spin.setValue(10.0)
    window.render_now()
    assert window.last_info["deformation_factor"] == 10.0
    # range lock, log, colormap
    window.range_lock.setChecked(True)
    window.range_min.setValue(0.0)
    window.range_max.setValue(2.0)
    window._range_edited()
    window.render_now()
    assert window.last_info["range"] == [0.0, 2.0]
    window.range_lock.setChecked(False)
    window.cmap_combo.setCurrentText("comsol_rainbow")
    window.log_check.setChecked(True)
    window.render_now()
    assert window.state.cmap == "comsol_rainbow" and window.state.log_scale
    window.log_check.setChecked(False)
    # clip and slice from the property panel
    window.clip_box.setChecked(True)
    window.clip_normal.setText("0, 1, 0")
    window._clip_edited()
    window.slice_box.setChecked(True)
    window._slice_edited()
    window.render_now()
    assert window.last_info["clip"]["cells"] > 0 and window.last_info["slice"]["cells"] > 0
    window.clip_box.setChecked(False)
    window.slice_box.setChecked(False)
    # overlays through the tree
    from PySide6.QtCore import Qt
    defs = window.tree.topLevelItem(0).child(1)
    assert defs.text(0) == "Definitions"
    restraints = defs.child(0)
    restraints.setCheckState(0, Qt.CheckState.Checked)
    window.render_now()
    assert any(o["kind"] == "restraint" for o in window.last_info["overlays"])
    assert restraints.childCount() == 5
    # body visibility through the tree
    mesh_item = window.tree.topLevelItem(0).child(0)
    body = mesh_item.child(0)
    assert body.text(0) == "bolt"
    body.setCheckState(0, Qt.CheckState.Unchecked)
    assert window.state.bodies == []
    body.setCheckState(0, Qt.CheckState.Checked)
    assert window.state.bodies is None
    # mesh quality + statistics tab
    window.quality_check.setChecked(True)
    window.render_now()
    assert window.last_info["location"] == "cell" and "Jacobian" in window.last_info["field"]
    window.tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert window.stats_table.rowCount() == 1 and "scaled_jacobian" in window.stats_text.toPlainText()
    window.quality_check.setChecked(False)
    # probe, screenshot, scene round trip
    window._picked(np.array([0.0, 5.0, 5.0]))
    assert "nearest node" in window.probe_text.toPlainText()
    png = window.screenshot(tmp_path / "shot.png")
    assert png.stat().st_size > 1000
    spec = tmp_path / "scene.json"
    window.state.view = "+x"
    window.save_scene(spec)
    window.load_scene(spec)
    assert window.state.view == "+x" and window.state.camera is not None
    summary = window.summary_text.toPlainText()
    assert "Restraints" in summary and "converged (2 steps, 10 increments" in summary and "diagnostic" not in summary.lower()
    window.close()


def test_run_app_offscreen_screenshot(short_screw, tmp_path):
    from fempost.app import run_app
    target = tmp_path / "view.png"
    assert run_app([str(short_screw)], offscreen=True, screenshot=str(target)) == 0
    window_png = target.with_name("view-window.png")
    assert target.exists() and window_png.exists()
    assert target.stat().st_size != window_png.stat().st_size      # the view is a real render, not the window grab
    import pyvista as pv
    image = pv.read(str(target))
    assert image.dimensions[0] >= 800 and image.dimensions[1] >= 600


def test_cli_gui_offscreen(short_screw, tmp_path):
    from fempost.__main__ import main
    target = tmp_path / "cli.png"
    assert main(["gui", str(short_screw), "--offscreen", "--screenshot", str(target)]) == 0
    assert target.exists()
