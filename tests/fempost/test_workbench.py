"""Shared-shell regressions; the original structural viewer remains usable in a tab."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_workbench_routing(tmp_path, short_screw):
    from fempost.workbench import target_kind
    assert target_kind(short_screw) == "structural"
    project = tmp_path / "saved-rf-project"
    project.mkdir()
    (project / "study.emerge").write_bytes(b"metadata only: routing does not load this file")
    assert target_kind(project) == "rf"
    assert target_kind(project / "study.emerge") == "rf"
    view = tmp_path / "comparison.rf-view.json"
    view.write_text(json.dumps({"schema": "fempost-rf-view-v1", "study": str(project), "state": {}}))
    assert target_kind(view) == "rf_view"
    with pytest.raises(FileNotFoundError):
        target_kind(tmp_path / "missing")


def test_embedded_structural_controls(qapp, short_screw):
    from fempost.workbench import WorkbenchWindow
    from PySide6.QtCore import Qt
    shell = WorkbenchWindow(offscreen=True)
    assert shell.stack.currentWidget() is shell.welcome
    structural = shell.open_target(short_screw)
    assert shell.tabs.count() == 1
    assert structural.windowFlags() & Qt.WindowType.WindowType_Mask == Qt.WindowType.Widget
    assert structural.case_combo.count() == 1
    assert structural.frame_combo.count() == 10
    structural.render_now()
    structural.field_combo.setCurrentIndex(structural.field_combo.findData("u_mag"))
    structural.deform_spin.setValue(0.0)
    structural.render_now()
    qapp.processEvents()
    assert structural.state.field == "u_mag"
    assert structural.last_info["deformation_factor"] == 0
    assert structural.last_info["location"] == "point"
    assert structural.settings is None  # embedded layout cannot clobber standalone settings
    shell.close_tab(0)
    assert shell.tabs.count() == 0
    assert shell.stack.currentWidget() is shell.welcome
    shell.close()


def test_workbench_cli_preserves_legacy_gui():
    from fempost.__main__ import build_parser, cmd_gui, cmd_workbench
    parser = build_parser()
    assert parser.parse_args(["gui"]).func is cmd_gui
    assert parser.parse_args(["workbench"]).func is cmd_workbench


def test_shared_shell_palette(qapp):
    from PySide6.QtGui import QPalette
    from fempost.workbench import configure_workbench_application
    configure_workbench_application(qapp)
    palette = qapp.palette()
    assert qapp.style().objectName().lower() == "fusion"
    assert palette.color(QPalette.ColorRole.Window).name() == "#f4f6f8"
    assert palette.color(QPalette.ColorRole.Text).name() == "#20354a"
    assert palette.color(QPalette.ColorRole.Base).name() == "#ffffff"
