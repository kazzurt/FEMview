"""Hidden desktop smoke check: drives the viewer offscreen and writes the review screenshot set.

Run from the project root with QT_QPA_PLATFORM=offscreen (nothing appears on screen):
    .venv-structural/Scripts/pythonw.exe -B tests/fempost/smoke_viewer.py
PNGs land in structural-fea-environment/temporary/fempost-review/<timestamp>/gui/.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_API", "pyside6")

import gmsh  # noqa: F401,E402  native DLL order
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fempost.app import REVIEW_DIR, ViewerWindow  # noqa: E402
from fempost.scene import DEFAULT_OVERLAYS, SceneState  # noqa: E402

FIXTURES = ROOT / "Dodec FEA/structural-fea-benchmark-runs/fixtures"
SHORT = FIXTURES / "explicit-pretension-analytical-final/explicit-bar-verification"
FRICTION = FIXTURES / "friction-native-tetra10-vector-v1/vector-mu-010"
KURTSFEM = ROOT / "Dodec FEA/structural-fea-benchmark-runs/assembly-35mm-two-layers-adaptive"
CODEX = Path(os.environ.get("FEMVIEW_STRUCTURAL_FIXTURE", str(Path(__file__).resolve().parent / "external/structural-run")))
REFINED = Path(os.environ.get("FEMVIEW_REFINED_FIXTURE", str(Path(__file__).resolve().parent / "external/refined-run")))


def main() -> int:
    out = REVIEW_DIR / datetime.now().strftime("%Y%m%d-%H%M%S") / "gui"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QFont("Segoe UI", 10))
    runs = [p for p in (FRICTION, SHORT, KURTSFEM, CODEX, REFINED) if (p / "job.frd").exists() or any(p.glob("*.frd"))]
    window = ViewerWindow(runs, offscreen=True, settings_path=str(out / "layout.ini"))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    def shot(name: str):
        window.render_now()
        app.processEvents()
        window.window_screenshot(out / f"{name}-window.png")
        window.screenshot(out / f"{name}.png")
        print(f"{name}: {window.status_label.text()}")

    assert window.case_combo.count() == len(runs), "every run should open as a case"
    # 1. friction fixture: contact pressure with overlays, last frame of step 1
    window.case_combo.setCurrentIndex(window.case_combo.findData("vector-mu-010"))
    window.frame_combo.setCurrentIndex(9)
    window.field_combo.setCurrentIndex(window.field_combo.findData("cpress"))
    for key in ("restraints", "contact"):
        window.state.overlays[key] = True
    window._populate_tree()
    shot("01-friction-cpress-overlays")
    assert "cpress" in window.state.field and window.state.frame == 9
    # 2. short screw: element mean from the DAT, deformation x10, clip + slice
    window.case_combo.setCurrentIndex(window.case_combo.findData("explicit-bar-verification"))
    assert window.location_combo.isEnabled(), "DAT stress exists for the explicit bar"
    window.location_combo.setCurrentIndex(1)
    window.deform_spin.setValue(10.0)
    window.clip_box.setChecked(True)
    window.clip_normal.setText("0, 1, 0")
    window._clip_edited()
    window.slice_box.setChecked(True)
    window.slice_normal.setText("1, 0, 0")
    window._slice_edited()
    shot("02-short-screw-element-mean-clip-slice")
    assert window.state.location == "cell" and window.state.clip and window.state.slice
    window.clip_box.setChecked(False)
    window.slice_box.setChecked(False)
    window.location_combo.setCurrentIndex(0)
    # 3. mesh quality view and statistics tab
    window.quality_check.setChecked(True)
    window.tabs.setCurrentIndex(1)
    app.processEvents()
    shot("03-short-screw-mesh-quality-stats")
    assert window.stats_table.rowCount() >= 1
    window.quality_check.setChecked(False)
    # 4. KurtsFEM loaded step with all overlays, auto deformation, then von Mises with a limit
    if window.case_combo.findData("assembly-35mm-two-layers-adaptive") >= 0:
        window.case_combo.setCurrentIndex(window.case_combo.findData("assembly-35mm-two-layers-adaptive"))
        window.frame_combo.setCurrentIndex(window.frame_combo.count() - 1)
        window.field_combo.setCurrentIndex(window.field_combo.findData("von_mises"))
        for key in ("restraints", "loads", "contact", "pretension"):
            window.state.overlays[key] = True
        window._populate_tree()
        window.auto_deform()
        shot("04-kurtsfem-loaded-overlays")
        assert "DIAGNOSTIC" in window.status_label.text()
        for key in ("restraints", "loads", "contact", "pretension"):
            window.state.overlays[key] = False
        window.limit_spin.setValue(240.0)
        window.limit_check.setChecked(True)
        window.set_view("-y")
        shot("05-kurtsfem-von-mises-limit-240")
        assert "volume above" in window.status_label.text()
        window.limit_check.setChecked(False)
        # preload step, displacement, pipe only
        window.frame_combo.setCurrentIndex(0)
        window.field_combo.setCurrentIndex(window.field_combo.findData("u_mag"))
        window.state.bodies = ["pipe", "screw_1", "screw_2", "screw_3"]
        window._populate_tree()
        window.set_view("iso")
        shot("06-kurtsfem-preload-umag-no-hub")
        window.state.bodies = None
        # probe + scene save/load round trip
        probe = window.scene.probe(window.state, [0.0, -19.0, -100.0])
        window._picked(np.array([0.0, -19.0, -100.0]))
        assert str(probe["node_id"]) in window.probe_text.toPlainText()
        spec = out / "scene-kurtsfem.json"
        window.save_scene(spec)
        window.load_scene(spec)
        shot("07-kurtsfem-scene-reloaded")
    # 5. Codex study run
    if window.case_combo.findData("matched_mouth_coarse") >= 0:
        window.case_combo.setCurrentIndex(window.case_combo.findData("matched_mouth_coarse"))
        window.field_combo.setCurrentIndex(window.field_combo.findData("von_mises"))
        window.cmap_combo.setCurrentText("comsol_rainbow")
        window.range_lock.setChecked(True)
        window.range_min.setValue(0.0)
        window.range_max.setValue(150.0)
        window._range_edited()
        for key in ("loads", "contact"):
            window.state.overlays[key] = True
        window._populate_tree()
        shot("08-codex-von-mises-rainbow-0-150-loads-contact")
        window.range_lock.setChecked(False)
        for key in ("loads", "contact"):
            window.state.overlays[key] = False
        window.field_combo.setCurrentIndex(window.field_combo.findData("cpress"))
        window.state.bodies = ["pipe", "insert-loaded"]
        window.clip_box.setChecked(True)
        window.clip_normal.setText("0, 0, 1")
        window._clip_edited()
        shot("09-codex-cpress-pipe-insert-clip")
        window.state.bodies = None
        window.clip_box.setChecked(False)
        window.field_combo.setCurrentIndex(window.field_combo.findData("s1"))
        window.set_view("+z")
        shot("10-codex-s1-diverging")
    # 6. aluminium prototype run Kurt opened live: overlays on by default, robust range, contact blanking,
    #    text size, convergence tab, preload-versus-loaded is not applicable (one step)
    if window.case_combo.findData("own_support_refined") >= 0:
        # the sections above switched overlays through the state directly; a freshly opened viewer
        # starts from the defaults (restraints + loads on), which is what Kurt sees on launch
        window.state.overlays = dict(DEFAULT_OVERLAYS)
        window._sync_overlay_widgets()
        window.case_combo.setCurrentIndex(window.case_combo.findData("own_support_refined"))
        window.field_combo.setCurrentIndex(window.field_combo.findData("von_mises"))
        shot("11-refined-default-camera-overlays")          # as opened: restraints + loads on, robust range
        assert window.overlay_actions["restraints"].isChecked() and window.overlay_actions["loads"].isChecked()
        assert window.last_info["range_mode"] == "robust" and "DIAGNOSTIC" not in window.status_label.text()
        window.set_view("iso")
        shot("12-refined-iso-overlays")
        window.set_view("-z")
        shot("13-refined-minus-z-overlays")
        window.range_max.setValue(300.0)
        window._range_typed()
        shot("14-refined-typed-range-0-300")
        assert window.range_lock.isChecked() and window.state.clim[1] == 300.0
        window.range_lock.setChecked(False)
        window.set_view("iso")
        window.field_combo.setCurrentIndex(window.field_combo.findData("cpress"))
        window.overlay_actions["contact"].setChecked(True)
        shot("15-refined-cpress-blanked-contact-overlay")
        window.overlay_actions["contact"].setChecked(False)
        window.field_combo.setCurrentIndex(window.field_combo.findData("von_mises"))
        window.text_spin.setValue(1.6)
        shot("16-refined-text-scale-1p6")
        window.text_spin.setValue(1.0)
        conv = [i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "Convergence"][0]
        window.tabs.setCurrentIndex(conv)
        app.processEvents()
        shot("17-refined-convergence-tab")
        assert window.conv_table.rowCount() == 1
        window.tabs.setCurrentIndex(0)
    # KurtsFEM two-step run: loaded minus preload preset, convergence tab
    if window.case_combo.findData("assembly-35mm-two-layers-adaptive") >= 0:
        window.case_combo.setCurrentIndex(window.case_combo.findData("assembly-35mm-two-layers-adaptive"))
        window.apply_plot_group("Loaded minus preload: von Mises (step 2 - step 1, same case)")
        shot("18-kurtsfem-loaded-minus-preload")
        assert window.state.minus_frame is not None
        window.minus_frame_combo.setCurrentIndex(0)
        conv = [i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "Convergence"][0]
        window.tabs.setCurrentIndex(conv)
        app.processEvents()
        shot("19-kurtsfem-convergence-tab")
        assert window.conv_table.rowCount() == 2
        window.tabs.setCurrentIndex(0)
    (out / "summary.txt").write_text("\n".join(window.log_lines), encoding="utf-8")
    print("PASS: cases, frames, fields, element data, clip/slice, quality stats, overlays, limit, probe, scene spec")
    print("screenshots:", out)
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
