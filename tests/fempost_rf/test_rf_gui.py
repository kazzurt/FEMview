"""Real Qt workspace checks against the retained benchmark samples, no solver."""
import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import gmsh  # DLL load order precedes Qt/VTK in the structural runtime.
import pytest
from PySide6 import QtWidgets

from fempost.rf_app import RFWorkspace


BASE = Path(os.environ.get("FEMVIEW_RF_FIXTURES", str(Path(__file__).resolve().parent / "external/rf")))


pytestmark = pytest.mark.skipif(not (BASE / "emerge-cylindrical-cavity").is_dir() or not (BASE / "emerge-wr340-waveguide").is_dir(), reason="External RF studies unavailable")

@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_loaded(app, widget, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if not widget.is_busy and widget._selection_valid:
            return
        time.sleep(.025)
    raise AssertionError(widget.status.text())


def test_rf_workspace_samples_views_and_exports(app, tmp_path, monkeypatch):
    monkeypatch.setenv("FEMPOST_RF_CACHE", str(tmp_path / "rf-cache"))
    cavity = RFWorkspace(BASE / "emerge-cylindrical-cavity")
    wave = RFWorkspace(BASE / "emerge-wr340-waveguide")
    try:
        wait_loaded(app, cavity)
        wait_loaded(app, wave)
        assert cavity.frequency_combo.count() == 10
        assert wave.frequency_combo.count() == 41
        assert not cavity.port_combo.isVisible()
        assert wave._active_index == 9
        assert cavity._active_index == 6
        assert cavity._info["unit"] == "V/m"
        assert wave._info["unit"] == "V/m"
        assert "Display error" not in wave.status.text()
        wave.phase_spin.setValue(90)
        wave.render_timer.stop()
        wave._render()
        assert wave.state.phase_deg == 90
        wave.field_combo.setCurrentIndex(wave.field_combo.findData("Savg"))
        wave.render_timer.stop()
        wave._render()
        assert not wave.play_button.isEnabled()
        assert wave._info["unit"] == "W/m^2"
        wave._probe_numeric()
        assert "requested_m" in wave.probe_text.toPlainText()
        wave.s_panel.mode.setCurrentIndex(2)
        wave.s_panel.traces["S12"].setChecked(True)
        wave.plotter.camera.parallel_scale = .12345
        saved = wave.save_view(tmp_path / "wave-view.json")
        assert json.loads(saved.read_text())["port"] == 1
        with pytest.raises(ValueError, match="read-only"):
            wave.save_view(BASE / "emerge-wr340-waveguide" / "bad-view.json")
        wave.load_view(saved)
        wait_loaded(app, wave)
        assert wave.state.field == "Savg"
        assert wave.state.phase_deg == 90
        assert wave.plotter.camera.parallel_scale == pytest.approx(.12345)
        assert wave.s_panel.mode.currentIndex() == 2
        assert wave.s_panel.traces["S12"].isChecked()
        target = wave.export_figure(tmp_path / "wave-field.png")
        assert target.stat().st_size > 1000
        assert target.with_name("wave-field-sparameters.png").stat().st_size > 1000
        cavity.frequency_combo.setCurrentIndex(0)
        wait_loaded(app, cavity)
        assert cavity.frame.index == 0
        assert "TM010" in cavity.frequency_combo.currentText()
    finally:
        cavity.shutdown()
        wave.shutdown()
        app.processEvents()


def test_native_frequency_selection_exact_probe_and_close(app, tmp_path, monkeypatch):
    monkeypatch.setenv("FEMPOST_RF_CACHE", str(tmp_path / "rf-cache"))
    wave = RFWorkspace(BASE / "emerge-wr340-waveguide")
    try:
        wait_loaded(app, wave)
        wave.frequency_combo.setCurrentIndex(0)
        app.processEvents()
        wave.frequency_combo.setCurrentIndex(40)
        wave.port_combo.setCurrentIndex(wave.port_combo.findData(2))
        wait_loaded(app, wave, timeout=90)
        assert wave.frame.index == 40
        assert wave.frame.port == 2
        assert wave._active_index == wave.frequency_combo.currentIndex() == 40
        assert wave.s_panel.selected == 40
        wave._probe_native()
        wait_loaded(app, wave, timeout=90)
        result = json.loads(wave.probe_text.toPlainText())
        assert result["valid"]
        assert result["Savg"][2] < 0
        assert result["units"]["E"] == "V/m"
        assert "Native EMerge" in result["source"]
        # A pending native read must not leave a running QThread when a tab closes.
        wave.frequency_combo.setCurrentIndex(12)
        app.processEvents()
        wave.shutdown()
        assert not wave.is_busy
    finally:
        wave.shutdown()
        app.processEvents()
