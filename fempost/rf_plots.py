"""Linked S-parameter plots, independent of the RF field renderer."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class _PlotToolbar(NavigationToolbar2QT):
    # Saving is routed through RFWorkspace's guarded figure export, which protects
    # the original saved-study tree and writes a reproducible view sidecar.
    toolitems = tuple(item for item in NavigationToolbar2QT.toolitems if item[3] != "save_figure")


class SParameterPanel(QtWidgets.QWidget):
    frequency_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.study = None
        self.selected = 0
        self.reference = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Magnitude (dB)", "Wrapped phase (deg)", "Unwrapped phase (deg)"])
        self.mode.currentIndexChanged.connect(self.draw)
        row.addWidget(self.mode)
        self.traces = {}
        for name in ("S11", "S21", "S12", "S22"):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(name in ("S11", "S21"))
            check.toggled.connect(self.draw)
            row.addWidget(check)
            self.traces[name] = check
        row.addStretch()
        self.note = QtWidgets.QLabel("Click a solved marker to browse its fields")
        row.addWidget(self.note)
        layout.addLayout(row)
        self.figure = Figure(figsize=(7.5, 2.5), dpi=100, constrained_layout=True)
        self.figure.set_facecolor("#f8fafc")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(185)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)
        toolbar = _PlotToolbar(self.canvas, self)
        toolbar.setMaximumHeight(30)
        layout.addWidget(toolbar)
        self.canvas.mpl_connect("button_press_event", self._click)
        self._toolbar = toolbar
        self.draw()

    def set_study(self, study, selected=0):
        self.study = study
        self.selected = selected
        self.draw()

    def select(self, selected):
        self.selected = int(selected)
        self.draw()

    def set_reference(self, study):
        """Overlay another solved study, without interpolating its frequency grid."""
        if study is not None and getattr(study, "sparameters", None) is None:
            raise ValueError("The reference study has no S parameters.")
        self.reference = study
        self.draw()

    def _values(self, s):
        mode = self.mode.currentIndex()
        if mode == 0:
            return 20 * np.log10(np.maximum(np.abs(s), 1e-15))
        angle = np.angle(s)
        return np.rad2deg(np.unwrap(angle) if mode == 2 else angle)

    def draw(self, *_):
        self.ax.clear()
        study = self.study
        if study is None or getattr(study, "sparameters", None) is None:
            self.ax.text(.5, .5, "Eigenmode studies have no port S parameters", ha="center", va="center", transform=self.ax.transAxes, color="#526377")
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return
        self.ax.set_axis_on()
        colors = {"S11": "#b43a3a", "S21": "#1265a8", "S12": "#6d4bb0", "S22": "#188571"}
        x = np.real(np.asarray(study.frequencies_hz)) / 1e9
        for name, check in self.traces.items():
            if not check.isChecked():
                continue
            i, j = int(name[1]) - 1, int(name[2]) - 1
            self.ax.plot(x, self._values(study.sparameters[:, i, j]), ".-", ms=4, lw=1.3, color=colors[name], label=name)
            if self.reference is not None:
                ref = self.reference
                self.ax.plot(np.real(np.asarray(ref.frequencies_hz)) / 1e9, self._values(ref.sparameters[:, i, j]), "--", lw=1, color=colors[name], alpha=.65, label=f"{name} · {ref.name}")
        if len(x):
            selected = min(self.selected, len(x) - 1)
            self.ax.axvline(x[selected], color="#2a3746", lw=1, alpha=.65)
            self.note.setText(f"{x[selected]:.4f} GHz · click to select solved frequency")
        # This warning applies only to the explicitly identified WR340 benchmark.
        if "wr340" in str(getattr(study, "name", "")).lower():
            cutoff = 3.471427258
            if len(x) and x.max() > cutoff:
                self.ax.axvspan(cutoff, x.max(), color="#e9be71", alpha=.15)
                self.ax.axvline(cutoff, color="#b08a47", ls=":", lw=.8)
        self.ax.set_xlabel("Solved frequency (GHz)")
        self.ax.set_ylabel(["Magnitude (dB)", "Phase (deg)", "Unwrapped phase (deg)"][self.mode.currentIndex()])
        self.ax.grid(True, color="#d8e0e7", lw=.6)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc="best", fontsize=8, ncol=min(4, len(handles)))
        self.ax.tick_params(labelsize=9)
        self.canvas.draw_idle()

    def _click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self.study is None:
            return
        if self._toolbar.mode:
            return
        x = np.real(np.asarray(self.study.frequencies_hz)) / 1e9
        if len(x):
            self.frequency_selected.emit(int(np.argmin(np.abs(x - event.xdata))))

    def export(self, path):
        self.figure.savefig(Path(path), dpi=200, facecolor=self.figure.get_facecolor())
