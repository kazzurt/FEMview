"""RF workspace for fempost: saved complex fields, linked plots, and reproducible views.

Qt and VTK live in the structural environment. Native EMerge interpolation is isolated
by rf_data in the existing RF environment; this module never starts a solve.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from .rf_data import RFStudy
from .rf_scene import RFScene, RFViewState, eigenmode_power_note
from .rf_plots import SParameterPanel


_FINISHING_WORKERS = set()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return str(value)


class _SampleWorker(QtCore.QThread):
    ready = QtCore.Signal(object)
    probed = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, generation, study_or_path, index, port, point=None):
        super().__init__()
        self.generation = generation
        self.study_or_path = study_or_path
        self.index = index
        self.port = port
        self.point = point
        self.study = None

    def run(self):
        try:
            study = self.study_or_path if hasattr(self.study_or_path, "sample") else RFStudy(self.study_or_path)
            self.study = study
            if self.isInterruptionRequested():
                study.close()
                return
            index = self.index
            if index is None:
                # Match the validated studies' useful opening views.
                if study.kind == "eigenmode":
                    modes = study.metadata.get("modes", [])
                    index = next((i for i, mode in enumerate(modes[:len(study.frequencies_hz)])
                                  if mode.get("analytic_family") == "TE112"),
                                 min(6, len(study.frequencies_hz) - 1))
                else:
                    index = int(np.argmin(np.abs(np.real(study.frequencies_hz) - 2.45e9)))
            if self.point is not None:
                result = study.probe(int(index), self.point, port=int(self.port))
                if not self.isInterruptionRequested():
                    self.probed.emit((self.generation, result, int(index), int(self.port)))
                return
            frame = study.sample(int(index), port=int(self.port))
            if self.isInterruptionRequested():
                study.close()
                return
            self.ready.emit((self.generation, study, frame, int(index), self.port))
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit((self.generation, f"{type(exc).__name__}: {exc}", traceback.format_exc()))


class RFWorkspace(QtWidgets.QWidget):
    """One RF study and view. Multiple workbench tabs provide independent comparisons."""
    title_changed = QtCore.Signal(str)
    frame_ready = QtCore.Signal()

    def __init__(self, path=None, parent=None):
        super().__init__(parent)
        self.study = None
        self.frame = None
        self.state = RFViewState()
        self._generation = 0
        self._worker = None
        self._pending = None
        self._closing = False
        self._building = True
        self._loaded_view = None
        self._info = {}
        self._active_index = 0
        self._active_port = 1
        self._selection_valid = False
        self._build()
        self._building = False
        self.animation = QtCore.QTimer(self)
        self.animation.setInterval(90)
        self.animation.timeout.connect(self._animate)
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(35)
        self.render_timer.timeout.connect(self._render)
        if path:
            self.open_study(path)

    @property
    def is_busy(self):
        return self._worker is not None

    def _combo(self, entries, changed=None):
        widget = QtWidgets.QComboBox()
        widget.setMinimumContentsLength(8)
        widget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for label, value in entries:
            widget.addItem(label, value)
        if changed:
            widget.currentIndexChanged.connect(changed)
        return widget

    def _check(self, label, checked=False):
        widget = QtWidgets.QCheckBox(label)
        widget.setChecked(checked)
        widget.toggled.connect(self._controls_changed)
        return widget

    def _number(self, low, high, value, decimals=4, step=.1):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.setMinimumWidth(72)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        return widget

    def _button(self, label, callback):
        widget = QtWidgets.QPushButton(label)
        widget.clicked.connect(callback)
        return widget

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 5)
        outer.setSpacing(7)
        header = QtWidgets.QHBoxLayout()
        self.study_title = QtWidgets.QLabel("RF study")
        self.study_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #18334a;")
        header.addWidget(self.study_title, 1)
        header.addWidget(self._button("Save view…", self._save_dialog))
        header.addWidget(self._button("Load view…", self._load_dialog))
        header.addWidget(self._button("Export figure…", self._export_dialog))
        outer.addLayout(header)

        selection = QtWidgets.QHBoxLayout()
        self.selection_label = QtWidgets.QLabel("Solution")
        selection.addWidget(self.selection_label)
        self.previous_button = self._button("‹", lambda: self._step(-1))
        self.previous_button.setMaximumWidth(30)
        selection.addWidget(self.previous_button)
        self.frequency_combo = QtWidgets.QComboBox()
        self.frequency_combo.setMinimumContentsLength(30)
        self.frequency_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.frequency_combo.currentIndexChanged.connect(self._selection_changed)
        selection.addWidget(self.frequency_combo, 1)
        self.next_button = self._button("›", lambda: self._step(1))
        self.next_button.setMaximumWidth(30)
        selection.addWidget(self.next_button)
        self.port_label = QtWidgets.QLabel("Independent excitation")
        selection.addWidget(self.port_label)
        self.port_combo = self._combo([("Port 1 · 1 W", 1), ("Port 2 · 1 W", 2)], self._selection_changed)
        selection.addWidget(self.port_combo)
        outer.addLayout(selection)

        self.normalization = QtWidgets.QLabel("Open a saved RF study from the workbench File menu.")
        self.normalization.setWordWrap(True)
        self.normalization.setStyleSheet("color: #4d6072;")
        outer.addWidget(self.normalization)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(self.main_splitter, 1)
        controls = QtWidgets.QTabWidget()
        controls.setMinimumWidth(340)
        controls.setMaximumWidth(400)
        self.controls = controls
        self.main_splitter.addWidget(controls)
        field_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(field_panel)
        form.setContentsMargins(10, 12, 10, 8)
        self.field_combo = self._combo([("Electric E", "E"), ("Magnetic H", "H"), ("Surface current Js", "Js"), ("Average Poynting", "Savg")], self._controls_changed)
        form.addRow("Field", self.field_combo)
        self.component_combo = self._combo([("Vector magnitude", "magnitude"), ("x component", "x"), ("y component", "y"), ("z component", "z")], self._controls_changed)
        form.addRow("Component", self.component_combo)
        self.representation_combo = self._combo([("Instantaneous field", "real"), ("Phasor amplitude (static)", "magnitude"), ("Imaginary quadrature", "imag"), ("Component phase", "phase")], self._controls_changed)
        form.addRow("Display", self.representation_combo)
        phase_row = QtWidgets.QHBoxLayout()
        self.phase_spin = self._number(-3600, 3600, 0, decimals=1, step=15)
        self.phase_spin.setSuffix(" °")
        self.phase_spin.valueChanged.connect(self._phase_changed)
        phase_row.addWidget(self.phase_spin, 1)
        self.play_button = QtWidgets.QPushButton("Play phase")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._play)
        phase_row.addWidget(self.play_button)
        form.addRow("RF phase", phase_row)
        self.phase_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.phase_slider.setRange(0, 359)
        self.phase_slider.valueChanged.connect(self.phase_spin.setValue)
        form.addRow(self.phase_slider)
        self.phase_note = QtWidgets.QLabel("Instantaneous colors and arrows follow the same RF phase. Play switches static phasor amplitude to instantaneous field. Average power is phase independent.")
        self.phase_note.setWordWrap(True)
        self.phase_note.setStyleSheet("color: #5d6a77; font-size: 11px;")
        form.addRow(self.phase_note)
        form.addRow(self._separator())
        self.vectors_check = self._check("Vectors: instantaneous real field", False)
        form.addRow(self.vectors_check)
        self.vector_density = QtWidgets.QSpinBox()
        self.vector_density.setRange(5, 41)
        self.vector_density.setValue(15)
        self.vector_density.setSingleStep(2)
        self.vector_density.setToolTip("Grid sites across the longest cut-plane span (5–41). Equal spacing in both directions, clipped to the cavity. Near-zero arrows are hidden. Also adjusts surface-arrow count.")
        self.vector_density.valueChanged.connect(self._controls_changed)
        form.addRow("Arrow density", self.vector_density)
        self.vector_scale = self._number(.1, 5, 1, decimals=2, step=.1)
        self.vector_scale.valueChanged.connect(self._controls_changed)
        form.addRow("Arrow size ×", self.vector_scale)
        self.cmap_combo = self._combo([("Automatic by field", "auto")] + [(s, s) for s in ("magma", "viridis", "copper", "inferno", "turbo", "coolwarm", "plasma", "cividis", "twilight_shifted")], self._controls_changed)
        form.addRow("Colormap", self.cmap_combo)
        self.range_combo = self._combo([("Full data range", "full"), ("Robust 0.5–99.5%", "robust"), ("Manual / locked", "manual")], self._controls_changed)
        form.addRow("Color range", self.range_combo)
        range_row = QtWidgets.QHBoxLayout()
        self.range_min = self._number(-1e30, 1e30, 0, decimals=8, step=.1)
        self.range_max = self._number(-1e30, 1e30, 1, decimals=8, step=.1)
        self.range_min.valueChanged.connect(self._manual_range)
        self.range_max.valueChanged.connect(self._manual_range)
        range_row.addWidget(self.range_min)
        range_row.addWidget(QtWidgets.QLabel("to"))
        range_row.addWidget(self.range_max)
        form.addRow(range_row)
        self.log_check = self._check("Logarithmic positive magnitude")
        form.addRow(self.log_check)
        self.range_info = QtWidgets.QLabel("")
        self.range_info.setWordWrap(True)
        self.range_info.setStyleSheet("color: #5d6a77; font-size: 11px;")
        form.addRow(self.range_info)
        form.addRow(self._button("Lock current color range", self._lock_range))
        comparison_note = QtWidgets.QLabel("Separate tabs retain independent views for comparison.")
        comparison_note.setWordWrap(True)
        form.addRow(comparison_note)
        self._scroll_tab(controls, field_panel, "Fields")

        geometry_panel = QtWidgets.QWidget()
        geometry = QtWidgets.QFormLayout(geometry_panel)
        self.display_combo = self._combo([("Three orthogonal slices", "slices"), ("Single movable slice", "slice"), ("Clipped geometry", "clip"), ("Surface", "surface")], self._controls_changed)
        geometry.addRow("Geometry", self.display_combo)
        self.axis_combo = self._combo([("x", "x"), ("y", "y"), ("z", "z")], self._controls_changed)
        geometry.addRow("Plane normal", self.axis_combo)
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(500)
        self.position_slider.valueChanged.connect(self._controls_changed)
        geometry.addRow("Plane position", self.position_slider)
        self.position_label = QtWidgets.QLabel("50% of sampled extent")
        geometry.addRow(self.position_label)
        self.invert_check = self._check("Invert clip side")
        geometry.addRow(self.invert_check)
        self.mesh_check = self._check("Overlay actual FEM mesh")
        geometry.addRow(self.mesh_check)
        self.smooth_check = self._check("Smooth circular slice edge (visual only)", True)
        geometry.addRow(self.smooth_check)
        self.opacity_spin = self._number(.05, 1, 1, decimals=2, step=.05)
        self.opacity_spin.valueChanged.connect(self._controls_changed)
        geometry.addRow("Opacity", self.opacity_spin)
        self.parallel_check = self._check("Parallel projection", True)
        geometry.addRow(self.parallel_check)
        geometry.addRow(self._separator())
        view_row = QtWidgets.QHBoxLayout()
        for title, view in (("Iso", "iso"), ("+X", "+x"), ("+Y", "+y"), ("+Z", "+z")):
            view_row.addWidget(self._button(title, lambda checked=False, name=view: self.set_view(name)))
        geometry.addRow(view_row)
        geometry.addRow(self._button("Fit view", self._fit))
        note = QtWidgets.QLabel("Coordinates are meters. The Cartesian grid contains interpolated complex fields; its cells and edges are visualization samples, not the FEM mesh.")
        note.setWordWrap(True)
        geometry.addRow(note)
        self._scroll_tab(controls, geometry_panel, "Geometry")

        probe_panel = QtWidgets.QWidget()
        probe_layout = QtWidgets.QVBoxLayout(probe_panel)
        probe_form = QtWidgets.QFormLayout()
        self.probe_xyz = []
        for axis in "xyz":
            spin = self._number(-1e6, 1e6, 0, decimals=7, step=.001)
            spin.setSuffix(" m")
            self.probe_xyz.append(spin)
            probe_form.addRow(axis, spin)
        probe_layout.addLayout(probe_form)
        probe_layout.addWidget(self._button("Probe nearest sample", self._probe_numeric))
        self.exact_probe_button = self._button("Evaluate native field at exact point", self._probe_native)
        self.exact_probe_button.setToolTip("Uses EMerge's edge-element interpolator on the saved solution. No new solve; first use can take several seconds.")
        probe_layout.addWidget(self.exact_probe_button)
        probe_note = QtWidgets.QLabel("Press P over a displayed surface to probe a sample. Values retain complex phase; probes are visualization-grid samples, not element-integral quantities.")
        probe_note.setWordWrap(True)
        probe_layout.addWidget(probe_note)
        self.probe_text = QtWidgets.QPlainTextEdit()
        self.probe_text.setReadOnly(True)
        probe_layout.addWidget(self.probe_text, 1)
        controls.addTab(probe_panel, "Probe")
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        controls.addTab(self.details, "Study")

        self.view_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.main_splitter.addWidget(self.view_splitter)
        self.plotter = QtInteractor(self)
        self.plotter.setMinimumSize(450, 300)
        self.plotter.set_background("#f3f6f9")
        self.scene = RFScene(self.plotter)
        self.view_splitter.addWidget(self.plotter.interactor)
        self.s_panel = SParameterPanel(self)
        self.s_panel.frequency_selected.connect(self.frequency_combo.setCurrentIndex)
        self.view_splitter.addWidget(self.s_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([340, 1050])
        self.view_splitter.setSizes([610, 260])
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #43586a; font-size: 11px;")
        outer.addWidget(self.status)
        try:
            self.plotter.enable_point_picking(callback=self._probe_point, show_message=False, show_point=True, use_picker=False, picker="cell", left_clicking=False)
        except Exception:
            # Numeric probes remain available even when a platform has no picker yet.
            pass
        self._set_selection_enabled(False)

    @staticmethod
    def _separator():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #d9e1e8;")
        return line

    @staticmethod
    def _scroll_tab(tabs, widget, title):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        tabs.addTab(scroll, title)

    def _set_selection_enabled(self, enabled):
        for widget in (self.frequency_combo, self.previous_button, self.next_button, self.port_combo):
            widget.setEnabled(enabled)

    def open_study(self, path):
        self._generation += 1
        self._selection_valid = False
        self._set_selection_enabled(False)
        self.play_button.setChecked(False)
        self.study_title.setText(Path(path).name)
        self.status.setText("Opening saved RF study and loading complex fields…")
        self._submit((self._generation, path, None, 1))

    def _submit(self, job):
        self._pending = job
        if self._worker is None:
            self._start_pending()

    def _start_pending(self):
        if self._closing or self._pending is None:
            return
        job, self._pending = self._pending, None
        self._worker = _SampleWorker(*job)
        self._worker.ready.connect(self._loaded)
        self._worker.probed.connect(self._native_probed)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    @QtCore.Slot()
    def _worker_finished(self):
        worker, self._worker = self._worker, None
        if worker:
            worker.deleteLater()
        self.exact_probe_button.setEnabled(True)
        self._start_pending()

    @QtCore.Slot(object)
    def _failed(self, result):
        generation, message, trace = result
        if generation != self._generation or self._closing:
            return
        self.status.setText("Could not load saved fields — " + message)
        if self._worker is not None and self._worker.point is not None:
            self.probe_text.setPlainText("Native point evaluation failed: " + message)
        self.details.setPlainText(trace)
        self._set_selection_enabled(self.study is not None)

    @QtCore.Slot(object)
    def _loaded(self, result):
        generation, study, frame, index, port = result
        if generation != self._generation or self._closing:
            if study is not self.study:
                study.close()
            return
        if self._pending is not None and self._pending[0] == generation:
            # The selector changed while interpolation was running. Never label stale fields
            # with a newer requested frequency; display only the latest completed selection.
            if study is not self.study:
                study.close()
            return
        first = self.study is not study
        previous_study = self.study
        self.study = study
        if first and previous_study is not None:
            previous_study.close()
        self.frame = frame
        self._active_index, self._active_port = index, int(port)
        self._building = True
        if first:
            self.frequency_combo.clear()
            for i, frequency in enumerate(study.frequencies_hz):
                label = study.labels[i] if i < len(study.labels) else f"{float(np.real(frequency))/1e9:.3f} GHz"
                self.frequency_combo.addItem(str(label), i)
            self.study_title.setText(study.name)
            self.title_changed.emit(study.name)
            self.selection_label.setText("Eigenmode" if study.kind == "eigenmode" else "Solved frequency")
            driven = study.kind == "driven"
            self.port_combo.clear()
            verified_power = study.metadata.get("incident_power_w") == 1.0
            for port_id in getattr(study, "port_ids", [1, 2]):
                self.port_combo.addItem(f"Port {port_id}" + (" · 1 W" if verified_power else " · stored amplitude"), port_id)
            self.port_label.setVisible(driven)
            self.port_combo.setVisible(driven)
            self.s_panel.setVisible(driven)
            self.normalization.setText(study.normalization)
            details = {"name": study.name, "kind": study.kind, "source": str(study.path), "normalization": study.normalization, "metadata": study.metadata}
            self.details.setPlainText(json.dumps(details, indent=2, default=_json_default))
            if self._loaded_view is None:
                self.state = RFViewState()
                if driven:
                    self.state.display = "slice"
                    self.state.axis = "y"
                    self.state.component = "y"
                    self.state.representation = "real"
                    self.state.cmap = "coolwarm"
                else:
                    # The benchmark opens on TE112: a quarter-length section
                    # exposes its electric antinode and useful vector arrows.
                    self.state.position = .25
            for spin, axis in zip(self.probe_xyz, frame.axes):
                spin.setValue(float((axis[0] + axis[-1]) / 2))
            self.s_panel.set_study(study, index)
        self.frequency_combo.setCurrentIndex(index)
        current_item = self.field_combo.model().item(self.field_combo.findData("Js"))
        current_item.setEnabled(frame.surface_current is not None)
        current_item.setToolTip("Native conductor sheet current (A/m)" if frame.surface_current is not None else "Requires saved native fields and verified closed conductor walls")
        self.port_combo.setCurrentIndex(self.port_combo.findData(int(port)))
        self._set_selection_enabled(True)
        self._selection_valid = True
        if self._loaded_view is not None:
            record = self._loaded_view
            self._loaded_view = None
            accepted = {field.name for field in fields(RFViewState)}
            self.state = RFViewState(**{k: v for k, v in record.get("state", {}).items() if k in accepted})
            s_view = record.get("sparameter_view", {})
            self.s_panel.mode.setCurrentIndex(int(s_view.get("mode", 0)))
            for name, checked in s_view.get("traces", {}).items():
                if name in self.s_panel.traces:
                    self.s_panel.traces[name].setChecked(bool(checked))
        self._sync_controls()
        self._building = False
        self._update_control_availability()
        self.s_panel.select(index)
        self._render(preserve_camera=not first)
        self.frame_ready.emit()

    def _step(self, offset):
        if self.frequency_combo.count():
            index = max(0, min(self.frequency_combo.count() - 1, self.frequency_combo.currentIndex() + offset))
            self.frequency_combo.setCurrentIndex(index)

    def _selection_changed(self, *_):
        if self._building or self.study is None:
            return
        index = self.frequency_combo.currentIndex()
        if index < 0:
            return
        self.play_button.setChecked(False)
        self._selection_valid = False
        self.status.setText(f"Loading saved fields for {self.frequency_combo.currentText()}… Previous field remains visible until loading completes.")
        self._submit((self._generation, self.study, index, int(self.port_combo.currentData() or 1)))

    def _sync_controls(self):
        state = self.state
        for combo, value in ((self.field_combo, state.field), (self.component_combo, state.component), (self.representation_combo, state.representation), (self.display_combo, state.display), (self.axis_combo, state.axis), (self.cmap_combo, state.cmap), (self.range_combo, state.range_mode)):
            found = combo.findData(value)
            if found >= 0:
                combo.setCurrentIndex(found)
        self.phase_spin.setValue(state.phase_deg)
        self.phase_slider.setValue(round(state.phase_deg) % 360)
        self.position_slider.setValue(round(state.position * 1000))
        for check, value in ((self.vectors_check, state.vectors), (self.invert_check, state.invert), (self.mesh_check, state.mesh), (self.parallel_check, state.parallel), (self.log_check, state.log_scale)):
            check.setChecked(bool(value))
        self.opacity_spin.setValue(state.opacity)
        self.vector_scale.setValue(state.vector_scale)
        self.vector_density.setValue(state.vector_density)
        self.smooth_check.setChecked(state.smooth_boundary)
        if state.clim is not None:
            self.range_min.setValue(state.clim[0])
            self.range_max.setValue(state.clim[1])

    def _phase_changed(self, value):
        blocker = QtCore.QSignalBlocker(self.phase_slider)
        self.phase_slider.setValue(round(value) % 360)
        del blocker
        self._controls_changed()

    def _manual_range(self, *_):
        if self._building:
            return
        self.range_combo.setCurrentIndex(self.range_combo.findData("manual"))
        self._controls_changed()

    def _controls_changed(self, *_):
        if self._building:
            return
        state = self.state
        previous_field = state.field
        state.field = self.field_combo.currentData()
        state.component = self.component_combo.currentData()
        state.representation = self.representation_combo.currentData()
        state.phase_deg = self.phase_spin.value()
        state.display = self.display_combo.currentData()
        state.axis = self.axis_combo.currentData()
        state.position = self.position_slider.value() / 1000
        state.invert = self.invert_check.isChecked()
        state.vectors = self.vectors_check.isChecked()
        state.vector_scale = self.vector_scale.value()
        state.vector_density = self.vector_density.value()
        state.cmap = self.cmap_combo.currentData()
        state.range_mode = self.range_combo.currentData()
        state.clim = [self.range_min.value(), self.range_max.value()] if state.range_mode == "manual" else None
        state.log_scale = self.log_check.isChecked()
        state.opacity = self.opacity_spin.value()
        state.mesh = self.mesh_check.isChecked()
        state.parallel = self.parallel_check.isChecked()
        state.smooth_boundary = self.smooth_check.isChecked()
        if state.field == "Js" and (previous_field != "Js" or state.display in {"slice", "slices"}):
            state.display = "surface"
            blocker = QtCore.QSignalBlocker(self.display_combo)
            self.display_combo.setCurrentIndex(self.display_combo.findData("surface"))
            del blocker
        elif previous_field == "Js" and state.field != "Js":
            state.display = "slices"
            blocker = QtCore.QSignalBlocker(self.display_combo)
            self.display_combo.setCurrentIndex(self.display_combo.findData("slices"))
            del blocker
        self._update_control_availability()
        if state.representation == "phase" and state.component == "magnitude":
            # A vector has no single complex phase. Choose a stated component.
            blocker = QtCore.QSignalBlocker(self.component_combo)
            self.component_combo.setCurrentIndex(self.component_combo.findData("x"))
            del blocker
            state.component = "x"
        positive = state.representation == "magnitude" or state.component == "magnitude"
        self.log_check.setEnabled(positive and state.representation != "phase")
        if not self.log_check.isEnabled():
            state.log_scale = False
        if self.frame is not None:
            axis = self.frame.axes["xyz".index(state.axis)]
            position = float(axis[0] + (axis[-1] - axis[0]) * state.position)
            self.position_label.setText(f"{state.axis} = {position:.6g} m · {100 * state.position:.1f}%")
        self.render_timer.start()

    def _update_control_availability(self):
        state = self.state
        currents = state.field == "Js"
        self.opacity_spin.setEnabled(not currents)
        if currents:
            state.opacity = 1.0
            blocker = QtCore.QSignalBlocker(self.opacity_spin)
            self.opacity_spin.setValue(1.0)
            del blocker
        for value in ("slice", "slices"):
            self.display_combo.model().item(self.display_combo.findData(value)).setEnabled(not currents)
        self.smooth_check.setEnabled(not currents and getattr(self.frame, "cylinder_geometry", None) is not None)
        self.axis_combo.setEnabled(not currents or state.display == "clip")
        self.position_slider.setEnabled(not currents or state.display == "clip")
        if state.field == "Savg":
            # Mean power is real and phase independent. Do not offer fictitious complex
            # mean-power components or imply that eigenmode residuals are physical flow.
            state.representation = "magnitude" if state.component == "magnitude" else "real"
            self.representation_combo.setEnabled(False)
            self.phase_spin.setEnabled(False)
            self.phase_slider.setEnabled(False)
            self.play_button.setEnabled(False)
            self.play_button.setChecked(False)
            self.vectors_check.setText("Vectors: time-average power flow")
        else:
            self.representation_combo.setEnabled(True)
            self.phase_spin.setEnabled(True)
            self.phase_slider.setEnabled(True)
            self.play_button.setEnabled(True)
            self.vectors_check.setText("Vectors: imaginary quadrature" if state.representation == "imag" else "Vectors: instantaneous field")

    def _render(self, preserve_camera=True):
        if self.frame is None or self._closing:
            return
        if preserve_camera and getattr(self.scene, "frame", None) is not None:
            self._capture_camera()
        try:
            self._info = self.scene.render(self.frame, self.state) or {}
            tab_title = f"{self.study.name} · {self.state.field}"
            if tab_title != getattr(self, "_last_tab_title", None):
                self._last_tab_title = tab_title
                self.title_changed.emit(tab_title)
            shown = self._info.get("shown_range", self._info.get("clim"))
            true = self._info.get("true_range")
            if shown is not None:
                self.range_info.setText(f"Shown: {shown[0]:.5g} … {shown[1]:.5g}" + (f"\nTrue sampled extrema: {true[0]:.5g} … {true[1]:.5g}" if true is not None else ""))
            kind = getattr(self.study, "kind", "")
            caveat = eigenmode_power_note(self.frame) if kind == "eigenmode" else getattr(self.frame, "normalization", self.study.normalization)
            if kind == "driven" and "wr340" in self.study.name.lower():
                caveat += " Above 3.4714 GHz, this WR340 model still terminates TE10 only."
            self.status.setText(f"{self.study.labels[self._active_index]} · {getattr(self.frame, 'source', 'saved fields')} · {caveat}")
            if self._info.get("warnings"):
                self.status.setText(self.status.text() + " " + " ".join(self._info["warnings"]))
        except Exception as exc:
            self.status.setText(f"Display error: {type(exc).__name__}: {exc}")
            self.details.appendPlainText(traceback.format_exc())

    def _lock_range(self):
        limits = self._info.get("shown_range", self._info.get("clim"))
        if limits is None:
            return
        self._building = True
        self.range_min.setValue(float(limits[0]))
        self.range_max.setValue(float(limits[1]))
        self.range_combo.setCurrentIndex(self.range_combo.findData("manual"))
        self._building = False
        self._controls_changed()

    def _play(self, checked):
        if checked and self.frame is not None and self.state.field != "Savg":
            if self.state.representation == "magnitude":
                self.representation_combo.setCurrentIndex(self.representation_combo.findData("real"))
            self.animation.start()
            self.play_button.setText("Pause")
        else:
            if hasattr(self, "animation"):
                self.animation.stop()
            self.play_button.setText("Play phase")

    def _animate(self):
        self.phase_spin.setValue((self.phase_spin.value() + 15) % 360)

    def set_view(self, name):
        self.scene.set_view(name)
        self._capture_camera()
        self.plotter.render()

    def _fit(self):
        self.plotter.reset_camera()
        self._capture_camera()
        self.plotter.render()

    def _probe_numeric(self):
        self._probe_point([spin.value() for spin in self.probe_xyz])

    def _probe_native(self):
        if self.study is None or self.frame is None or not self._selection_valid:
            self.probe_text.setPlainText("Wait for the selected field to finish loading.")
            return
        point = [spin.value() for spin in self.probe_xyz]
        self.exact_probe_button.setEnabled(False)
        self.probe_text.setPlainText("Evaluating the saved edge-element solution at the exact point…")
        self._submit((self._generation, self.study, self._active_index, self._active_port, point))

    @QtCore.Slot(object)
    def _native_probed(self, response):
        generation, result, index, port = response
        if self._closing or generation != self._generation or not self._selection_valid or index != self._active_index or port != self._active_port:
            return
        from .rf_scene import eigen_gauge
        result["units"] = {"E": "V/m", "H": "A/m", "Savg": "W/m^2", "point_m": "m"}
        result["display_common_gauge_deg"] = float(np.angle(eigen_gauge(self.frame.E), deg=True)) if self.study.kind == "eigenmode" else 0.0
        result["display_phase_deg"] = self.state.phase_deg
        result["phase_note"] = "E and H below are raw saved complex phasors. The display adds the common gauge and RF phase to both."
        self.probe_text.setPlainText(json.dumps(result, indent=2, default=_json_default))

    def _probe_point(self, point):
        if self.frame is None:
            return
        try:
            result = self.scene.probe(point)
            self.probe_text.setPlainText(json.dumps(result, indent=2, default=_json_default))
            self.controls.setCurrentIndex(2)
        except Exception as exc:
            self.probe_text.setPlainText(str(exc))

    def _default_folder(self):
        folder = Path(__file__).resolve().parents[1] / "structural-fea-environment" / "temporary" / "fempost-rf-views"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _output_path(self, path):
        path = Path(path).expanduser().resolve()
        if self.study is not None:
            source = Path(self.study.path).resolve()
            if source.is_file():
                source = source.parent
            for parent in [source, *source.parents]:
                if parent.name.lower() == "results":
                    source = parent.parent
                    break
            if path.is_relative_to(source):
                raise ValueError("Choose a destination outside the original RF study. Saved results are read-only.")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _capture_camera(self):
        self.state.camera = [list(item) for item in self.plotter.camera_position]
        self.state.parallel_scale = float(self.plotter.camera.parallel_scale)
        self.state.view_angle = float(self.plotter.camera.view_angle)

    def view_record(self):
        if self.study is None or self.frame is None or not self._selection_valid:
            raise ValueError("Wait until the selected field has finished loading before saving or exporting.")
        self._capture_camera()
        return {"schema": "fempost.rf.view/v1", "study": str(self.study.path), "index": self._active_index, "port": self._active_port, "state": asdict(self.state), "sparameter_view": {"mode": self.s_panel.mode.currentIndex(), "traces": {name: check.isChecked() for name, check in self.s_panel.traces.items()}}, "created_utc": datetime.now(timezone.utc).isoformat(), "normalization": self.study.normalization, "coordinates": "m", "sample_source": getattr(self.frame, "source", ""), "plot": self._info}

    def save_view(self, path):
        path = self._output_path(path)
        record = self.view_record()
        path.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
        self.status.setText(f"Saved view: {path}")
        return path

    def load_view(self, path):
        path = Path(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "fempost.rf.view/v1":
            raise ValueError("This file is not a fempost RF view (fempost.rf.view/v1).")
        study_path = Path(record["study"])
        if not study_path.is_absolute():
            study_path = path.parent / study_path
        self._generation += 1
        self._loaded_view = record
        self._selection_valid = False
        self._set_selection_enabled(False)
        self.play_button.setChecked(False)
        self.status.setText("Restoring RF study and saved view…")
        self._submit((self._generation, study_path, int(record.get("index", 0)), int(record.get("port", 1))))

    def export_figure(self, path):
        path = self._output_path(path)
        record = self.view_record()
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        # Use the same scene renderer in a dedicated offscreen plotter. Qt GL
        # contexts are not guaranteed current when another tab/dialog has focus.
        import pyvista as pv
        export_plotter = pv.Plotter(off_screen=True, window_size=(1800, 1100))
        try:
            export_scene = RFScene(export_plotter)
            export_scene.render(self.frame, RFViewState.from_dict(record["state"]))
            export_plotter.show(screenshot=str(path), auto_close=False)
        finally:
            export_plotter.close()
        path.with_suffix(".view.json").write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
        if getattr(self.study, "sparameters", None) is not None:
            self.s_panel.export(path.with_name(path.stem + "-sparameters.png"))
        self.status.setText(f"Exported figure and reproducible view: {path}")
        return path

    def _save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save RF view", str(self._default_folder() / "rf-view.json"), "RF view (*.json)")
        if path:
            self._guard(lambda: self.save_view(path))

    def _load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load RF view", str(self._default_folder()), "RF view (*.json)")
        if path:
            self._guard(lambda: self.load_view(path))

    def _export_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export field figure", str(self._default_folder() / "rf-field.png"), "PNG image (*.png)")
        if path:
            self._guard(lambda: self.export_figure(path))

    def _guard(self, callback):
        try:
            callback()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "RF workspace", str(exc))

    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        self.animation.stop()
        self.render_timer.stop()
        self._pending = None
        if self._worker is not None:
            # Cancel the owned read-only native reader, waking its response queue, then
            # join the Python worker before a tab or the application is destroyed.
            worker, self._worker = self._worker, None
            worker.requestInterruption()
            worker.ready.disconnect(self._loaded)
            worker.probed.disconnect(self._native_probed)
            worker.failed.disconnect(self._failed)
            worker.finished.disconnect(self._worker_finished)
            if worker.study is not None:
                worker.study.close()
            if worker.wait(30000):
                worker.deleteLater()
            else:
                _FINISHING_WORKERS.add(worker)
                worker.finished.connect(lambda w=worker: (_FINISHING_WORKERS.discard(w), w.deleteLater()))
                app = QtWidgets.QApplication.instance()
                if app:
                    app.aboutToQuit.connect(lambda w=worker: w.wait(30000))
        elif self.study is not None:
            self.study.close()
        self.plotter.close()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
