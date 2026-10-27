"""Desktop viewer: PySide6 + pyvistaqt, post-processing only.

Layout (COMSOL-shaped): left dock model tree (cases, mesh bodies, definitions with counts, result
fields), centre 3D view with a toolbar (case, frame, view presets, fit, parallel, edges, clip widget,
deformation factor with Auto, ghost, screenshot, scene save/load), right dock properties (field,
data location, colormap, range lock, log, opacity, limit, body filter, clip and slice planes),
bottom dock tabs (Summary, Mesh statistics, Probe, Log). Every control edits one ``SceneState`` and
re-renders through ``Scene.render`` so figures from ``fempost render`` match what Kurt sees.
Nothing here writes beside a run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import convergence as convergence_module
from .bundle import RunBundle, _json_default
from .fields import FIELD_SPECS
from .frd import CACHE_DIR, ROOT
from .scene import (COLORMAPS, DEFAULT_OVERLAYS, OVERLAY_KEYS, STATUS_COLORS, Scene, SceneState, VIEWS, animation_factors,
                    apply_plot_group, plot_group_names)

REVIEW_DIR = ROOT / "structural-fea-environment/temporary/fempost-review"
TOOLBAR_OVERLAYS = (("restraints", "Restraints"), ("loads", "Loads"), ("contact", "Contact"))
ANIMATION_FRAMES = 16


def _qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def evidence_lines(acceptance: dict) -> list[str]:
    """Solver evidence block: where each number of the status line comes from."""
    lines = [f"Status line: {acceptance['label']}", f"  kind: {acceptance.get('kind')} (accepted = {acceptance.get('accepted')})"]
    if acceptance.get("diagnostic_reasons"):
        lines.append("  diagnostic because: " + "; ".join(acceptance["diagnostic_reasons"]))
    conv = acceptance.get("convergence") or {}
    sources = conv.get("sources") or {}
    names = {k: Path(v).name for k, v in sources.items() if v}
    if conv.get("steps"):
        lines.append(f"Convergence ({', '.join(names.get(k, '') for k in ('sta', 'cvg', 'log') if names.get(k))}):")
        for row in conv["steps"]:
            lines.append(f"  step {row['step']}: {row['increments']} increments, {row['attempts']} attempts"
                         + (f" ({row['unconverged_attempts']} unconverged)" if row.get("unconverged_attempts") else "")
                         + (f", {row['cutbacks']} cutbacks" if row.get("cutbacks") else "")
                         + f", {row['iterations']} iterations, step time reached {row.get('end_step_time')}"
                         + (f", increment size {row['min_increment']:g}..{row['max_increment']:g}" if row.get("min_increment") is not None else ""))
        final = conv.get("final_iteration") or {}
        if final:
            lines.append(f"  last iteration: residual force {final.get('residual_force_pct')} %, displacement correction "
                         f"{final.get('disp_correction_pct')} % ({names.get('cvg', '.cvg')}); largest correction "
                         f"{final.get('largest_disp_correction')} at node {final.get('correction_node')} dof {final.get('correction_dof')}, "
                         f"active contact elements {final.get('contact_elements')} ({names.get('log', 'stdout log')})")
        if conv.get("total_time_s") is not None:
            lines.append(f"  Total CalculiX time {conv['total_time_s']:.0f} s ({names.get('log', 'stdout log')}); equations {conv.get('equations')}")
        if conv.get("converged") is False:
            lines.append("  NOT converged: " + "; ".join(conv.get("reasons") or []))
    else:
        lines.append("Convergence: no .sta/.cvg/stdout record beside the run")
    for entry in acceptance.get("equilibrium") or []:
        text = (f"Equilibrium ({entry['source']}, stage {entry['stage']}"
                + (f", t={entry['time']}" if entry.get("time") is not None else "") + "): "
                f"force residual {np.round(entry['force_residual_N'], 4).tolist()} N, |F| = {entry['force_residual_norm_N']:.4g} N")
        if entry.get("moment_residual_norm_Nmm") is not None:
            text += f"; moment residual {np.round(entry['moment_residual_Nmm'], 3).tolist()} N·mm, |M| = {entry['moment_residual_norm_Nmm']:.4g} N·mm"
        lines.append(text)
        if entry.get("applied_force_norm_N"):
            lines.append(f"  applied load |F| = {entry['applied_force_norm_N']:.4g} N ({entry.get('applied_source')}): "
                         f"|F residual| / |F applied| = {entry.get('force_fraction', float('nan')):.3e}"
                         + (f"; {entry['moment_fraction_basis']} = {entry['moment_fraction']:.3e}" if entry.get("moment_fraction") is not None else ""))
        if entry.get("frd_complete") is not None:
            lines.append(f"  frd_complete = {entry['frd_complete']}, max displacement {entry.get('max_displacement_mm')} mm (results.json)")
        if entry.get("body_balance"):
            worst = sorted(entry["body_balance"].items(), key=lambda kv: -(kv[1].get("force_residual_norm_N") or 0))[:4]
            lines.append("  per-body balance (results.json, |F| N / |M| N·mm): "
                         + ", ".join(f"{name} {v.get('force_residual_norm_N', 0):.3g} / {v.get('moment_residual_norm_Nmm') or 0:.3g}" for name, v in worst))
    return lines


def summary_text(bundle: RunBundle) -> str:
    """Readable summary of a run: solver status, acceptance, mesh, definitions and sidecar numbers."""
    s = bundle.summary(include_frames=True)
    lines = [f"Run: {s['run']['name']}  [{s['run']['family']}]", f"Folder: {s['run']['folder']}",
             f"Heading: {s['run']['heading']}", f"Units: {s['run']['units']}", ""]
    solver = s["solver"]
    lines += [f"Solver status: {solver['solver_status']}  ·  {solver['label']}",
              f"Completed steps: {solver.get('completed_steps')} of {solver.get('expected_steps')}  ·  CalculiX {solver.get('solver_version')}  ·  "
              f"{solver.get('elapsed_s') or 0:.0f} s on {solver.get('threads')} threads (run.json)", ""]
    lines += evidence_lines(solver) + [""]
    if solver.get("gates"):
        lines.append("Acceptance gates (verbatim from results.json):")
        for gate in solver["gates"]:
            lines.append(f"  {gate.get('status', '?'):7s} {gate.get('name')}: measured {gate.get('measured')} / limit {gate.get('limit')}")
        lines.append("")
    mesh = s.get("mesh", {})
    if mesh:
        lines += [f"Mesh: {mesh['nodes']} nodes, {mesh['elements']} elements  {mesh['element_types']}",
                  f"Bounds mm: {np.round(mesh['bounds_mm']['min'], 2).tolist()} .. {np.round(mesh['bounds_mm']['max'], 2).tolist()}  diagonal {mesh['diagonal_mm']:.1f}",
                  f"Mesh hash: {mesh['mesh_hash'][:16]}…  FRD parse: {'cache' if mesh['frd'].get('cache_hit') else 'parsed'} in {mesh['frd'].get('elapsed_s', 0):.1f} s", ""]
    if s.get("bodies"):
        lines.append("Bodies:")
        for body in s["bodies"]:
            lines.append(f"  {body['name']:16s} elset {body['elset']:14s} {body['elements']:8d} elements {body['nodes']:8d} nodes  material {body['material']} E={body['E_MPa']} MPa")
        lines.append("")
    if s.get("steps"):
        lines.append("Steps: " + "; ".join(f"{st['number']}: {st['procedure']}{' NLGEOM' if st['nlgeom'] else ''} {st['procedure_data']}" for st in s["steps"]))
    if s.get("frames"):
        lines.append("Frames: " + ", ".join(f"[{f['index']}] step {f['step']} t={f['time']:g} ({', '.join(f['fields'])})" for f in s["frames"][:12])
                     + (" …" if len(s["frames"]) > 12 else ""))
    lines.append("")
    if s.get("restraints"):
        lines.append("Restraints (*BOUNDARY):")
        for r in s["restraints"]:
            lines.append(f"  step {r['step']}: {r['target']} {r['dofs']} = {r['value']:g}  ({r['nodes']} nodes){'  FIXED' if r.get('fixed') else ''}")
    loads = s.get("loads", {})
    for c in loads.get("cload", []):
        lines.append(f"Load step {c['step']} (*CLOAD): {c['nodes']} nodes, resultant {np.round(c['resultant_N'], 3).tolist()} N = {c['magnitude_N']:.4g} N at {np.round(c['centroid_mm'], 2).tolist()} mm"
                     + ("  OP=NEW" if c.get("op_new") else ""))
    for d in loads.get("dload", []):
        lines.append(f"Load step {d['step']} (*DLOAD): {d['target']} {d['labels_text']} = {d['value']:g} MPa ({d['faces']} faces)")
    if s.get("contact_pairs"):
        lines.append("Contact pairs:")
        for p in s["contact_pairs"]:
            extra = f" [{p['type']}]" if p.get("type") else ""
            fr = f" mu={p['friction'][0]:g}" if p.get("friction") else ""
            lines.append(f"  {p['slave']} -> {p['master']}{extra}: {p['category']} ({p['behavior']} {p['stiffness']}{fr}); {p['slave_faces']} slave / {p['master_faces']} master faces")
    for t in s.get("ties", []):
        lines.append(f"Tie {t['name']}: {t['slave']} -> {t['master']} ({t['slave_faces']} / {t['master_faces']} faces)")
    for p in s.get("pretension", []):
        lines.append(f"Pretension: {json.dumps(p, default=_json_default)}")
    if s.get("materials"):
        lines.append("Materials: " + "; ".join(f"{n}: E={m['E_MPa']} MPa nu={m['nu']}" for n, m in s["materials"].items()))
    lines.append(f"Equations: {s.get('equations', 0)}")
    lines.append("")
    side = s.get("sidecars", {})
    if side.get("results"):
        res = side["results"]
        lines.append("Pipeline results.json (verbatim subset):")
        for key in ("force_residual_N", "moment_residual_Nmm", "max_displacement_mm", "metrics", "limitations", "diagnostic_only"):
            if key in res:
                lines.append(f"  {key}: {json.dumps(res[key], default=_json_default)}")
        if res.get("regions"):
            lines.append("  Region peaks (nodal vm max / ip vm max MPa):")
            for name, values in res["regions"].items():
                lines.append(f"    {name:24s} {values.get('nodal_vm_max_MPa', float('nan')):9.1f} / {values.get('ip_vm_max_MPa', float('nan')):9.1f}  umax {values.get('max_displacement_mm', float('nan')):.4g} mm")
        if res.get("interfaces"):
            lines.append("  Interface resultants (N):")
            for entry in res["interfaces"]:
                lines.append(f"    {entry.get('label', '')}: {entry.get('slave')} -> {entry.get('master')}  |F| = {entry.get('resultant_N', float('nan')):.4g}")
        if res.get("stages"):
            for stage, values in res["stages"].items():
                eq = values.get("equilibrium") or {}
                lines.append(f"  Stage {stage} t={values.get('time')}: support {eq.get('support_resultant_N')} applied {eq.get('applied_resultant_N')} "
                             f"force imbalance {eq.get('force_imbalance_fraction')}  contact {values.get('contact_status')}")
    if side.get("model"):
        model = side["model"]
        lines.append("Codex model.json: " + json.dumps({k: model.get(k) for k in ("fix", "contact_note", "preload_N", "applied_force_N", "material_E_MPa")}, default=_json_default))
    if side.get("assembly"):
        asm = side["assembly"]
        lines.append("KurtsFEM assembly.json: " + json.dumps({k: asm.get(k) for k in ("load", "anchorage_enforcement", "fixed_hub_stress", "stages", "final_artificial_supports")}, default=_json_default))
    if s["run"].get("messages"):
        lines += ["", "Messages:"] + [f"  {m}" for m in s["run"]["messages"]]
    return "\n".join(lines)


def _vec_text(values) -> str:
    return ", ".join(f"{float(v):g}" for v in values)


def _parse_vec(text: str, default):
    try:
        parts = [float(p) for p in text.replace(";", ",").split(",") if p.strip()]
        if len(parts) == 3:
            return parts
    except ValueError:
        pass
    return list(default)


class ViewerWindow:
    """Factory for the Qt main window (kept out of module scope so ``info`` never imports Qt)."""

    def __new__(cls, runs, *, offscreen=False, scene_spec=None, settings_path=None):
        QtCore, QtGui, QtWidgets = _qt()
        from pyvistaqt import QtInteractor

        class _Window(QtWidgets.QMainWindow):
            def __init__(self):
                super().__init__()
                self.setWindowTitle("fempost · read-only CalculiX post-processing (N-mm-MPa)")
                self.setObjectName("fempostMainWindow")
                self.resize(1500, 950)
                self.offscreen = offscreen
                self.scenes: dict[str, Scene] = {}
                self.state = SceneState()
                self._render_pending = False
                self._stats_cache: dict[str, dict] = {}
                self._conv_cache: dict[str, dict] = {}
                self._building = True
                self._picking_enabled = False
                self._anim = None
                self.log_lines: list[str] = []
                self.docks: dict[str, QtWidgets.QDockWidget] = {}
                self.overlay_actions: dict[str, QtGui.QAction] = {}
                if settings_path:
                    self.settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)
                elif not offscreen:
                    self.settings = QtCore.QSettings("KurtsFEM", "fempost")
                else:
                    self.settings = None
                self._build_central()
                self._build_docks()
                self._build_toolbar()
                self._build_menus()
                self._default_layout = self.saveState()
                self._default_geometry = self.saveGeometry()
                self._building = False
                self.restore_layout()
                for run in runs:
                    self.open_run(run)
                if scene_spec:
                    self.load_scene(scene_spec)
                if self.scenes and not self._render_pending:
                    self.request_render()

            # ---- construction ---------------------------------------------------------
            def _build_central(self):
                central = QtWidgets.QWidget()
                central.setMinimumSize(480, 320)      # the docks can never swallow the 3D view
                layout = QtWidgets.QVBoxLayout(central)
                layout.setContentsMargins(0, 0, 0, 0)
                self.plotter = QtInteractor(central, off_screen=self.offscreen)
                self.plotter.set_background("#f4f6f8")
                layout.addWidget(self.plotter.interactor, 1)
                self.setCentralWidget(central)
                self.status_label = QtWidgets.QLabel("Open a run folder (toolbar) or pass it on the command line.")
                self.statusBar().addWidget(self.status_label, 1)
                self.plotter.interactor.installEventFilter(self)   # double-click probe fallback (no VTK picker needed)

            def _build_menus(self):
                menu = self.menuBar().addMenu("&View")
                for name, dock in self.docks.items():
                    menu.addAction(dock.toggleViewAction())
                menu.addAction(self.toolbar.toggleViewAction())
                menu.addSeparator()
                for key, title in TOOLBAR_OVERLAYS:
                    menu.addAction(self.overlay_actions[key])
                menu.addSeparator()
                reset = menu.addAction("Reset layout")
                reset.setToolTip("Restore the default dock arrangement and window size")
                reset.triggered.connect(self.reset_layout)
                if self.settings is not None:
                    save = menu.addAction("Save layout now")
                    save.triggered.connect(self.save_layout)
                self.view_menu = menu
                button = QtWidgets.QToolButton()
                button.setText("View ▾")
                button.setToolTip("Show or hide docks and toolbars, reset the layout")
                button.setMenu(menu)
                button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
                self.toolbar.addSeparator()
                self.toolbar.addWidget(button)

            def _build_toolbar(self):
                bar = QtWidgets.QToolBar("Main toolbar")
                bar.setObjectName("fempostMainToolbar")
                bar.setMovable(True)
                self.addToolBar(bar)
                self.toolbar = bar
                open_action = bar.addAction("Open run…")
                open_action.triggered.connect(self.open_dialog)
                bar.addSeparator()
                bar.addWidget(QtWidgets.QLabel(" Case "))
                self.case_combo = QtWidgets.QComboBox()
                self.case_combo.setMinimumWidth(220)
                self.case_combo.currentIndexChanged.connect(self._case_changed)
                bar.addWidget(self.case_combo)
                bar.addWidget(QtWidgets.QLabel(" Frame "))
                self.frame_combo = QtWidgets.QComboBox()
                self.frame_combo.setMinimumWidth(260)
                self.frame_combo.currentIndexChanged.connect(self._frame_changed)
                bar.addWidget(self.frame_combo)
                bar.addSeparator()
                for name in ("iso", "+x", "-x", "+y", "-y", "+z", "-z"):
                    action = bar.addAction(name.upper() if name != "iso" else "Iso")
                    action.triggered.connect(lambda checked=False, n=name: self.set_view(n))
                fit = bar.addAction("Fit")
                fit.triggered.connect(self.fit_view)
                self.parallel_action = bar.addAction("Parallel")
                self.parallel_action.setCheckable(True)
                self.parallel_action.toggled.connect(lambda on: self._set("parallel", on))
                self.edges_action = bar.addAction("Edges")
                self.edges_action.setCheckable(True)
                self.edges_action.toggled.connect(lambda on: self._set("edges", on))
                self.clip_widget_action = bar.addAction("Clip widget")
                self.clip_widget_action.setCheckable(True)
                self.clip_widget_action.toggled.connect(self.toggle_clip_widget)
                bar.addSeparator()
                bar.addWidget(QtWidgets.QLabel(" Deformation × "))
                self.deform_spin = QtWidgets.QDoubleSpinBox()
                self.deform_spin.setRange(0.0, 1e7)
                self.deform_spin.setDecimals(3)
                self.deform_spin.setSingleStep(1.0)
                self.deform_spin.setToolTip("Deformation exaggeration factor: 10 means 10×; 0 shows the undeformed shape")
                self.deform_spin.valueChanged.connect(self._deform_edited)
                bar.addWidget(self.deform_spin)
                auto = bar.addAction("Auto")
                auto.setToolTip("Scale the peak displacement to 10 % of the model diagonal")
                auto.triggered.connect(self.auto_deform)
                self.ghost_action = bar.addAction("Ghost")
                self.ghost_action.setCheckable(True)
                self.ghost_action.setChecked(True)
                self.ghost_action.setToolTip("Undeformed wireframe ghost")
                self.ghost_action.toggled.connect(lambda on: self._set("ghost", on))
                self.play_action = bar.addAction("Play")
                self.play_action.setCheckable(True)
                self.play_action.setToolTip(f"Animate the deformation from 0 to the current factor and back ({ANIMATION_FRAMES} frames, "
                                            "each a full render); click again to stop")
                self.play_action.toggled.connect(self.play_animation)
                bar.addSeparator()
                for key, title in TOOLBAR_OVERLAYS:
                    action = bar.addAction(title)
                    action.setCheckable(True)
                    action.setChecked(bool(self.state.overlays.get(key)))
                    action.setToolTip(f"Show the {title.lower()} overlay (same as the tick in the model tree)")
                    action.toggled.connect(lambda on, k=key: self._overlay_toggled(k, on))
                    self.overlay_actions[key] = action
                bar.addSeparator()
                shot = bar.addAction("Screenshot")
                shot.triggered.connect(lambda: self.screenshot())
                save_scene = bar.addAction("Save scene")
                save_scene.triggered.connect(self.save_scene_dialog)
                load_scene = bar.addAction("Load scene")
                load_scene.triggered.connect(self.load_scene_dialog)

            def _build_docks(self):
                # left: model tree
                self.tree = QtWidgets.QTreeWidget()
                self.tree.setHeaderLabels(["Model", "Count"])
                self.tree.setColumnWidth(0, 300)
                self.tree.itemChanged.connect(self._tree_changed)
                self.tree.itemClicked.connect(self._tree_clicked)
                left = QtWidgets.QDockWidget("Model tree", self)
                left.setObjectName("dockModelTree")
                left.setWidget(self.tree)
                self.tree.setMinimumWidth(220)
                self.tree.setMaximumWidth(640)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, left)
                self.docks["tree"] = left
                # right: properties
                props = QtWidgets.QWidget()
                form = QtWidgets.QFormLayout(props)
                self.field_combo = QtWidgets.QComboBox()
                self.field_combo.currentIndexChanged.connect(self._field_changed)
                form.addRow("Field", self.field_combo)
                self.location_combo = QtWidgets.QComboBox()
                self.location_combo.addItem("Nodal display (FRD, extrapolated)", "point")
                self.location_combo.addItem("Element mean (DAT integration points)", "cell")
                self.location_combo.currentIndexChanged.connect(lambda i: self._set("location", self.location_combo.currentData()))
                form.addRow("Data", self.location_combo)
                self.minus_combo = QtWidgets.QComboBox()
                self.minus_combo.addItem("none", None)
                self.minus_combo.setToolTip("A − B: subtract another open case on the same solver mesh")
                self.minus_combo.currentIndexChanged.connect(lambda i: self._set("minus_case", self.minus_combo.currentData()))
                form.addRow("Minus case", self.minus_combo)
                self.minus_frame_combo = QtWidgets.QComboBox()
                self.minus_frame_combo.addItem("none", None)
                self.minus_frame_combo.setToolTip("A − B within this case: subtract another frame (loaded minus preload)")
                self.minus_frame_combo.currentIndexChanged.connect(lambda i: self._set("minus_frame", self.minus_frame_combo.currentData()))
                form.addRow("Minus frame", self.minus_frame_combo)
                self.cmap_combo = QtWidgets.QComboBox()
                self.cmap_combo.addItems(COLORMAPS)
                self.cmap_combo.currentTextChanged.connect(lambda t: self._set("cmap", t))
                form.addRow("Colormap", self.cmap_combo)
                range_row = QtWidgets.QHBoxLayout()
                self.range_min = QtWidgets.QDoubleSpinBox()
                self.range_max = QtWidgets.QDoubleSpinBox()
                self._range_timer = QtCore.QTimer(self)
                self._range_timer.setSingleShot(True)
                self._range_timer.setInterval(350)
                self._range_timer.timeout.connect(self._range_typed)
                for spin in (self.range_min, self.range_max):
                    spin.setRange(-1e12, 1e12)
                    spin.setDecimals(4)
                    spin.setKeyboardTracking(False)
                    spin.setToolTip("Typed values apply immediately and tick Lock; untick Lock for the automatic range")
                    spin.valueChanged.connect(lambda _v: None if self._building else self._range_timer.start())
                    spin.editingFinished.connect(self._range_typed)
                    range_row.addWidget(spin)
                self.range_lock = QtWidgets.QCheckBox("Lock")
                self.range_lock.setToolTip("Ticked: keep this range when the frame or field changes. Unticked: automatic range")
                self.range_lock.toggled.connect(self._range_edited)
                range_row.addWidget(self.range_lock)
                form.addRow("Range", range_row)
                auto_row = QtWidgets.QHBoxLayout()
                self.range_mode_combo = QtWidgets.QComboBox()
                self.range_mode_combo.addItem("Robust (0.5–99.5 % of the visible bodies)", "robust")
                self.range_mode_combo.addItem("Full (true min … max of the visible bodies)", "full")
                self.range_mode_combo.setToolTip("Automatic range: robust percentiles ignore singular peaks; the true min and max "
                                                 "are always printed in the scalar bar title")
                self.range_mode_combo.currentIndexChanged.connect(self._range_mode_changed)
                auto_row.addWidget(self.range_mode_combo, 1)
                fit = QtWidgets.QPushButton("Fit to visible")
                fit.setToolTip("Lock the range to the true min and max of the bodies currently shown")
                fit.clicked.connect(self.fit_range_to_visible)
                auto_row.addWidget(fit)
                form.addRow("Auto range", auto_row)
                self.log_check = QtWidgets.QCheckBox("Log scale")
                self.log_check.toggled.connect(lambda on: self._set("log_scale", on))
                form.addRow("", self.log_check)
                self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.opacity_slider.setRange(5, 100)
                self.opacity_slider.setValue(100)
                self.opacity_slider.sliderReleased.connect(lambda: self._set("opacity", self.opacity_slider.value() / 100.0))
                form.addRow("Opacity", self.opacity_slider)
                limit_row = QtWidgets.QHBoxLayout()
                self.limit_check = QtWidgets.QCheckBox("Limit")
                self.limit_spin = QtWidgets.QDoubleSpinBox()
                self.limit_spin.setRange(-1e12, 1e12)
                self.limit_spin.setDecimals(3)
                self.limit_spin.setValue(240.0)
                self.limit_check.toggled.connect(self._limit_edited)
                self.limit_spin.editingFinished.connect(self._limit_edited)
                limit_row.addWidget(self.limit_check)
                limit_row.addWidget(self.limit_spin)
                form.addRow("Clamp above", limit_row)
                self.quality_check = QtWidgets.QCheckBox("Mesh quality (scaled Jacobian)")
                self.quality_check.toggled.connect(lambda on: self._set("mesh_quality", on))
                form.addRow("", self.quality_check)
                self.text_spin = QtWidgets.QDoubleSpinBox()
                self.text_spin.setRange(0.5, 3.0)
                self.text_spin.setSingleStep(0.1)
                self.text_spin.setDecimals(1)
                self.text_spin.setValue(1.0)
                self.text_spin.setKeyboardTracking(False)
                self.text_spin.setToolTip("Multiplies every 3D text size (titles, legend, scalar bar, labels); fonts also grow with the window")
                self.text_spin.valueChanged.connect(self._text_scale_edited)
                form.addRow("Text size ×", self.text_spin)
                # clip
                clip_box = QtWidgets.QGroupBox("Clip plane")
                clip_box.setCheckable(True)
                clip_box.setChecked(False)
                clip_form = QtWidgets.QFormLayout(clip_box)
                self.clip_normal = QtWidgets.QLineEdit("1, 0, 0")
                self.clip_origin = QtWidgets.QLineEdit("")
                self.clip_origin.setPlaceholderText("model centre")
                self.clip_invert = QtWidgets.QCheckBox("Invert side")
                self.clip_ghost = QtWidgets.QCheckBox("Ghost removed half")
                self.clip_ghost.setChecked(True)
                clip_form.addRow("Normal", self.clip_normal)
                clip_form.addRow("Origin", self.clip_origin)
                clip_form.addRow("", self.clip_invert)
                clip_form.addRow("", self.clip_ghost)
                self.clip_box = clip_box
                for widget in (self.clip_normal, self.clip_origin):
                    widget.editingFinished.connect(self._clip_edited)
                for widget in (self.clip_invert, self.clip_ghost):
                    widget.toggled.connect(self._clip_edited)
                clip_box.toggled.connect(self._clip_edited)
                form.addRow(clip_box)
                # slice
                slice_box = QtWidgets.QGroupBox("Slice plane")
                slice_box.setCheckable(True)
                slice_box.setChecked(False)
                slice_form = QtWidgets.QFormLayout(slice_box)
                self.slice_normal = QtWidgets.QLineEdit("0, 0, 1")
                self.slice_origin = QtWidgets.QLineEdit("")
                self.slice_origin.setPlaceholderText("model centre")
                self.slice_contours = QtWidgets.QSpinBox()
                self.slice_contours.setRange(0, 60)
                self.slice_contours.setValue(10)
                self.slice_only = QtWidgets.QCheckBox("Slice only (hide the solid)")
                slice_form.addRow("Normal", self.slice_normal)
                slice_form.addRow("Origin", self.slice_origin)
                slice_form.addRow("Contours", self.slice_contours)
                slice_form.addRow("", self.slice_only)
                self.slice_box = slice_box
                for widget in (self.slice_normal, self.slice_origin):
                    widget.editingFinished.connect(self._slice_edited)
                self.slice_contours.valueChanged.connect(self._slice_edited)
                self.slice_only.toggled.connect(self._slice_edited)
                slice_box.toggled.connect(self._slice_edited)
                form.addRow(slice_box)
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(props)
                scroll.setMinimumWidth(300)
                scroll.setMaximumWidth(560)
                right = QtWidgets.QDockWidget("Properties", self)
                right.setObjectName("dockProperties")
                right.setWidget(scroll)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, right)
                self.docks["properties"] = right
                # bottom: tabs
                self.tabs = QtWidgets.QTabWidget()
                self.tabs.setMinimumHeight(140)
                self.tabs.setMaximumHeight(560)
                self.summary_text = QtWidgets.QPlainTextEdit()
                self.summary_text.setReadOnly(True)
                self.summary_text.setFont(QtGui.QFont("Consolas", 9))
                self.tabs.addTab(self.summary_text, "Summary")
                stats = QtWidgets.QWidget()
                stats_layout = QtWidgets.QHBoxLayout(stats)
                self.stats_table = QtWidgets.QTableWidget()
                stats_layout.addWidget(self.stats_table, 3)
                self.stats_text = QtWidgets.QPlainTextEdit()
                self.stats_text.setReadOnly(True)
                self.stats_text.setFont(QtGui.QFont("Consolas", 9))
                stats_layout.addWidget(self.stats_text, 3)
                hist_column = QtWidgets.QVBoxLayout()
                self.hist_labels = [QtWidgets.QLabel() for _ in range(2)]
                for label in self.hist_labels:
                    hist_column.addWidget(label)
                stats_layout.addLayout(hist_column, 2)
                self.tabs.addTab(stats, "Mesh statistics")
                conv = QtWidgets.QWidget()
                conv_layout = QtWidgets.QHBoxLayout(conv)
                self.conv_figure = QtWidgets.QLabel()
                self.conv_figure.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
                conv_scroll = QtWidgets.QScrollArea()
                conv_scroll.setWidgetResizable(True)
                conv_scroll.setWidget(self.conv_figure)
                conv_layout.addWidget(conv_scroll, 5)
                conv_right = QtWidgets.QVBoxLayout()
                self.conv_table = QtWidgets.QTableWidget()
                conv_right.addWidget(self.conv_table, 2)
                self.conv_text = QtWidgets.QPlainTextEdit()
                self.conv_text.setReadOnly(True)
                self.conv_text.setFont(QtGui.QFont("Consolas", 9))
                conv_right.addWidget(self.conv_text, 3)
                conv_layout.addLayout(conv_right, 4)
                self.tabs.addTab(conv, "Convergence")
                self.beam_table = QtWidgets.QTableWidget()
                self.beam_table.setToolTip("Beam runs only: FRD section-force components at both end nodes of every element "
                                           "(*EL FILE, SECTION FORCES)")
                self.tabs.addTab(self.beam_table, "Beams")
                self.probe_text = QtWidgets.QPlainTextEdit()
                self.probe_text.setReadOnly(True)
                self.probe_text.setPlainText("Click a point on the model to probe the nearest node (double-click also works).")
                self.tabs.addTab(self.probe_text, "Probe")
                self.log_text = QtWidgets.QPlainTextEdit()
                self.log_text.setReadOnly(True)
                self.tabs.addTab(self.log_text, "Log")
                self.tabs.currentChanged.connect(self._tab_changed)
                bottom = QtWidgets.QDockWidget("Details", self)
                bottom.setObjectName("dockDetails")
                bottom.setWidget(self.tabs)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, bottom)
                self.docks["details"] = bottom
                self.resizeDocks([bottom], [260], QtCore.Qt.Orientation.Vertical)
                self._enable_picking()

            def _enable_picking(self) -> bool:
                """pyvista point picking needs a live interactor; try now and again after the first render."""
                if self._picking_enabled:
                    return True
                if getattr(self.plotter, "iren", None) is None:
                    return False
                try:
                    self.plotter.enable_point_picking(callback=self._picked, show_message=False, use_picker=True,
                                                      show_point=True, color="#ffeb3b", left_clicking=True)
                    self._picking_enabled = True
                except Exception as exc:
                    self.log(f"point picking unavailable ({exc}); double-click picking uses the screen-ray fallback")
                    self._picking_enabled = None       # do not retry every render
                return bool(self._picking_enabled)

            def eventFilter(self, watched, event):
                if watched is self.plotter.interactor and event.type() == QtCore.QEvent.Type.MouseButtonDblClick \
                        and event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.position() if hasattr(event, "position") else event.pos()
                    self.pick_screen(float(pos.x()), float(pos.y()))
                    return True
                return super().eventFilter(watched, event)

            def pick_screen(self, x: float, y: float) -> dict | None:
                """Probe from a Qt widget point: VTK cell picker when it works, else the nearest visible node
                to the view ray through that pixel (works offscreen and without an interactor)."""
                scene = self.scene
                if scene is None:
                    return None
                renderer = self.plotter.renderer
                height = self.plotter.interactor.height() or 600
                vx, vy = float(x), float(height - y)
                point = None
                try:
                    import vtk
                    picker = vtk.vtkCellPicker()
                    picker.SetTolerance(0.005)
                    if picker.Pick(vx, vy, 0.0, renderer):
                        point = list(picker.GetPickPosition())
                except Exception:
                    point = None
                try:
                    if point is not None:
                        probe = scene.probe(self.state, point)
                        probe["pick"] = "vtkCellPicker"
                    else:
                        near, far = [], []
                        for depth, target in ((0.0, near), (1.0, far)):
                            renderer.SetDisplayPoint(vx, vy, depth)
                            renderer.DisplayToWorld()
                            w = renderer.GetWorldPoint()
                            target.extend([w[i] / w[3] for i in range(3)] if w[3] else list(w[:3]))
                        probe = scene.probe_ray(self.state, near, np.subtract(far, near))
                        probe["pick"] = "screen ray"
                        point = probe["coordinates_mm"]
                except Exception as exc:
                    self.probe_text.setPlainText(f"probe failed at screen point ({x:.0f}, {y:.0f}): {exc}")
                    self.tabs.setCurrentWidget(self.probe_text)
                    return None
                self._show_probe(point, probe)
                return probe

            # ---- cases --------------------------------------------------------------------
            def log(self, message: str):
                stamp = time.strftime("%H:%M:%S")
                self.log_lines.append(f"{stamp} {message}")
                self.log_text.appendPlainText(f"{stamp} {message}")

            def open_dialog(self):
                if self.offscreen:
                    return
                folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a CalculiX run folder", str(ROOT / "Dodec FEA"))
                if folder:
                    self.open_run(folder)
                    self.request_render()

            def open_run(self, target) -> str | None:
                try:
                    bundle = RunBundle(target, log=self.log)
                    scene = Scene(bundle)
                    bundle.mesh  # parse now so errors surface here
                except Exception as exc:
                    self.log(f"cannot open {target}: {exc}")
                    self.status_label.setText(f"Cannot open {target}: {exc}")
                    return None
                name = bundle.name
                if name in self.scenes:
                    name = f"{name} ({len(self.scenes) + 1})"
                self.scenes[name] = scene
                self.case_combo.blockSignals(True)
                self.case_combo.addItem(name, name)
                self.case_combo.setCurrentIndex(self.case_combo.count() - 1)
                self.case_combo.blockSignals(False)
                self.state.case = name
                self.log(f"opened {name}: {bundle.mesh.n_nodes} nodes, {bundle.mesh.n_elements} elements, {len(bundle.frames)} frames; "
                         f"{bundle.acceptance()['label']}")
                self._populate_for_case()
                return name

            @property
            def scene(self) -> Scene | None:
                if not self.scenes:
                    return None
                return self.scenes.get(self.state.case) or next(iter(self.scenes.values()))

            def _case_changed(self, index):
                if self._building or index < 0:
                    return
                self.state.case = self.case_combo.itemData(index)
                self.state.camera = None
                self._populate_for_case()
                self.request_render()

            def _populate_for_case(self):
                scene = self.scene
                if scene is None:
                    return
                self._building = True
                bundle = scene.bundle
                labels = bundle.step_labels()
                self.frame_combo.clear()
                self.minus_frame_combo.clear()
                self.minus_frame_combo.addItem("none", None)
                for frame in bundle.frames:
                    self.frame_combo.addItem(f"{frame.label}  ·  {labels.get(frame.step, '')}"[:90], frame.index)
                    self.minus_frame_combo.addItem(f"{frame.label}  ·  {labels.get(frame.step, '')}"[:60], frame.index)
                self.frame_combo.setCurrentIndex(max(len(bundle.frames) - 1, 0))
                self.state.frame = len(bundle.frames) - 1 if bundle.frames else 0
                self.state.minus_frame = None
                self._populate_fields()
                self._populate_minus()
                self._populate_tree()
                self.summary_text.setPlainText(summary_text(bundle))
                self.stats_table.clear()
                self.stats_text.clear()
                self.conv_table.clear()
                self.conv_text.clear()
                self.conv_figure.clear()
                self._building = False
                self._tab_changed(self.tabs.currentIndex())

            def _populate_fields(self):
                scene = self.scene
                frame = scene.bundle.frames[scene.frame_index(self.state)]
                current = self.state.field
                self.field_combo.blockSignals(True)
                self.field_combo.clear()
                for key in scene.available_fields(frame):
                    spec = FIELD_SPECS[key]
                    self.field_combo.addItem(f"{spec.group}: {spec.label} [{spec.unit}]", key)
                index = self.field_combo.findData(current)
                if index < 0:
                    index = max(self.field_combo.findData("von_mises"), 0)
                self.field_combo.setCurrentIndex(index)
                self.state.field = self.field_combo.currentData() or current
                self.field_combo.blockSignals(False)
                has_dat = scene.has_element_stress(frame)
                self.location_combo.setEnabled(has_dat)
                self.location_combo.setToolTip("" if has_dat else "No *EL PRINT stress table in the DAT at this time")

            def _populate_minus(self):
                scene = self.scene
                self.minus_combo.blockSignals(True)
                self.minus_combo.clear()
                self.minus_combo.addItem("none", None)
                for name, candidate in self.scenes.items():
                    if name != self.state.case and candidate.mesh_hash() == scene.mesh_hash():
                        self.minus_combo.addItem(name, name)
                index = self.minus_combo.findData(self.state.minus_case) if self.state.minus_case else 0
                if index < 0:
                    self.state.minus_case = None
                    index = 0
                self.minus_combo.setCurrentIndex(index)
                self.minus_combo.setEnabled(self.minus_combo.count() > 1)
                self.minus_combo.blockSignals(False)

            def _populate_tree(self):
                scene = self.scene
                bundle = scene.bundle
                self.tree.blockSignals(True)
                self.tree.clear()
                root = QtWidgets.QTreeWidgetItem([f"Case: {self.state.case}", ""])
                self.tree.addTopLevelItem(root)
                mesh_item = QtWidgets.QTreeWidgetItem(["Mesh", f"{bundle.mesh.n_elements} el"])
                root.addChild(mesh_item)
                bodies = bundle.bodies()
                visible = set(self.state.bodies) if self.state.bodies else set(bodies)
                for body, rows in bodies.items():
                    item = QtWidgets.QTreeWidgetItem([body, str(rows.size)])
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, QtCore.Qt.CheckState.Checked if body in visible else QtCore.Qt.CheckState.Unchecked)
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("body", body))
                    mesh_item.addChild(item)
                deck = bundle.deck
                if deck is not None:
                    sets = QtWidgets.QTreeWidgetItem(["Sets and surfaces", ""])
                    mesh_item.addChild(sets)
                    for name in deck.nsets:
                        sets.addChild(QtWidgets.QTreeWidgetItem([f"NSET {name}", str(deck.resolve_nodes(name).size)]))
                    for name in deck.elsets:
                        sets.addChild(QtWidgets.QTreeWidgetItem([f"ELSET {name}", str(deck.resolve_elements(name).size)]))
                    for name in deck.surfaces:
                        sets.addChild(QtWidgets.QTreeWidgetItem([f"SURFACE {name}", str(bundle.surface_face_count(name))]))
                defs = QtWidgets.QTreeWidgetItem(["Definitions", ""])
                root.addChild(defs)
                groups = {"restraints": ("Restraints (*BOUNDARY)", bundle.restraints()),
                          "loads": ("Loads (*CLOAD, *DLOAD)", bundle.cload_groups() + bundle.dloads() if bundle.deck else []),
                          "contact": ("Contact pairs", bundle.contact_pairs()),
                          "ties": ("Ties", bundle.ties()),
                          "pretension": ("Pretension", bundle.pretension()),
                          "reactions": ("Reaction forces (FORC on restrained sets)", bundle.restraints())}
                for key, (title, entries) in groups.items():
                    group = QtWidgets.QTreeWidgetItem([title, str(len(entries))])
                    group.setFlags(group.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                    group.setCheckState(0, QtCore.Qt.CheckState.Checked if self.state.overlays.get(key) else QtCore.Qt.CheckState.Unchecked)
                    group.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("overlay", key))
                    defs.addChild(group)
                    for entry in entries:
                        group.addChild(QtWidgets.QTreeWidgetItem(self._definition_row(key, entry)))
                results = QtWidgets.QTreeWidgetItem(["Results", ""])
                root.addChild(results)
                frame = bundle.frames[scene.frame_index(self.state)]
                presets = QtWidgets.QTreeWidgetItem(["Plot groups (click to apply)", ""])
                results.addChild(presets)
                n_steps = len({f.step for f in bundle.frames})
                for name in plot_group_names(scene.available_fields(frame), n_steps):
                    leaf = QtWidgets.QTreeWidgetItem([name, ""])
                    leaf.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("plot_group", name))
                    presets.addChild(leaf)
                by_group: dict[str, QtWidgets.QTreeWidgetItem] = {}
                for key in scene.available_fields(frame):
                    spec = FIELD_SPECS[key]
                    parent = by_group.get(spec.group)
                    if parent is None:
                        parent = QtWidgets.QTreeWidgetItem([spec.group, ""])
                        results.addChild(parent)
                        by_group[spec.group] = parent
                    leaf = QtWidgets.QTreeWidgetItem([spec.label, spec.unit])
                    leaf.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("field", key))
                    parent.addChild(leaf)
                quality = QtWidgets.QTreeWidgetItem(["Mesh quality (scaled Jacobian)", ""])
                quality.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("quality", None))
                results.addChild(quality)
                self.tree.expandItem(root)
                self.tree.expandItem(mesh_item)
                self.tree.expandItem(defs)
                self.tree.expandItem(results)
                self.tree.expandItem(presets)
                self.tree.blockSignals(False)

            def apply_plot_group(self, name: str):
                self.state = apply_plot_group(self.state, name, self.scene.bundle if self.scene else None)
                self._sync_controls()
                self.request_render()

            def _overlay_toggled(self, key: str, on: bool):
                """Toolbar overlay buttons mirror the model-tree ticks (and the other way round)."""
                if self._building:
                    return
                self.state.overlays[key] = bool(on)
                self._sync_overlay_widgets()
                self.request_render()

            def _sync_overlay_widgets(self):
                was_building = self._building
                self._building = True
                for key, action in self.overlay_actions.items():
                    action.setChecked(bool(self.state.overlays.get(key)))
                self.tree.blockSignals(True)
                root = self.tree.topLevelItem(0)
                if root is not None:
                    for k in range(root.childCount()):
                        group = root.child(k)
                        for j in range(group.childCount()):
                            item = group.child(j)
                            payload = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                            if payload and payload[0] == "overlay":
                                item.setCheckState(0, QtCore.Qt.CheckState.Checked if self.state.overlays.get(payload[1])
                                                   else QtCore.Qt.CheckState.Unchecked)
                self.tree.blockSignals(False)
                self._building = was_building

            @staticmethod
            def _definition_row(key, entry) -> list[str]:
                if key == "restraints":
                    return [f"step {entry['step']}: {entry['target']} {entry['dofs']}" + (f" = {entry['value']:g}" if entry['value'] else ""), f"{entry['nodes']} nodes"]
                if key == "loads":
                    if "resultant_N" in entry:
                        return [f"step {entry['step']} CLOAD: {_vec_text(np.round(entry['resultant_N'], 3))} N", f"{entry['nodes']} nodes"]
                    return [f"step {entry['step']} DLOAD {entry['target']} {entry['labels_text']} = {entry['value']:g} MPa", f"{entry['faces']} faces"]
                if key == "contact":
                    extra = f" [{entry['type']}]" if entry.get("type") else ""
                    return [f"{entry['slave']} → {entry['master']}{extra}: {entry['category']}", f"{entry['slave_faces']} / {entry['master_faces']} faces"]
                if key == "ties":
                    return [f"{entry['name']}: {entry['slave']} → {entry['master']}", f"{entry['slave_faces']} / {entry['master_faces']}"]
                if entry.get("source") == "assembly.json":
                    return [f"screw {entry.get('screw')} node {entry.get('node')} target {entry.get('target_N')} N", "1"]
                return [f"PRE-TENSION SECTION {entry.get('surface')} node {entry.get('node')}", str(entry.get("faces", ""))]

            # ---- state changes ------------------------------------------------------------
            def _set(self, attribute, value):
                if self._building:
                    return
                setattr(self.state, attribute, value)
                self.request_render()

            def _frame_changed(self, index):
                if self._building or index < 0:
                    return
                self.state.frame = int(self.frame_combo.itemData(index))
                self._populate_fields()
                self.request_render()

            def _field_changed(self, index):
                if self._building or index < 0:
                    return
                self.state.field = self.field_combo.itemData(index)
                self.state.mesh_quality = False
                self.quality_check.blockSignals(True)
                self.quality_check.setChecked(False)
                self.quality_check.blockSignals(False)
                self.request_render()

            def _range_edited(self, *_):
                """Lock ticked: the spin-box values are the range. Unticked: automatic range (robust or full)."""
                if self._building:
                    return
                if self.range_lock.isChecked():
                    self.state.clim = [self.range_min.value(), self.range_max.value()]
                else:
                    self.state.clim = None
                self.request_render()

            def _range_typed(self, *_):
                """A typed min or max applies immediately and ticks Lock on the user's behalf."""
                if self._building:
                    return
                self._range_timer.stop()
                lo, hi = self.range_min.value(), self.range_max.value()
                if hi <= lo:
                    self.status_label.setText(f"Range max must exceed min ({lo:g} .. {hi:g})")
                    return
                if not self.range_lock.isChecked():
                    self.range_lock.blockSignals(True)
                    self.range_lock.setChecked(True)
                    self.range_lock.blockSignals(False)
                self.state.clim = [lo, hi]
                self.request_render()

            def _range_mode_changed(self, index):
                if self._building:
                    return
                self.state.range_mode = self.range_mode_combo.itemData(index) or "robust"
                if self.range_lock.isChecked():            # switching the automatic mode means: back to automatic
                    self.range_lock.blockSignals(True)
                    self.range_lock.setChecked(False)
                    self.range_lock.blockSignals(False)
                self.state.clim = None
                self.request_render()

            def fit_range_to_visible(self):
                """Lock the range to the true min and max of the bodies currently shown."""
                info = getattr(self, "last_info", None) or {}
                true = info.get("range_true")
                if not true or self.scene is None:
                    return
                lo, hi = float(true[0]), float(true[1])
                if hi <= lo:
                    hi = lo + (abs(lo) * 0.05 or 1.0)
                self._building = True
                self.range_min.setValue(lo)
                self.range_max.setValue(hi)
                self.range_lock.setChecked(True)
                self._building = False
                self.state.clim = [lo, hi]
                self.request_render()

            def _text_scale_edited(self, value):
                if self._building:
                    return
                self.state.text_scale = float(value)
                if self.settings is not None:
                    self.settings.setValue("text_scale", float(value))
                self.request_render()

            def _limit_edited(self, *_):
                if self._building:
                    return
                self.state.limit = self.limit_spin.value() if self.limit_check.isChecked() else None
                self.request_render()

            def _deform_edited(self, value):
                if self._building:
                    return
                self.state.deform = float(value)
                self.request_render()

            def auto_deform(self):
                self.state.deform = None
                self.request_render()

            def _clip_edited(self, *_):
                if self._building:
                    return
                if self.clip_box.isChecked():
                    origin = _parse_vec(self.clip_origin.text(), [np.nan] * 3)
                    self.state.clip = {"normal": _parse_vec(self.clip_normal.text(), [1, 0, 0]),
                                       "origin": None if any(np.isnan(origin)) else origin,
                                       "invert": self.clip_invert.isChecked(), "ghost": self.clip_ghost.isChecked()}
                else:
                    self.state.clip = None
                self.request_render()

            def _slice_edited(self, *_):
                if self._building:
                    return
                if self.slice_box.isChecked():
                    origin = _parse_vec(self.slice_origin.text(), [np.nan] * 3)
                    self.state.slice = {"normal": _parse_vec(self.slice_normal.text(), [0, 0, 1]),
                                        "origin": None if any(np.isnan(origin)) else origin,
                                        "contours": self.slice_contours.value(), "only": self.slice_only.isChecked()}
                else:
                    self.state.slice = None
                self.request_render()

            def _tree_changed(self, item, column):
                if self._building:
                    return
                data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if not data:
                    return
                kind, key = data
                if kind == "body":
                    checked = []
                    mesh_item = item.parent()
                    for k in range(mesh_item.childCount()):
                        child = mesh_item.child(k)
                        payload = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                        if payload and payload[0] == "body" and child.checkState(0) == QtCore.Qt.CheckState.Checked:
                            checked.append(payload[1])
                    self.state.bodies = None if len(checked) == mesh_item.childCount() - 1 or len(checked) == len(self.scene.bundle.bodies()) else checked
                    if not checked:
                        self.state.bodies = checked
                elif kind == "overlay":
                    self.state.overlays[key] = item.checkState(0) == QtCore.Qt.CheckState.Checked
                    self._sync_overlay_widgets()
                self.request_render()

            def _tree_clicked(self, item, column):
                data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if not data:
                    return
                kind, key = data
                if kind == "field":
                    index = self.field_combo.findData(key)
                    if index >= 0:
                        self.field_combo.setCurrentIndex(index)
                elif kind == "quality":
                    self.quality_check.setChecked(True)
                elif kind == "plot_group":
                    self.apply_plot_group(key)

            def set_view(self, name):
                self.state.view = name
                self.state.camera = None
                self.request_render()

            def fit_view(self):
                self.plotter.reset_camera()
                self.plotter.camera.zoom(0.85)
                self.plotter.render()

            def toggle_clip_widget(self, on):
                if on:
                    scene = self.scene
                    if scene is None:
                        return
                    center = list(scene.grid.center)
                    self.clip_box.setChecked(True)

                    def callback(normal, origin):
                        self.clip_normal.setText(_vec_text(np.round(normal, 4)))
                        self.clip_origin.setText(_vec_text(np.round(origin, 3)))
                        self._clip_edited()

                    try:
                        self.plotter.add_plane_widget(callback, normal=_parse_vec(self.clip_normal.text(), [1, 0, 0]),
                                                      origin=_parse_vec(self.clip_origin.text(), center) if self.clip_origin.text() else center,
                                                      implicit=True, normal_rotation=True)
                    except Exception as exc:
                        self.log(f"clip widget: {exc}")
                else:
                    try:
                        self.plotter.clear_plane_widgets()
                    except Exception:
                        pass

            def _picked(self, point, *args):
                scene = self.scene
                if scene is None:
                    return
                try:
                    probe = scene.probe(self.state, point)
                except Exception as exc:
                    self.probe_text.setPlainText(f"probe failed: {exc}")
                    return
                self._show_probe(point, probe)

            def _show_probe(self, point, probe: dict):
                lines = [f"Picked {np.round(point, 3).tolist()} mm → nearest node {probe['node_id']} at {np.round(probe['coordinates_mm'], 3).tolist()} mm  ({probe['frame']})"
                         + (f"  [{probe['pick']}]" if probe.get("pick") else ""),
                         f"Bodies: {probe.get('bodies')}  elements: {probe.get('elements')}"]
                for key, spec in FIELD_SPECS.items():
                    if key in probe:
                        lines.append(f"  {spec.label:46s} {probe[key]:12.5g} {spec.unit}")
                self.probe_text.setPlainText("\n".join(lines))
                self.tabs.setCurrentWidget(self.probe_text)

            def _tab_changed(self, index):
                if self.tabs.tabText(index) == "Mesh statistics" and self.scene is not None:
                    self._fill_statistics()
                elif self.tabs.tabText(index) == "Convergence" and self.scene is not None:
                    self._fill_convergence()
                elif self.tabs.tabText(index) == "Beams" and self.scene is not None:
                    self._fill_beams()

            def _fill_beams(self):
                bundle = self.scene.bundle
                table = self.beam_table
                table.clear()
                info = bundle.beam_info()
                if not info["is_beam"]:
                    table.setColumnCount(1)
                    table.setRowCount(1)
                    table.setHorizontalHeaderLabels(["Beams"])
                    table.setItem(0, 0, QtWidgets.QTableWidgetItem("not a beam run (no B31/B32 elements in the FRD)"))
                    table.resizeColumnsToContents()
                    return
                frame = bundle.frames[self.scene.frame_index(self.state)]
                rows = bundle.beam_table(frame)
                components = ["F1 (SXX)", "F2 (SYY)", "F3 (SZZ)", "M1 (SXY)", "M2 (SYZ)", "M3 (SZX)"]
                columns = ["element", "body", "material", "dims_mm", "node_a", "node_b", "|F|max N", "|M|max N·mm"] \
                    + [f"{c} @a" for c in components] + [f"{c} @b" for c in components]
                table.setColumnCount(len(columns))
                table.setHorizontalHeaderLabels(columns)
                table.setRowCount(len(rows))
                for r, entry in enumerate(rows):
                    cells = [entry["element"], entry["body"], entry["material"] or "", _vec_text(entry["dims_mm"] or []),
                             entry["node_a"], entry["node_b"], f"{entry['force_max_N']:.4g}", f"{entry['moment_max_Nmm']:.4g}"] \
                        + [f"{v:.4g}" for v in entry["end_a"]] + [f"{v:.4g}" for v in entry["end_b"]]
                    for c, value in enumerate(cells):
                        table.setItem(r, c, QtWidgets.QTableWidgetItem(str(value)))
                table.resizeColumnsToContents()
                if not info["section_forces"]:
                    self.log("Beams tab: the deck has no *EL FILE, SECTION FORCES; the STRESS columns are stresses, not section forces")

            def _fill_convergence(self):
                scene = self.scene
                bundle = scene.bundle
                name = self.state.case
                record = self._conv_cache.get(name)
                if record is None:
                    try:
                        summary = bundle.convergence
                        acceptance = bundle.acceptance()
                        conv_dir = CACHE_DIR / "convergence"
                        conv_dir.mkdir(parents=True, exist_ok=True)
                        key = "-".join(Path(p).stat().st_mtime_ns.__str__() for p in (bundle.files.sta, bundle.files.cvg, bundle.files.log) if p)
                        png = conv_dir / f"{bundle.mesh.mesh_hash()[:12]}-{abs(hash(key)) % 10**10}.png"
                        if not png.exists():
                            convergence_module.convergence_figure(summary, png, size=(7.2, 6.4), dpi=100,
                                                                  title=f"{bundle.name}: Newton convergence",
                                                                  evidence=acceptance.get("equilibrium"))
                        record = {"summary": summary, "acceptance": acceptance, "png": png}
                    except Exception as exc:
                        self.conv_text.setPlainText(f"convergence view failed: {exc}")
                        return
                    self._conv_cache[name] = record
                summary = record["summary"]
                columns = ["step", "increments", "attempts", "unconverged_attempts", "cutbacks", "iterations", "end_step_time",
                           "end_total_time", "min_increment", "max_increment"]
                rows = convergence_module.table_rows(summary)
                self.conv_table.setColumnCount(len(columns))
                self.conv_table.setHorizontalHeaderLabels(columns)
                self.conv_table.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    for c, key in enumerate(columns):
                        value = row.get(key)
                        text = f"{value:.4g}" if isinstance(value, float) else ("" if value is None else str(value))
                        self.conv_table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                self.conv_table.resizeColumnsToContents()
                self.conv_text.setPlainText("\n".join(evidence_lines(record["acceptance"])))
                self.conv_figure.setPixmap(QtGui.QPixmap(str(record["png"])))

            # ---- animation ------------------------------------------------------------------
            def play_animation(self, on: bool):
                """Sweep the deformation factor 0 → current → 0 with one full render per frame."""
                if not on:
                    self._anim = None
                    return
                scene = self.scene
                if scene is None:
                    self.play_action.setChecked(False)
                    return
                peak = scene.deformation_factor(self.state)
                if peak <= 0:
                    self.log("animation: the deformation factor is 0; set Auto or a factor first")
                    self.play_action.setChecked(False)
                    return
                self._anim = {"factors": animation_factors(peak, ANIMATION_FRAMES), "k": 0, "deform": self.state.deform,
                              "camera": self.state.camera, "started": time.perf_counter()}
                self.state.camera = [list(map(float, p)) for p in self.plotter.camera_position]
                QtCore.QTimer.singleShot(0, self._animation_step)

            def _animation_step(self):
                anim = self._anim
                if anim is None or not self.play_action.isChecked():
                    self._finish_animation()
                    return
                k = anim["k"]
                if k >= len(anim["factors"]):
                    self._finish_animation()
                    return
                self.state.deform = anim["factors"][k]
                anim["k"] = k + 1
                self.render_now()
                QtCore.QTimer.singleShot(30, self._animation_step)

            def _finish_animation(self):
                anim = self._anim
                self._anim = None
                if anim is not None:
                    self.state.deform = anim["deform"]
                    self.state.camera = anim["camera"]
                    self.log(f"animation: {anim['k']} frames in {time.perf_counter() - anim['started']:.1f} s")
                if self.play_action.isChecked():
                    self.play_action.blockSignals(True)
                    self.play_action.setChecked(False)
                    self.play_action.blockSignals(False)
                self.request_render()

            # ---- layout -----------------------------------------------------------------------
            def save_layout(self):
                if self.settings is None:
                    return
                self.settings.setValue("geometry", self.saveGeometry())
                self.settings.setValue("windowState", self.saveState())
                self.settings.setValue("text_scale", float(self.state.text_scale))
                self.settings.sync()

            def restore_layout(self) -> bool:
                if self.settings is None:
                    return False
                restored = False
                geometry = self.settings.value("geometry")
                if geometry is not None:
                    restored = bool(self.restoreGeometry(geometry))
                layout = self.settings.value("windowState")
                if layout is not None:
                    restored = bool(self.restoreState(layout)) or restored
                scale = self.settings.value("text_scale")
                if scale is not None:
                    try:
                        self.state.text_scale = float(scale)
                        self._building = True
                        self.text_spin.setValue(self.state.text_scale)
                        self._building = False
                    except (TypeError, ValueError):
                        pass
                return restored

            def reset_layout(self):
                """Default arrangement: tree left, properties right, details bottom, toolbar top, 1500×950."""
                self.restoreState(self._default_layout)
                for dock in self.docks.values():
                    dock.setFloating(False)
                    dock.show()
                self.toolbar.show()
                self.resize(1500, 950)
                self.resizeDocks([self.docks["details"]], [260], QtCore.Qt.Orientation.Vertical)
                self.resizeDocks([self.docks["tree"], self.docks["properties"]], [320, 340], QtCore.Qt.Orientation.Horizontal)
                if self.settings is not None:
                    self.settings.remove("geometry")
                    self.settings.remove("windowState")

            def _fill_statistics(self):
                from . import mesh_stats
                scene = self.scene
                name = self.state.case
                stats = self._stats_cache.get(name)
                if stats is None:
                    try:
                        stats = mesh_stats.statistics(scene.bundle, scene.quality)
                    except Exception as exc:
                        self.stats_text.setPlainText(f"statistics failed: {exc}")
                        return
                    self._stats_cache[name] = stats
                columns = ["body", "elset", "elements", "nodes", "material", "volume_mm3", "min_scaled_jacobian", "max_aspect_ratio", "min_edge_mm", "max_edge_mm"]
                self.stats_table.setColumnCount(len(columns))
                self.stats_table.setHorizontalHeaderLabels(columns)
                self.stats_table.setRowCount(len(stats["bodies"]))
                for row, body in enumerate(stats["bodies"]):
                    for col, key in enumerate(columns):
                        value = body.get(key)
                        text = f"{value:.4g}" if isinstance(value, float) else str(value)
                        self.stats_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
                self.stats_table.resizeColumnsToContents()
                q = stats["quality"]
                lines = [f"Nodes {stats['counts']['nodes']}  elements {stats['counts']['elements']}  {stats['counts']['element_types']}",
                         f"Unused nodes {stats['counts']['unused_nodes']}  inverted corner elements {q['inverted_elements']}", ""]
                for measure in ("scaled_jacobian", "aspect_ratio", "min_edge_mm", "max_edge_mm", "volume_mm3"):
                    summary = q[measure]
                    if summary.get("count"):
                        lines.append(f"{measure:16s} min {summary['min']:.4g}  p1 {summary['p1']:.4g}  mean {summary['mean']:.4g}  p99 {summary['p99']:.4g}  max {summary['max']:.4g}")
                for note in q.get("notes", []):
                    lines.append(f"note: {note}")
                if stats.get("regions"):
                    lines += ["", "Regions (elements): " + ", ".join(f"{k} {v}" for k, v in stats["regions"].items())]
                if stats.get("mesher"):
                    lines += ["", "Mesher mesh.json: " + json.dumps(stats["mesher"], default=_json_default)[:1500]]
                sets = stats["sets"]
                lines += ["", f"Node sets: {len(sets['node_sets'])}, element sets: {len(sets['element_sets'])}, surfaces: {len(sets['surfaces'])} (see the model tree)"]
                self.stats_text.setPlainText("\n".join(lines))
                hist_dir = CACHE_DIR / "histograms"
                hist_dir.mkdir(parents=True, exist_ok=True)
                for label, (measure, title) in zip(self.hist_labels, (("scaled_jacobian", "Scaled Jacobian (corner nodes)"), ("aspect_ratio", "Aspect ratio"))):
                    png = hist_dir / f"{stats['mesh_hash'][:16]}-{measure}.png"
                    try:
                        if not png.exists():
                            mesh_stats.histogram_png(stats["histograms"][measure], title, png)
                        label.setPixmap(QtGui.QPixmap(str(png)))
                    except Exception as exc:
                        label.setText(f"{title}: {exc}")

            # ---- rendering ----------------------------------------------------------------
            def request_render(self):
                if self._building or self.scene is None or self._render_pending:
                    return
                self._render_pending = True
                QtCore.QTimer.singleShot(0, self.render_now)

            def render_now(self):
                self._render_pending = False
                scene = self.scene
                if scene is None:
                    return
                started = time.perf_counter()
                other = self.scenes.get(self.state.minus_case) if self.state.minus_case else None
                try:
                    info = scene.render(self.plotter, self.state, other)
                except Exception as exc:
                    self.log(f"render failed: {exc}")
                    self.status_label.setText(f"Render failed: {exc}")
                    return
                self.plotter.render()
                self._building = True
                if self.state.deform is None:
                    self.deform_spin.setValue(info["deformation_factor"])
                if not self.range_lock.isChecked():
                    self.range_min.setValue(info["range"][0])
                    self.range_max.setValue(info["range"][1])
                self._building = False
                text = f"{self.state.case} · {info['frame']} · {info['field']} · {info['location']} · deformation ×{info['deformation_factor']:.4g}"
                text += f" · range {info.get('range_mode')} {info['range'][0]:.4g}..{info['range'][1]:.4g} (true {info['range_true'][0]:.4g}..{info['range_true'][1]:.4g})"
                if info.get("limit"):
                    text += f" · volume above {info['limit']['limit']:g}: {100 * info['limit']['fraction']:.2f} %"
                text += f" · {info['acceptance']} · {time.perf_counter() - started:.2f} s"
                self.status_label.setText(text)
                for warning in info.get("warnings", []):
                    self.log(warning)
                self.last_info = info
                if self._picking_enabled is False:
                    self._enable_picking()

            def screenshot(self, path=None):
                if path is None:
                    if self.offscreen:
                        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        path = REVIEW_DIR / stamp / "gui-screenshot.png"
                    else:
                        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screenshot", str(Path.home() / "fempost.png"), "PNG (*.png)")
                        if not path:
                            return None
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                rendered = False
                if not self.offscreen:
                    try:
                        self.plotter.render()
                        self.plotter.screenshot(str(path))
                        rendered = True
                    except Exception as exc:
                        self.log(f"QtInteractor screenshot unavailable ({exc}); rendering offscreen instead")
                if not rendered:
                    # Under QT_QPA_PLATFORM=offscreen the Qt GL widget never paints, so the same
                    # scene state is rendered by the same Scene code into an offscreen plotter
                    # (the fempost render path). On the desktop the embedded view is used directly.
                    self.offscreen_screenshot(path)
                self.log(f"screenshot {path}")
                return path

            def offscreen_screenshot(self, path):
                from .scene import offscreen_plotter
                scene = self.scene
                if scene is None:
                    return None
                size = self.plotter.interactor.size()
                plotter = offscreen_plotter((max(size.width(), 800), max(size.height(), 600)))
                other = self.scenes.get(self.state.minus_case) if self.state.minus_case else None
                try:
                    info = scene.render(plotter, self.state, other)
                    plotter.screenshot(str(path))
                finally:
                    plotter.close()
                self.last_info = info
                return path

            def window_screenshot(self, path):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.grab().save(str(path))
                return path

            def save_scene(self, path):
                state = SceneState.from_dict(self.state.to_dict())
                state.camera = [list(map(float, p)) for p in self.plotter.camera_position]
                state.save(path)
                self.log(f"scene saved {path}")

            def save_scene_dialog(self):
                if self.offscreen:
                    return
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save scene spec", str(Path.home() / "scene.json"), "JSON (*.json)")
                if path:
                    self.save_scene(path)

            def load_scene(self, path):
                state = SceneState.load(path)
                if state.case and state.case not in self.scenes:
                    candidate = Path(state.case)
                    if candidate.exists():
                        state.case = self.open_run(candidate) or state.case
                if state.case not in self.scenes:
                    state.case = self.state.case
                self.state = state
                index = self.case_combo.findData(state.case)
                if index >= 0:
                    self.case_combo.blockSignals(True)
                    self.case_combo.setCurrentIndex(index)
                    self.case_combo.blockSignals(False)
                self._populate_for_case()
                self._sync_controls()
                self.request_render()

            def load_scene_dialog(self):
                if self.offscreen:
                    return
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load scene spec", str(Path.home()), "JSON (*.json)")
                if path:
                    self.load_scene(path)

            def _sync_controls(self):
                """Push the current state into the widgets (after loading a scene spec)."""
                self._building = True
                s = self.state
                if s.frame is not None and self.frame_combo.count():
                    idx = self.frame_combo.findData(s.frame if s.frame >= 0 else self.frame_combo.count() + s.frame)
                    self.frame_combo.setCurrentIndex(max(idx, 0))
                idx = self.field_combo.findData(s.field)
                if idx >= 0:
                    self.field_combo.setCurrentIndex(idx)
                self.location_combo.setCurrentIndex(max(self.location_combo.findData(s.location), 0))
                self._populate_minus()
                idx = self.minus_frame_combo.findData(s.minus_frame) if s.minus_frame is not None else 0
                self.minus_frame_combo.setCurrentIndex(max(idx, 0))
                if s.cmap in COLORMAPS:
                    self.cmap_combo.setCurrentText(s.cmap)
                self.range_lock.setChecked(s.clim is not None)
                if s.clim:
                    self.range_min.setValue(s.clim[0])
                    self.range_max.setValue(s.clim[1])
                self.range_mode_combo.setCurrentIndex(max(self.range_mode_combo.findData(s.range_mode or "robust"), 0))
                self.text_spin.setValue(float(s.text_scale or 1.0))
                for key, action in self.overlay_actions.items():
                    action.setChecked(bool(s.overlays.get(key)))
                self.log_check.setChecked(s.log_scale)
                self.opacity_slider.setValue(int(s.opacity * 100))
                self.limit_check.setChecked(s.limit is not None)
                if s.limit is not None:
                    self.limit_spin.setValue(s.limit)
                self.quality_check.setChecked(s.mesh_quality)
                self.deform_spin.setValue(s.deform or 0.0)
                self.ghost_action.setChecked(s.ghost)
                self.edges_action.setChecked(s.edges)
                self.parallel_action.setChecked(s.parallel)
                self.clip_box.setChecked(s.clip is not None)
                if s.clip:
                    self.clip_normal.setText(_vec_text(s.clip.get("normal", [1, 0, 0])))
                    self.clip_origin.setText(_vec_text(s.clip["origin"]) if s.clip.get("origin") else "")
                    self.clip_invert.setChecked(bool(s.clip.get("invert")))
                    self.clip_ghost.setChecked(bool(s.clip.get("ghost", True)))
                self.slice_box.setChecked(s.slice is not None)
                if s.slice:
                    self.slice_normal.setText(_vec_text(s.slice.get("normal", [0, 0, 1])))
                    self.slice_origin.setText(_vec_text(s.slice["origin"]) if s.slice.get("origin") else "")
                    self.slice_contours.setValue(int(s.slice.get("contours", 10)))
                    self.slice_only.setChecked(bool(s.slice.get("only")))
                self._populate_tree()
                self._building = False

            def closeEvent(self, event):
                self._anim = None
                try:
                    self.save_layout()
                except Exception as exc:
                    self.log(f"layout not saved: {exc}")
                try:
                    self.plotter.close()
                except Exception:
                    pass
                super().closeEvent(event)

        return _Window()


def run_app(runs, *, offscreen: bool = False, screenshot: str | None = None, scene: str | None = None,
            settings_path: str | None = None) -> int:
    """Start the viewer. Offscreen (tests, agents): render, write the screenshot(s) and exit."""
    if offscreen or screenshot:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        offscreen = True
    QtCore, QtGui, QtWidgets = _qt()
    if sys.platform == "win32" and not offscreen:
        # Give the process its own taskbar identity; otherwise Windows groups the window under
        # python.exe and shows Python's icon instead of ours.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("KurtsFEM.fempost.viewer")
        except Exception:  # pragma: no cover - cosmetic only
            pass
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    icon_path = Path(__file__).with_name("resources") / "fempost.ico"
    if icon_path.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    if not QtGui.QFontDatabase.families():
        QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    window = ViewerWindow(runs, offscreen=offscreen, scene_spec=scene, settings_path=settings_path)
    if icon_path.is_file():
        window.setWindowIcon(QtGui.QIcon(str(icon_path)))
    if offscreen:
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    if offscreen:
        window.render_now()
        app.processEvents()
        if screenshot:
            path = Path(screenshot)
            window.screenshot(path)
            window.window_screenshot(path.with_name(path.stem + "-window.png"))
        window.close()
        return 0
    return app.exec()
