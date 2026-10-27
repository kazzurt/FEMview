"""Follow-up program (live-test fixes, convergence, Phase 3): items 1-12 of fempost/STATUS.md."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from fempost import convergence as cv
from fempost.__main__ import main
from fempost.bundle import RunBundle
from fempost.scene import (DEFAULT_OVERLAYS, PLOT_GROUPS, Scene, SceneState, animate, animation_factors, apply_plot_group,
                           offscreen_plotter, plot_group_names)

from conftest import CODEX_RUN, FIXTURES

REFINED = Path(os.environ.get("FEMVIEW_REFINED_FIXTURE", str(Path(__file__).resolve().parent / "external/refined-run")))
UNCONVERGED_STA = FIXTURES / "backend-tests-cancel-lifecycle-final/test_sta_ignores_unconverged_a0/job.sta"


# ---- item 9: convergence parsers -------------------------------------------------------------
def test_parse_sta_keeps_unconverged_attempts():
    if not UNCONVERGED_STA.is_file():
        pytest.skip("unconverged STA fixture missing")
    rows = cv.parse_sta(UNCONVERGED_STA)
    assert [(r["attempt"], r["converged"], r["iterations"]) for r in rows] == [(1, False, 20), (2, True, 5)]
    summary = cv.summarize(UNCONVERGED_STA, None, None)
    assert summary["converged"] is True and summary["totals"] == {"increments": 1, "iterations": 25, "cutbacks": 1, "steps": 1}
    assert cv.short_text(summary) == "converged (1 increment, 25 iterations, 1 cutback)"


def test_convergence_fixture_two_steps(short_screw, tmp_path):
    sta, cvg, log = (short_screw / f"short_screw.{ext}" for ext in ("sta", "cvg", "stdout.log"))
    summary = cv.summarize(sta, cvg, log, expected_steps=2, expected_step_times=[1.0, 1.0])
    assert summary["converged"] is True and summary["totals"]["steps"] == 2 and summary["totals"]["increments"] == 10
    assert len(summary["iterations"]) == 20 and summary["iterations"][0]["disp_correction_pct"] == pytest.approx(100.0)
    assert summary["iterations"][-1]["converged"] is True        # merged from the stdout log
    assert summary["total_time_s"] is not None and summary["job_finished"]
    rows = cv.table_rows(summary)
    assert [r["increments"] for r in rows] == [5, 5] and rows[1]["end_total_time"] == pytest.approx(2.0)
    png = cv.convergence_figure(summary, tmp_path / "conv.png")
    assert png.stat().st_size > 5000


def test_convergence_log_cutbacks_and_new_steps(tmp_path):
    log = tmp_path / "x.stdout.log"
    log.write_text("\n".join([
        " increment 1 attempt 1 ", " increment size= 1.000000e+000", " iteration 1", " Number of contact spring elements=10",
        " average force= 1.0", " largest residual force= 2.0 in node 5 and dof 1", " largest increment of disp= 1.0",
        " largest correction to disp= 0.5 in node 7 and dof 2", " no convergence", " iteration 2", " largest residual force= 0.1 in node 5 and dof 1",
        " largest correction to disp= 0.01 in node 7 and dof 2", "divergence allowed: number of contact elements stabilized",
        " no convergence", " iteration 3", " convergence", " increment 1 attempt 1 ", " increment size= 5.000000e-001",
        " iteration 1", " no convergence", " the increment size is decreased to 2.500000e-001", " increment 1 attempt 2 ",
        " increment size= 2.500000e-001", " iteration 1", " convergence; the increment size is increased to 3.750000e-001",
        " Job finished", "Total CalculiX Time: 12.5"]))
    parsed = cv.parse_log(log)
    assert parsed["steps_seen"] == 2 and parsed["total_time_s"] == 12.5 and parsed["job_finished"]
    kinds = [e["kind"] for e in parsed["events"]]
    assert "cutback" in kinds and "reattempt" in kinds and "divergence_allowed" in kinds and "size_increase" in kinds
    assert parsed["iterations"][0]["largest_residual_force"] == 2.0 and parsed["iterations"][0]["residual_node"] == 5
    assert parsed["iterations"][1]["messages"] and parsed["iterations"][2]["converged"] is True
    assert [i["step"] for i in parsed["increments"]] == [1, 2, 2] and parsed["increments"][2]["attempt"] == 2


def test_convergence_prototype_run():
    if not (REFINED / "job.cvg").is_file():
        pytest.skip("own_support_refined missing")
    summary = cv.summarize(REFINED / "job.sta", REFINED / "job.cvg", REFINED / "job.stdout.log")
    its = summary["iterations"]
    assert len(its) == 9 and summary["converged"] is True
    assert its[0]["disp_correction_pct"] == pytest.approx(100.0) and its[-1]["disp_correction_pct"] == pytest.approx(0.02852)
    assert its[0]["contact_elements"] == 1841268 and its[-1]["contact_elements"] == 1362203
    assert its[-1]["largest_disp_correction"] == pytest.approx(5.41516e-4) and its[-1]["correction_node"] == 54894
    assert summary["equations"] == 696552


# ---- item 5: acceptance label from the run's own evidence ----------------------------------
def test_acceptance_kinds(short_screw, kurtsfem_run, tmp_path):
    bar = RunBundle(short_screw).acceptance()
    assert bar["kind"] == "converged" and bar["label"].startswith("completed · converged (2 steps, 10 increments, 20 iterations)")
    assert "diagnostic" not in bar["label"] and bar["accepted"] is False and "|" not in bar["short_label"]
    kf = RunBundle(kurtsfem_run).acceptance()
    assert kf["kind"] == "diagnostic" and "UNACCEPTED · DIAGNOSTIC ONLY" in kf["label"]
    assert "converged (2 steps, 14 increments, 70 iterations)" in kf["label"]
    assert "force_equilibrium" in kf["label"] and len(kf["gates"]) >= 10          # gates kept verbatim
    assert kf["equilibrium"] and kf["equilibrium"][-1]["source"].endswith("results.json")
    # a failed / unconverged solve stays diagnostic only
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("short_screw.inp", "short_screw.frd"):
        shutil.copy(short_screw / name, broken / name)
    (broken / "short_screw.run.json").write_text(json.dumps({"status": "failed", "expected_steps": 2, "completed_steps": [1]}))
    (broken / "short_screw.sta").write_text("SUMMARY OF JOB INFORMATION\n  STEP      INC     ATT  ITRS     TOT TIME     STEP TIME      INC TIME\n"
                                             "     1          1     1U   20  0.100000E+00  0.100000E+00  0.100000E+00\n")
    bad = RunBundle(broken, use_cache=False).acceptance()
    assert bad["kind"] == "diagnostic" and "diagnostic only" in bad["label"] and "NOT converged" in bad["label"]
    assert "1 of 2 steps completed" in bad["label"]


def test_acceptance_evidence_codex(codex_run):
    acc = RunBundle(codex_run).acceptance()
    assert acc["kind"] == "evidence" and acc["label"].startswith("completed · converged (")
    entry = acc["equilibrium"][0]
    assert entry["source"] == "results.json" and entry["applied_source"] == "model.json"
    assert 0 < entry["force_fraction"] < 1e-3 and "model diagonal" in entry["moment_fraction_basis"]
    assert "equilibrium residual |F| =" in acc["label"] and "diagnostic" not in acc["label"]


# ---- items 2, 3, 6, 7, 8: scene --------------------------------------------------------------
def test_default_overlays_and_glyph_sizes(friction):
    state = SceneState()
    assert state.overlays == DEFAULT_OVERLAYS and state.overlays["restraints"] and state.overlays["loads"] and not state.overlays["contact"]
    scene = Scene(RunBundle(friction))
    plotter = offscreen_plotter((800, 600))
    info = scene.render(plotter, SceneState(field="von_mises", frame=9))
    restraints = [o for o in info["overlays"] if o["kind"] == "restraint"]
    assert restraints, "restraints are drawn by default"
    diagonal = scene.bundle.mesh.diagonal
    for entry in restraints:
        assert 0.008 * diagonal - 1e-9 <= entry["glyph_radius_mm"] <= 0.02 * diagonal + 1e-9
        assert len(entry["extent_mm"]) == 3 and entry["count"] > 0
    assert info["font_scale"] == pytest.approx(1.0)
    info = scene.render(plotter, SceneState(field="von_mises", frame=9, text_scale=1.5))
    assert info["font_scale"] == pytest.approx(1.5)
    plotter.close()
    plotter = offscreen_plotter((800, 900))
    info = scene.render(plotter, SceneState(field="von_mises", frame=9))
    assert info["font_scale"] == pytest.approx(1.25)                 # fonts grow with the window height
    plotter.close()


def test_load_overlay_resultant_and_faces(short_screw):
    scene = Scene(RunBundle(short_screw))
    plotter = offscreen_plotter((800, 600))
    info = scene.render(plotter, SceneState(field="von_mises"))
    loads = [o for o in info["overlays"] if o["kind"] == "load"]
    assert loads and loads[0]["magnitude"] == pytest.approx(100.0) and "resultant_tail" in loads[0]
    assert "resultant_text" in loads[0] and "100 N" in loads[0]["resultant_text"]
    plotter.close()


def test_robust_range_and_true_extrema(codex_run):
    scene = Scene(RunBundle(codex_run))
    plotter = offscreen_plotter((800, 600))
    info = scene.render(plotter, SceneState(field="von_mises", overlays={}))
    assert info["range_mode"] == "robust" and info["range"][1] < info["range_true"][1]
    values = scene.field_result(SceneState(field="von_mises")).values
    assert info["range_true"][1] == pytest.approx(float(np.nanmax(values)))
    assert info["range"][1] == pytest.approx(float(np.nanpercentile(values, 99.5)), rel=1e-6)
    full = scene.render(plotter, SceneState(field="von_mises", range_mode="full"))
    assert full["range_mode"] == "full" and full["range"] == pytest.approx(full["range_true"])
    manual = scene.render(plotter, SceneState(field="von_mises", clim=[0, 150]))
    assert manual["range_mode"] == "manual" and manual["range"] == [0.0, 150.0]
    # visible bodies only: hiding the peak body changes the true maximum
    pipe = scene.render(plotter, SceneState(field="von_mises", bodies=["pipe"]))
    assert pipe["range_true"][1] <= info["range_true"][1]
    # signed field keeps a diverging band centred on zero under the robust range
    signed = scene.render(plotter, SceneState(field="s1"))
    assert signed["range"][0] == pytest.approx(-signed["range"][1])
    plotter.close()


def test_contact_field_blanking(friction):
    scene = Scene(RunBundle(friction))
    state = SceneState(field="cpress", frame=9)
    result = scene.field_result(state)
    mask = scene.contact_node_mask()
    assert mask.sum() > 0 and np.isnan(result.values[~mask]).all() and np.isfinite(result.values[mask]).all()
    assert "CPRESS > 0 is compression" in result.note and "blanked" in result.note
    plotter = offscreen_plotter((600, 450))
    info = scene.render(plotter, SceneState(field="cpress", frame=9, overlays={"contact": True}))
    assert info["range_true"][0] >= float(np.nanmin(result.values[mask])) - 1e-9
    plotter.close()


def test_probe_ray(short_screw):
    scene = Scene(RunBundle(short_screw))
    probe = scene.probe_ray(SceneState(deform=0), [0.0, 50.0, 5.0], [0.0, -1.0, 0.0])
    assert probe["coordinates_mm"][0] == pytest.approx(0.0) and probe["coordinates_mm"][2] == pytest.approx(5.0)
    assert probe["ray_distance_mm"] == pytest.approx(0.0, abs=1e-6) and "von_mises" in probe
    with pytest.raises(ValueError):                      # every node lies behind the ray origin
        scene.probe_ray(SceneState(deform=0), [0, 50, 5], [0, 1, 0])


# ---- items 10, 11: preload-versus-loaded preset and animation ------------------------------
def test_loaded_minus_preload_plot_group(short_screw):
    bundle = RunBundle(short_screw)
    scene = Scene(bundle)
    names = plot_group_names(scene.available_fields(bundle.frames[-1]), n_steps=2)
    preset = [n for n in names if n.startswith("Loaded minus preload: von Mises")][0]
    assert not [n for n in plot_group_names(scene.available_fields(bundle.frames[-1]), n_steps=1) if "Loaded minus" in n]
    state = apply_plot_group(SceneState(), preset, bundle)
    assert state.frame == bundle.last_frame_of_step(2).index and state.minus_frame == bundle.last_frame_of_step(1).index
    assert state.overlays["restraints"] and state.cmap == "coolwarm"
    result = scene.field_result(state)
    a = scene.field_result(SceneState(field="von_mises", frame=state.frame)).values
    b = scene.field_result(SceneState(field="von_mises", frame=state.minus_frame)).values
    assert "change A - B" in result.label and np.allclose(np.nan_to_num(result.values), np.nan_to_num(a - b))
    plotter = offscreen_plotter((600, 450))
    info = scene.render(plotter, state)
    assert "minus" in info["frame"] or info["range"][0] == pytest.approx(-info["range"][1])
    plotter.close()
    assert len(PLOT_GROUPS) == 14


def test_animation_gif_and_png_sequence(short_screw, tmp_path):
    scene = Scene(RunBundle(short_screw))
    plotter = offscreen_plotter((400, 300))
    factors = animation_factors(10.0, 8)
    assert len(factors) == 8 and factors[0] == 0.0 and max(factors) == pytest.approx(10.0, rel=0.05)
    record = animate(scene, plotter, SceneState(field="von_mises", deform=10.0), tmp_path / "sweep.gif", frames=4, fps=5)
    assert (tmp_path / "sweep.gif").stat().st_size > 1000 and record["frames"] == 4 and record["peak_factor"] == 10.0
    record = animate(scene, plotter, SceneState(field="von_mises", deform=10.0), tmp_path / "seq.png", frames=3)
    assert len(record["files"]) == 3 and all(Path(f).exists() for f in record["files"])
    plotter.close()


# ---- item 12: DAT contact prints as per-face fields ------------------------------------------
def test_dat_contact_prints_as_fields(friction, tmp_path, capsys):
    bundle = RunBundle(friction)
    scene = Scene(bundle)
    frame = bundle.frames[9]                                   # step 1 · inc 10 · t = 1
    prints = bundle.dat.contact_prints(frame.time)
    assert prints["stress"] is not None and prints["displacement"] is not None
    assert prints["stress"]["rows"] > len(prints["stress"]["element_ids"])      # several contact points per slave face
    keys = scene.available_fields(frame)
    assert {"dat_cpress", "dat_gap", "dat_slip", "dat_cshear"} <= set(keys) and "dat_cpress" not in scene.available_fields.__self__.bundle.frames[0].fields
    record = scene.dat_contact(frame)
    assert len(record["faces"]) == 8 == len(record["values"]["dat_cpress"])      # the eight SLAVE faces of the deck
    result = scene.field_result(SceneState(field="dat_cpress", frame=9))
    assert result.location == "face" and "CONTACT PRINT" in result.note
    frd = scene.field_result(SceneState(field="cpress", frame=9)).values
    assert np.nanmean(result.values) == pytest.approx(np.nanmean(frd[np.isfinite(frd)]), rel=0.05)   # DAT face mean vs FRD nodal
    gap = scene.field_result(SceneState(field="dat_gap", frame=9)).values
    assert np.all(gap < 0)                                     # linear pressure-overclosure: pressure means penetration
    plotter = offscreen_plotter((640, 480))
    info = scene.render(plotter, SceneState(field="dat_cpress", frame=9, deform=0, overlays={"contact": True}, clip={"normal": [0, 0, 1]}))
    assert info["location"] == "face" and info["dat_contact"]["faces"] == 8 and any("clip" in w for w in info["warnings"])
    plotter.close()
    names = plot_group_names(keys)
    assert "Contact pressure from DAT prints (per slave face)" in names
    out = tmp_path / "dat.png"
    assert main(["render", str(friction), "--out", str(out), "--field", "dat_cpress", "--frame", "9", "--size", "480,360"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["info"]["location"] == "face" and out.stat().st_size > 1000


# ---- item 13: frame / beam runs ---------------------------------------------------------------
CUBE_RUN = Path(os.environ.get("FEMVIEW_CUBE_FIXTURE", str(Path(__file__).resolve().parent / "external/cube-run")))


def test_dat_rows_with_text_columns():
    from fempost.dat import _parse_rows
    rows = _parse_rows(["  1   1 -1.780763E+00 -1.764689E+00 -6.475996E+00  3.284674E-02 -3.612015E-01  1.715814E-01 OR1",
                        "  1   2  1.719698E+00  1.712196E+00 -6.291392E+00  1.225225E-01 -1.094409E-01 -1.194825E-01 OR1"])
    assert rows.shape == (2, 8) and rows[0, 0] == 1 and rows[1, 1] == 2 and rows[1, 2] == pytest.approx(1.719698)


def test_beam_run_tubes_section_forces_and_table(tmp_path, capsys):
    if not (CUBE_RUN / "cube.frd").is_file():
        pytest.skip("cube frame run missing")
    bundle = RunBundle(CUBE_RUN)
    info = bundle.beam_info()
    assert info["is_beam"] and info["section_forces"] and info["output_2d"] and info["radius_mm"] == pytest.approx(16.17, abs=0.05)
    assert set(bundle.bodies()) == {"beams rigidoffset", "beams equivalent"}       # 410 one-member elsets grouped by material
    scene = Scene(bundle)
    frame = bundle.frames[-1]
    keys = scene.available_fields(frame)
    assert "sf_fmag" in keys and "sf_m3" in keys and "von_mises" not in keys       # section forces replace stresses
    table = bundle.beam_table(frame)
    assert len(table) == 410 and table[0]["element"] == 1 and table[0]["material"] == "RIGIDOFFSET"
    # the FRD STRESS block equals the section resultants Codex's pipeline wrote beside the run
    sidecar = json.loads((CUBE_RUN / "section-forces.json").read_text())
    entry = sidecar[0]
    row = int(bundle.mesh.node_index([entry["node"]])[0])
    assert np.allclose(frame.fields["STRESS"][row, :6], entry["section_resultants"], rtol=1e-4)
    record = bundle.dat.element_mean_stress(frame.time)
    assert record is not None and len(record["element_ids"]) == 410                # rows carry the orientation name
    plotter = offscreen_plotter((640, 480))
    rendered = scene.render(plotter, SceneState(field="sf_fmag"))
    assert rendered["beam"]["tubes"] and rendered["beam"]["section_forces"] and rendered["range_true"][1] == pytest.approx(1617.7, abs=0.5)
    plotter.close()
    out = tmp_path / "beam.png"
    assert main(["render", str(CUBE_RUN), "--out", str(out), "--field", "sf_m3", "--deform", "0", "--size", "480,360"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["info"]["beam"]["tubes"] and out.stat().st_size > 1000
    assert main(["info", str(CUBE_RUN), "--no-frames", "--compact"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["beams"]["elements"] == 410 and summary["beams"]["section_forces"] is True


# ---- CLI ---------------------------------------------------------------------------------------
def test_render_cli_convergence_animate_and_overlay_none(short_screw, tmp_path, capsys):
    out = tmp_path / "conv.png"
    assert main(["render", str(short_screw), "--convergence", str(out)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert out.stat().st_size > 5000 and record["converged"] is True and record["totals"]["steps"] == 2
    assert record["steps"][0]["iterations"] == 10
    fig = tmp_path / "fig.png"
    assert main(["render", str(short_screw), "--out", str(fig), "--overlay", "none", "--range", "full", "--text-scale", "1.4",
                 "--size", "640,480", "--convergence"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["overlays"] == [] and record["info"]["range_mode"] == "full" and record["info"]["font_scale"] == pytest.approx(1.4 * (0.5 + 0.5 * 480 / 600))
    assert Path(record["convergence"]["convergence"]).exists()
    assert main(["render", str(short_screw), "--out", str(fig), "--no-overlay", "loads", "--size", "480,360"]) == 0
    record = json.loads(capsys.readouterr().out)
    kinds = {o["kind"] for o in record["overlays"]}
    assert "restraint" in kinds and "load" not in kinds
    gif = tmp_path / "anim.gif"
    assert main(["render", str(short_screw), "--animate", str(gif), "--frames", "3", "--size", "320,240"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert gif.exists() and record["animation"]["frames"] == 3
    assert main(["render", str(short_screw), "--out", str(fig), "--minus-frame", "4", "--step", "2", "--size", "320,240"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert "change A - B" in record["info"]["field"]
    assert main(["render", str(short_screw), "--out", str(fig), "--plot-group", "Loaded minus preload: von Mises (step 2 - step 1, same case)",
                 "--size", "320,240"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["state"]["minus_frame"] == 4 and record["state"]["frame"] == 9


# ---- items 1, 3, 4, 7, 8, 9, 11: the viewer window -------------------------------------------
def test_viewer_followup(short_screw, friction, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from fempost.app import ViewerWindow
    app = QApplication.instance() or QApplication([])
    ini = tmp_path / "layout.ini"
    window = ViewerWindow([friction, short_screw], offscreen=True, settings_path=str(ini))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()
    window.render_now()
    # item 3: overlays on by default, toolbar buttons mirror the tree both ways
    assert window.state.overlays["restraints"] and window.overlay_actions["restraints"].isChecked()
    assert any(o["kind"] == "restraint" for o in window.last_info["overlays"])
    window.overlay_actions["contact"].setChecked(True)
    window.render_now()
    defs = window.tree.topLevelItem(0).child(1)
    contact_item = [defs.child(k) for k in range(defs.childCount()) if defs.child(k).text(0).startswith("Contact")][0]
    assert contact_item.checkState(0) == Qt.CheckState.Checked and window.state.overlays["contact"]
    contact_item.setCheckState(0, Qt.CheckState.Unchecked)
    assert not window.overlay_actions["contact"].isChecked() and not window.state.overlays["contact"]
    # item 1: a typed max applies immediately and ticks Lock; unticking Lock returns to automatic; Fit to visible
    assert not window.range_lock.isChecked()
    window.range_max.setValue(2.0)
    window._range_typed()
    window.render_now()
    assert window.range_lock.isChecked() and window.state.clim[1] == 2.0 and window.last_info["range_mode"] == "manual"
    window.range_lock.setChecked(False)
    window.render_now()
    assert window.state.clim is None and window.last_info["range_mode"] in ("robust", "full")
    window.fit_range_to_visible()
    window.render_now()
    assert window.range_lock.isChecked() and window.state.clim == pytest.approx(window.last_info["range_true"])
    window.range_lock.setChecked(False)
    window.range_mode_combo.setCurrentIndex(1)
    window.render_now()
    assert window.state.range_mode == "full" and window.last_info["range_mode"] == "full"
    window.range_mode_combo.setCurrentIndex(0)
    # item 8: text size setting reaches the render
    window.text_spin.setValue(1.5)
    window.render_now()
    assert window.last_info["font_scale"] == pytest.approx(1.5 * window.scene.font_scale(window.plotter, SceneState()))
    # item 7: picking from a screen point (cell picker or screen-ray fallback), both offscreen
    probe = window.pick_screen(window.plotter.interactor.width() / 2, window.plotter.interactor.height() / 2)
    assert probe is not None and probe["pick"] in ("vtkCellPicker", "screen ray") and "node_id" in probe
    assert str(probe["node_id"]) in window.probe_text.toPlainText()
    # item 9: convergence tab
    index = [i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "Convergence"][0]
    window.tabs.setCurrentIndex(index)
    app.processEvents()
    assert window.state.case == "explicit-bar-verification"      # the last run opened is the current case
    assert window.conv_table.rowCount() == 2 and "Convergence (short_screw.sta, short_screw.cvg, short_screw.stdout.log)" in window.conv_text.toPlainText()
    assert window.conv_figure.pixmap() is not None and not window.conv_figure.pixmap().isNull()
    # item 10: minus frame control
    window.minus_frame_combo.setCurrentIndex(1)
    window.render_now()
    assert window.state.minus_frame == 0 and "change A - B" in window.last_info["field"]
    window.minus_frame_combo.setCurrentIndex(0)
    # item 11: play animation runs its frames and restores the factor
    window.auto_deform()
    window.render_now()
    window.play_action.setChecked(True)
    import time
    deadline = time.perf_counter() + 90
    while window._anim is not None and time.perf_counter() < deadline:      # frames fire from 30 ms single-shot timers
        app.processEvents()
        time.sleep(0.02)
    assert window._anim is None and window.state.deform is None and not window.play_action.isChecked()
    assert any("animation:" in line for line in window.log_lines)
    # item 4: View menu, floating docks dock back, reset layout, QSettings round trip
    titles = [a.text() for a in window.view_menu.actions()]
    assert {"Model tree", "Properties", "Details", "Main toolbar", "Reset layout"} <= set(titles)
    details = window.docks["details"]
    details.setFloating(True)
    app.processEvents()
    assert details.isFloating()
    details.setFloating(False)
    app.processEvents()
    assert not details.isFloating()
    window.docks["tree"].hide()
    window.reset_layout()
    app.processEvents()
    assert window.docks["tree"].isVisible() and window.centralWidget().minimumWidth() >= 480
    window.docks["properties"].hide()
    window.text_spin.setValue(1.3)
    window.close()
    app.processEvents()
    assert ini.exists()
    again = ViewerWindow([], offscreen=True, settings_path=str(ini))
    assert not again.docks["properties"].isVisible() and again.state.text_scale == pytest.approx(1.3)
    again.close()
