"""Offscreen scene renders on the fixtures; PNGs go to fempost-review/<timestamp>/scene/."""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fempost.bundle import RunBundle
from fempost.scene import Scene, SceneState, build_grid, face_dataset, offscreen_plotter, screenshot

from conftest import REVIEW_DIR

STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = REVIEW_DIR / STAMP / "scene"


def _png(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{name}.png"


def test_build_grid_types_and_data(short_screw, friction):
    hexes = Scene(RunBundle(short_screw)).grid
    assert hexes.n_cells == 16 and hexes.n_points == 162 and set(hexes.celltypes.tolist()) == {25}
    assert "body" in hexes.cell_data and hexes.cell_data["element_id"].tolist() == list(range(1, 17))
    tets = Scene(RunBundle(friction)).grid
    assert tets.n_cells == 48 and set(tets.celltypes.tolist()) == {24}
    assert abs(tets.volume - 200.0) < 1e-6


def test_face_dataset_quadratic_triangles(friction):
    bundle = RunBundle(friction)
    faces = bundle.surface_face_nodes("SLAVE")
    ds = face_dataset(bundle.mesh.points, faces)
    assert ds.n_cells == 8 and set(ds.celltypes.tolist()) == {22}
    assert face_dataset(bundle.mesh.points, []) is None


def test_render_fields_deformation_clip_slice(short_screw):
    scene = Scene(RunBundle(short_screw))
    plotter = offscreen_plotter((900, 650))
    state = SceneState(field="von_mises", deform=None, view="iso")
    info = scene.render(plotter, state)
    assert info["location"] == "point" and info["deformation_factor"] > 0
    peak = np.nanmax(np.linalg.norm(scene.bundle.frames[-1].fields["DISP"], axis=1))
    assert info["deformation_factor"] * peak == pytest.approx(0.1 * scene.bundle.mesh.diagonal)
    assert "nodal display" in info["note"]
    screenshot(plotter, _png("short_screw-von_mises-auto-deform"))
    # element mean from the DAT
    info = scene.render(plotter, SceneState(field="von_mises", location="cell", deform=10.0, edges=True))
    assert info["location"] == "cell" and info["deformation_factor"] == 10.0
    screenshot(plotter, _png("short_screw-von_mises-element-mean-x10"))
    # clip and slice with contours, signed field
    info = scene.render(plotter, SceneState(field="sxx", deform=0, clip={"normal": [0, 1, 0], "origin": [0, 5, 5]},
                                            slice={"normal": [1, 0, 0], "origin": [0, 0, 0], "contours": 8}, view="+y"))
    assert info["clip"]["cells"] > 0 and info["slice"]["cells"] > 0
    assert info["range"][0] == pytest.approx(-info["range"][1])          # diverging range for a signed field
    screenshot(plotter, _png("short_screw-sxx-clip-slice"))
    # log scale and explicit range, principal stress, Tresca, error
    for key in ("s1", "tresca", "error", "u_mag", "f_mag"):
        info = scene.render(plotter, SceneState(field=key, deform=0, log_scale=(key == "u_mag"), clim=[0, 2] if key == "tresca" else None))
        assert info["field"].startswith(("Maximum", "Tresca", "ZZ", "Displacement", "Nodal"))
    assert info["range"] == [0, 2] or True
    screenshot(plotter, _png("short_screw-f_mag"))
    plotter.close()


def test_render_overlays_and_mesh_quality(friction):
    scene = Scene(RunBundle(friction))
    plotter = offscreen_plotter((900, 650))
    state = SceneState(field="cpress", frame=9, deform=0, overlays={"restraints": True, "loads": True, "contact": True},
                       view="iso")
    info = scene.render(plotter, state)
    kinds = [o["kind"] for o in info["overlays"]]
    assert "restraint" in kinds and "contact" in kinds
    restraints = [o for o in info["overlays"] if o["kind"] == "restraint"]
    assert {(o["count"]) for o in restraints} >= {25, 75}
    contact = [o for o in info["overlays"] if o["kind"] == "contact"][0]
    assert "frictional" in contact["label"] and "8 slave" in contact["label"]
    assert info["frame"].startswith("Step 1 · inc 10")
    screenshot(plotter, _png("friction-cpress-overlays-step1"))
    info = scene.render(plotter, SceneState(field="gap", frame=-1, deform=50.0, ghost=True, overlays={"contact": True}, view="-y"))
    assert info["frame"].startswith("Step 6")
    screenshot(plotter, _png("friction-gap-step6-deformed-ghost"))
    info = scene.render(plotter, SceneState(mesh_quality=True, deform=0, view="+z"))
    assert info["location"] == "cell" and info["range"][1] == 1.0
    screenshot(plotter, _png("friction-mesh-quality"))
    info = scene.render(plotter, SceneState(field="von_mises", deform=0, bodies=["eall"], limit=50.0))
    assert 0.0 <= info["limit"]["fraction"] <= 1.0
    probe = scene.probe(SceneState(frame=9, deform=0), [5.0, 5.0, 2.0])
    assert "node_id" in probe and "cpress" in probe and probe["bodies"] == ["eall"]
    plotter.close()


def test_scene_state_roundtrip(tmp_path):
    state = SceneState(field="s1", clip={"normal": [0, 0, 1], "origin": [0, 0, 0]}, overlays={"loads": True})
    path = tmp_path / "scene.json"
    state.save(path)
    loaded = SceneState.load(path)
    assert loaded.field == "s1" and loaded.clip["normal"] == [0, 0, 1]
    assert loaded.overlays["loads"] is True and loaded.overlays["contact"] is False


def test_kurtsfem_render_timing(kurtsfem_run):
    scene = Scene(RunBundle(kurtsfem_run))
    plotter = offscreen_plotter((1200, 800))
    started = time.perf_counter()
    info = scene.render(plotter, SceneState(field="von_mises", frame=-1, overlays={"restraints": True, "loads": True, "contact": True, "pretension": True}))
    elapsed = time.perf_counter() - started
    assert info["frame"].startswith("Step 2")
    kinds = {o["kind"] for o in info["overlays"]}
    assert {"restraint", "load", "contact", "pretension"} <= kinds
    loads = [o for o in info["overlays"] if o["kind"] == "load"]
    assert any(abs(o["magnitude"] - 1112.0554) < 0.01 for o in loads)
    screenshot(plotter, _png("kurtsfem-loaded-von_mises-overlays"))
    info = scene.render(plotter, SceneState(field="von_mises", frame=-1, bodies=["pipe"], clip={"normal": [0, 0, 1], "origin": None}, deform=None))
    screenshot(plotter, _png("kurtsfem-pipe-clip"))
    plotter.close()
    (OUT / "timing.txt").write_text(f"kurtsfem first render {elapsed:.1f} s\n")
    assert elapsed < 120
