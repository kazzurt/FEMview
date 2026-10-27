"""Run discovery, FRD arrays, frames/steps, cache and the JSON summary on every run family."""
import json

import numpy as np
import pytest

from fempost import frd as frd_module
from fempost.bundle import RunBundle, _json_default, discover


def test_discover_from_folder_and_members(short_screw):
    files = discover(short_screw)
    assert files.stem == "short_screw"
    assert files.frd.name == "short_screw.frd" and files.inp and files.dat and files.sta
    assert files.run_json and files.run_json.name == "short_screw.run.json"
    for member in ("short_screw.frd", "short_screw.inp", "short_screw.dat", "short_screw.run.json"):
        assert discover(short_screw / member).stem == "short_screw"


def test_frd_arrays_and_vtk_types(short_screw):
    bundle = RunBundle(short_screw)
    mesh = bundle.mesh
    assert mesh.n_nodes == 162 and mesh.n_elements == 16
    assert set(mesh.element_types.tolist()) == {4}
    assert set(mesh.vtk_types.tolist()) == {25}            # C3D20 -> VTK_QUADRATIC_HEXAHEDRON
    assert mesh.connectivity.shape == (16, 20) and (mesh.connectivity >= 0).all()
    assert mesh.material_names == ["STEEL"]
    assert mesh.node_index([1, 163, 142]).tolist() == [0, 161, -1]   # reference node 142 is not in the FRD
    assert np.allclose(mesh.points[mesh.node_index(1)], [-10.0, 0.0, 0.0])


def test_frames_and_steps_from_sta(short_screw):
    bundle = RunBundle(short_screw)
    frames = bundle.frames
    assert len(frames) == 10
    assert [(f.step, f.increment) for f in frames] == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                                                       (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
    assert np.allclose([f.time for f in frames], np.arange(1, 11) * 0.2)
    last = frames[-1]
    assert set(last.fields) == {"DISP", "STRESS", "FORC", "ERROR"}
    assert last.fields["DISP"].shape == (162, 3) and last.fields["STRESS"].shape == (162, 6)
    assert last.components["STRESS"] == ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]
    assert np.isfinite(last.fields["DISP"]).all()
    assert bundle.last_frame_of_step(2) is last


def test_friction_fixture_contact_and_tet10(friction):
    bundle = RunBundle(friction)
    mesh = bundle.mesh
    assert mesh.n_nodes == 150 and mesh.n_elements == 48
    assert set(mesh.vtk_types.tolist()) == {24}            # C3D10 -> VTK_QUADRATIC_TETRA
    frames = bundle.frames
    assert len(frames) == 60
    assert sorted({f.step for f in frames}) == [1, 2, 3, 4, 5, 6]
    assert [f.increment for f in frames if f.step == 3] == list(range(1, 11))
    assert frames[-1].components["CONTACT"] == ["COPEN", "CSLIP1", "CSLIP2", "CPRESS", "CSHEAR1", "CSHEAR2"]
    pairs = bundle.contact_pairs()
    assert len(pairs) == 1 and pairs[0]["slave"] == "SLAVE" and pairs[0]["category"] == "frictional"
    assert pairs[0]["slave_faces"] == 8 and pairs[0]["master_faces"] == 8
    faces = bundle.surface_face_nodes("SLAVE")
    assert len(faces) == 8 and all(len(f) == 6 for f in faces)


def test_cache_roundtrip(short_screw):
    first = RunBundle(short_screw)
    first.mesh
    assert first.frd_provenance["cache"].endswith(".frd.npz")
    second = RunBundle(short_screw)
    second.mesh
    assert second.frd_provenance["cache_hit"] is True
    assert np.array_equal(first.mesh.connectivity, second.mesh.connectivity)
    for a, b in zip(first.frames, second.frames):
        assert (a.step, a.increment, a.time) == (b.step, b.increment, b.time)
        for name in a.fields:
            assert np.allclose(a.fields[name], b.fields[name], equal_nan=True)
    assert first.mesh.mesh_hash() == second.mesh.mesh_hash()


def test_no_cache_option(short_screw):
    bundle = RunBundle(short_screw, use_cache=False)
    bundle.mesh
    assert bundle.frd_provenance["cache_hit"] is False and "cache" not in bundle.frd_provenance


def test_summary_is_json_and_complete(short_screw):
    bundle = RunBundle(short_screw)
    summary = bundle.summary()
    text = json.dumps(summary, default=_json_default)
    assert json.loads(text)["mesh"]["elements"] == 16
    assert summary["run"]["family"] == "generic"
    assert summary["solver"]["solver_status"] == "completed" and summary["solver"]["accepted"] is False
    assert summary["solver"]["kind"] == "converged" and "converged (2 steps, 10 increments, 20 iterations)" in summary["solver"]["label"]
    assert "diagnostic" not in summary["solver"]["label"]          # converged solve without an acceptance record is not "diagnostic only"
    assert summary["bodies"][0]["name"] == "bolt" and summary["bodies"][0]["material"] == "STEEL"
    assert [s["number"] for s in summary["steps"]] == [1, 2]
    restraints = {(r["step"], r["target"]): r for r in summary["restraints"]}
    assert restraints[(0, "NALL")]["dofs"] == "Uy+Uz" and restraints[(0, "NALL")]["nodes"] == 141
    assert restraints[(2, "TIP")]["value"] == pytest.approx(5e-5)
    cload = summary["loads"]["cload"][0]
    assert cload["step"] == 1 and cload["resultant_N"] == [100.0, 0.0, 0.0]
    assert np.allclose(cload["centroid_mm"], [-10.0, 0.0, 0.0])   # reference node 142 located from the deck
    assert summary["dat"]["stress_times"] == pytest.approx(list(np.arange(1, 11) * 0.2))
    assert len(summary["frames"]) == 10


def test_cli_info_writes_json(short_screw, tmp_path, capsys):
    from fempost.__main__ import main
    out = tmp_path / "info.json"
    assert main(["info", str(short_screw), "--out", str(out), "--no-frames"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mesh"]["nodes"] == 162 and "frames" not in data
    assert main(["info", str(short_screw / "short_screw.frd"), "--compact"]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["stem"] == "short_screw"
    assert main(["info", str(tmp_path / "missing")]) == 2


def test_kurtsfem_family(kurtsfem_run):
    bundle = RunBundle(kurtsfem_run)
    assert bundle.family == "kurtsfem"
    summary = bundle.summary(include_frames=True)
    assert [f["step"] for f in summary["frames"]] == [1, 2]
    assert "preload" in summary["frames"][0]["step_label"]
    assert {b["name"] for b in summary["bodies"]} == {"pipe", "hub", "screw_1", "screw_2", "screw_3"}
    assert "UNACCEPTED · DIAGNOSTIC ONLY" in summary["solver"]["label"] and summary["solver"]["kind"] == "diagnostic"
    gates = {g["name"]: g["status"] for g in summary["sidecars"]["results"]["gates"]}
    assert gates["solver_completed"] == "passed"
    assert summary["sidecars"]["results"]["stages"]["loaded"]["components"]["pipe"]["vtu"] == "loaded-pipe.vtu"
    pretension = [p for p in summary["pretension"] if p.get("source") == "assembly.json"]
    assert [p["node"] for p in pretension] == [176843, 176844, 176845]
    loads = {c["step"]: c for c in summary["loads"]["cload"]}
    assert loads[1]["resultant_N"] == pytest.approx([300.0, 0.0, 0.0])
    assert loads[2]["resultant_N"][1] == pytest.approx(1112.0554, abs=1e-3)
    assert len(summary["contact_pairs"]) == 4
    assert discover(kurtsfem_run / "structural-results/results.json").stem == "dodec_assembly"
    assert discover(kurtsfem_run / "dodec_assembly.assembly.json").stem == "dodec_assembly"


def test_codex_family(codex_run):
    bundle = RunBundle(codex_run)
    assert bundle.family == "study" and bundle.name == "matched_mouth_coarse"
    bodies = bundle.bodies()
    assert list(bodies) == ["hub", "pipe", "insert-loaded", "insert-other1", "insert-other2", "screw-1", "screw-2", "screw-3"]
    assert bundle.body_elset("hub") == "BODY0"
    codes, names = bundle.regions()
    assert len(codes) == bundle.mesh.n_elements and "HUB_BODY" in names
    pairs = {(p["slave"], p["master"]): p for p in bundle.contact_pairs()}
    assert pairs[("BORE", "INSERT_BEARING")]["type"] == "BEARING"
    assert pairs[("BITE_0", "SCREW_BITE_0")]["category"] == "tied"
    assert bundle.model_arrays() is not None and "forces" in bundle.model_arrays()
    group = bundle.cload_groups()[0]
    assert group["nodes"] == 4949 and np.allclose(group["resultant_N"], [-1108.058, -1378.556, 48.8496], atol=1e-3)
    assert discover(codex_run / "fields.npz").stem == "job"
    assert discover(codex_run / "results.json").stem == "job"
    summary = bundle.summary(include_frames=False)
    assert summary["sidecars"]["mesh"]["tet10"] == 68963
    assert "HUB_BODY" in summary["sidecars"]["results"]["regions"]


def test_frd_scan_block_times_handles_glued_format(tmp_path):
    sample = tmp_path / "t.frd"
    sample.write_text("  100CL  1012.00000E-001         162                     0    1           1\n"
                      "  100CL  101 1.000000000      119941                     0    1           1\n")
    assert frd_module._scan_block_times(sample) == [0.2, 1.0]
