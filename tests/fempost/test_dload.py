"""Item 21: the lines of a ``*DLOAD`` block are grouped into logical loads (face-by-face pressure decks)."""
import json
import shutil

import pytest

from fempost.__main__ import main
from fempost.bundle import RunBundle
from fempost.inp import read_deck
from fempost.scene import Scene, SceneState, offscreen_plotter


def _face_by_face_run(friction, tmp_path):
    """The friction fixture (C3D10) with its steps replaced by one step of per-element *DLOAD lines."""
    folder = tmp_path / "dl"
    folder.mkdir()
    shutil.copy(friction / "friction.frd", folder / "dl.frd")
    text = (friction / "friction.inp").read_text(encoding="utf-8", errors="replace")
    head = text[:text.index("*STEP")]
    elements = read_deck(friction / "friction.inp").element_ids.tolist()
    assert len(elements) >= 40
    block1 = "".join(f"{e},{'P1' if k % 2 else 'P2'},0.25\n" for k, e in enumerate(elements[:40]))
    block2 = "".join(f"{e},P3,-1.5\n" for e in elements[:10])
    (folder / "dl.inp").write_text(head + "*STEP\n*STATIC\n*DLOAD\n" + block1 + "*DLOAD\n" + block2
                                   + "*NODE FILE\nU\n*END STEP\n", encoding="utf-8")
    return folder


def test_dload_lines_grouped_into_logical_loads(friction, tmp_path, capsys):
    folder = _face_by_face_run(friction, tmp_path)
    bundle = RunBundle(folder, use_cache=False)
    assert len(bundle.deck.dloads) == 50                     # parsed records stay one per line
    groups = bundle.dloads()
    assert len(groups) == 2
    assert groups[0]["faces"] == 40 and groups[0]["labels"] == ["P1", "P2"] and groups[0]["value"] == 0.25
    assert groups[0]["label"] == "DLOAD 0.25 MPa on 40 faces (P1,P2), step 1" and groups[0]["elements"] == 40
    assert groups[1]["faces"] == 10 and groups[1]["labels"] == ["P3"] and groups[1]["value"] == -1.5
    assert groups[1]["label"] == "DLOAD -1.5 MPa on 10 faces (P3), step 1"
    assert len(groups[0]["targets"]) == 40 and groups[0]["element_ids"].shape == (40,) and set(groups[0]["face_numbers"]) == {1, 2}
    # offscreen render: one actor / legend line per group, arrows sized from the patch, counts kept
    scene = Scene(bundle)
    plotter = offscreen_plotter((640, 480))
    info = scene.render(plotter, SceneState(field="von_mises"))
    dloads = [o for o in info["overlays"] if o["kind"] == "dload"]
    assert [d["count"] for d in dloads] == [40, 10] and all(1 <= d["shown"] <= d["count"] for d in dloads)
    diagonal = bundle.mesh.diagonal
    for d in dloads:
        assert 0.03 * diagonal - 1e-9 <= d["glyph_length_mm"] <= 0.06 * diagonal + 1e-9
    assert [line for line in info["legend"] if line.startswith("DLOAD")] == [groups[0]["label"], groups[1]["label"]]
    assert plotter.renderer.actors and len(plotter.renderer.actors) < 20
    plotter.close()
    # info JSON: two entries, no per-line explosion
    assert main(["info", str(folder), "--no-cache"]) == 0
    summary = json.loads(capsys.readouterr().out)
    entries = summary["loads"]["dload"]
    assert len(entries) == 2 and entries[0]["faces"] == 40 and entries[0]["labels"] == ["P1", "P2"]
    assert entries[0]["targets_total"] == 40 and len(entries[0]["targets"]) == 20 and "element_ids" not in entries[0]


def test_dload_elset_target_keeps_single_entry(tmp_path):
    folder = tmp_path / "elset"
    folder.mkdir()
    (folder / "deck.inp").write_text("""*HEADING
elset target
*NODE
1,0,0,0
2,1,0,0
3,0,1,0
4,0,0,1
5,1,1,1
6,2,0,0
7,2,1,0
8,2,0,1
*ELEMENT,TYPE=C3D4,ELSET=E1
1,1,2,3,4
2,2,6,7,8
*MATERIAL,NAME=M
*ELASTIC
1000,0.3
*SOLID SECTION,ELSET=E1,MATERIAL=M
*STEP
*STATIC
*DLOAD
E1,P2,-3.5
*END STEP
""")
    groups = RunBundle(folder, use_cache=False).dloads()
    assert len(groups) == 1
    group = groups[0]
    assert group["label"] == "DLOAD E1 P2 = -3.5 MPa (2 faces, step 1)"
    assert group["target"] == "E1" and group["labels"] == ["P2"] and group["labels_text"] == "P2"
    assert group["faces"] == 2 and group["elements"] == 2 and group["targets"] == ["E1"]
    assert group["element_ids"].tolist() == [1, 2] and group["face_numbers"].tolist() == [2, 2]


def test_dload_labels_text():
    from fempost.bundle import _labels_text
    assert _labels_text(["P1", "P2", "P3", "P4"]) == "P1–P4"
    assert _labels_text(["P1", "P3"]) == "P1,P3"
    assert _labels_text(["P2"]) == "P2"
    assert _labels_text(["GRAV"]) == "GRAV"
