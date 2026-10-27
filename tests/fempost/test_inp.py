"""Deck reader on the fixture decks, the Codex deck and the KurtsFEM deck plus synthetic corners."""
import numpy as np
import pytest

from fempost.inp import read_deck, tokenize


def test_short_screw_deck(short_screw):
    deck = read_deck(short_screw / "short_screw.inp")
    assert deck.heading.startswith("Native solid pretension")
    assert len(deck.node_ids) == 163 and len(deck.element_ids) == 16
    assert deck.element_types[0] == "C3D20" and len(deck.element_nodes[0]) == 20
    assert deck.element_nodes[0][:4] == [1, 2, 3, 4] and deck.element_nodes[0][-1] == 20
    assert deck.resolve_nodes("ROOT").size == 21 and deck.resolve_nodes("142").tolist() == [142]
    assert deck.resolve_nodes("NALL").size == 141 and deck.resolve_nodes("EXPLICIT_CUT_HEAD").size == 21
    assert deck.resolve_elements("BOLT").size == 16
    assert deck.surface_faces("CUT") == [(5, 4), (6, 4), (7, 4), (8, 4)]
    steps = [(b.step, b.target, b.dofs, b.value) for b in deck.boundaries]
    assert (0, "ROOT", "Ux", 0.0) in steps and (2, "TIP", "Ux", 5e-5) in steps
    assert any("FIXED" in b.params for b in deck.boundaries if b.step == 2)
    assert [(c.step, c.target, c.dof, c.value) for c in deck.cloads] == [(1, "142", 1, 100.0)]
    assert deck.materials["STEEL"].elastic == [200000.0, 0.0]
    assert deck.solid_sections == [{"elset": "BOLT", "material": "STEEL", "line": 406}]
    assert deck.body_material("bolt") == "STEEL"
    assert [s.number for s in deck.steps] == [1, 2] and all(s.nlgeom for s in deck.steps)
    assert deck.steps[0].procedure == "STATIC" and deck.steps[0].procedure_data == [0.2, 1.0, 1e-6, 0.2]
    assert deck.equation_count == 63


def test_friction_deck(friction):
    deck = read_deck(friction / "friction.inp")
    assert deck.resolve_nodes("NALL").size == 150          # nested NSET NALL = NLOW, NUP
    assert deck.resolve_elements("EALL").size == 48
    assert deck.surfaces["SLAVE"].kind == "ELEMENT" and len(deck.surface_faces("MASTER")) == 8
    pair = deck.contact_pairs[0]
    assert (pair.slave, pair.master, pair.interaction, pair.kind) == ("SLAVE", "MASTER", "FRICTION", "SURFACE TO SURFACE")
    interaction = deck.interactions["FRICTION"]
    assert interaction.pressure_overclosure == "LINEAR" and interaction.friction[0] == pytest.approx(0.1)
    assert interaction.kind == "frictional"
    assert len(deck.steps) == 6 and deck.steps[0].params.get("INC") == "200"
    drive = [b for b in deck.boundaries if b.target == "DRIVE" and b.step == 3]
    assert len(drive) == 2 and drive[0].value == pytest.approx(0.00707106781186547)


def test_codex_deck(codex_run):
    deck = read_deck(codex_run / "job.inp")
    assert len(deck.node_ids) == 119941 and len(deck.element_ids) == 68963
    assert {s["elset"] for s in deck.solid_sections} == {f"BODY{k}" for k in range(8)}
    assert deck.body_material("BODY5") == "STEEL" and deck.body_material("BODY0") == "ALUMINUM"
    assert deck.resolve_nodes("FIX").size == 344
    assert len(deck.contact_pairs) == 14
    assert deck.interactions["THREAD"].kind == "tied" and deck.interactions["BEARING"].kind == "frictionless"
    assert len(deck.surface_faces("BORE")) == 2082
    assert sum(1 for c in deck.cloads if c.step == 1) == 14847
    assert deck.steps[0].params == {"INC": "100"} and not deck.steps[0].nlgeom


def test_kurtsfem_deck(kurtsfem_run):
    deck = read_deck(kurtsfem_run / "dodec_assembly.inp")
    assert len(deck.node_ids) == 179932 and len(deck.element_ids) == 87985
    assert deck.resolve_elements("PIPE").size == 50016 and deck.resolve_nodes("HUB_FIXED").size == 35328
    assert deck.resolve_nodes("PRETENSION_BOLT_1").tolist() == [176843]
    assert len(deck.surface_faces("LOAD_PATCH")) == 308
    assert [(p.slave, p.master) for p in deck.contact_pairs][0] == ("PIPE_BORE", "HUB_SNOUT")
    assert deck.interactions["SEPARATING_CONTACT"].behavior_values == [100000.0]
    assert len(deck.steps) == 2 and deck.equation_count == 9261
    fixed = [b for b in deck.boundaries if b.step == 2 and "FIXED" in b.params]
    assert [b.target for b in fixed] == ["176843", "176844", "176845"]
    assert sum(1 for c in deck.cloads if c.step == 2) == 665


def test_generate_include_dload_tie_pretension(tmp_path):
    include = tmp_path / "sets.inp"
    include.write_text("*NSET,NSET=GEN,GENERATE\n1,9,2\n*ELSET,ELSET=TOP\n3,4\n")
    deck_path = tmp_path / "deck.inp"
    deck_path.write_text("""*HEADING
synthetic ** not a comment inside heading
*NODE
1,0,0,0
2,1,0,0
3,0,1,0
4,0,0,1
5,1,1,1
6,2,0,0
7,2,1,0
8,2,0,1
9,3,0,0
*ELEMENT,TYPE=C3D4,ELSET=E1
1,1,2,3,4
2,2,6,7,8
** a comment line
*INCLUDE,INPUT=sets.inp
*SURFACE,NAME=S1
E1,S2
*SURFACE,NAME=NS,TYPE=NODE
GEN
*TIE,NAME=T1,POSITION TOLERANCE=0.1
S1,NS
*PRE-TENSION SECTION,SURFACE=S1,NODE=9
1.0,0.0,0.0
*MATERIAL,NAME=M
*ELASTIC
1000,0.3
*DENSITY
7.8e-9
*SOLID SECTION,ELSET=E1,MATERIAL=M
*STEP
*STATIC
*DLOAD
E1,P2,-3.5
*BOUNDARY
GEN,1,3
*END STEP
""")
    deck = read_deck(deck_path)
    assert deck.resolve_nodes("GEN").tolist() == [1, 3, 5, 7, 9]
    assert deck.resolve_elements("TOP").tolist() == [3, 4]
    assert deck.surface_faces("S1") == [(1, 2), (2, 2)]
    assert deck.surface_nodes("NS").tolist() == [1, 3, 5, 7, 9]
    assert deck.ties[0].slave == "S1" and deck.ties[0].params["POSITION TOLERANCE"] == "0.1"
    assert deck.pretension_sections[0].surface == "S1" and deck.pretension_sections[0].normal == [1.0, 0.0, 0.0]
    assert deck.dloads[0].label == "P2" and deck.dloads[0].value == -3.5 and deck.dloads[0].step == 1
    assert deck.materials["M"].density == pytest.approx(7.8e-9)
    assert deck.boundaries[0].step == 1 and deck.boundaries[0].dofs == "Ux+Uy+Uz"
    assert "INCLUDE" not in deck.keyword_counts and deck.keyword_counts["NSET"] == 1
    assert [k.name for k in tokenize(deck_path)][:3] == ["HEADING", "NODE", "ELEMENT"]
