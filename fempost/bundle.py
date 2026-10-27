"""One ``RunBundle`` per case: discovery, the CalculiX triple, sidecars, labels and a JSON summary.

Ground truth is the FRD (mesh and fields) and the INP (sets, restraints, loads, contact, ties,
pretension, materials, steps). The DAT is read lazily for integration-point stress and totals.
Sidecars only add names and summary numbers. Nothing is written beside the run; the FRD arrays
are cached under ``structural-fea-environment/temporary/fempost-cache``. No VTK import here so
``fempost info`` stays light.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import convergence
from . import inp as inp_module
from . import sidecars
from .dat import DatReader
from .frd import ROOT, FRD_TYPE_NAMES, Frame, FrdMesh, load_frd
from .inp import FACE_NODES, FRD_CODE_TO_TYPE, Deck, read_deck

SIDE_CAR_NAMES = {"fields_npz": "fields.npz", "model_json": "model.json", "model_arrays_npz": "model-arrays.npz",
                  "results_json": "results.json", "project_json": "project.json", "manifest_json": "manifest.json",
                  "solver_mesh_npz": "solver-mesh.npz", "structural_mesh_npz": "structural-mesh.npz",
                  "mesh_npz": "mesh.npz", "mesh_json": "mesh.json"}


@dataclass
class RunFiles:
    folder: Path
    stem: str | None = None
    inp: Path | None = None
    frd: Path | None = None
    dat: Path | None = None
    sta: Path | None = None
    cvg: Path | None = None
    log: Path | None = None
    run_json: Path | None = None
    sidecars: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {"folder": str(self.folder), "stem": self.stem}
        for key in ("inp", "frd", "dat", "sta", "cvg", "log", "run_json"):
            value = getattr(self, key)
            out[key] = str(value) if value else None
        out["sidecars"] = {k: str(v) for k, v in self.sidecars.items()}
        out["warnings"] = list(self.warnings)
        return out


def _from_stem(folder: Path, stem: str | None, warnings: list[str] | None = None) -> RunFiles:
    files = RunFiles(folder, stem, warnings=list(warnings or []))
    if stem:
        for ext in ("inp", "frd", "dat", "sta", "cvg"):
            candidate = folder / f"{stem}.{ext}"
            if candidate.is_file():
                setattr(files, ext, candidate)
        run_json = folder / f"{stem}.run.json"
        if run_json.is_file():
            files.run_json = run_json
        for name in (f"{stem}.stdout.log", f"{stem}.log", f"{stem}.out", "stdout.log", "ccx.log"):
            candidate = folder / name
            if candidate.is_file():
                files.log = candidate
                break
        assembly = folder / f"{stem}.assembly.json"
        if assembly.is_file():
            files.sidecars["assembly_json"] = assembly
    for key, name in SIDE_CAR_NAMES.items():
        candidate = folder / name
        if candidate.is_file():
            files.sidecars[key] = candidate
    report = folder / "structural-results" / "results.json"
    if report.is_file():
        files.sidecars["kurtsfem_results_json"] = report
    model = sidecars.load_json(files.sidecars.get("model_json"))
    if model and model.get("level") and "mesh_npz" not in files.sidecars:
        mesh_dir = folder.parent.parent / "meshes" / str(model["level"])
        for key, name in (("mesh_npz", "mesh.npz"), ("mesh_json", "mesh.json")):
            candidate = mesh_dir / name
            if candidate.is_file():
                files.sidecars[key] = candidate
    return files


def _from_folder(folder: Path, warnings: list[str] | None = None) -> RunFiles:
    warnings = list(warnings or [])
    frds = sorted(folder.glob("*.frd"))
    stem = None
    if len(frds) == 1:
        stem = frds[0].stem
    elif len(frds) > 1:
        with_inp = [f for f in frds if f.with_suffix(".inp").is_file()]
        pick = (with_inp or frds)[0]
        stem = pick.stem
        warnings.append(f"{len(frds)} FRD files in folder; using {pick.name}")
    else:
        inps = sorted(folder.glob("*.inp"))
        if len(inps) == 1:
            stem = inps[0].stem
            warnings.append("no FRD beside the deck; results unavailable")
        report = folder / "results.json"
        data = sidecars.load_json(report) if report.is_file() else None
        if data and data.get("source_run"):
            source = Path(str(data["source_run"]))
            if not source.is_absolute():
                source = folder / source
            if source.is_file():
                return _from_stem(source.parent, source.name.replace(".run.json", "").split(".")[0], warnings)
    return _from_stem(folder, stem, warnings)


def discover(target: str | Path) -> RunFiles:
    """Locate the CalculiX triple and sidecars from a folder, a triple member, results.json or fields.npz."""
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run target does not exist: {path}")
    if path.is_dir():
        return _from_folder(path)
    suffix = path.suffix.lower()
    if suffix in (".frd", ".inp", ".dat", ".sta", ".cvg"):
        return _from_stem(path.parent, path.stem)
    if path.name.endswith(".run.json"):
        return _from_stem(path.parent, path.name[:-len(".run.json")])
    if path.name.endswith(".assembly.json"):
        return _from_stem(path.parent, path.name[:-len(".assembly.json")])
    if path.name == "results.json":
        data = sidecars.load_json(path) or {}
        if data.get("source_run"):
            source = Path(str(data["source_run"]))
            if not source.is_absolute():
                source = path.parent / source
            if source.is_file():
                return _from_stem(source.parent, source.name.replace(".run.json", ""))
            return _from_folder(path.parent.parent)
        return _from_folder(path.parent)
    if path.name in SIDE_CAR_NAMES.values():
        return _from_folder(path.parent)
    raise ValueError(f"Not a recognised run target: {path}")


def _structural_parse_sta(path: Path) -> list[dict]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from structural.backends.calculix import parse_sta
    return parse_sta(path)


def _label_key(label: str):
    """Sort face labels numerically (P1, P2, ..., P10) and other labels after them."""
    if label.startswith("P") and label[1:].isdigit():
        return (0, int(label[1:]), label)
    return (1, 0, label)


def _labels_text(labels: list[str]) -> str:
    """``P1–P4`` for a contiguous run of three or more face labels, otherwise ``P1,P3``."""
    numbers = [int(l[1:]) for l in labels if l.startswith("P") and l[1:].isdigit()]
    if len(numbers) >= 3 and len(numbers) == len(labels) and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        return f"P{numbers[0]}–P{numbers[-1]}"
    return ",".join(labels)


class RunBundle:
    """Read-only view of one finished run. Every heavy member is lazy and cached."""

    def __init__(self, target: str | Path, *, use_cache: bool = True, log=None):
        self.files = discover(target)
        self.use_cache = use_cache
        self.log = log or (lambda message: None)
        self.messages: list[str] = list(self.files.warnings)
        self._deck: Deck | None = None
        self._deck_loaded = False
        self._mesh: FrdMesh | None = None
        self._frames: list[Frame] | None = None
        self._frd_provenance: dict = {}
        self._dat: DatReader | None = None
        self._sta: list[dict] | None = None
        self._json: dict[str, dict | None] = {}
        self._npz: dict[str, dict | None] = {}
        self._bodies: dict[str, np.ndarray] | None = None
        self._body_names: dict[str, str] = {}
        self._regions: tuple[np.ndarray, list[str]] | None = None
        self._model_arrays_ok: bool | None = None
        self._convergence: dict | None = None

    # ---- identity -------------------------------------------------------------------------
    @property
    def folder(self) -> Path:
        return self.files.folder

    @property
    def name(self) -> str:
        model = self.json("model_json")
        if model and model.get("tag"):
            return str(model["tag"])
        return self.folder.name

    @property
    def family(self) -> str:
        if "assembly_json" in self.files.sidecars or "kurtsfem_results_json" in self.files.sidecars:
            return "kurtsfem"
        if "model_json" in self.files.sidecars or "fields_npz" in self.files.sidecars:
            return "study"
        return "generic"

    # ---- lazy members ---------------------------------------------------------------------
    def json(self, key: str) -> dict | None:
        if key not in self._json:
            self._json[key] = sidecars.load_json(self.files.sidecars.get(key)) if key != "run_json" \
                else sidecars.load_json(self.files.run_json)
        return self._json[key]

    def npz(self, key: str) -> dict[str, np.ndarray] | None:
        if key not in self._npz:
            self._npz[key] = sidecars.load_npz(self.files.sidecars.get(key))
        return self._npz[key]

    @property
    def deck(self) -> Deck | None:
        if not self._deck_loaded:
            self._deck_loaded = True
            if self.files.inp:
                try:
                    self._deck = read_deck(self.files.inp)
                    self.messages.extend(f"deck: {w}" for w in self._deck.warnings[:10])
                except Exception as exc:  # a broken deck must not hide the results
                    self.messages.append(f"deck unreadable: {exc}")
        return self._deck

    def _load_frd(self) -> None:
        if self._mesh is not None:
            return
        if not self.files.frd:
            raise FileNotFoundError(f"No FRD in {self.folder}")
        self._mesh, self._frames, self._frd_provenance = load_frd(self.files.frd, use_cache=self.use_cache, log=self.log)
        self._assign_steps()
        if self.beam_info()["section_forces"]:
            for frame in self._frames:
                frame.derived["section_forces"] = True

    @property
    def mesh(self) -> FrdMesh:
        self._load_frd()
        return self._mesh

    @property
    def frames(self) -> list[Frame]:
        self._load_frd()
        return self._frames

    @property
    def frd_provenance(self) -> dict:
        self._load_frd()
        return self._frd_provenance

    @property
    def sta(self) -> list[dict]:
        if self._sta is None:
            self._sta = _structural_parse_sta(self.files.sta) if self.files.sta else []
        return self._sta

    @property
    def dat(self) -> DatReader | None:
        if self._dat is None and self.files.dat:
            self._dat = DatReader(self.files.dat, use_cache=self.use_cache)
        return self._dat

    def _assign_steps(self) -> None:
        """Step numbers from the STA increment table, else from the deck's step periods."""
        frames = self._frames or []
        rows = self.sta
        periods: list[float] = []
        if self.deck is not None:
            for step in self.deck.steps:
                data = step.procedure_data
                periods.append(float(data[1]) if len(data) > 1 and data[1] > 0 else 1.0)
        boundaries = np.cumsum(periods) if periods else np.zeros(0)
        counters: dict[int, int] = {}
        for frame in frames:
            step = None
            for row in rows:
                if abs(row["total_time"] - frame.time) <= 1e-6 * max(1.0, abs(frame.time)):
                    step = int(row["step"])
                    break
            if step is None and len(boundaries):
                step = int(np.searchsorted(boundaries, frame.time - 1e-9)) + 1
                step = min(step, len(boundaries))
            if step is None:
                step = 1
            counters[step] = counters.get(step, 0) + 1
            frame.step = step
            frame.increment = counters[step]

    # ---- frames ---------------------------------------------------------------------------
    def frame(self, index: int) -> Frame:
        return self.frames[index]

    def last_frame_of_step(self, step: int) -> Frame | None:
        candidates = [f for f in self.frames if f.step == step]
        return candidates[-1] if candidates else None

    def step_labels(self) -> dict[int, str]:
        labels: dict[int, str] = {}
        assembly = self.json("assembly_json")
        stages = (assembly or {}).get("stages")
        for frame in self.frames:
            text = f"Step {frame.step}"
            if isinstance(stages, list) and 0 < frame.step <= len(stages):
                text += f" · {stages[frame.step - 1]}"
            labels.setdefault(frame.step, text)
        return labels

    # ---- element grouping -----------------------------------------------------------------
    def bodies(self) -> dict[str, np.ndarray]:
        """Body name -> FRD element rows. Solid-section element sets first, then FRD materials."""
        if self._bodies is not None:
            return self._bodies
        mesh = self.mesh
        bodies: dict[str, np.ndarray] = {}
        names: dict[str, str] = {}
        deck = self.deck
        if deck is not None:
            sections = [s["elset"] for s in deck.solid_sections if s["elset"] in deck.elsets]
            candidates = sections or [n for n in deck.elsets if n not in ("ALL_ELEMENTS", "EALL", "ROI")]
            if not sections and deck.beam_sections:
                # beam decks carry one *BEAM SECTION elset per member: group the members by material
                by_material: dict[str, list[str]] = {}
                for section in deck.beam_sections:
                    if section["elset"] in deck.elsets:
                        by_material.setdefault(section["material"] or "BEAM", []).append(section["elset"])
                for material, elsets in by_material.items():
                    rows = np.unique(np.concatenate([mesh.element_index(deck.resolve_elements(e)) for e in elsets]))
                    rows = rows[rows >= 0]
                    if rows.size:
                        display = f"beams {material.lower()}"
                        bodies[display] = rows
                        names[display] = f"{len(elsets)} BEAM SECTION elsets ({material})"
                if bodies:
                    candidates = []
            model = self.json("model_json")
            body_names = list((model or {}).get("body_names") or [])
            for elset in candidates:
                rows = mesh.element_index(deck.resolve_elements(elset))
                rows = rows[rows >= 0]
                if rows.size == 0:
                    continue
                display = elset.lower()
                if elset.startswith("BODY") and elset[4:].isdigit() and int(elset[4:]) < len(body_names):
                    display = str(body_names[int(elset[4:])])
                bodies[display] = rows
                names[display] = elset
        if not bodies:
            for material in np.unique(mesh.element_materials):
                rows = np.flatnonzero(mesh.element_materials == material)
                label = mesh.material_names[int(material) - 1] if 0 < material <= len(mesh.material_names) else f"material {material}"
                bodies[label.lower()] = rows
                names[label.lower()] = f"FRD material {int(material)}"
        self._bodies = bodies
        self._body_names = names
        return bodies

    def body_elset(self, body: str) -> str:
        self.bodies()
        return self._body_names.get(body, body)

    def cell_body_ids(self) -> tuple[np.ndarray, list[str]]:
        """(m,) index into the body list, -1 when an element belongs to no body."""
        bodies = self.bodies()
        ids = np.full(self.mesh.n_elements, -1, dtype=np.int32)
        for k, rows in enumerate(bodies.values()):
            ids[rows] = k
        return ids, list(bodies)

    def regions(self) -> tuple[np.ndarray, list[str]] | None:
        """Per-element region labels from fields.npz/model-arrays.npz (Codex runs) as codes + names."""
        if self._regions is not None:
            return self._regions
        for key in ("fields_npz", "model_arrays_npz"):
            data = self.npz(key)
            if data and "labels" in data and len(data["labels"]) == self.mesh.n_elements:
                labels = np.asarray(data["labels"]).astype(str)
                names, codes = np.unique(labels, return_inverse=True)
                self._regions = (codes.astype(np.int32), [str(n) for n in names])
                return self._regions
        return None

    def node_set_rows(self, name: str) -> np.ndarray:
        deck = self.deck
        if deck is None:
            return np.zeros(0, dtype=np.int64)
        rows = self.mesh.node_index(deck.resolve_nodes(name))
        return rows[rows >= 0]

    # ---- faces of element surfaces --------------------------------------------------------
    def surface_face_nodes(self, name: str) -> list[np.ndarray]:
        """Node rows of each face of an ELEMENT surface (3 or 6 per triangle, 4 per quad)."""
        deck = self.deck
        if deck is None:
            return []
        mesh = self.mesh
        out: list[np.ndarray] = []
        faces = deck.surface_faces(name)
        if not faces:
            nodes = deck.surface_nodes(name)
            return [mesh.node_index(nodes)] if nodes.size else []
        element_ids = np.array([e for e, _ in faces], dtype=np.int64)
        face_numbers = np.array([f for _, f in faces], dtype=np.int64)
        rows = mesh.element_index(element_ids)
        for row, face in zip(rows, face_numbers):
            if row < 0:
                continue
            etype = FRD_CODE_TO_TYPE.get(int(mesh.element_types[row]))
            table = FACE_NODES.get(etype or "")
            if not table or face not in table:
                continue
            local = table[face]
            out.append(mesh.connectivity[row, list(local)])
        return out

    def surface_face_count(self, name: str) -> int:
        deck = self.deck
        if deck is None:
            return 0
        surface = deck.surfaces.get(name.upper())
        if surface is None:
            return 0
        if surface.kind == "NODE":
            return int(deck.surface_nodes(name).size)
        return len(deck.surface_faces(name))

    # ---- definitions ----------------------------------------------------------------------
    def restraints(self) -> list[dict]:
        deck = self.deck
        if deck is None:
            return []
        out = []
        for boundary in deck.boundaries:
            rows = self.node_set_rows(boundary.target) if self._mesh is not None or self.files.frd else np.zeros(0)
            count = int(rows.size) if rows.size else int(deck.resolve_nodes(boundary.target).size)
            out.append({"step": boundary.step, "target": boundary.target, "dofs": boundary.dofs,
                        "first_dof": boundary.first_dof, "last_dof": boundary.last_dof, "value": boundary.value,
                        "nodes": count, "fixed": "FIXED" in boundary.params, "line": boundary.line})
        return out

    def cload_groups(self) -> list[dict]:
        """Concentrated loads per step as nodal force vectors (rows, F) with resultant and centroid."""
        deck = self.deck
        if deck is None:
            return []
        out = []
        mesh = self.mesh
        steps = sorted({c.step for c in deck.cloads})
        for step in steps:
            forces: dict[int, np.ndarray] = {}
            targets: set[str] = set()
            for load in deck.cloads:
                if load.step != step or not (1 <= load.dof <= 3):
                    continue
                targets.add(load.target)
                for node in deck.resolve_nodes(load.target):
                    vector = forces.setdefault(int(node), np.zeros(3))
                    vector[load.dof - 1] += load.value
            if not forces:
                continue
            ids = np.array(list(forces), dtype=np.int64)
            vectors = np.array(list(forces.values()))
            rows = mesh.node_index(ids)
            ok = rows >= 0
            points = np.full((len(ids), 3), np.nan)
            points[ok] = mesh.points[rows[ok]]
            # Reference nodes (pretension, kinematic) are absent from the FRD: take the deck coordinates.
            if (~ok).any() and deck.node_ids.size:
                deck_rows = {int(n): k for k, n in enumerate(deck.node_ids)}
                for k in np.flatnonzero(~ok):
                    row = deck_rows.get(int(ids[k]))
                    if row is not None:
                        points[k] = deck.node_xyz[row]
            located = np.isfinite(points).all(1)
            resultant = vectors.sum(0)
            magnitude = np.linalg.norm(vectors, axis=1)
            if located.any() and magnitude[located].sum() > 0:
                centroid = (points[located] * magnitude[located, None]).sum(0) / magnitude[located].sum()
            else:
                centroid = np.full(3, np.nan)
            out.append({"step": step, "rows": rows[ok], "forces": vectors[ok], "node_ids": ids, "nodes": int(len(ids)),
                        "points": points[located], "point_forces": vectors[located],
                        "resultant_N": resultant.tolist(), "magnitude_N": float(np.linalg.norm(resultant)),
                        "centroid_mm": centroid.tolist(), "targets": sorted(targets)[:20],
                        "nodes_in_frd": int(ok.sum()), "nodes_located": int(located.sum()),
                        "op_new": any(l.op and l.op.upper() == "NEW" for l in deck.cloads if l.step == step)})
        return out

    def dloads(self) -> list[dict]:
        """Logical distributed loads: the lines of one ``*DLOAD`` block grouped by (step, OP, value).

        The cage decks apply a pressure face by face (thousands of ``<element>,P<n>,<value>`` lines in one
        block); those become one load per value whose ``labels`` list the face labels. A block with a named
        elset target and one label keeps its one entry. ``element_ids``/``face_numbers`` are the resolved
        (element, face) pairs (face 0 for non-face labels such as GRAV), ``faces`` their count.
        """
        deck = self.deck
        if deck is None:
            return []
        groups: dict[tuple, dict] = {}
        for load in deck.dloads:
            key = (load.line, load.step, (load.op or "").upper(), float(f"{load.value:.12g}"))
            group = groups.get(key)
            if group is None:
                group = groups[key] = {"step": load.step, "op": load.op, "value": load.value, "line": load.line,
                                       "labels": [], "targets": [], "_ids": [], "_faces": []}
            if load.label not in group["labels"]:
                group["labels"].append(load.label)
            group["targets"].append(load.target)
            ids = deck.resolve_elements(load.target)
            face = int(load.label[1:]) if load.label.startswith("P") and load.label[1:].isdigit() else 0
            group["_ids"].append(ids)
            group["_faces"].append(np.full(ids.size, face, dtype=np.int64))
        out = []
        for group in groups.values():
            ids = np.concatenate(group.pop("_ids")) if group["_ids"] else np.zeros(0, dtype=np.int64)
            faces = np.concatenate(group.pop("_faces")) if group["_faces"] else np.zeros(0, dtype=np.int64)
            group["labels"] = sorted(group["labels"], key=_label_key)
            group["labels_text"] = _labels_text(group["labels"])
            group["element_ids"] = ids
            group["face_numbers"] = faces
            group["faces"] = int(ids.size)
            group["elements"] = int(np.unique(ids).size)
            named = [t for t in group["targets"] if not t.lstrip("-").isdigit()]
            if named and len(named) == len(group["targets"]):
                group["target"] = ", ".join(dict.fromkeys(named))
                group["label"] = (f"DLOAD {group['target']} {group['labels_text']} = {group['value']:g} MPa "
                                  f"({group['faces']} faces, step {group['step']})")
            else:
                group["target"] = f"{group['elements']} elements" + (f" + {', '.join(dict.fromkeys(named))}" if named else "")
                group["label"] = (f"DLOAD {group['value']:.4g} MPa on {group['faces']:,} faces ({group['labels_text']}), "
                                  f"step {group['step']}")
            out.append(group)
        return out

    def contact_pairs(self) -> list[dict]:
        deck = self.deck
        if deck is None:
            return []
        model = self.json("model_json") or {}
        labels = {}
        for entry in model.get("pairs", []):
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                labels[(str(entry[0]).upper(), str(entry[1]).upper())] = {"type": entry[2],
                                                                           "label": entry[3] if len(entry) > 3 else ""}
        out = []
        for pair in deck.contact_pairs:
            interaction = deck.interactions.get(pair.interaction)
            extra = labels.get((pair.slave, pair.master), {})
            out.append({"slave": pair.slave, "master": pair.master, "interaction": pair.interaction, "kind": pair.kind,
                        "behavior": interaction.pressure_overclosure if interaction else None,
                        "stiffness": interaction.behavior_values if interaction else [],
                        "friction": interaction.friction if interaction else [],
                        "category": interaction.kind if interaction else "unspecified",
                        "type": extra.get("type"), "label": extra.get("label"), "step": pair.step,
                        "slave_faces": self.surface_face_count(pair.slave),
                        "master_faces": self.surface_face_count(pair.master), "line": pair.line})
        return out

    def ties(self) -> list[dict]:
        deck = self.deck
        if deck is None:
            return []
        return [{"name": t.name, "slave": t.slave, "master": t.master, "params": t.params,
                 "slave_faces": self.surface_face_count(t.slave), "master_faces": self.surface_face_count(t.master)}
                for t in deck.ties]

    def pretension(self) -> list[dict]:
        deck = self.deck
        out = []
        if deck is not None:
            for section in deck.pretension_sections:
                out.append({"surface": section.surface, "node": section.node, "element": section.element,
                            "normal": section.normal, "faces": self.surface_face_count(section.surface or "")})
        assembly = self.json("assembly_json")
        for entry in (assembly or {}).get("pretension", []) or []:
            if isinstance(entry, dict):
                out.append({"source": "assembly.json", "screw": entry.get("screw"), "node": entry.get("reference_node"),
                            "node_set": entry.get("node_set"), "axis": entry.get("axis"), "target_N": entry.get("target_n"),
                            "formulation": (assembly or {}).get("pretension_formulation")})
        return out

    def materials(self) -> dict:
        deck = self.deck
        if deck is None:
            return {}
        return {name: {"E_MPa": m.elastic[0] if m.elastic else None, "nu": m.elastic[1] if len(m.elastic) > 1 else None,
                       "density": m.density, "extra": m.extra} for name, m in deck.materials.items()}

    # ---- model-arrays acceleration (Codex) ------------------------------------------------
    def model_arrays(self) -> dict[str, np.ndarray] | None:
        data = self.npz("model_arrays_npz")
        if not data:
            return None
        if self._model_arrays_ok is None:
            self._model_arrays_ok = sidecars.points_match(data.get("points"), self.mesh.points)
            if not self._model_arrays_ok:
                self.messages.append("model-arrays.npz points do not match the FRD mesh; ignored")
        return data if self._model_arrays_ok else None

    # ---- beam (frame) runs ----------------------------------------------------------------
    def beam_info(self) -> dict:
        """Beam runs: every FRD element is a B31/B32 line (CalculiX ``*NODE FILE, OUTPUT=2D`` keeps the
        beam axis in the FRD instead of the expanded solids). With ``*EL FILE, SECTION FORCES`` the FRD
        STRESS block carries the section forces and moments, not stresses."""
        cached = getattr(self, "_beam_info", None)
        if cached is not None:
            return cached
        mesh = self.mesh
        codes = {int(c) for c in np.unique(mesh.element_types)}
        is_beam = bool(codes) and codes <= {11, 12}
        info = {"is_beam": is_beam, "section_forces": False, "output_2d": False, "radius_mm": None, "sections": 0,
                "materials": [], "dims_mm": None}
        deck = self.deck
        if is_beam and deck is not None:
            for request in deck.output_requests:
                params = request.get("params") or {}
                if request["keyword"] == "EL FILE" and "SECTION FORCES" in params:
                    info["section_forces"] = True
                if request["keyword"] == "NODE FILE" and str(params.get("OUTPUT", "")).upper() == "2D":
                    info["output_2d"] = True
            dims = [s["dims"] for s in deck.beam_sections if s.get("dims")]
            info["sections"] = len(deck.beam_sections)
            info["materials"] = sorted({s["material"] for s in deck.beam_sections if s.get("material")})
            if dims:
                info["dims_mm"] = [float(np.mean([d[k] for d in dims if len(d) > k])) for k in range(2) if any(len(d) > k for d in dims)]
                info["radius_mm"] = 0.5 * float(np.mean(info["dims_mm"]))
        if is_beam and info["radius_mm"] is None:
            info["radius_mm"] = 0.004 * mesh.diagonal
        self._beam_info = info
        return info

    def beam_table(self, frame: Frame) -> list[dict]:
        """Per beam element: body, material, section dims and the FRD section-force components at both
        end nodes (``*EL FILE, SECTION FORCES``: STRESS columns SXX SYY SZZ = forces, SXY SYZ SZX = moments)."""
        info = self.beam_info()
        if not info["is_beam"] or "STRESS" not in frame.fields:
            return []
        mesh = self.mesh
        values = frame.fields["STRESS"]
        ids, names = self.cell_body_ids()
        deck = self.deck
        by_elset: dict[int, dict] = {}
        if deck is not None:
            for section in deck.beam_sections:
                for element in deck.resolve_elements(section["elset"]):
                    by_elset[int(element)] = section
        out = []
        for row in range(mesh.n_elements):
            conn = mesh.connectivity[row]
            conn = conn[conn >= 0]
            if conn.size < 2:
                continue
            a, b = int(conn[0]), int(conn[-1] if conn.size == 2 else conn[1])
            section = by_elset.get(int(mesh.element_ids[row]), {})
            entry = {"element": int(mesh.element_ids[row]), "body": names[ids[row]] if ids[row] >= 0 else "",
                     "material": section.get("material"), "dims_mm": section.get("dims"),
                     "node_a": int(mesh.node_ids[a]), "node_b": int(mesh.node_ids[b]),
                     "end_a": [float(v) for v in values[a, :6]], "end_b": [float(v) for v in values[b, :6]]}
            forces = np.array([entry["end_a"][:3], entry["end_b"][:3]])
            moments = np.array([entry["end_a"][3:6], entry["end_b"][3:6]])
            entry["force_max_N"] = float(np.nanmax(np.linalg.norm(forces, axis=1)))
            entry["moment_max_Nmm"] = float(np.nanmax(np.linalg.norm(moments, axis=1)))
            out.append(entry)
        return out

    def nonphysical_stress_bodies(self) -> dict[str, str]:
        """Bodies whose stress the pipeline declares non-physical (KurtsFEM fixed rigid hub)."""
        assembly = self.json("assembly_json") or {}
        out: dict[str, str] = {}
        if str(assembly.get("fixed_hub_stress", "")).startswith("undefined") and "hub" in self.bodies():
            out["hub"] = "Fixed rigid hub stress is not a physical result (blanked, as in structural/results.py)"
        return out

    # ---- solver progress and acceptance ---------------------------------------------------
    @property
    def convergence(self) -> dict:
        """Merged STA / CVG / stdout-log solver progress (``convergence.summarize``)."""
        if self._convergence is None:
            run = self.json("run_json") or {}
            self._convergence = convergence.summarize(self.files.sta, self.files.cvg, self.files.log,
                                                      expected_steps=run.get("expected_steps"),
                                                      expected_step_times=run.get("expected_step_times"))
        return self._convergence

    def equilibrium_evidence(self) -> list[dict]:
        """Final equilibrium residuals recorded by the pipeline (results.json / KurtsFEM stages), with sources."""
        diagonal = None
        if self.files.frd:
            try:
                diagonal = self.mesh.diagonal
            except Exception:
                diagonal = None
        return convergence.equilibrium_evidence(self.json("results_json"), self.json("model_json"),
                                                self.json("kurtsfem_results_json"), diagonal)

    def acceptance(self) -> dict:
        """Status line built from the run's own evidence.

        ``kind`` is one of ``accepted`` (KurtsFEM results.json says accepted), ``evidence`` (completed,
        converged, equilibrium residuals recorded by the pipeline), ``converged`` (completed and
        converged, no equilibrium record) or ``diagnostic`` (failed or incomplete solve, unconverged
        last increment, KurtsFEM gates failed, or no solver record). Nothing here claims acceptance
        the files do not support; every number names its file.
        """
        run = self.json("run_json") or {}
        report = self.json("kurtsfem_results_json")
        conv = self.convergence
        evidence = self.equilibrium_evidence()
        status = str(run.get("status") or "unknown")
        if report and report.get("solver_status"):
            status = str(report["solver_status"])
        accepted = bool(run.get("accepted", False))
        gates: list = []
        completed = run.get("completed_steps")
        expected = run.get("expected_steps")
        reasons: list[str] = []
        if status not in ("completed", "unknown"):
            reasons.append(f"solver status {status}")
        if conv.get("errors"):
            reasons.append(conv["errors"][0][:80])
        try:
            if expected is not None and completed is not None and len(completed) < int(expected):
                reasons.append(f"{len(completed)} of {expected} steps completed")
        except TypeError:
            pass
        if conv.get("converged") is False:
            reasons.append("last increment not converged" if not conv.get("reasons") else conv["reasons"][0])
        parts = [status]
        has_progress = bool(conv.get("steps") or conv.get("iterations"))
        if has_progress:
            parts.append(convergence.short_text(conv))
        kind = "diagnostic"
        if report:
            assessment = report.get("assessment") or report.get("acceptance") or {}
            gates = assessment.get("gates", [])
            accepted = bool(report.get("accepted", False)) and not report.get("diagnostic_only", False)
            failed_gates = [str(g.get("name")) for g in gates if str(g.get("status")) == "failed"]
            if accepted:
                kind = "accepted"
                parts.append("numerically accepted · design adequacy not assessed")
            else:
                text = "UNACCEPTED · DIAGNOSTIC ONLY"
                if failed_gates:
                    text += f" (gates failed: {', '.join(failed_gates)})"
                elif reasons:
                    text += f" ({'; '.join(reasons)})"
                parts.append(text)
        elif reasons:
            parts.append("diagnostic only (" + "; ".join(reasons) + ")")
        elif evidence:
            kind = "evidence"
            entry = evidence[0]
            text = f"equilibrium residual |F| = {entry['force_residual_norm_N']:.2g} N"
            if entry.get("moment_residual_norm_Nmm") is not None:
                text += f", |M| = {entry['moment_residual_norm_Nmm']:.2g} N·mm"
            if entry.get("force_fraction") is not None:
                text += f" ({entry['force_fraction']:.1e} of the applied load)"
            text += f" [{entry['source']}]"
            parts.append(text)
        elif conv.get("converged"):
            kind = "converged"
            parts.append("no equilibrium record")
        elif status == "completed":
            parts.append("no convergence record · diagnostic only")
        else:
            parts.append("no solver record · diagnostic only")
        label = " · ".join(p for p in parts if p)
        # short form for the 3D view (VTK's font has no '|'; keep it to about 90 characters)
        short = [status]
        if has_progress:
            totals = conv.get("totals") or {}
            short.append(("converged" if conv.get("converged") else "NOT converged" if conv.get("converged") is False else "progress")
                         + f" ({totals.get('increments', 0)} inc, {totals.get('iterations', 0)} it"
                         + (f", {totals['cutbacks']} cutbacks" if totals.get("cutbacks") else "") + ")")
        if kind == "accepted":
            short.append("numerically accepted")
        elif report:
            failed_gates = [str(g.get("name")) for g in gates if str(g.get("status")) == "failed"]
            short.append("UNACCEPTED · DIAGNOSTIC ONLY" + (f" ({len(failed_gates)} gates failed)" if failed_gates else ""))
        elif reasons:
            short.append("DIAGNOSTIC ONLY (" + reasons[0] + ")")
        elif kind == "evidence":
            entry = evidence[0]
            text = f"residual F {entry['force_residual_norm_N']:.2g} N"
            if entry.get("moment_residual_norm_Nmm") is not None:
                text += f", M {entry['moment_residual_norm_Nmm']:.2g} N·mm"
            short.append(text)
        elif kind == "converged":
            short.append("no equilibrium record")
        else:
            short.append("no solver record · DIAGNOSTIC ONLY")
        short_label = " · ".join(short)
        return {"solver_status": status, "accepted": accepted, "kind": kind, "label": label, "short_label": short_label, "gates": gates,
                "completed_steps": completed, "expected_steps": expected,
                "solver_version": run.get("solver_version"), "elapsed_s": run.get("elapsed_s"), "threads": run.get("threads"),
                "convergence": {"converged": conv.get("converged"), "reasons": conv.get("reasons"), "totals": conv.get("totals"),
                                "steps": convergence.table_rows(conv), "sources": conv.get("sources"),
                                "total_time_s": conv.get("total_time_s"), "final_iteration": conv.get("final_iteration"),
                                "equations": conv.get("equations")},
                "equilibrium": evidence, "diagnostic_reasons": reasons}

    # ---- summary --------------------------------------------------------------------------
    def summary(self, *, include_frames: bool = True) -> dict:
        deck = self.deck
        mesh = self.mesh if self.files.frd else None
        out: dict = {"run": {"name": self.name, "family": self.family, "folder": str(self.folder), "stem": self.files.stem,
                             "heading": deck.heading if deck else (mesh.heading if mesh else ""), "files": self.files.as_dict(),
                             "units": "N-mm-MPa", "messages": list(self.messages)}}
        out["solver"] = self.acceptance()
        if mesh is not None:
            lo, hi = mesh.bounds
            out["mesh"] = {"nodes": mesh.n_nodes, "elements": mesh.n_elements, "element_types": mesh.type_counts(),
                           "bounds_mm": {"min": lo.tolist(), "max": hi.tolist()}, "diagonal_mm": mesh.diagonal,
                           "mesh_hash": mesh.mesh_hash(), "frd_materials": mesh.material_names,
                           "frd": {k: v for k, v in self.frd_provenance.items() if k != "frd"}}
            bodies = []
            for body, rows in self.bodies().items():
                elset = self.body_elset(body)
                material = deck.body_material(elset) if deck else None
                nodes = np.unique(mesh.connectivity[rows]); nodes = nodes[nodes >= 0]
                bodies.append({"name": body, "elset": elset, "elements": int(rows.size), "nodes": int(nodes.size),
                               "material": material, "E_MPa": (self.materials().get(material) or {}).get("E_MPa")})
            out["bodies"] = bodies
            regions = self.regions()
            if regions:
                codes, names = regions
                counts = np.bincount(codes, minlength=len(names))
                out["regions"] = {name: int(count) for name, count in zip(names, counts)}
        if deck is not None:
            out["node_sets"] = {name: int(deck.resolve_nodes(name).size) for name in deck.nsets}
            out["element_sets"] = {name: int(deck.resolve_elements(name).size) for name in deck.elsets}
            out["surfaces"] = {name: {"kind": s.kind, "faces": self.surface_face_count(name)} for name, s in deck.surfaces.items()}
            out["steps"] = [{"number": s.number, "procedure": s.procedure, "nlgeom": s.nlgeom, "params": s.params,
                             "procedure_data": s.procedure_data} for s in deck.steps]
            out["restraints"] = self.restraints()
            loads = {"cload": [], "dload": []}
            for group in self.dloads():         # one entry per logical load; the (element, face) arrays stay out
                entry = {k: v for k, v in group.items() if k not in ("element_ids", "face_numbers", "targets")}
                entry["targets"] = group["targets"][:20]
                entry["targets_total"] = len(group["targets"])
                loads["dload"].append(entry)
            if mesh is not None:
                for group in self.cload_groups():
                    loads["cload"].append({k: v for k, v in group.items()
                                           if k not in ("rows", "forces", "node_ids", "points", "point_forces")})
            else:
                for step in sorted({c.step for c in deck.cloads}):
                    loads["cload"].append({"step": step, "entries": sum(1 for c in deck.cloads if c.step == step)})
            out["loads"] = loads
            out["contact_pairs"] = self.contact_pairs()
            out["interactions"] = {name: {"behavior": i.pressure_overclosure, "values": i.behavior_values,
                                          "friction": i.friction, "category": i.kind} for name, i in deck.interactions.items()}
            out["ties"] = self.ties()
            out["pretension"] = self.pretension()
            out["materials"] = self.materials()
            out["equations"] = deck.equation_count
            out["keyword_counts"] = deck.keyword_counts
        if mesh is not None:
            beam = self.beam_info()
            if beam["is_beam"]:
                out["beams"] = dict(beam)
                table = self.beam_table(self.frames[-1]) if self.frames else []
                if table:
                    out["beams"]["elements"] = len(table)
                    out["beams"]["force_max_N"] = max(t["force_max_N"] for t in table)
                    out["beams"]["moment_max_Nmm"] = max(t["moment_max_Nmm"] for t in table)
                    out["beams"]["frame"] = self.frames[-1].label
        if mesh is not None and include_frames:
            labels = self.step_labels()
            out["frames"] = [{"index": f.index, "step": f.step, "step_label": labels.get(f.step), "increment": f.increment,
                              "time": f.time, "fields": {name: comps for name, comps in f.components.items()}}
                             for f in self.frames]
        if self.dat is not None:
            try:
                out["dat"] = {"tables": len(self.dat.tables), "stress_times": self.dat.stress_times(),
                              "kinds": sorted({t.kind for t in self.dat.tables})}
            except Exception as exc:
                out["dat"] = {"error": str(exc)}
        out["sidecars"] = {}
        if self.family == "study":
            out["sidecars"]["model"] = sidecars.codex_model_summary(self.json("model_json"))
            out["sidecars"]["results"] = sidecars.codex_results_summary(self.json("results_json"))
            out["sidecars"]["mesh"] = sidecars.mesh_json_summary(self.json("mesh_json"))
        elif self.family == "kurtsfem":
            out["sidecars"]["assembly"] = sidecars.kurtsfem_assembly_summary(self.json("assembly_json"))
            out["sidecars"]["results"] = sidecars.kurtsfem_results_summary(self.json("kurtsfem_results_json"))
            manifest = self.json("manifest_json")
            if manifest:
                out["sidecars"]["manifest"] = manifest
        out["sidecars"]["run"] = sidecars.run_json_summary(self.json("run_json"))
        return out


def summary_json(target: str | Path, **kwargs) -> str:
    bundle = RunBundle(target, **kwargs)
    return json.dumps(bundle.summary(), indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
