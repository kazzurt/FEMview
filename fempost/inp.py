"""Generic read-only CalculiX input deck reader.

Reads the keywords the viewer needs to show assumptions: ``*NODE``/``*ELEMENT`` (decks that carry
the mesh inline), ``*NSET``, ``*ELSET``, ``*SURFACE``, ``*BOUNDARY``, ``*CLOAD``, ``*DLOAD``,
``*CONTACT PAIR`` with ``*SURFACE INTERACTION``/``*SURFACE BEHAVIOR``/``*FRICTION``, ``*TIE``,
``*PRE-TENSION SECTION``, ``*MATERIAL``/``*ELASTIC``/``*SOLID SECTION``, ``*STEP`` and
``*INCLUDE``. Everything else is kept as a raw keyword record. Step 0 means model definition;
step n means inside the n-th ``*STEP`` block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NODES_PER_TYPE = {"C3D4": 4, "C3D10": 10, "C3D10T": 10, "C3D8": 8, "C3D8R": 8, "C3D8I": 8, "C3D20": 20,
                  "C3D20R": 20, "C3D6": 6, "C3D15": 15, "S3": 3, "S4": 4, "S4R": 4, "S6": 6, "S8": 8, "S8R": 8,
                  "CPS3": 3, "CPE3": 3, "CAX3": 3, "CPS4": 4, "CPE4": 4, "CAX4": 4, "CPS4R": 4, "CPE4R": 4,
                  "CAX4R": 4, "CPS6": 6, "CPE6": 6, "CAX6": 6, "CPS8": 8, "CPE8": 8, "CAX8": 8, "CPS8R": 8,
                  "CPE8R": 8, "CAX8R": 8, "B31": 2, "B31R": 2, "B32": 3, "B32R": 3, "T3D2": 2, "T3D3": 3,
                  "SPRING1": 1, "SPRINGA": 2, "DASHPOTA": 2, "GAPUNI": 2, "MASS": 1}
# CalculiX face labels -> zero-based element-local nodes (corners first, then the edge nodes of
# that face for quadratic tetrahedra). Orientation is outward for positively oriented elements.
FACE_NODES = {
    "C3D4": {1: (0, 1, 2), 2: (0, 3, 1), 3: (1, 3, 2), 4: (2, 3, 0)},
    "C3D10": {1: (0, 1, 2, 4, 5, 6), 2: (0, 3, 1, 7, 8, 4), 3: (1, 3, 2, 8, 9, 5), 4: (2, 3, 0, 9, 7, 6)},
    "C3D8": {1: (0, 1, 2, 3), 2: (4, 7, 6, 5), 3: (0, 4, 5, 1), 4: (1, 5, 6, 2), 5: (2, 6, 7, 3), 6: (3, 7, 4, 0)},
    "C3D6": {1: (0, 1, 2), 2: (3, 5, 4), 3: (0, 3, 4, 1), 4: (1, 4, 5, 2), 5: (2, 5, 3, 0)},
}
FACE_NODES["C3D20"] = FACE_NODES["C3D8"]
FACE_NODES["C3D15"] = FACE_NODES["C3D6"]
FRD_CODE_TO_TYPE = {1: "C3D8", 2: "C3D6", 3: "C3D4", 4: "C3D20", 5: "C3D15", 6: "C3D10"}
DOF_NAMES = {1: "Ux", 2: "Uy", 3: "Uz", 4: "Rx", 5: "Ry", 6: "Rz", 11: "T"}


@dataclass
class Keyword:
    name: str
    params: dict[str, str | None]
    data: list[str]
    line: int
    step: int
    source: str = ""

    def get(self, key: str, default=None):
        return self.params.get(key.upper(), default)


@dataclass
class Surface:
    name: str
    kind: str                                                     # ELEMENT or NODE
    faces: list[tuple[str, str]] = field(default_factory=list)    # (element id or elset name, face label)
    nodes: list[str] = field(default_factory=list)                # node ids or nset names (NODE surfaces)
    line: int = 0


@dataclass
class Boundary:
    step: int
    target: str
    first_dof: int
    last_dof: int
    value: float
    params: dict = field(default_factory=dict)
    line: int = 0

    @property
    def dofs(self) -> str:
        return "+".join(DOF_NAMES.get(d, str(d)) for d in range(self.first_dof, self.last_dof + 1))


@dataclass
class CLoad:
    step: int
    target: str
    dof: int
    value: float
    op: str | None = None
    line: int = 0


@dataclass
class DLoad:
    step: int
    target: str
    label: str
    value: float
    op: str | None = None
    line: int = 0


@dataclass
class ContactPair:
    slave: str
    master: str
    interaction: str
    kind: str
    params: dict = field(default_factory=dict)
    step: int = 0
    line: int = 0


@dataclass
class Interaction:
    name: str
    pressure_overclosure: str | None = None
    behavior_values: list[float] = field(default_factory=list)
    friction: list[float] = field(default_factory=list)
    line: int = 0

    @property
    def kind(self) -> str:
        po = (self.pressure_overclosure or "").upper()
        if po == "TIED":
            return "tied"
        if self.friction:
            return "frictional"
        if po:
            return "frictionless"
        return "unspecified"


@dataclass
class Tie:
    name: str
    slave: str
    master: str
    params: dict = field(default_factory=dict)
    line: int = 0


@dataclass
class PreTension:
    surface: str | None
    node: str | None
    element: str | None
    normal: list[float]
    line: int = 0


@dataclass
class Material:
    name: str
    elastic: list[float] = field(default_factory=list)
    density: float | None = None
    extra: dict = field(default_factory=dict)
    line: int = 0


@dataclass
class Step:
    number: int
    params: dict
    procedure: str | None = None
    procedure_params: dict = field(default_factory=dict)
    procedure_data: list[float] = field(default_factory=list)
    line: int = 0

    @property
    def nlgeom(self) -> bool:
        return "NLGEOM" in self.params


def _split_params(text: str) -> tuple[str, dict[str, str | None]]:
    parts = [p.strip() for p in text.split(",")]
    name = re.sub(r"\s+", " ", parts[0].lstrip("*").strip().upper())
    params: dict[str, str | None] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().upper()] = value.strip()
        else:
            params[part.strip().upper()] = None
    return name, params


def _expand_generate(tokens: list[str]) -> list[str]:
    numbers = [int(t) for t in tokens if t.lstrip("-").isdigit()]
    if len(numbers) < 2:
        return tokens
    start, stop = numbers[0], numbers[1]
    inc = numbers[2] if len(numbers) > 2 else 1
    return [str(k) for k in range(start, stop + 1, max(inc, 1))]


def tokenize(path: str | Path, *, _depth: int = 0) -> list[Keyword]:
    """Split a deck into keyword records, following ``*INCLUDE`` read-only."""
    path = Path(path)
    keywords: list[Keyword] = []
    current: Keyword | None = None
    step = 0
    with path.open("r", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                name, params = _split_params(line)
                if name == "INCLUDE" and _depth < 8:
                    target = params.get("INPUT")
                    if target:
                        include = Path(target)
                        if not include.is_absolute():
                            include = path.parent / include
                        if include.is_file():
                            for included in tokenize(include, _depth=_depth + 1):
                                included.step = step
                                keywords.append(included)
                    current = None
                    continue
                if name == "STEP":
                    step += 1
                current = Keyword(name, params, [], number, step, str(path))
                keywords.append(current)
                if name == "END STEP":
                    current = None
            elif current is not None:
                if "**" in line:
                    line = line.split("**", 1)[0].strip()
                if line:
                    current.data.append(line)
    return keywords


@dataclass
class Deck:
    path: Path
    keywords: list[Keyword]
    heading: str = ""
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    node_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    element_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    element_types: list[str] = field(default_factory=list)
    element_nodes: list[list[int]] = field(default_factory=list)
    nsets: dict[str, np.ndarray] = field(default_factory=dict)
    elsets: dict[str, np.ndarray] = field(default_factory=dict)
    surfaces: dict[str, Surface] = field(default_factory=dict)
    boundaries: list[Boundary] = field(default_factory=list)
    cloads: list[CLoad] = field(default_factory=list)
    dloads: list[DLoad] = field(default_factory=list)
    contact_pairs: list[ContactPair] = field(default_factory=list)
    interactions: dict[str, Interaction] = field(default_factory=dict)
    ties: list[Tie] = field(default_factory=list)
    pretension_sections: list[PreTension] = field(default_factory=list)
    materials: dict[str, Material] = field(default_factory=dict)
    solid_sections: list[dict] = field(default_factory=list)
    beam_sections: list[dict] = field(default_factory=list)      # *BEAM SECTION: elset, material, section, dims, direction
    output_requests: list[dict] = field(default_factory=list)    # *EL FILE / *NODE FILE / *EL PRINT / *NODE PRINT / *CONTACT ...
    steps: list[Step] = field(default_factory=list)
    equation_count: int = 0
    mpc_count: int = 0
    keyword_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- set resolution -------------------------------------------------------------------
    def resolve_nodes(self, target: str) -> np.ndarray:
        """Node ids of a node id or (possibly nested) NSET name; empty when unknown."""
        return self._resolve(str(target), self.nsets, set())

    def resolve_elements(self, target: str) -> np.ndarray:
        return self._resolve(str(target), self.elsets, set())

    def _resolve(self, target: str, table: dict[str, np.ndarray], seen: set) -> np.ndarray:
        key = target.strip().upper()
        if key.lstrip("-").isdigit():
            return np.array([int(key)], dtype=np.int64)
        if key in seen or key not in table:
            return np.zeros(0, dtype=np.int64)
        seen.add(key)
        raw = table[key]
        if raw.dtype.kind in "US":
            parts = [self._resolve(str(t), table, seen) for t in raw.tolist()]
            return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        return raw

    def surface_faces(self, name: str) -> list[tuple[int, int]]:
        """(element id, face number) pairs of an ELEMENT surface, elset references expanded."""
        surface = self.surfaces.get(name.strip().upper())
        if surface is None or surface.kind != "ELEMENT":
            return []
        out: list[tuple[int, int]] = []
        for target, label in surface.faces:
            face = int(re.sub(r"\D", "", label) or 0)
            if target.isdigit():
                out.append((int(target), face))
            else:
                out.extend((int(e), face) for e in self.resolve_elements(target))
        return out

    def surface_nodes(self, name: str) -> np.ndarray:
        surface = self.surfaces.get(name.strip().upper())
        if surface is None or surface.kind != "NODE":
            return np.zeros(0, dtype=np.int64)
        parts = [self.resolve_nodes(t) for t in surface.nodes]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def body_material(self, elset: str) -> str | None:
        for section in self.solid_sections + self.beam_sections:
            if section["elset"] == elset.upper():
                return section["material"]
        return None

    @property
    def has_mesh(self) -> bool:
        return len(self.node_ids) > 0 and len(self.element_ids) > 0


def _finish_sets(raw: dict[str, list[str]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, tokens in raw.items():
        if tokens and all(t.lstrip("-").isdigit() for t in tokens):
            out[name] = np.unique(np.array([int(t) for t in tokens], dtype=np.int64))
        elif not tokens:
            out[name] = np.zeros(0, dtype=np.int64)
        else:
            out[name] = np.array([t.upper() for t in tokens])  # nested set names, resolved lazily
    return out


def _floats(lines: list[str]) -> list[float]:
    out: list[float] = []
    for line in lines:
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                pass
    return out


def read_deck(path: str | Path) -> Deck:
    path = Path(path)
    keywords = tokenize(path)
    deck = Deck(path, keywords)
    nodes: list[tuple[int, float, float, float]] = []
    element_ids: list[int] = []
    nset_raw: dict[str, list[str]] = {}
    elset_raw: dict[str, list[str]] = {}
    current_interaction: Interaction | None = None
    current_material: Material | None = None
    for kw in keywords:
        deck.keyword_counts[kw.name] = deck.keyword_counts.get(kw.name, 0) + 1
        name = kw.name
        if name == "HEADING":
            deck.heading = " ".join(kw.data)
        elif name == "NODE":
            setname = (kw.get("NSET") or "").upper()
            ids_here: list[str] = []
            for line in kw.data:
                parts = [p.strip() for p in line.split(",")]
                if not parts[0]:
                    continue
                try:
                    nid = int(parts[0])
                    xyz = [float(p) for p in parts[1:4] if p]
                except ValueError:
                    continue
                xyz = (xyz + [0.0, 0.0, 0.0])[:3]
                nodes.append((nid, xyz[0], xyz[1], xyz[2]))
                ids_here.append(parts[0])
            if setname:
                nset_raw.setdefault(setname, []).extend(ids_here)
        elif name == "ELEMENT":
            etype = (kw.get("TYPE") or "").upper()
            setname = (kw.get("ELSET") or "").upper()
            count = NODES_PER_TYPE.get(etype)
            buffer: list[str] = []
            ids_here = []
            for line in kw.data:
                buffer.extend(p.strip() for p in line.split(",") if p.strip())
                complete = (count is not None and len(buffer) >= count + 1) or (count is None and not line.endswith(","))
                if complete:
                    try:
                        eid = int(buffer[0])
                        nodal = [int(t) for t in (buffer[1:count + 1] if count else buffer[1:])]
                    except ValueError:
                        buffer = []
                        continue
                    deck.element_types.append(etype)
                    deck.element_nodes.append(nodal)
                    ids_here.append(str(eid))
                    element_ids.append(eid)
                    buffer = buffer[count + 1:] if count else []
            if setname:
                elset_raw.setdefault(setname, []).extend(ids_here)
        elif name in ("NSET", "ELSET"):
            setname = (kw.get(name) or "").upper()
            if not setname:
                continue
            table = nset_raw if name == "NSET" else elset_raw
            is_generate = "GENERATE" in kw.params
            tokens: list[str] = []
            for line in kw.data:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                tokens.extend(_expand_generate(parts) if is_generate else parts)
            table.setdefault(setname, []).extend(tokens)
        elif name == "SURFACE":
            sname = (kw.get("NAME") or "").upper()
            kind = (kw.get("TYPE") or "ELEMENT").upper()
            surface = deck.surfaces.setdefault(sname, Surface(sname, kind, line=kw.line))
            for line in kw.data:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if kind == "NODE":
                    surface.nodes.extend(p.upper() for p in parts)
                elif len(parts) >= 2:
                    surface.faces.append((parts[0].upper(), parts[1].upper()))
                elif parts:
                    surface.faces.append((parts[0].upper(), "S0"))
        elif name == "BOUNDARY":
            for line in kw.data:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2 or not parts[0]:
                    continue
                try:
                    first = int(parts[1])
                    last = int(parts[2]) if len(parts) > 2 and parts[2] else first
                    value = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
                except ValueError:
                    deck.warnings.append(f"line {kw.line}: unreadable *BOUNDARY data {line!r}")
                    continue
                deck.boundaries.append(Boundary(kw.step, parts[0].upper(), first, last, value, dict(kw.params), kw.line))
        elif name == "CLOAD":
            op = kw.get("OP")
            for line in kw.data:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    deck.cloads.append(CLoad(kw.step, parts[0].upper(), int(parts[1]), float(parts[2]), op, kw.line))
                except ValueError:
                    deck.warnings.append(f"line {kw.line}: unreadable *CLOAD data {line!r}")
        elif name == "DLOAD":
            op = kw.get("OP")
            for line in kw.data:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    deck.dloads.append(DLoad(kw.step, parts[0].upper(), parts[1].upper(), float(parts[2]), op, kw.line))
                except ValueError:
                    deck.warnings.append(f"line {kw.line}: unreadable *DLOAD data {line!r}")
        elif name == "CONTACT PAIR":
            interaction = (kw.get("INTERACTION") or "").upper()
            kind = (kw.get("TYPE") or "NODE TO SURFACE").upper()
            for line in kw.data:
                parts = [p.strip().upper() for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    deck.contact_pairs.append(ContactPair(parts[0], parts[1], interaction, kind, dict(kw.params),
                                                          kw.step, kw.line))
        elif name == "SURFACE INTERACTION":
            current_interaction = Interaction((kw.get("NAME") or "").upper(), line=kw.line)
            deck.interactions[current_interaction.name] = current_interaction
        elif name == "SURFACE BEHAVIOR" and current_interaction is not None:
            current_interaction.pressure_overclosure = (kw.get("PRESSURE-OVERCLOSURE") or "").upper()
            current_interaction.behavior_values = _floats(kw.data)
        elif name == "FRICTION" and current_interaction is not None:
            current_interaction.friction = _floats(kw.data)
        elif name == "TIE":
            for line in kw.data:
                parts = [p.strip().upper() for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    deck.ties.append(Tie((kw.get("NAME") or "").upper(), parts[0], parts[1], dict(kw.params), kw.line))
        elif name == "PRE-TENSION SECTION":
            deck.pretension_sections.append(PreTension((kw.get("SURFACE") or None), kw.get("NODE"), kw.get("ELEMENT"),
                                                       _floats(kw.data), kw.line))
        elif name == "MATERIAL":
            current_material = Material((kw.get("NAME") or "").upper(), line=kw.line)
            deck.materials[current_material.name] = current_material
            current_interaction = None
        elif name == "ELASTIC" and current_material is not None:
            current_material.elastic = _floats(kw.data[:1])
        elif name == "DENSITY" and current_material is not None:
            values = _floats(kw.data[:1])
            current_material.density = values[0] if values else None
        elif name in ("PLASTIC", "EXPANSION", "CONDUCTIVITY", "SPECIFIC HEAT") and current_material is not None:
            current_material.extra[name] = len(kw.data)
        elif name == "SOLID SECTION":
            deck.solid_sections.append({"elset": (kw.get("ELSET") or "").upper(),
                                        "material": (kw.get("MATERIAL") or "").upper(), "line": kw.line})
        elif name == "BEAM SECTION":
            deck.beam_sections.append({"elset": (kw.get("ELSET") or "").upper(), "material": (kw.get("MATERIAL") or "").upper(),
                                       "section": (kw.get("SECTION") or "").upper(), "orientation": (kw.get("ORIENTATION") or "").upper(),
                                       "dims": _floats(kw.data[:1]), "direction": _floats(kw.data[1:2]), "line": kw.line})
        elif name in ("EL FILE", "NODE FILE", "EL PRINT", "NODE PRINT", "CONTACT FILE", "CONTACT PRINT"):
            deck.output_requests.append({"keyword": name, "params": {str(k).upper(): v for k, v in dict(kw.params).items()},
                                         "keys": [k.strip().upper() for k in ",".join(kw.data).split(",") if k.strip()],
                                         "step": kw.step, "line": kw.line})
        elif name == "STEP":
            deck.steps.append(Step(kw.step, dict(kw.params), line=kw.line))
        elif name in ("STATIC", "DYNAMIC", "FREQUENCY", "BUCKLE", "HEAT TRANSFER", "MODAL DYNAMIC") and deck.steps:
            step = deck.steps[-1]
            step.procedure = name
            step.procedure_params = dict(kw.params)
            step.procedure_data = _floats(kw.data[:1])
        elif name == "EQUATION":
            deck.equation_count += 1
        elif name == "MPC":
            deck.mpc_count += 1
    if nodes:
        array = np.array(nodes, dtype=np.float64)
        deck.node_ids = array[:, 0].astype(np.int64)
        deck.node_xyz = array[:, 1:4]
    deck.element_ids = np.array(element_ids, dtype=np.int64)
    deck.nsets = _finish_sets(nset_raw)
    deck.elsets = _finish_sets(elset_raw)
    return deck
