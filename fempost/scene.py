"""Scene construction shared by the GUI and the offscreen CLI.

``Scene`` wraps one ``RunBundle``: it builds the pyvista grid once, derives fields per frame and
renders a ``SceneState`` onto any pyvista plotter (``QtInteractor`` in the GUI, ``Plotter(off_screen)``
for ``render``/tests). Everything the state describes (frame, field, data location, colormap,
range, log, deformation factor or Auto, clip, slice with contours, overlays, mesh quality view,
undeformed ghost, view preset, camera) is JSON-able so an agent can write it and Kurt can open it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dc_field
from pathlib import Path

import numpy as np

from . import fields as fields_module
from . import mesh_stats
from .bundle import RunBundle
from .fields import FieldResult, FIELD_SPECS, ascii_text
from .inp import FACE_NODES, FRD_CODE_TO_TYPE

COLORMAPS = ["turbo", "viridis", "coolwarm", "comsol_rainbow", "jet", "plasma", "RdYlGn", "Greys"]
VIEWS = {"iso": ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)), "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
         "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), "+z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
         "-z": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))}
OVERLAY_KEYS = ("restraints", "loads", "contact", "ties", "pretension", "reactions")
# Restraints and loads are on when a case opens; contact pairs, ties, pretension and reactions stay off.
DEFAULT_OVERLAYS = {"restraints": True, "loads": True, "contact": False, "ties": False, "pretension": False, "reactions": False}
ROBUST_PERCENTILES = (0.5, 99.5)
STATUS_COLORS = {"accepted": "#2e7d32", "evidence": "#1565c0", "converged": "#1565c0", "diagnostic": "#e65100"}
MAX_GLYPHS = 3000          # sphere/arrow glyphs per overlay set; larger sets are subsampled (count stays in the label)
CONTACT_COLORS = {"tied": "#1f77b4", "frictionless": "#ff7f0e", "frictional": "#d62728", "unspecified": "#7f7f7f"}
DOF_COLORS = {"Ux+Uy+Uz": "#c62828", "Ux": "#ef6c00", "Uy": "#2e7d32", "Uz": "#1565c0", "Ux+Uy": "#8e24aa",
              "Uy+Uz": "#00838f", "Ux+Uz": "#6d4c41"}
FACE_VTK = {3: 5, 6: 22, 4: 9, 8: 23}
BACKGROUND = "#f4f6f8"
# Standard plot groups (plan section "Standard plot groups created for every case"). Each entry
# resets the view-related state and applies these overrides; explicit options still win afterwards.
PLOT_GROUPS: list[tuple[str, dict]] = [
    ("Stress: von Mises on the deformed shape", {"field": "von_mises", "deform": None}),
    ("Displacement magnitude", {"field": "u_mag", "deform": None}),
    ("Principal stress S1 (diverging)", {"field": "s1", "deform": None, "cmap": "coolwarm"}),
    ("Principal stress S3 (diverging)", {"field": "s3", "deform": None, "cmap": "coolwarm"}),
    ("Stress component Sxx (pick others in Field)", {"field": "sxx", "deform": None, "cmap": "coolwarm"}),
    ("Contact pressure with pair overlay", {"field": "cpress", "deform": 0.0, "overlays": {"contact": True}}),
    ("Contact gap with pair overlay", {"field": "gap", "deform": 0.0, "cmap": "coolwarm", "overlays": {"contact": True}}),
    ("Mesh quality (scaled Jacobian)", {"mesh_quality": True, "deform": 0.0}),
    ("Reaction forces at restrained sets", {"field": "u_mag", "deform": 0.0, "opacity": 0.35,
                                            "overlays": {"restraints": True, "reactions": True}}),
    ("Solver error estimate (ZZ)", {"field": "error", "deform": 0.0, "cmap": "viridis"}),
    ("Limit overlay: von Mises above 240 MPa", {"field": "von_mises", "deform": None, "limit": 240.0}),
    ("Contact pressure from DAT prints (per slave face)", {"field": "dat_cpress", "deform": 0.0, "overlays": {"contact": True}}),
    # Two-step (KurtsFEM preload / loaded) runs: last frame of step 2 minus last frame of step 1 of the same case.
    ("Loaded minus preload: von Mises (step 2 - step 1, same case)", {"field": "von_mises", "deform": None, "cmap": "coolwarm",
                                                                       "frame": "step:2", "minus_frame": "step:1", "steps": 2}),
    ("Loaded minus preload: displacement (step 2 - step 1, same case)", {"field": "u_mag", "deform": None, "cmap": "coolwarm",
                                                                          "frame": "step:2", "minus_frame": "step:1", "steps": 2}),
]


def plot_group_names(available_fields, n_steps: int = 1) -> list[str]:
    names = []
    for name, overrides in PLOT_GROUPS:
        field = overrides.get("field")
        if overrides.get("steps", 1) > n_steps:
            continue
        if overrides.get("mesh_quality") or field in available_fields:
            names.append(name)
    return names


def apply_plot_group(state: "SceneState", name: str, bundle=None) -> "SceneState":
    """Preset ``name`` applied to a copy of ``state``. ``frame``/``minus_frame`` values of the form
    ``step:n`` resolve to the last frame of step n when a bundle is given (else last / first frame)."""
    overrides = dict(PLOT_GROUPS).get(name)
    if overrides is None:
        raise KeyError(f"unknown plot group {name!r}; choose from {[n for n, _ in PLOT_GROUPS]}")
    new = replace(state)
    new.mesh_quality = False
    new.limit = None
    new.clim = None
    new.cmap = "turbo"
    new.opacity = 1.0
    new.minus_frame = None
    new.overlays = dict(DEFAULT_OVERLAYS)
    for key, value in overrides.items():
        if key == "overlays":
            new.overlays.update(value)
        elif key == "steps":
            continue
        elif key in ("frame", "minus_frame") and isinstance(value, str) and value.startswith("step:"):
            step = int(value.split(":", 1)[1])
            frame = bundle.last_frame_of_step(step) if bundle is not None else None
            if frame is not None:
                setattr(new, key, frame.index)
            else:
                setattr(new, key, -1 if key == "frame" else 0)
        else:
            setattr(new, key, value)
    return new


def colormap(name: str):
    """A matplotlib colormap object or name; ``comsol_rainbow`` is COMSOL's blue-cyan-green-yellow-red."""
    if name == "comsol_rainbow":
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("comsol_rainbow",
                                                 ["#0000ff", "#00bfff", "#00ff80", "#ffff00", "#ff8000", "#ff0000"])
    return name


@dataclass
class SceneState:
    case: str | None = None
    frame: int = -1                       # frame index, -1 = last frame
    field: str = "von_mises"
    location: str = "point"               # point (FRD nodal display) | cell (DAT integration-point element mean)
    cmap: str = "turbo"
    clim: list[float] | None = None       # None = data range (symmetric about 0 for signed fields)
    log_scale: bool = False
    opacity: float = 1.0
    bodies: list[str] | None = None       # visible bodies, None = all
    deform: float | None = None           # None = Auto (10 % of diagonal), 0 = undeformed
    ghost: bool = True                    # undeformed ghost when deformed
    edges: bool = False
    clip: dict | None = None              # {"normal": [..], "origin": [..], "invert": bool, "ghost": bool}
    slice: dict | None = None             # {"normal": [..], "origin": [..], "contours": int, "only": bool}
    overlays: dict = dc_field(default_factory=lambda: dict(DEFAULT_OVERLAYS))
    mesh_quality: bool = False            # colour by corner scaled Jacobian instead of the field
    limit: float | None = None            # clamp colour above this value, report volume fraction
    view: str = "iso"
    parallel: bool = False
    camera: list | None = None            # [[position], [focal point], [view up]] overrides view
    show_scalar_bar: bool = True
    show_legend: bool = True
    title: str | None = None
    minus_case: str | None = None         # A − B: subtract this case's field (same mesh hash required)
    minus_frame: int | None = None        # A − B within the same case: subtract this frame index
    range_mode: str = "robust"            # robust = percentiles of the visible bodies; full = true min..max
    robust_percentiles: list[float] = dc_field(default_factory=lambda: list(ROBUST_PERCENTILES))
    text_scale: float = 1.0               # multiplies every 3D text size (fonts also scale with the window height)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        state = cls(**known)
        overlays = dict(DEFAULT_OVERLAYS)
        overlays.update(state.overlays or {})
        state.overlays = overlays
        return state

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SceneState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


def build_grid(mesh):
    """pyvista UnstructuredGrid of every element with a VTK type, in FRD element order."""
    import pyvista as pv
    counts = mesh.nodes_per_element
    keep = (mesh.vtk_types > 0) & (counts > 0)
    conn = mesh.connectivity[keep]
    counts = counts[keep]
    total = int((counts + 1).sum())
    cells = np.empty(total, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts + 1)[:-1]]) if len(counts) else np.zeros(0, dtype=np.int64)
    cells[starts] = counts
    for k in range(conn.shape[1]):
        rows = np.flatnonzero(counts > k)
        cells[starts[rows] + 1 + k] = conn[rows, k]
    grid = pv.UnstructuredGrid(cells, mesh.vtk_types[keep], mesh.points.copy())
    grid.point_data["node_id"] = mesh.node_ids
    grid.cell_data["element_id"] = mesh.element_ids[keep]
    grid.cell_data["frd_row"] = np.flatnonzero(keep)
    return grid


def face_dataset(points: np.ndarray, faces: list[np.ndarray]):
    """UnstructuredGrid of surface faces (3/6-node triangles, 4/8-node quads) on the given points."""
    import pyvista as pv
    if not faces:
        return None
    cells = []
    types = []
    for face in faces:
        face = np.asarray(face)
        face = face[face >= 0]
        vtk_type = FACE_VTK.get(len(face))
        if vtk_type is None:
            continue
        cells.append(np.concatenate([[len(face)], face]))
        types.append(vtk_type)
    if not cells:
        return None
    return pv.UnstructuredGrid(np.concatenate(cells), np.array(types, dtype=np.uint8), points)


class Scene:
    """Grid, derived fields and overlays of one run; renders a SceneState on a pyvista plotter."""

    def __init__(self, bundle: RunBundle):
        self.bundle = bundle
        self._grid = None
        self._quality = None
        self._subgrids: dict = {}
        self._element_stress: dict = {}
        self.last_info: dict = {}

    # ---- grid ----------------------------------------------------------------------------
    @property
    def grid(self):
        if self._grid is None:
            self._grid = build_grid(self.bundle.mesh)
            ids, names = self.bundle.cell_body_ids()
            self._grid.cell_data["body"] = ids[self._grid.cell_data["frd_row"]]
            regions = self.bundle.regions()
            if regions:
                self._grid.cell_data["region"] = regions[0][self._grid.cell_data["frd_row"]]
        return self._grid

    @property
    def linear_grid(self):
        """Linear version of the grid for clipping and slicing (clean planar cuts, no cracks).

        Quadratic tetrahedra are split into the same eight sub-tetrahedra the structural pipeline
        uses for its field meshes (``structural.results.TET10_SUBTETS``); quadratic hexahedra and
        wedges keep their corner nodes only. ``cell_data["parent"]`` maps back to the grid's cells.
        """
        if getattr(self, "_linear", None) is None:
            import pyvista as pv
            grid = self.grid
            mesh = self.bundle.mesh
            rows = grid.cell_data["frd_row"]
            types = mesh.element_types[rows]
            conn = mesh.connectivity[rows]
            try:
                import sys
                from .frd import ROOT
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from structural.results import TET10_SUBTETS
                subtets = np.asarray(TET10_SUBTETS)
            except Exception:
                subtets = np.array([[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3],
                                    [4, 5, 6, 8], [4, 8, 6, 7], [6, 8, 9, 7], [6, 5, 9, 8]])
            cells, kinds, parent = [], [], []
            for code, vtk_type, count in ((6, 10, 4), (3, 10, 4), (1, 12, 8), (4, 12, 8), (2, 13, 6), (5, 13, 6)):
                sel = np.flatnonzero(types == code)
                if sel.size == 0:
                    continue
                if code == 6:
                    sub = conn[sel][:, subtets].reshape(-1, 4)
                    cells.append(np.hstack([np.full((len(sub), 1), 4), sub]).ravel())
                    kinds.append(np.full(len(sub), 10, dtype=np.uint8))
                    parent.append(np.repeat(sel, len(subtets)))
                else:
                    sub = conn[sel][:, :count]
                    cells.append(np.hstack([np.full((len(sub), 1), count), sub]).ravel())
                    kinds.append(np.full(len(sub), vtk_type, dtype=np.uint8))
                    parent.append(sel)
            if cells:
                linear = pv.UnstructuredGrid(np.concatenate(cells), np.concatenate(kinds), mesh.points.copy())
                linear.cell_data["parent"] = np.concatenate(parent)
            else:
                linear = grid.copy()
                linear.cell_data["parent"] = np.arange(grid.n_cells)
            self._linear = linear
        return self._linear

    @property
    def is_beam(self) -> bool:
        try:
            return bool(self.bundle.beam_info()["is_beam"])
        except Exception:
            return False

    @staticmethod
    def tubes(dataset, radius: float, n_sides: int = 12):
        """Line cells (beam axes) as tubes; point data (fields, U) rides along."""
        lines = dataset.extract_surface(algorithm="dataset_surface") if dataset.__class__.__name__ != "PolyData" else dataset
        return lines.tube(radius=radius, n_sides=n_sides, capping=True)

    @property
    def quality(self) -> mesh_stats.QualityArrays:
        if self._quality is None:
            self._quality = mesh_stats.quality(self.bundle.mesh)
        return self._quality

    def frame_index(self, state: SceneState) -> int:
        frames = self.bundle.frames
        index = state.frame if state.frame is not None else -1
        if index < 0:
            index = len(frames) + index
        return int(min(max(index, 0), len(frames) - 1))

    def body_names(self) -> list[str]:
        return list(self.bundle.bodies())

    def visible_rows_mask(self, state: SceneState) -> np.ndarray:
        ids, names = self.bundle.cell_body_ids()
        if not state.bodies:
            return np.ones(len(ids), dtype=bool)
        wanted = [names.index(b) for b in state.bodies if b in names]
        return np.isin(ids, wanted)

    # ---- fields ----------------------------------------------------------------------------
    def element_stress(self, frame):
        key = round(frame.time, 9)
        if key not in self._element_stress:
            reader = self.bundle.dat
            self._element_stress[key] = reader.element_mean_stress(frame.time) if reader is not None else None
        return self._element_stress[key]

    def has_element_stress(self, frame) -> bool:
        try:
            return self.element_stress(frame) is not None
        except Exception:
            return False

    def available_fields(self, frame) -> list[str]:
        keys = fields_module.available(frame)
        try:
            record = self.dat_contact(frame)
        except Exception:
            record = None
        if record is not None:
            keys += [k for k in fields_module.DAT_CONTACT_KEYS if k in record["values"]]
        return keys

    def dat_contact(self, frame) -> dict | None:
        """*CONTACT PRINT CSTR/CDIS values per slave face at the frame time, with the face node rows."""
        key = round(frame.time, 9)
        cache = getattr(self, "_dat_contact", None)
        if cache is None:
            cache = self._dat_contact = {}
        if key in cache:
            return cache[key]
        record = None
        reader = self.bundle.dat
        prints = reader.contact_prints(frame.time) if reader is not None else None
        if prints is not None:
            mesh = self.bundle.mesh
            pairs: dict[tuple[int, int], int] = {}
            for kind in ("stress", "displacement"):
                table = prints.get(kind)
                if table is None:
                    continue
                for element, face in zip(table["element_ids"], table["faces"]):
                    pairs.setdefault((int(element), int(face)), len(pairs))
            faces: list[np.ndarray] = []
            keep: list[int] = []
            for (element, face), index in pairs.items():
                row = int(mesh.element_index([element])[0])
                if row < 0:
                    continue
                etype = FRD_CODE_TO_TYPE.get(int(mesh.element_types[row]))
                table = FACE_NODES.get(etype or "")
                if not table or face not in table:
                    continue
                faces.append(mesh.connectivity[row, list(table[face])])
                keep.append(index)
            if faces:
                n = len(pairs)
                values: dict[str, np.ndarray] = {}
                stress, disp = prints.get("stress"), prints.get("displacement")
                if stress is not None:
                    press = np.full(n, np.nan)
                    shear = np.full(n, np.nan)
                    for element, face, mean in zip(stress["element_ids"], stress["faces"], stress["mean"]):
                        k = pairs[(int(element), int(face))]
                        press[k] = mean[0]
                        shear[k] = float(np.hypot(mean[1], mean[2]))
                    values["dat_cpress"] = press
                    values["dat_cshear"] = shear
                if disp is not None:
                    gap = np.full(n, np.nan)
                    slip = np.full(n, np.nan)
                    for element, face, mean in zip(disp["element_ids"], disp["faces"], disp["mean"]):
                        k = pairs[(int(element), int(face))]
                        gap[k] = mean[0]
                        slip[k] = float(np.hypot(mean[1], mean[2]))
                    values["dat_gap"] = gap
                    values["dat_slip"] = slip
                keep_array = np.asarray(keep, dtype=np.int64)
                record = {"faces": faces, "values": {k: v[keep_array] for k, v in values.items()},
                          "pairs": [list(pairs)[i] for i in keep], "time": frame.time,
                          "points_per_face": int(np.max(stress["points"])) if stress is not None else None}
        cache[key] = record
        return record

    def mesh_hash(self) -> str:
        if getattr(self, "_hash", None) is None:
            self._hash = self.bundle.mesh.mesh_hash()
        return self._hash

    def difference(self, result: FieldResult, state: SceneState, other: "Scene", other_frame: int | None = None) -> FieldResult:
        """A − B of the same field on a shared mesh. B is ``other`` at the same frame index (clamped),
        or at ``other_frame`` (the preload-versus-loaded preset uses the same case with another frame)."""
        if other.mesh_hash() != self.mesh_hash():
            raise ValueError(f"{other.bundle.name} does not share the solver mesh of {self.bundle.name}; no A - B possible")
        frame = other_frame if other_frame is not None else min(self.frame_index(state), len(other.bundle.frames) - 1)
        other_state = replace(state, frame=frame, minus_case=None, minus_frame=None)
        other_result = other._field_result(other_state)
        if other_result.location != result.location or other_result.values.shape != result.values.shape:
            raise ValueError("A and B fields are not on the same data location")
        b_label = other.bundle.frames[other.frame_index(other_state)].label
        if other is self:
            note = f"A = {self.bundle.frames[self.frame_index(state)].label} minus B = {b_label} (same case); {result.note}"
            label = f"{result.label} change A - B"
        else:
            note = f"A = {self.bundle.name} minus B = {other.bundle.name} ({b_label}); {result.note}"
            label = f"{result.label} difference A - B"
        return FieldResult(result.key, label, result.unit, result.values - other_result.values, result.location, True, note)

    def field_result(self, state: SceneState, other: "Scene | None" = None) -> FieldResult:
        result = self._field_result(state)
        if state.mesh_quality:
            return result
        if other is not None:
            result = self.difference(result, state, other)
        elif state.minus_frame is not None:
            result = self.difference(result, state, self, other_frame=int(state.minus_frame))
        return result

    def contact_node_mask(self) -> np.ndarray:
        """Nodes that belong to a contact pair's slave surface (CalculiX writes CONTACT values only
        there; every other node carries zero). Without a deck: nodes with any non-zero CONTACT value."""
        if getattr(self, "_contact_mask", None) is None:
            mesh = self.bundle.mesh
            mask = np.zeros(mesh.n_nodes, dtype=bool)
            pairs = self.bundle.contact_pairs()
            for pair in pairs:
                for face in self.bundle.surface_face_nodes(pair["slave"]):
                    rows = np.asarray(face)
                    mask[rows[rows >= 0]] = True
            if not mask.any():
                for frame in self.bundle.frames:
                    if "CONTACT" in frame.fields:
                        mask |= (np.nan_to_num(frame.fields["CONTACT"]) != 0).any(axis=1)
            self._contact_mask = mask
        return self._contact_mask

    def _field_result(self, state: SceneState) -> FieldResult:
        frame = self.bundle.frames[self.frame_index(state)]
        if state.mesh_quality:
            values = self.quality.scaled_jacobian[self.grid.cell_data["frd_row"]]
            return FieldResult("scaled_jacobian", "Scaled Jacobian (corner nodes)", "-", values, "cell", False,
                               "vtkMeshQuality on the corner-node element; 1 = ideal hexahedron, 0.707 = equilateral tetrahedron")
        key = state.field if state.field in FIELD_SPECS else "u_mag"
        if key in fields_module.DAT_CONTACT_KEYS:
            record = self.dat_contact(frame)
            if record is not None and key in record["values"]:
                spec = FIELD_SPECS[key]
                note = f"{fields_module.DAT_CONTACT_NOTE}; {len(record['faces'])} slave faces printed at t={frame.time:g}"
                return FieldResult(key, spec.label, spec.unit, np.array(record["values"][key], dtype=float), "face", spec.signed, note)
            key = "cpress" if "cpress" in fields_module.available(frame) else "u_mag"
        if key not in fields_module.available(frame):
            available = fields_module.available(frame)
            key = "u_mag" if "u_mag" in available else (available[0] if available else None)
            if key is None:
                raise ValueError("frame has no fields")
        result = None
        if state.location == "cell" and FIELD_SPECS[key].source == "STRESS":
            record = self.element_stress(frame)
            if record is not None:
                result = fields_module.compute_element(key, self.bundle.mesh, record, frame.derived)
                result.values = self._blank_nonphysical(result.values.copy(), "cell")[self.grid.cell_data["frd_row"]]
        if result is None:
            result = fields_module.compute(key, frame)
            if state.location == "cell":
                result.note += "; DAT element stress unavailable at this time, showing nodal data"
            if FIELD_SPECS[key].source == "STRESS":
                result.values = self._blank_nonphysical(np.array(result.values, dtype=float), "point")
            if FIELD_SPECS[key].source == "CONTACT":
                mask = self.contact_node_mask()
                values = np.array(result.values, dtype=float)
                values[~mask] = np.nan
                result.values = values
                result.note = (f"{fields_module.CONTACT_SIGN_NOTE}; {int(mask.sum())} slave-surface nodes carry values, "
                               f"{int((~mask).sum())} other nodes blanked (grey); colour range over the slave nodes only")
        blanked = self.bundle.nonphysical_stress_bodies()
        if blanked and FIELD_SPECS[key].source == "STRESS":
            result.note += "; " + "; ".join(blanked.values())
        return result

    def _blank_nonphysical(self, values: np.ndarray, location: str) -> np.ndarray:
        """NaN the stress of bodies the pipeline declares non-physical (FRD order arrays)."""
        blanked = self.bundle.nonphysical_stress_bodies()
        if not blanked:
            return values
        bodies = self.bundle.bodies()
        mesh = self.bundle.mesh
        for name in blanked:
            rows = bodies.get(name)
            if rows is None:
                continue
            if location == "cell":
                values[rows] = np.nan
            else:
                nodes = np.unique(mesh.connectivity[rows])
                values[nodes[nodes >= 0]] = np.nan
        return values

    # ---- deformation -----------------------------------------------------------------------
    def deformation_factor(self, state: SceneState) -> float:
        frame = self.bundle.frames[self.frame_index(state)]
        if state.deform is None:
            return fields_module.auto_deformation_factor(frame, self.bundle.mesh.diagonal)
        return float(state.deform)

    def displacement(self, state: SceneState) -> np.ndarray:
        frame = self.bundle.frames[self.frame_index(state)]
        if "DISP" in frame.fields:
            u = np.nan_to_num(frame.fields["DISP"][:, :3])
        else:
            u = np.zeros((self.bundle.mesh.n_nodes, 3))
        return u

    # ---- overlays (datasets only; no plotter) ----------------------------------------------
    def _warped_points(self, state: SceneState, factor: float) -> np.ndarray:
        return self.bundle.mesh.points + factor * self.displacement(state)

    @property
    def surface(self):
        """Outer surface of the full grid with original point ids (cached; used to tint the faces of node sets)."""
        if getattr(self, "_surface", None) is None:
            surface = self.grid.extract_surface(pass_pointid=True, pass_cellid=True, algorithm="dataset_surface")
            self._surface = surface
            ids = surface.point_data["vtkOriginalPointIds"]
            faces = surface.faces if surface.n_cells else np.zeros(0, dtype=np.int64)
            self._surface_face_nodes = None
            if faces.size:
                # flat list of (face, original node) pairs for fast "all nodes of the face are in the set" tests
                offsets = []
                k = 0
                while k < faces.size:
                    n = int(faces[k])
                    offsets.append((k + 1, n))
                    k += n + 1
                face_index = np.concatenate([np.full(n, i) for i, (_, n) in enumerate(offsets)])
                node_rows = np.concatenate([faces[start:start + n] for start, n in offsets])
                self._surface_face_nodes = (face_index, ids[node_rows], len(offsets))
        return self._surface

    def set_faces(self, node_mask: np.ndarray, points: np.ndarray):
        """Surface faces whose nodes all lie in ``node_mask`` (FRD rows), on the given (warped) points."""
        surface = self.surface
        if getattr(self, "_surface_face_nodes", None) is None:
            return None
        face_index, nodes, n_faces = self._surface_face_nodes
        inside = node_mask[nodes]
        all_in = np.ones(n_faces, dtype=bool)
        np.logical_and.at(all_in, face_index, inside)
        picked = np.flatnonzero(all_in)
        if picked.size == 0:
            return None
        subset = surface.extract_cells(picked)
        original = subset.point_data["vtkOriginalPointIds"]
        subset.points = points[original]
        return subset

    @staticmethod
    def _subsample(rows: np.ndarray, limit: int = MAX_GLYPHS) -> np.ndarray:
        if rows.size <= limit:
            return rows
        stride = int(np.ceil(rows.size / limit))
        return rows[::stride]

    def _glyph_radius(self, set_points: np.ndarray) -> float:
        """Sphere radius for a node set: 4 % of the set's extent, clamped to 0.8–2 % of the model diagonal."""
        diagonal = self.bundle.mesh.diagonal
        extent = float(np.linalg.norm(set_points.max(axis=0) - set_points.min(axis=0))) if len(set_points) > 1 else 0.0
        return float(np.clip(0.04 * extent, 0.008 * diagonal, 0.02 * diagonal))

    def restraint_overlays(self, state: SceneState, points: np.ndarray) -> list[dict]:
        import pyvista as pv
        frame = self.bundle.frames[self.frame_index(state)]
        out = []
        mesh = self.bundle.mesh
        for entry in self.bundle.restraints():
            if entry["step"] > frame.step:
                continue
            rows = self.bundle.node_set_rows(entry["target"])
            if rows.size == 0:
                continue
            color = DOF_COLORS.get(entry["dofs"], "#555555")
            label = f"{entry['target']} {entry['dofs']}" + (f" = {entry['value']:g}" if entry["value"] else "") + f" ({rows.size} nodes"
            label += f", step {entry['step']})" if entry["step"] else ")"
            set_points = points[rows]
            radius = self._glyph_radius(set_points)
            cloud = pv.PolyData(points[self._subsample(rows)])
            spheres = cloud.glyph(geom=pv.Sphere(radius=radius, theta_resolution=10, phi_resolution=8), scale=False, orient=False)
            mask = np.zeros(mesh.n_nodes, dtype=bool)
            mask[rows] = True
            faces = self.set_faces(mask, points)
            extent = (set_points.max(axis=0) - set_points.min(axis=0)).tolist()
            out.append({"kind": "restraint", "dataset": spheres, "faces": faces, "color": color, "label": label,
                        "count": int(rows.size), "step": entry["step"], "glyph_radius_mm": radius,
                        "extent_mm": extent, "centroid_mm": set_points.mean(axis=0).tolist()})
        return out

    def load_overlays(self, state: SceneState, points: np.ndarray) -> list[dict]:
        import pyvista as pv
        frame = self.bundle.frames[self.frame_index(state)]
        diagonal = self.bundle.mesh.diagonal
        out = []
        groups = [g for g in self.bundle.cload_groups() if g["step"] <= frame.step]
        active = []
        for group in groups:        # OP=NEW in a later step replaces the earlier concentrated loads
            if group["op_new"]:
                active = []
            active.append(group)
        mesh = self.bundle.mesh
        for group in active:
            vectors = group["forces"]
            arrows = None
            faces = None
            rows = np.asarray(group["rows"])
            if len(vectors) and rows.size:
                set_points = points[rows]
                extent = float(np.linalg.norm(set_points.max(axis=0) - set_points.min(axis=0))) if rows.size > 1 else 0.0
                magnitudes = np.linalg.norm(vectors, axis=1)
                peak = float(magnitudes.max())
                # arrow length between 1.5 % and 6 % of the diagonal (or 30 % of the patch extent), so a
                # patch of small nodal forces still reads; the shaft thickness follows the length.
                longest = float(np.clip(0.3 * extent, 0.03 * diagonal, 0.06 * diagonal)) if extent > 0 else 0.06 * diagonal
                shortest = 0.4 * longest
                lengths = shortest + (longest - shortest) * (magnitudes / peak if peak > 0 else 1.0)
                keep = self._subsample(np.arange(rows.size))
                cloud = pv.PolyData(set_points[keep])
                cloud["F"] = vectors[keep]
                cloud["L"] = lengths[keep]
                # arrows start on the node and point along the force; the resultant arrow points at the patch
                arrows = cloud.glyph(orient="F", scale="L", factor=1.0,
                                     geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.045))
                mask = np.zeros(mesh.n_nodes, dtype=bool)
                mask[rows] = True
                faces = self.set_faces(mask, points)
            resultant = np.asarray(group["resultant_N"])
            magnitude = float(np.linalg.norm(resultant))
            centroid = np.asarray(group["centroid_mm"], dtype=float)
            total = float(np.abs(np.asarray(group["forces"])).sum()) if len(group["forces"]) else 0.0
            self_equilibrated = total > 0 and magnitude <= 1e-6 * total     # e.g. the cube frame's balanced load sets
            label = f"CLOAD step {group['step']}: {group['nodes']} nodes, ΣF = {magnitude:.4g} N"
            if self_equilibrated:
                label += f" (self-equilibrated; Σ|F| = {total:.4g} N)"
            entry = {"kind": "load", "dataset": arrows, "faces": faces, "color": "#e53935", "label": label, "count": group["nodes"],
                     "step": group["step"], "resultant": resultant.tolist(), "magnitude": magnitude,
                     "sum_abs_N": total, "self_equilibrated": self_equilibrated}
            if magnitude > 0 and np.isfinite(centroid).all() and not self_equilibrated:
                direction = resultant / magnitude
                length = 0.28 * diagonal
                entry["resultant_arrow"] = pv.Arrow(start=centroid - direction * length, direction=direction, scale=length,
                                                    tip_length=0.22, tip_radius=0.075, shaft_radius=0.03)
                entry["resultant_point"] = centroid
                entry["resultant_tail"] = (centroid - direction * length).tolist()
                entry["resultant_text"] = (f"ΣF = {magnitude:.4g} N\n({resultant[0]:.4g}, {resultant[1]:.4g}, {resultant[2]:.4g}) N"
                                           f"\nat ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm")
            out.append(entry)
        for load in self.bundle.dloads():
            if load["step"] > frame.step:
                continue
            entry = self._dload_overlay(load, points, diagonal)
            if entry:
                out.append(entry)
        return out

    def _dload_overlay(self, load: dict, points: np.ndarray, diagonal: float) -> dict | None:
        """One arrow set for a logical ``*DLOAD`` group (all its (element, face) pairs at once)."""
        import pyvista as pv
        mesh = self.bundle.mesh
        face_numbers = np.asarray(load["face_numbers"])
        wanted = face_numbers > 0                               # P<n> labels only (GRAV, CENTRIF ... have no face)
        if not wanted.any():
            return None
        rows = mesh.element_index(np.asarray(load["element_ids"])[wanted])
        faces = face_numbers[wanted]
        ok = rows >= 0
        rows, faces = rows[ok], faces[ok]
        centroids = []
        normals = []
        for row, face in zip(rows.tolist(), faces.tolist()):
            etype = FRD_CODE_TO_TYPE.get(int(mesh.element_types[row]))
            table = FACE_NODES.get(etype or "")
            if not table or face not in table:
                continue
            local = table[face]
            nodes = mesh.connectivity[row, list(local[:3])]
            p = points[nodes]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            corners = mesh.connectivity[row, list(local[:4 if len(local) in (4, 8) else 3])]
            centroids.append(points[corners].mean(axis=0))
            normals.append(_unit(normal))
        if not centroids:
            return None
        centroids = np.array(centroids)
        normals = np.array(normals)
        # arrow length from the loaded patch's extent, clamped to 3-6 % of the diagonal (as the CLOAD arrows)
        extent = float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))) if len(centroids) > 1 else 0.0
        length = float(np.clip(0.3 * extent, 0.03 * diagonal, 0.06 * diagonal)) if extent > 0 else 0.06 * diagonal
        # thin the arrows to one per grid cell of 0.35 arrow lengths (a face-by-face pressure band has a
        # face every few mm: drawing every one of them at 3-6 % of the diagonal hides the part under them)
        cells = np.floor(centroids / (0.35 * length)).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        keep = self._subsample(np.sort(first))
        cloud = pv.PolyData(centroids[keep])
        sign = -1.0 if load["value"] > 0 else 1.0           # positive pressure pushes into the face
        cloud["N"] = sign * normals[keep]
        arrows = cloud.glyph(orient="N", scale=False, factor=length, geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.045))
        return {"kind": "dload", "dataset": arrows, "color": "#8e24aa", "count": int(len(centroids)), "shown": int(keep.size),
                "step": load["step"], "value": load["value"], "labels": list(load["labels"]),
                "glyph_length_mm": length, "extent_mm": extent, "label": load["label"]}

    def surface_dataset(self, name: str, points: np.ndarray):
        faces = self.bundle.surface_face_nodes(name)
        if not faces:
            return None
        if len(faces) == 1 and faces[0].ndim == 1 and self.bundle.deck.surfaces.get(name.upper(), None) and \
                self.bundle.deck.surfaces[name.upper()].kind == "NODE":
            import pyvista as pv
            rows = faces[0]
            return pv.PolyData(points[rows[rows >= 0]])
        return face_dataset(points, faces)

    def contact_overlays(self, state: SceneState, points: np.ndarray, kind: str = "contact") -> list[dict]:
        out = []
        pairs = self.bundle.contact_pairs() if kind == "contact" else self.bundle.ties()
        for pair in pairs:
            slave = self.surface_dataset(pair["slave"], points)
            master = self.surface_dataset(pair["master"], points)
            if kind == "contact":
                category = pair.get("category", "unspecified")
                color = CONTACT_COLORS.get(category, "#7f7f7f")
                type_label = f" [{pair['type']}]" if pair.get("type") else ""
                friction = f", μ={pair['friction'][0]:g}" if pair.get("friction") else ""
                label = (f"{pair['slave']} → {pair['master']}{type_label}: {category}{friction} "
                         f"({pair['slave_faces']} slave / {pair['master_faces']} master faces)")
            else:
                color = "#43a047"
                label = f"TIE {pair['name']} {pair['slave']} → {pair['master']} ({pair['slave_faces']} / {pair['master_faces']} faces)"
            out.append({"kind": kind, "slave": slave, "master": master, "color": color, "label": label,
                        "count": pair.get("slave_faces", 0), "pair": pair})
        return out

    def pretension_overlays(self, state: SceneState, points: np.ndarray) -> list[dict]:
        import pyvista as pv
        out = []
        mesh = self.bundle.mesh
        diagonal = self.bundle.mesh.diagonal
        deck = self.bundle.deck
        for entry in self.bundle.pretension():
            if entry.get("source") == "assembly.json":
                node = entry.get("node")
                row = int(mesh.node_index([node])[0]) if node is not None else -1
                position = points[row] if row >= 0 else None
                if position is None and deck is not None and deck.node_ids.size:
                    hit = np.flatnonzero(deck.node_ids == node)
                    position = deck.node_xyz[hit[0]] if hit.size else None
                if position is None:
                    continue
                axis = _unit(entry.get("axis") or [1, 0, 0])
                arrow = pv.Arrow(start=position, direction=axis, scale=0.12 * diagonal, tip_length=0.25, tip_radius=0.08, shaft_radius=0.03)
                out.append({"kind": "pretension", "dataset": arrow, "marker": pv.PolyData(np.atleast_2d(position)), "color": "#f9a825",
                            "label": f"Pretension screw {entry.get('screw')} node {node}: target {entry.get('target_N')} N",
                            "count": 1})
            elif entry.get("surface"):
                surface = self.surface_dataset(entry["surface"], points)
                if surface is not None:
                    out.append({"kind": "pretension", "slave": surface, "color": "#f9a825",
                                "label": f"PRE-TENSION SECTION {entry['surface']} ({entry.get('faces', 0)} faces)", "count": entry.get("faces", 0)})
        return out

    def reaction_overlays(self, state: SceneState, points: np.ndarray) -> list[dict]:
        """Reaction force per restrained node set: FRD FORC (RF) summed over the constrained DOFs,
        with the DAT ``total force`` of the same set when the deck printed it."""
        import pyvista as pv
        frame = self.bundle.frames[self.frame_index(state)]
        if "FORC" not in frame.fields:
            return []
        forces = np.nan_to_num(frame.fields["FORC"][:, :3])
        diagonal = self.bundle.mesh.diagonal
        totals: dict[str, list] = {}
        if self.bundle.dat is not None:
            try:
                for entry in self.bundle.dat.total_forces(frame.time):
                    totals[str(entry["set"]).upper()] = entry["force_N"]
            except Exception:
                pass
        per_target: dict[str, set] = {}
        for entry in self.bundle.restraints():
            if entry["step"] <= frame.step:
                per_target.setdefault(entry["target"], set()).update(
                    d - 1 for d in range(entry["first_dof"], entry["last_dof"] + 1) if 1 <= d <= 3)
        out = []
        for target, dofs in per_target.items():
            rows = self.bundle.node_set_rows(target)
            if rows.size == 0 or not dofs:
                continue
            dofs = sorted(dofs)
            vector = np.zeros(3)
            vector[dofs] = forces[rows][:, dofs].sum(axis=0)
            magnitude = float(np.linalg.norm(vector))
            centroid = points[rows].mean(axis=0)
            label = (f"RF {target} = ({vector[0]:.4g}, {vector[1]:.4g}, {vector[2]:.4g}) N, |RF| = {magnitude:.4g} N "
                     f"({rows.size} nodes)")
            if target in totals:
                label += f"; DAT total {np.round(totals[target], 3).tolist()}"
            entry = {"kind": "reaction", "color": "#00897b", "label": label, "count": int(rows.size), "target": target,
                     "force": vector.tolist(), "magnitude": magnitude, "dat_total": totals.get(target)}
            if magnitude > 0:
                direction = vector / magnitude
                length = 0.2 * diagonal
                entry["dataset"] = pv.Arrow(start=centroid, direction=direction, scale=length, tip_length=0.22,
                                            tip_radius=0.07, shaft_radius=0.028)
                entry["point"] = centroid
                entry["text"] = f"RF {target}\n|RF| = {magnitude:.4g} N"
            out.append(entry)
        return out

    # ---- render ----------------------------------------------------------------------------
    def render(self, plotter, state: SceneState, other: "Scene | None" = None) -> dict:
        import pyvista as pv
        plotter.clear()
        plotter.set_background(BACKGROUND)
        bundle = self.bundle
        frame = bundle.frames[self.frame_index(state)]
        result = self.field_result(state, other)
        grid = self.grid
        factor = 0.0 if state.mesh_quality and state.deform is None else self.deformation_factor(state)
        info: dict = {"frame": frame.label, "frame_index": frame.index, "field": result.title, "location": result.location,
                      "note": result.note, "deformation_factor": factor, "overlays": [], "legend": [], "warnings": []}
        grid.point_data["U"] = self.displacement(state)
        face_mode = result.location == "face"          # DAT contact prints live on slave faces, not on the grid
        name = result.key + ("_cell" if result.location == "cell" else "")
        if face_mode:
            pass
        elif result.location == "cell":
            grid.cell_data[name] = result.values
        else:
            grid.point_data[name] = result.values
        mask = self.visible_rows_mask(state)[grid.cell_data["frd_row"]]
        shown = grid if mask.all() else grid.extract_cells(np.flatnonzero(mask))
        warped = shown.warp_by_vector("U", factor=factor) if factor else shown
        cut_source = None
        beam = self.is_beam
        if beam:
            # beam axes (B31/B32 lines) are drawn as tubes with the *BEAM SECTION half-size so the field reads
            radius = float(self.bundle.beam_info()["radius_mm"] or 0.004 * self.bundle.mesh.diagonal)
            try:
                warped = self.tubes(warped, radius)
                info["beam"] = {"tubes": True, "radius_mm": radius, "section_forces": bool(frame.derived.get("section_forces"))}
            except Exception as exc:
                info["warnings"].append(f"beam tubes: {exc}")
                info["beam"] = {"tubes": False, "radius_mm": radius}
        if face_mode and (state.clip or state.slice):
            info["warnings"].append("clip and slice are not applied to DAT contact-print faces")
        if (state.clip or state.slice) and beam and not face_mode:
            cut_source = warped
        elif (state.clip or state.slice) and not face_mode:
            linear = self.linear_grid
            parent = linear.cell_data["parent"]
            linear.point_data["U"] = grid.point_data["U"]
            if result.location == "cell":
                linear.cell_data[name] = grid.cell_data[name][parent]
            else:
                linear.point_data[name] = grid.point_data[name]
            cut_source = linear if mask.all() else linear.extract_cells(np.flatnonzero(mask[parent]))
            if factor:
                cut_source = cut_source.warp_by_vector("U", factor=factor)
        epsilon = 1e-6 * self.bundle.mesh.diagonal     # keep cut planes off element faces
        fs = self.font_scale(plotter, state)
        # colour range: true min/max of the visible bodies, robust percentiles by default
        if face_mode:
            visible_values = result.values
        else:
            visible_values = shown.cell_data[name] if result.location == "cell" else shown.point_data[name]
        finite = np.asarray(visible_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            true_lo, true_hi = float(finite.min()), float(finite.max())
        else:
            true_lo, true_hi = result.finite_range()
        lo, hi = true_lo, true_hi
        range_mode = "full"
        if state.clim is not None and len(state.clim) == 2:
            lo, hi = float(state.clim[0]), float(state.clim[1])
            range_mode = "manual"
        elif state.range_mode == "robust" and not state.mesh_quality and finite.size > 20:
            p_lo, p_hi = state.robust_percentiles or ROBUST_PERCENTILES
            r_lo, r_hi = (float(v) for v in np.percentile(finite, [p_lo, p_hi]))
            if r_hi > r_lo:
                lo, hi = r_lo, r_hi
                range_mode = "robust"
        if range_mode != "manual" and result.signed and not state.mesh_quality:
            span = max(abs(lo), abs(hi))
            lo, hi = -span, span
        if state.mesh_quality and state.clim is None:
            lo, hi = min(0.0, true_lo), 1.0
            range_mode = "full"
        if state.limit is not None and state.clim is None:
            hi = float(state.limit)
        if state.log_scale:
            lo = max(lo, hi * 1e-4 if hi > 0 else 1e-6)
        if hi - lo <= 1e-9 * max(abs(lo), abs(hi), 1e-30):   # uniform field: open a readable band
            pad = 0.05 * abs(hi) if abs(hi) > 0 else 1.0
            lo, hi = lo - pad, hi + pad
        cmap = "RdYlGn" if state.mesh_quality and state.cmap == "turbo" else state.cmap
        if result.signed and state.cmap == "turbo" and not state.mesh_quality:
            cmap = "coolwarm"
        bar_title = result.bar_title
        if range_mode == "robust":
            p_lo, p_hi = state.robust_percentiles or ROBUST_PERCENTILES
            bar_title += f"\nshown: {p_lo:g}-{p_hi:g} %\ntrue min {true_lo:.3g}\ntrue max {true_hi:.3g}"
        elif range_mode == "manual":
            bar_title += f"\nmanual range\ntrue min {true_lo:.3g}\ntrue max {true_hi:.3g}"
        elif not state.mesh_quality:
            bar_title += f"\nfull range\nmin {true_lo:.3g}\nmax {true_hi:.3g}"
        common = dict(scalars=name, cmap=colormap(cmap), clim=(lo, hi), log_scale=state.log_scale, opacity=state.opacity,
                      show_edges=state.edges or state.mesh_quality, nan_color="#c8c8c8", nan_opacity=0.5,
                      show_scalar_bar=state.show_scalar_bar,
                      scalar_bar_args={"title": bar_title, "vertical": True, "position_x": 0.86, "position_y": 0.16,
                                       "height": 0.66, "width": 0.05, "title_font_size": int(round(13 * fs)),
                                       "label_font_size": int(round(12 * fs)), "n_labels": 7, "fmt": "%.4g",
                                       "font_family": "arial", "shadow": False})
        if state.limit is not None:
            common["above_color"] = "#ff00ff"
        slice_only = bool(state.slice and state.slice.get("only"))
        if face_mode:
            plotter.add_mesh(warped, color="#cfd8dc", opacity=0.35, show_scalar_bar=False)
            record = self.dat_contact(frame)
            points_now = self._warped_points(state, factor)
            faces = face_dataset(points_now, record["faces"])
            if faces is not None:
                faces.cell_data[name] = result.values
                plotter.add_mesh(faces, **common)
            info["dat_contact"] = {"faces": len(record["faces"]), "time": record["time"],
                                   "points_per_face": record.get("points_per_face")}
        elif state.clip:
            normal = _unit(state.clip.get("normal", [1, 0, 0]))
            origin = np.asarray(state.clip.get("origin") or warped.center, dtype=float) + epsilon * normal
            invert = bool(state.clip.get("invert", False))
            kept = cut_source.clip(normal=normal, origin=origin, invert=invert)
            if kept.n_cells:
                plotter.add_mesh(kept, **common)
            if state.clip.get("ghost", True):
                removed = cut_source.clip(normal=normal, origin=origin, invert=not invert)
                if removed.n_cells:
                    plotter.add_mesh(removed, color="#b0bec5", opacity=0.12, show_scalar_bar=False)
            info["clip"] = {"normal": normal.tolist(), "origin": origin.tolist(), "cells": int(kept.n_cells)}
        elif not slice_only:
            plotter.add_mesh(warped, **common)
        if state.slice and not face_mode:
            normal = _unit(state.slice.get("normal", [0, 0, 1]))
            origin = np.asarray(state.slice.get("origin") or warped.center, dtype=float) + epsilon * normal
            cut = cut_source.slice(normal=normal, origin=origin)
            if cut.n_cells:
                slice_args = dict(common)
                slice_args["opacity"] = 1.0
                slice_args["show_scalar_bar"] = state.show_scalar_bar and (slice_only or state.clip is None)
                if state.clip is None and not slice_only:
                    slice_args["show_scalar_bar"] = False
                plotter.add_mesh(cut, **slice_args)
                contours = int(state.slice.get("contours", 10) or 0)
                if contours > 0:
                    source = cut.cell_data_to_point_data() if result.location == "cell" else cut
                    try:
                        lines = source.contour(isosurfaces=contours, scalars=name, rng=(lo, hi))
                        if lines.n_cells:
                            plotter.add_mesh(lines, color="black", line_width=1.5, show_scalar_bar=False)
                    except Exception as exc:
                        info["warnings"].append(f"contours: {exc}")
                info["slice"] = {"normal": normal.tolist(), "origin": origin.tolist(), "cells": int(cut.n_cells)}
            if slice_only and state.clip is None:
                plotter.add_mesh(warped, color="#b0bec5", opacity=0.08, show_scalar_bar=False)
        if factor and state.ghost:
            plotter.add_mesh(shown, style="wireframe", color="#90a4ae", opacity=0.25, line_width=1, show_scalar_bar=False)
        info["range"] = [lo, hi]
        info["range_true"] = [true_lo, true_hi]
        info["range_mode"] = range_mode
        info["font_scale"] = fs
        if state.limit is not None and face_mode:
            info["warnings"].append("limit overlay is not applied to DAT contact-print faces")
        elif state.limit is not None:
            volumes = self.quality.volume[grid.cell_data["frd_row"]]
            conn = bundle.mesh.connectivity[grid.cell_data["frd_row"]]
            if result.location == "point":
                values = result.values
            else:
                values = result.values
            info["limit"] = fields_module.volume_fraction_above(values, state.limit, volumes, conn, result.location)
            try:
                source = warped if result.location == "point" else warped.cell_data_to_point_data()
                iso = source.contour([float(state.limit)], scalars=name)
                if iso.n_cells:
                    plotter.add_mesh(iso, color="#ff00ff", opacity=0.55, show_scalar_bar=False)
                info["limit"]["isosurface_cells"] = int(iso.n_cells)
            except Exception as exc:
                info["warnings"].append(f"limit isosurface: {exc}")
        # overlays
        points = self._warped_points(state, factor)
        legend = []
        overlays = state.overlays or {}
        label_font = int(round(13 * fs))
        if overlays.get("restraints"):
            for entry in self.restraint_overlays(state, points):
                if entry.get("faces") is not None:
                    plotter.add_mesh(entry["faces"], color=entry["color"], opacity=0.85, show_scalar_bar=False)
                plotter.add_mesh(entry["dataset"], color=entry["color"], show_scalar_bar=False, smooth_shading=True)
                legend.append([entry["label"], entry["color"]])
                info["overlays"].append({k: v for k, v in entry.items() if k not in ("dataset", "faces")})
        if overlays.get("loads"):
            for entry in self.load_overlays(state, points):
                if entry.get("faces") is not None:
                    plotter.add_mesh(entry["faces"], color="#ff8a80", opacity=0.75, show_scalar_bar=False)
                if entry.get("dataset") is not None:
                    plotter.add_mesh(entry["dataset"], color=entry["color"], show_scalar_bar=False)
                if entry.get("resultant_arrow") is not None:
                    plotter.add_mesh(entry["resultant_arrow"], color="#b71c1c", show_scalar_bar=False, smooth_shading=True)
                    plotter.add_point_labels([entry["resultant_tail"]], [ascii_text(entry["resultant_text"])], font_size=label_font,
                                             point_size=1, shape_opacity=0.85, shape_color="white", text_color="#b71c1c",
                                             bold=True, always_visible=True, margin=6)
                legend.append([entry["label"], entry["color"]])
                info["overlays"].append({k: v for k, v in entry.items() if k not in ("dataset", "faces", "resultant_arrow", "resultant_point")})
        for kind, flag in (("contact", overlays.get("contact")), ("ties", overlays.get("ties"))):
            if not flag:
                continue
            entries = self.contact_overlays(state, points, "contact" if kind == "contact" else "ties")
            grouped: dict[str, list] = {}
            # with a contact field on screen the slave faces are drawn as wireframe so the colours show through
            contact_field = FIELD_SPECS.get(result.key) is not None and FIELD_SPECS[result.key].source in ("CONTACT", "DATCONTACT")
            for entry in entries:
                if entry["slave"] is not None:
                    if contact_field:
                        plotter.add_mesh(entry["slave"], color=entry["color"], style="wireframe", line_width=1.5, opacity=0.9,
                                         show_scalar_bar=False)
                    else:
                        plotter.add_mesh(entry["slave"], color=entry["color"], opacity=0.9, show_scalar_bar=False)
                if entry["master"] is not None:
                    plotter.add_mesh(entry["master"], color=entry["color"], style="wireframe", line_width=1.2, opacity=0.8, show_scalar_bar=False)
                info["overlays"].append({k: v for k, v in entry.items() if k not in ("slave", "master", "pair")})
                category = entry["pair"].get("category", "tie") if kind == "contact" else "tie"
                grouped.setdefault(category, []).append(entry)
            if len(entries) <= 6:
                legend += [[entry["label"], entry["color"]] for entry in entries]
            else:           # many pairs: one legend line per category; every pair is listed in the model tree
                for category, group in grouped.items():
                    faces = sum(int(e.get("count") or 0) for e in group)
                    types = sorted({str(e["pair"].get("type")) for e in group if e["pair"].get("type")})
                    legend.append([f"{kind}: {category} x{len(group)} pairs{' [' + ', '.join(types) + ']' if types else ''}, "
                                   f"{faces} slave faces (pairs in the model tree)", group[0]["color"]])
        if overlays.get("pretension"):
            for entry in self.pretension_overlays(state, points):
                for key in ("dataset", "slave"):
                    if entry.get(key) is not None:
                        plotter.add_mesh(entry[key], color=entry["color"], show_scalar_bar=False)
                if entry.get("marker") is not None:
                    plotter.add_mesh(entry["marker"], color="#f57f17", point_size=int(14 * fs), render_points_as_spheres=True, show_scalar_bar=False)
                legend.append([entry["label"], entry["color"]])
                info["overlays"].append({k: v for k, v in entry.items() if k not in ("dataset", "slave", "marker")})
        if overlays.get("reactions"):
            for entry in self.reaction_overlays(state, points):
                if entry.get("dataset") is not None:
                    plotter.add_mesh(entry["dataset"], color=entry["color"], show_scalar_bar=False, smooth_shading=True)
                    plotter.add_point_labels([entry["point"]], [ascii_text(entry["text"])], font_size=label_font, point_size=1,
                                             shape_opacity=0.85, shape_color="white", text_color="#00695c", bold=True,
                                             always_visible=True, margin=6)
                legend.append([entry["label"], entry["color"]])
                info["overlays"].append({k: v for k, v in entry.items() if k not in ("dataset", "point")})
        if legend and state.show_legend:
            limit_entries = 10
            shown_entries = legend[:limit_entries]
            if len(legend) > limit_entries:
                shown_entries.append([f"... {len(legend) - limit_entries} more (see the model tree)", "#555555"])
            info["legend"] = [text for text, _ in shown_entries]
            try:
                # the legend box scales its text to the box: height per entry follows the text scale,
                # the width follows the longest label so long entries are not shrunk to fit
                longest = max(len(text) for text, _ in shown_entries)
                width = float(np.clip((0.16 + 0.0055 * min(longest, 110)) * fs, 0.25, 0.75))
                height = min(0.55, (0.030 * fs) * len(shown_entries) + 0.012)
                plotter.add_legend(labels=[[ascii_text(text)[:110], color] for text, color in shown_entries], bcolor="white",
                                   border=True, size=(width, height), loc="lower left", face=None)
            except Exception as exc:
                info["warnings"].append(f"legend: {exc}")
        # annotations
        acceptance = bundle.acceptance()
        header = state.title or f"{bundle.name} · {frame.label}"
        if other is not None:
            header += f"  minus  {other.bundle.name}"
        elif state.minus_frame is not None:
            header += f"  minus  {bundle.frames[min(max(int(state.minus_frame), 0), len(bundle.frames) - 1)].label}"
        location_text = {"cell": "element mean (DAT integration points)", "face": "slave-face mean (DAT contact prints)"}.get(
            result.location, "nodal display (FRD)")
        header += f"\n{result.ascii_title} · {location_text}"
        if factor:
            header += f" · deformation x{factor:.4g}"
        # pyvista doubles add_text font sizes (points -> pixels); 9 reads as 18 px at scale 1
        plotter.add_text(ascii_text(header), position="upper_left", font_size=max(6, int(round(9 * fs))), color="#263238", font="arial")
        status_color = STATUS_COLORS.get(acceptance.get("kind", "diagnostic"), "#e65100")
        status_text = ascii_text(acceptance.get("short_label") or acceptance["label"])
        try:
            width = int(plotter.window_size[0])
        except Exception:
            width = 800
        # keep the status line inside the window: about 1.1 px per point per character in this font
        fit = max(5, int(0.95 * width / (1.1 * max(len(status_text), 1))))
        plotter.add_text(status_text, position="lower_edge", font_size=max(5, min(int(round(7 * fs)), fit)),
                         color=status_color, font="arial")
        plotter.add_axes(viewport=(0.86, 0.0, 1.0, 0.16))
        self.apply_camera(plotter, state, warped.bounds)
        info["acceptance"] = acceptance["label"]
        info["acceptance_kind"] = acceptance.get("kind")
        self.last_info = info
        return info

    @staticmethod
    def font_scale(plotter, state: SceneState) -> float:
        """3D text multiplier: 1 at a 600 px tall view, growing with the window height, times ``text_scale``."""
        try:
            width, height = plotter.window_size
        except Exception:
            width, height = 800, 600
        height = max(int(height or 600), 200)
        return float(state.text_scale or 1.0) * (0.5 + 0.5 * height / 600.0)

    def apply_camera(self, plotter, state: SceneState, bounds=None) -> None:
        if state.camera:
            plotter.camera_position = state.camera
        else:
            direction, up = VIEWS.get(state.view or "iso", VIEWS["iso"])
            b = np.asarray(bounds if bounds is not None else self.grid.bounds, dtype=float).reshape(3, 2)
            center = b.mean(axis=1)
            diagonal = float(np.linalg.norm(b[:, 1] - b[:, 0])) or 1.0
            plotter.camera_position = [tuple(center + _unit(direction) * diagonal * 2.5), tuple(center), up]
            plotter.reset_camera()
            plotter.camera.zoom(0.85)          # VTK fits the bounding sphere tightly; leave room for the titles
        if state.parallel:
            plotter.enable_parallel_projection()
        else:
            plotter.disable_parallel_projection()

    def visible_node_mask(self, state: SceneState) -> np.ndarray:
        rows = self.visible_rows_mask(state)
        if rows.all():
            return np.ones(self.bundle.mesh.n_nodes, dtype=bool)
        conn = self.bundle.mesh.connectivity[rows]
        mask = np.zeros(self.bundle.mesh.n_nodes, dtype=bool)
        mask[conn[conn >= 0]] = True
        return mask

    def probe_ray(self, state: SceneState, origin, direction) -> dict:
        """Nearest visible node to the ray origin + t·direction (screen-point picking without a VTK picker)."""
        factor = self.deformation_factor(state) if not state.mesh_quality else 0.0
        points = self._warped_points(state, factor)
        origin = np.asarray(origin, dtype=float)
        direction = _unit(direction)
        rel = points - origin
        t = rel @ direction
        perpendicular = np.linalg.norm(rel - np.outer(t, direction), axis=1)
        perpendicular[(t < 0) | ~self.visible_node_mask(state)] = np.inf
        row = int(np.argmin(perpendicular))
        if not np.isfinite(perpendicular[row]):
            raise ValueError("no visible node in front of the ray")
        out = self.probe(state, points[row])
        out["ray_distance_mm"] = float(perpendicular[row])
        return out

    def probe(self, state: SceneState, point) -> dict:
        """Nearest node to ``point`` (in the displayed, possibly deformed frame) with its values."""
        frame = self.bundle.frames[self.frame_index(state)]
        factor = self.deformation_factor(state) if not state.mesh_quality else 0.0
        points = self._warped_points(state, factor)
        row = int(np.argmin(np.linalg.norm(points - np.asarray(point, dtype=float), axis=1)))
        out = {"node_id": int(self.bundle.mesh.node_ids[row]), "row": row,
               "coordinates_mm": self.bundle.mesh.points[row].tolist(), "frame": frame.label}
        for key in fields_module.available(frame):
            try:
                out[key] = float(fields_module.compute(key, frame).values[row])
            except Exception:
                pass
        ids, names = self.bundle.cell_body_ids()
        touching = np.flatnonzero((self.bundle.mesh.connectivity == row).any(axis=1))
        if touching.size:
            out["bodies"] = sorted({names[i] for i in ids[touching] if i >= 0})
            out["elements"] = self.bundle.mesh.element_ids[touching][:8].tolist()
        return out


def screenshot(plotter, path: str | Path, *, transparent: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path), transparent_background=transparent)
    return path


def animation_factors(peak: float, frames: int = 24) -> list[float]:
    """Deformation factors of one sweep 0 → peak → 0 (sine eased), ``frames`` values."""
    frames = max(int(frames), 2)
    phase = np.linspace(0.0, np.pi, frames, endpoint=False)
    return [float(peak * np.sin(p)) for p in phase]


def animate(scene: Scene, plotter, state: SceneState, path: str | Path, *, frames: int = 24, fps: int = 10,
            other: Scene | None = None, log=None) -> dict:
    """Sweep the deformation factor 0 → current → 0 and write a GIF (Pillow) or a PNG sequence.

    ``path`` ending in ``.gif`` gets one animated file; any other suffix gets ``<stem>-000.png`` …
    Every frame is a full ``Scene.render`` so overlays, clips and labels stay consistent.
    """
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = scene.deformation_factor(state)
    camera = state.camera
    images = []
    written = []
    record = None
    for k, factor in enumerate(animation_factors(peak, frames)):
        frame_state = replace(state, deform=factor, camera=camera)
        record = scene.render(plotter, frame_state, other)
        if camera is None:          # keep the first frame's camera for the whole sweep
            camera = [list(map(float, p)) for p in plotter.camera_position]
        image = plotter.screenshot(None, return_img=True)
        if path.suffix.lower() == ".gif":
            images.append(Image.fromarray(np.asarray(image)))
        else:
            target = path.with_name(f"{path.stem}-{k:03d}{path.suffix or '.png'}")
            Image.fromarray(np.asarray(image)).save(target)
            written.append(str(target))
        if log:
            log(f"animation frame {k + 1}/{frames} factor {factor:.4g}")
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / max(fps, 1)), loop=0)
        written.append(str(path))
    return {"files": written, "frames": frames, "peak_factor": peak, "fps": fps, "last": record}


def offscreen_plotter(size=(1600, 1000)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    return pv.Plotter(off_screen=True, window_size=list(size))
