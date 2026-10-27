"""FRD results as numpy arrays, cached by the FRD file's SHA-256.

The text parser is ``structural.backends.calculix.parse_frd`` (imported read-only). This module
converts its dict-of-lists output to arrays, recovers each result block's time from the ``100CL``
header records (the structural reader returns the node count as the time for the glued
fixed-width form ``100CL  1012.00000E-001`` written by the pinned MinGW build), groups blocks into
frames and caches everything as one ``.npz`` under ``fempost-cache``. FRD STRESS is extrapolated
nodal display data; integration-point stress lives in the DAT (see ``dat.py``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("FEMPOST_CACHE", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "FEMview" / "fempost_cache")))
CACHE_VERSION = 3

# FRD element type code -> (VTK cell type, node count). For tetrahedra the FRD node order equals
# VTK's (corners, then edges 01 12 20 03 13 23; the structural package relies on this). The FRD
# writer lists 20-node bricks as corners, bottom mid-edges, VERTICAL mid-edges, top mid-edges
# (verified on the explicit-bar fixture against the deck), whereas VTK and the CalculiX input deck
# want corners, bottom, top, vertical: FRD_PERMUTATION swaps them. The 15-node wedge is assumed to
# follow the same FRD convention (no fixture available to verify).
FRD_TO_VTK = {1: (12, 8), 2: (13, 6), 3: (10, 4), 4: (25, 20), 5: (26, 15), 6: (24, 10),
              7: (5, 3), 8: (22, 6), 9: (9, 4), 10: (23, 8), 11: (3, 2), 12: (21, 3)}
FRD_PERMUTATION = {4: list(range(12)) + [16, 17, 18, 19, 12, 13, 14, 15],
                   5: list(range(9)) + [12, 13, 14, 9, 10, 11]}
FRD_TYPE_NAMES = {1: "C3D8 hexahedron", 2: "C3D6 wedge", 3: "C3D4 tetrahedron", 4: "C3D20 quadratic hexahedron",
                  5: "C3D15 quadratic wedge", 6: "C3D10 quadratic tetrahedron", 7: "S3 triangle",
                  8: "S6 quadratic triangle", 9: "S4 quadrilateral", 10: "S8 quadratic quadrilateral",
                  11: "B31 beam", 12: "B32 quadratic beam"}
SOLID_TYPES = {1, 2, 3, 4, 5, 6}
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?")


def _structural():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from structural.backends import calculix  # read-only import of the pipeline's readers
    return calculix


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _IdIndex:
    """Vectorised id -> row lookup; -1 for ids that are absent."""

    def __init__(self, ids: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int64)
        self._table = None
        self._dict = None
        if self.ids.size and 0 <= self.ids.min() and self.ids.max() < 50_000_000:
            self._table = np.full(int(self.ids.max()) + 1, -1, dtype=np.int64)
            self._table[self.ids] = np.arange(len(self.ids))
        else:
            self._dict = {int(i): k for k, i in enumerate(self.ids)}

    def __call__(self, ids) -> np.ndarray:
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        if self._table is not None:
            out = np.full(ids.shape, -1, dtype=np.int64)
            ok = (ids >= 0) & (ids < len(self._table))
            out[ok] = self._table[ids[ok]]
            return out
        return np.array([self._dict.get(int(i), -1) for i in ids], dtype=np.int64)


@dataclass
class FrdMesh:
    node_ids: np.ndarray          # (n,) solver node ids in FRD order
    points: np.ndarray            # (n, 3) mm
    element_ids: np.ndarray       # (m,) solver element ids in FRD order
    element_types: np.ndarray     # (m,) FRD type codes
    element_groups: np.ndarray    # (m,) FRD group column
    element_materials: np.ndarray # (m,) FRD material column (1-based index into material_names)
    connectivity: np.ndarray      # (m, k) zero-based node rows, -1 padded
    vtk_types: np.ndarray         # (m,) VTK cell type codes
    material_names: list[str] = field(default_factory=list)
    heading: str = ""

    def __post_init__(self):
        self._node_index = _IdIndex(self.node_ids)
        self._element_index = _IdIndex(self.element_ids)

    def node_index(self, ids) -> np.ndarray:
        return self._node_index(ids)

    def element_index(self, ids) -> np.ndarray:
        return self._element_index(ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_elements(self) -> int:
        return len(self.element_ids)

    @property
    def nodes_per_element(self) -> np.ndarray:
        return np.array([FRD_TO_VTK.get(int(t), (0, 0))[1] for t in self.element_types], dtype=np.int64)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.points.min(0), self.points.max(0)])

    @property
    def diagonal(self) -> float:
        lo, hi = self.bounds
        return float(np.linalg.norm(hi - lo))

    def mesh_hash(self) -> str:
        """Geometry/topology hash shared by runs on the same solver mesh (independent of results)."""
        digest = hashlib.sha256()
        for array in (self.node_ids, self.points, self.element_ids, self.element_types, self.connectivity):
            array = np.ascontiguousarray(array)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def type_counts(self) -> dict[str, int]:
        codes, counts = np.unique(self.element_types, return_counts=True)
        return {FRD_TYPE_NAMES.get(int(c), f"FRD type {int(c)}"): int(k) for c, k in zip(codes, counts)}


@dataclass
class Frame:
    index: int
    step: int
    increment: int
    time: float
    fields: dict[str, np.ndarray]           # name -> (n_nodes, ncomp), NaN where a node has no value
    components: dict[str, list[str]]        # name -> component labels as written in the FRD
    derived: dict = field(default_factory=dict, compare=False, repr=False)   # memoised derived arrays

    @property
    def label(self) -> str:
        return f"Step {self.step} · inc {self.increment} · t={self.time:g}"


def _scan_block_times(path: Path) -> list[float]:
    """Times of every ``100CL`` nodal-result header, in file order."""
    times: list[float] = []
    with Path(path).open("rb") as stream:
        for raw in stream:
            if raw.startswith(b"  100C"):
                text = raw.decode("ascii", "replace")
                token = text[12:24].strip()
                try:
                    times.append(float(token))
                    continue
                except ValueError:
                    pass
                found = _FLOAT.findall(text[7:40])
                times.append(float(found[1]) if len(found) > 1 else float("nan"))
    return times


def _header_info(path: Path) -> tuple[list[str], str]:
    materials: list[str] = []
    heading = ""
    with Path(path).open("rb") as stream:
        for k, raw in enumerate(stream):
            text = raw.decode("ascii", "replace").rstrip()
            if text.startswith("    1UMAT"):
                materials.append(text[14:].strip())
            elif text.startswith("    1U") and k == 1:
                heading = text[6:].strip()
            elif text.startswith("    2C") or k > 200:
                break
    return materials, heading


def _frd_to_arrays(frd: dict, path: Path) -> tuple[FrdMesh, list[Frame]]:
    node_ids = np.fromiter(frd["nodes"].keys(), dtype=np.int64, count=len(frd["nodes"]))
    points = np.array(list(frd["nodes"].values()), dtype=np.float64).reshape(-1, 3)
    element_ids = np.fromiter(frd["elements"].keys(), dtype=np.int64, count=len(frd["elements"]))
    types = np.array([e["type"] for e in frd["elements"].values()], dtype=np.int64)
    meta = np.array([(list(e["metadata"]) + [0, 0])[:2] for e in frd["elements"].values()], dtype=np.int64).reshape(-1, 2)
    width = max([len(e["nodes"]) for e in frd["elements"].values()] + [1])
    lookup = _IdIndex(node_ids)
    connectivity = np.full((len(element_ids), width), -1, dtype=np.int64)
    for row, element in enumerate(frd["elements"].values()):
        nodes = element["nodes"]
        order = FRD_PERMUTATION.get(int(element["type"]))
        if order is not None and len(nodes) == len(order):
            nodes = [nodes[k] for k in order]
        connectivity[row, :len(nodes)] = lookup(nodes)
    vtk_types = np.array([FRD_TO_VTK.get(int(t), (0, 0))[0] for t in types], dtype=np.uint8)
    material_names, heading = _header_info(path)
    mesh = FrdMesh(node_ids, points, element_ids, types, meta[:, 0], meta[:, 1], connectivity, vtk_types,
                   material_names, heading)
    times = _scan_block_times(path)
    if len(times) != len(frd["fields"]):
        times = [float(f["time"]) if f["time"] is not None else float("nan") for f in frd["fields"]]
    # The FRD carries no step number (the 1PSTEP/100CL integers are global increment counters);
    # frames are grouped by time here and the bundle assigns steps from the STA table or the deck.
    frames: list[Frame] = []
    current_key = None
    for block, time_value in zip(frd["fields"], times):
        key = round(time_value, 9)
        if key != current_key:
            frames.append(Frame(len(frames), 1, len(frames) + 1, float(time_value), {}, {}))
            current_key = key
        frame = frames[-1]
        values = block["values"]
        if not values:
            continue
        ids = np.fromiter(values.keys(), dtype=np.int64, count=len(values))
        data = np.array(list(values.values()), dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        rows = lookup(ids)
        ok = rows >= 0
        array = np.full((len(node_ids), data.shape[1]), np.nan)
        array[rows[ok]] = data[ok]
        name = block["name"]
        if name in frame.fields:  # the same field twice in one frame: keep both, suffix the repeat
            name = f"{name}_{sum(1 for k in frame.fields if k.startswith(block['name']))}"
        frame.fields[name] = array
        frame.components[name] = list(block["components"])[:data.shape[1]]
    return mesh, frames


def cache_path(digest: str) -> Path:
    return CACHE_DIR / f"{digest}.frd.npz"


def _save_cache(target: Path, mesh: FrdMesh, frames: list[Frame], source: Path, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"node_ids": mesh.node_ids, "points": mesh.points, "element_ids": mesh.element_ids,
               "element_types": mesh.element_types, "element_groups": mesh.element_groups,
               "element_materials": mesh.element_materials, "connectivity": mesh.connectivity,
               "vtk_types": mesh.vtk_types}
    meta = {"cache_version": CACHE_VERSION, "source": str(source), "sha256": digest,
            "material_names": mesh.material_names, "heading": mesh.heading,
            "frames": [{"index": f.index, "step": f.step, "increment": f.increment, "time": f.time,
                        "fields": list(f.fields), "components": f.components} for f in frames]}
    for frame in frames:
        for name, array in frame.fields.items():
            payload[f"F{frame.index}__{name}"] = array
    payload["meta_json"] = np.array(json.dumps(meta))
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    os.replace(temporary, target)


def _load_cache(target: Path, digest: str) -> tuple[FrdMesh, list[Frame]] | None:
    try:
        with np.load(target, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            if meta.get("cache_version") != CACHE_VERSION or meta.get("sha256") != digest:
                return None
            mesh = FrdMesh(data["node_ids"], data["points"], data["element_ids"], data["element_types"],
                           data["element_groups"], data["element_materials"], data["connectivity"], data["vtk_types"],
                           list(meta.get("material_names", [])), meta.get("heading", ""))
            frames = []
            for entry in meta["frames"]:
                fields = {name: data[f"F{entry['index']}__{name}"] for name in entry["fields"]}
                frames.append(Frame(entry["index"], entry["step"], entry["increment"], entry["time"], fields,
                                    entry["components"]))
            return mesh, frames
    except Exception:
        return None


def load_frd(path: str | Path, *, use_cache: bool = True, log=None) -> tuple[FrdMesh, list[Frame], dict]:
    """Return (mesh, frames, provenance). The FRD is never modified; the cache is keyed by its SHA-256."""
    path = Path(path)
    started = time.perf_counter()
    digest = sha256_file(path)
    provenance = {"frd": str(path), "sha256": digest, "size_bytes": path.stat().st_size, "cache_hit": False}
    target = cache_path(digest)
    if use_cache and target.exists():
        loaded = _load_cache(target, digest)
        if loaded is not None:
            provenance.update(cache_hit=True, cache=str(target), elapsed_s=time.perf_counter() - started)
            if log:
                log(f"FRD cache hit {target.name} ({provenance['elapsed_s']:.2f} s)")
            return loaded[0], loaded[1], provenance
    if log:
        log(f"Parsing {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    frd = _structural().parse_frd(path)
    mesh, frames = _frd_to_arrays(frd, path)
    del frd
    if use_cache:
        try:
            _save_cache(target, mesh, frames, path, digest)
            provenance["cache"] = str(target)
        except OSError as exc:
            provenance["cache_error"] = str(exc)
    provenance["elapsed_s"] = time.perf_counter() - started
    if log:
        log(f"Parsed {mesh.n_nodes} nodes, {mesh.n_elements} elements, {len(frames)} frames in "
            f"{provenance['elapsed_s']:.1f} s")
    return mesh, frames, provenance
