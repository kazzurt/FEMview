"""Mesh statistics: counts per body and set, element quality, edge lengths, histograms.

Quality uses ``vtkMeshQuality`` through pyvista on the corner-node (linear) version of every solid
element, because the filter returns -1 for quadratic cells in VTK 9.6. Scaled Jacobian is VTK's
definition (1 for a unit cube; sqrt(2)/2 ≈ 0.707 for an equilateral tetrahedron). The mesher's own
``mesh.json`` numbers (Codex runs) are reported verbatim beside the recomputed ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frd import FRD_TO_VTK, SOLID_TYPES, FrdMesh

LINEAR_VTK = {1: (12, 8), 2: (13, 6), 3: (10, 4), 4: (12, 8), 5: (13, 6), 6: (10, 4)}
EDGES = {
    10: ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
    12: ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)),
    13: ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)),
}


@dataclass
class QualityArrays:
    scaled_jacobian: np.ndarray
    aspect_ratio: np.ndarray
    volume: np.ndarray
    min_edge: np.ndarray
    max_edge: np.ndarray
    corner_vtk_types: np.ndarray
    notes: list[str] = field(default_factory=list)


def corner_grid(mesh: FrdMesh):
    """pyvista UnstructuredGrid of the corner nodes of every solid element (linear cell types)."""
    import pyvista as pv
    cells = []
    types = []
    for row in range(mesh.n_elements):
        code = int(mesh.element_types[row])
        if code not in LINEAR_VTK:
            continue
        vtk_type, count = LINEAR_VTK[code]
        cells.append(np.concatenate([[count], mesh.connectivity[row, :count]]))
        types.append(vtk_type)
    if not cells:
        return None, np.zeros(0, dtype=np.int64)
    rows = np.array([r for r in range(mesh.n_elements) if int(mesh.element_types[r]) in LINEAR_VTK])
    grid = pv.UnstructuredGrid(np.concatenate(cells), np.array(types, dtype=np.uint8), mesh.points)
    return grid, rows


def edge_lengths(mesh: FrdMesh) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) corner-edge length per element (NaN for non-solid elements)."""
    minimum = np.full(mesh.n_elements, np.nan)
    maximum = np.full(mesh.n_elements, np.nan)
    for code in np.unique(mesh.element_types):
        code = int(code)
        if code not in LINEAR_VTK:
            continue
        vtk_type, count = LINEAR_VTK[code]
        rows = np.flatnonzero(mesh.element_types == code)
        conn = mesh.connectivity[rows, :count]
        pts = mesh.points[conn]                          # (k, count, 3)
        edges = np.array(EDGES[vtk_type])
        lengths = np.linalg.norm(pts[:, edges[:, 0]] - pts[:, edges[:, 1]], axis=2)
        minimum[rows] = lengths.min(axis=1)
        maximum[rows] = lengths.max(axis=1)
    return minimum, maximum


def quality(mesh: FrdMesh) -> QualityArrays:
    n = mesh.n_elements
    out = QualityArrays(np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                        np.zeros(n, dtype=np.uint8))
    grid, rows = corner_grid(mesh)
    if grid is None:
        out.notes.append("no solid elements; quality not computed")
        return out
    out.corner_vtk_types[rows] = grid.celltypes
    for measure, target in (("scaled_jacobian", out.scaled_jacobian), ("aspect_ratio", out.aspect_ratio),
                            ("volume", out.volume)):
        try:
            values = np.asarray(grid.cell_quality(measure).cell_data[measure], dtype=float)
            values[values == -1.0] = np.nan      # VTK's null value for unsupported cell/measure pairs
            target[rows] = values
        except Exception as exc:                # a missing measure must not break the statistics
            out.notes.append(f"{measure}: {exc}")
    out.min_edge, out.max_edge = edge_lengths(mesh)
    missing = ~np.isfinite(out.aspect_ratio) & np.isfinite(out.min_edge) & (out.min_edge > 0)
    if missing.any():
        # vtkMeshQuality has no aspect ratio for hexahedra/wedges: use the corner-edge ratio there.
        out.aspect_ratio[missing] = out.max_edge[missing] / out.min_edge[missing]
        out.notes.append(f"aspect ratio of {int(missing.sum())} non-tetrahedral elements is the max/min corner-edge ratio")
    quadratic = any(int(t) in (4, 5, 6) for t in np.unique(mesh.element_types))
    if quadratic:
        out.notes.append("quality measured on the corner nodes of quadratic elements (vtkMeshQuality is linear-only); "
                         "a negative corner scaled Jacobian does not by itself invalidate the quadratic element")
    return out


def _summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {"count": int(finite.size), "min": float(finite.min()), "p1": float(np.percentile(finite, 1)),
            "mean": float(finite.mean()), "median": float(np.median(finite)), "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max())}


def _histogram(values: np.ndarray, bins: int = 24, limits: tuple[float, float] | None = None) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(finite, bins=bins, range=limits)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def statistics(bundle, quality_arrays: QualityArrays | None = None) -> dict:
    """All mesh statistics of a bundle as a JSON-able dict (tables, summaries, histograms)."""
    mesh = bundle.mesh
    q = quality_arrays or quality(mesh)
    bodies = []
    for name, rows in bundle.bodies().items():
        nodes = np.unique(mesh.connectivity[rows])
        nodes = nodes[nodes >= 0]
        elset = bundle.body_elset(name)
        deck = bundle.deck
        bodies.append({"body": name, "elset": elset, "elements": int(rows.size), "nodes": int(nodes.size),
                       "material": deck.body_material(elset) if deck else None,
                       "volume_mm3": float(np.nansum(q.volume[rows])),
                       "min_scaled_jacobian": float(np.nanmin(q.scaled_jacobian[rows])) if np.isfinite(q.scaled_jacobian[rows]).any() else None,
                       "max_aspect_ratio": float(np.nanmax(q.aspect_ratio[rows])) if np.isfinite(q.aspect_ratio[rows]).any() else None,
                       "min_edge_mm": float(np.nanmin(q.min_edge[rows])) if np.isfinite(q.min_edge[rows]).any() else None,
                       "max_edge_mm": float(np.nanmax(q.max_edge[rows])) if np.isfinite(q.max_edge[rows]).any() else None})
    deck = bundle.deck
    sets = {"node_sets": {}, "element_sets": {}, "surfaces": {}}
    if deck is not None:
        sets["node_sets"] = {name: int(deck.resolve_nodes(name).size) for name in deck.nsets}
        sets["element_sets"] = {name: int(deck.resolve_elements(name).size) for name in deck.elsets}
        sets["surfaces"] = {name: bundle.surface_face_count(name) for name in deck.surfaces}
    counts = {"nodes": mesh.n_nodes, "elements": mesh.n_elements, "element_types": mesh.type_counts(),
              "solid_elements": int(np.isin(mesh.element_types, list(SOLID_TYPES)).sum()),
              "unused_nodes": int(mesh.n_nodes - np.unique(mesh.connectivity[mesh.connectivity >= 0]).size)}
    lo, hi = mesh.bounds
    quality_summary = {"scaled_jacobian": _summary(q.scaled_jacobian), "aspect_ratio": _summary(q.aspect_ratio),
                       "min_edge_mm": _summary(q.min_edge), "max_edge_mm": _summary(q.max_edge),
                       "volume_mm3": _summary(q.volume), "notes": list(q.notes),
                       "inverted_elements": int(np.nansum(q.scaled_jacobian < 0)) if np.isfinite(q.scaled_jacobian).any() else 0}
    histograms = {"scaled_jacobian": _histogram(q.scaled_jacobian, limits=(min(0.0, float(np.nanmin(q.scaled_jacobian))) if np.isfinite(q.scaled_jacobian).any() else 0.0, 1.0)),
                  "aspect_ratio": _histogram(q.aspect_ratio), "min_edge_mm": _histogram(q.min_edge),
                  "max_edge_mm": _histogram(q.max_edge)}
    regions = bundle.regions()
    region_counts = {}
    if regions:
        codes, names = regions
        region_counts = {name: int(c) for name, c in zip(names, np.bincount(codes, minlength=len(names)))}
    mesher = {}
    mesh_json = bundle.json("mesh_json")
    if mesh_json:
        mesher = {k: mesh_json[k] for k in ("level", "nodes", "tet10", "minDetJac_mm3", "sizes_mm", "elapsed_s",
                                             "gmsh_version", "surface_counts", "mesh_sha256") if k in mesh_json}
    return {"counts": counts, "bounds_mm": {"min": lo.tolist(), "max": hi.tolist(), "diagonal": mesh.diagonal},
            "bodies": bodies, "sets": sets, "regions": region_counts, "quality": quality_summary,
            "histograms": histograms, "mesher": mesher, "mesh_hash": mesh.mesh_hash()}


def histogram_png(histogram: dict, title: str, path, *, color="#2f6fb3") -> None:
    """Write one histogram as a PNG with matplotlib (Agg backend, nothing on screen)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    edges = np.asarray(histogram.get("edges", []))
    counts = np.asarray(histogram.get("counts", []))
    figure, axis = plt.subplots(figsize=(4.2, 2.4), dpi=110)
    if edges.size > 1:
        axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color, edgecolor="white", linewidth=0.5)
    axis.set_title(title, fontsize=9)
    axis.tick_params(labelsize=7)
    axis.set_ylabel("elements", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
