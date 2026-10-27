"""Read-only EMerge subprocess protocol. Run with the RF Python environment.

No solver or Simulation is constructed here. EMerge's own basis interpolation
evaluates its edge unknowns and curl-derived H; these are visualization samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np


def _verify_published(output, frequency_hz, shape, axes):
    """Verify an immutable entry; tolerate transient Windows sharing handles."""
    for attempt in range(12):
        try:
            with np.load(output, allow_pickle=False) as published:
                same_entry = (
                    np.isclose(float(published["frequency_hz"]), frequency_hz, rtol=0, atol=1)
                    and published["E"].shape == shape
                    and published["H"].shape == shape
                    and all(np.array_equal(published[key], axis) for key, axis in zip(("x", "y", "z"), axes))
                )
            if not same_entry:
                raise ValueError("Concurrent cache entry does not match this result")
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.1)


def conductor_surface_fields(field):
    """Air-side H and equivalent conductor sheet current on exterior FEM faces.

    The caller must establish that these are conductor walls, not ports or open
    boundaries. The outward normal points from the air domain into the metal;
    Js = H_air cross n_out = n_metal-to-air cross H_air (A/m).
    Each face uses its one adjacent element explicitly, avoiding boundary-search
    ambiguities and the visual sample grid. All values retain the saved phase.
    """
    mesh = field.mesh
    adjacent = np.asarray(mesh.tri_to_tet)
    boundary_ids = np.flatnonzero(np.sum(adjacent >= 0, axis=0) == 1)
    if not len(boundary_ids):
        raise ValueError("No exterior FEM faces found")
    tris = mesh.tris[:, boundary_ids].T
    xyz = mesh.nodes.T
    vertices = xyz[tris]
    centers = vertices.mean(axis=1)
    area_vectors = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
    twice_area = np.linalg.norm(area_vectors, axis=1)
    if np.any(twice_area <= 0):
        raise ValueError("Degenerate boundary triangle")
    normals = area_vectors/twice_area[:, None]
    parents = np.max(adjacent[:, boundary_ids], axis=0)
    tet_centers = xyz[mesh.tets[:, parents].T].mean(axis=1)
    flip = np.sum(normals*(tet_centers-centers), axis=1) > 0
    normals[flip] *= -1
    field.interpolate(*centers.T, mapping=parents, usenan=True)
    magnetic = np.column_stack((field.Hx, field.Hy, field.Hz))
    if not np.isfinite(magnetic).all():
        raise ValueError("Nonfinite native boundary field")
    current = np.cross(magnetic, normals)
    nodes, inverse = np.unique(tris, return_inverse=True)
    return dict(points=xyz[nodes], triangles=inverse.reshape(-1, 3),
                centers=centers, normals=normals, areas=twice_area/2,
                H=magnetic, Js=current, frequency_hz=float(np.real(field.freq)))


def publish_surface_cache(payload, request, source):
    output = Path(request["output"]).resolve()
    cache_root = Path(request["cache_root"]).resolve()
    source_root = source.parent.parent if source.parent.name.lower().endswith(".emresults") else source.parent
    if not output.is_relative_to(cache_root) or output.is_relative_to(source_root):
        raise ValueError("Surface cache must be outside the saved study")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + "." + uuid.uuid4().hex + ".pending.npz")
    try:
        np.savez_compressed(temporary, **payload)
        try:
            if os.name == "nt":
                temporary.rename(output)
            else:
                os.link(temporary, output)
        except (PermissionError, FileExistsError):
            if not output.is_file():
                raise
        for attempt in range(12):
            try:
                with np.load(output, allow_pickle=False) as saved:
                    if (saved["Js"].shape != payload["Js"].shape or
                        not np.isclose(saved["frequency_hz"], payload["frequency_hz"], rtol=0, atol=1)):
                        raise ValueError("Surface cache does not match selected result")
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(.1)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output)}


def serve(source: Path):
    from emerge._emerge.file import load_object
    from loguru import logger
    logger.remove()
    packed = load_object(str(source))
    fields = sorted(list(packed["simdata"].mw.field), key=lambda f: float(np.real(f.freq)))
    if not fields:
        raise ValueError("Native result contains no saved microwave fields")
    mapping = None
    coordinates = None
    mapped_mesh = None
    last_axes = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "close":
                return
            if command == "inspect":
                first = fields[0]
                nodes = first.mesh.nodes
                scalars = sorted(list(packed["simdata"].mw.scalar), key=lambda s: float(np.real(s.freq)))
                sdata = []
                if first._mode_field is None and first.port_modes:
                    ids = [p.smat_index for p in first.port_modes]
                    for s in scalars:
                        sdata.append([[[complex(s.S(i, j)).real, complex(s.S(i, j)).imag]
                                       for j in ids] for i in ids])
                result = {"kind": "eigenmode" if first._mode_field is not None else "driven",
                          "frequencies_hz": [float(np.real(f.freq)) for f in fields],
                          "imag_frequencies_hz": [float(np.imag(f.freq)) for f in fields],
                          "bounds_m": [[float(np.min(v)), float(np.max(v))] for v in nodes],
                          "tetrahedra": int(first.mesh.tets.shape[1]),
                          "port_ids": [p.smat_index for p in first.port_modes], "S": sdata}
            else:
                index = int(request["index"])
                if not 0 <= index < len(fields):
                    raise IndexError("Saved field index is outside the native result")
                field = fields[index]
                if field._mode_field is None:
                    port = request.get("port", 1)
                    if port not in field._fields:
                        raise ValueError(f"Port {port} is not in this saved result")
                    field.excite_port(port, 1.0 + 0.0j)
                field._silent = True
                if command == "surface_current":
                    if field._mode_field is None or not request.get("closed_conductor_walls"):
                        raise ValueError("Surface currents require verified closed conductor eigenmode walls")
                    result = publish_surface_cache(conductor_surface_fields(field), request, source)
                elif command == "sample":
                    axes = [np.asarray(a, dtype=float) for a in request["axes"]]
                    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in axes):
                        raise ValueError("Sampling axes must be finite one-dimensional coordinates")
                    same_axes = last_axes is not None and all(np.array_equal(a, b) for a, b in zip(axes, last_axes))
                    same_mesh = mapped_mesh is not None and (field.mesh is mapped_mesh or
                        (np.array_equal(field.mesh.nodes, mapped_mesh.nodes) and
                         np.array_equal(field.mesh.tets, mapped_mesh.tets)))
                    if not same_axes or not same_mesh:
                        coordinates = np.meshgrid(*axes, indexing="ij")
                        mapping = field.basis.interpolate_index(*(v.ravel() for v in coordinates), usenan=True)
                        last_axes, mapped_mesh = axes, field.mesh
                    field.interpolate(*coordinates, mapping=mapping, usenan=True)
                    electric = np.stack((field.Ex, field.Ey, field.Ez), axis=-1)
                    magnetic = np.stack((field.Hx, field.Hy, field.Hz), axis=-1)
                    output = Path(request["output"]).resolve()
                    # Defense in depth: all writes must be in the caller's separate cache.
                    cache_root = Path(request["cache_root"]).resolve()
                    source_root = source.parent.parent if source.parent.name.lower().endswith(".emresults") else source.parent
                    if not output.is_relative_to(cache_root) or output.is_relative_to(source_root):
                        raise ValueError("Worker output must be in the separate viewer cache")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    # Independent tabs can interpolate the same cache entry at once.
                    # Each owns its temporary file. First-writer publication is
                    # atomic and never replaces an entry another tab may be reading.
                    temporary = output.with_name(output.stem + "." + uuid.uuid4().hex + ".pending.npz")
                    try:
                        np.savez_compressed(temporary, E=electric, H=magnetic,
                                            x=axes[0], y=axes[1], z=axes[2],
                                            frequency_hz=float(np.real(field.freq)))
                        try:
                            if os.name == "nt":
                                # Windows rename fails if destination already exists.
                                temporary.rename(output)
                            else:
                                # POSIX rename replaces; a hard link instead gives
                                # atomic exclusive publication on the same volume.
                                os.link(temporary, output)
                        except (PermissionError, FileExistsError):
                            # Another process won publication. No writer mutates
                            # this destination; all participants only read it.
                            if not output.is_file():
                                raise
                            _verify_published(output, float(np.real(field.freq)), electric.shape, axes)
                    finally:
                        temporary.unlink(missing_ok=True)
                    result = {"path": str(output), "frequency_hz": float(np.real(field.freq))}
                elif command == "probe":
                    point = np.asarray(request["point_m"], dtype=float)
                    if point.shape != (3,) or not np.isfinite(point).all():
                        raise ValueError("Probe coordinates must be three finite meter values")
                    field.interpolate(*(float(v) for v in point), usenan=True)
                    e = np.array([field.Ex[0], field.Ey[0], field.Ez[0]])
                    h = np.array([field.Hx[0], field.Hy[0], field.Hz[0]])
                    valid = bool(np.isfinite(e).all() and np.isfinite(h).all())
                    result = {"point_m": point.tolist(), "valid": valid,
                              "E": [[float(v.real), float(v.imag)] for v in e] if valid else None,
                              "H": [[float(v.real), float(v.imag)] for v in h] if valid else None,
                              "frequency_hz": float(np.real(field.freq))}
                else:
                    raise ValueError(f"Unknown worker command: {command}")
            print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    try:
        serve(args.source.resolve())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"Native result load failed: {exc}"}), flush=True)
        raise SystemExit(1)
