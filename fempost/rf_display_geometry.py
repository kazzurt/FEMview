"""Display-only cylinder sections; original FEM fields and probes are untouched."""
from __future__ import annotations
import numpy as np


class VisualFieldSampler:
    """Resample complex phasors on cosmetic geometry, never analysis data.

    The last grid cells near the wall can contain NaNs. For this display only,
    those values use their nearest valid sample; coordinates outside the saved
    grid are clamped to its extent. Interpolation is performed on complex E/H
    before taking the phase-dependent norm, not on precomputed magnitudes.
    """
    def __init__(self, frame):
        from scipy.interpolate import RegularGridInterpolator
        from scipy.spatial import cKDTree
        self.axes = frame.axes
        valid = np.isfinite(frame.E).all(axis=-1) & np.isfinite(frame.H).all(axis=-1)
        if not np.any(valid):
            raise ValueError("No valid visualization samples")
        values = np.concatenate((frame.E, frame.H), axis=-1).copy()
        if not np.all(valid):
            xyz = np.stack(np.meshgrid(*frame.axes, indexing="ij"), axis=-1)
            tree = cKDTree(xyz[valid])
            nearest = tree.query(xyz[~valid])[1]
            values[~valid] = values[valid][nearest]
        self.interpolate = RegularGridInterpolator(frame.axes, values, bounds_error=True)
        self.validity = RegularGridInterpolator(frame.axes, valid.astype(float),
                                               bounds_error=False, fill_value=0.)

    def valid_at(self, points):
        """Conservative clipping when no explicit cavity boundary is available."""
        return self.validity(points) >= 1.-1e-9

    def __call__(self, points):
        clipped = np.array(points, dtype=float, copy=True)
        for axis, coordinates in enumerate(self.axes):
            clipped[:, axis] = np.clip(clipped[:, axis], coordinates[0], coordinates[-1])
        values = self.interpolate(clipped)
        return {"E": values[:, :3], "H": values[:, 3:]}


def cartesian_plane_points(bounds, axis, origin, density=15, cylinder=None):
    """Uniform square lattice, centered on the bounds and clipped to the cavity.

    Density is the number of sites across the longest in-plane bounding span.
    The shorter span uses the same physical spacing. Neither field values nor
    display-mesh vertex order influence seed locations. Sliding a plane keeps
    its in-plane lattice fixed; only its intersection with the cavity changes.
    """
    bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
    fixed = "xyz".index(axis)
    free = [i for i in range(3) if i != fixed]
    spans = np.diff(bounds, axis=1).ravel()
    density = int(np.clip(density, 5, 41))
    spacing = max(spans[free])/(density-1)
    if spacing <= 0 or not bounds[fixed, 0] <= origin[fixed] <= bounds[fixed, 1]:
        return np.empty((0, 3))
    coordinates = []
    for i in free:
        count = int(np.floor(spans[i]/spacing+1e-9))+1
        coordinates.append(bounds[i].mean() + (np.arange(count)-(count-1)/2)*spacing)
    grid = np.meshgrid(*coordinates, indexing="ij")
    points = np.empty((grid[0].size, 3))
    points[:, fixed] = origin[fixed]
    for i, values in zip(free, grid):
        points[:, i] = values.ravel()
    if cylinder is not None:
        radius, zmin, zmax = cylinder["radius"], cylinder["zmin"], cylinder["zmax"]
        tolerance = max(radius, zmax-zmin)*1e-10
        inside = ((np.hypot(points[:, 0], points[:, 1]) <= radius+tolerance)
                  & (points[:, 2] >= zmin-tolerance) & (points[:, 2] <= zmax+tolerance))
        points = points[inside]
    return points


def cylinder_section(geometry, axis, origin):
    """A smooth circular XY disk or exact rectangular axial cylinder section."""
    import pyvista as pv
    radius, zmin, zmax = geometry["radius"], geometry["zmin"], geometry["zmax"]
    if axis == "z":
        nr, nt = 24, 128
        theta = np.arange(nt)*2*np.pi/nt
        rings = np.arange(1, nr+1)*radius/nr
        points = np.zeros((1+nr*nt, 3))
        points[:, 2] = origin[2]
        points[1:, 0] = (rings[:, None]*np.cos(theta)).ravel()
        points[1:, 1] = (rings[:, None]*np.sin(theta)).ravel()
        faces = []
        for j in range(nt):
            nxt = (j+1) % nt
            faces.append((3, 0, 1+j, 1+nxt))
        for ring in range(nr-1):
            first, second = 1+ring*nt, 1+(ring+1)*nt
            for j in range(nt):
                nxt = (j+1) % nt
                faces.extend(((3, first+j, second+j, second+nxt),
                              (3, first+j, second+nxt, first+nxt)))
        return pv.PolyData(points, np.asarray(faces).ravel())
    fixed = "xyz".index(axis)
    span = np.sqrt(max(0., radius*radius-origin[fixed]**2))
    if span <= 0:
        return pv.PolyData()
    cross, vertical = np.meshgrid(np.linspace(-span, span, 49), np.linspace(zmin, zmax, 49), indexing="ij")
    points = np.zeros((cross.size, 3))
    points[:, fixed] = origin[fixed]
    points[:, 1-fixed] = cross.ravel()
    points[:, 2] = vertical.ravel()
    faces = []
    for i in range(48):
        for j in range(48):
            a = i*49+j
            faces.extend(((3, a, a+49, a+50), (3, a, a+50, a+1)))
    return pv.PolyData(points, np.asarray(faces).ravel())
