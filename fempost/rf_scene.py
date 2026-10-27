"""RF rendering on explicitly labelled SI visualization samples, never nodal FEM fields.

Complex peak phasors use Re(F exp(+j phase)); the eigenmode gauge rotates E and H
together. The only power field is 0.5 Re(E cross H*), with no display normalization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import numpy as np

FIELD_COLORMAPS = {"E": "magma", "H": "viridis", "Js": "copper", "Savg": "cividis"}


@dataclass
class RFViewState:
    field: str = "E"
    component: str = "magnitude"
    representation: str = "real"
    phase_deg: float = 0.0
    display: str = "slices"
    axis: str = "z"
    position: float = 0.5
    invert: bool = False
    vectors: bool = False
    vector_scale: float = 1.0
    vector_density: int = 15
    cmap: str = "auto"
    range_mode: str = "full"
    clim: list | None = None
    log_scale: bool = False
    opacity: float = 1.0
    mesh: bool = False
    camera: list | None = None
    parallel_scale: float | None = None
    view_angle: float | None = None
    parallel: bool = True
    smooth_boundary: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def average_poynting(E, H):
    return 0.5 * np.real(np.cross(E, np.conjugate(H)))


def eigen_gauge(E):
    """A reproducible common phase, not independent E/H normalization."""
    values = np.asarray(E).reshape(-1)
    good = np.isfinite(values)
    if not np.any(good):
        return 1.0 + 0j
    reference = values[np.flatnonzero(good)[np.argmax(np.abs(values[good]))]]
    return np.exp(-1j * np.angle(reference)) if reference != 0 else 1.0 + 0j


def eigenmode_power_note(frame):
    loss_model = getattr(frame, "loss_model", "unknown")
    if loss_model == "lossy":
        return "Lossy decaying eigenmode: arbitrary amplitude; power flow is not a driven steady-state result."
    if loss_model == "lossless":
        return "Lossless standing eigenmode: mean power should vanish; displayed residuals are numerical, not transported power."
    return "Eigenmode amplitude is arbitrary; mean power needs a validated loss model and may include numerical residuals."


def field_values(frame, state, samples=None):
    """Return scalars, phase-snapshot vectors, phase-stable scale data and labels."""
    if state.field not in {"E", "H", "Js", "Savg"}:
        raise ValueError("Choose E, H, Js or Savg")
    if state.component not in {"magnitude", "x", "y", "z"}:
        raise ValueError("Choose magnitude, x, y or z")
    if state.representation not in {"magnitude", "real", "imag", "phase"}:
        raise ValueError("Unknown complex-field representation")
    eigen = getattr(frame, "kind", "driven") == "eigenmode"
    phase = np.exp(1j * np.deg2rad(state.phase_deg))
    gauge = eigen_gauge(frame.E) if eigen else 1.0
    component = None if state.component == "magnitude" else "xyz".index(state.component)
    unit = {"E": "V/m", "H": "A/m", "Js": "A/m", "Savg": "W/m^2"}[state.field]
    suffix = " (arbitrary amplitude)" if eigen else ""
    warnings = []
    if state.field == "Savg":
        vector = average_poynting(samples["E"], samples["H"]) if samples is not None else average_poynting(frame.E, frame.H)
        values = np.linalg.norm(vector, axis=-1) if component is None else vector[..., component]
        scale = values
        signed = component is not None
        label = "Mean Poynting " + ("magnitude" if component is None else state.component)
        if eigen:
            warnings.append(eigenmode_power_note(frame))
    else:
        if state.field == "Js":
            surface = getattr(frame, "surface_current", None)
            if surface is None:
                raise ValueError("Surface currents are unavailable: verified conductor boundary fields are required")
            phasor = surface["Js"] * gauge
        else:
            phasor = (samples[state.field] if samples is not None else getattr(frame, state.field)) * gauge
        vector = np.imag(phasor * phase) if state.representation == "imag" else np.real(phasor * phase)
        amp = np.linalg.norm(phasor, axis=-1) if component is None else np.abs(phasor[..., component])
        value = phasor if component is None else phasor[..., component]
        signed = component is not None and state.representation in {"real", "imag"}
        if state.representation == "magnitude":
            values = amp
        elif state.representation == "phase":
            if component is None:
                raise ValueError("Vector magnitude has no unique phase. Choose x, y or z to display phase.")
            values = np.rad2deg(np.angle(value * phase))
            # Phase of numerical zero has no physical meaning.
            peak = np.nanmax(amp) if np.any(np.isfinite(amp)) else 0
            values = np.where(amp > max(peak * 1e-8, np.finfo(float).tiny), values, np.nan)
            unit, suffix = "deg", ""
        else:
            snapshot = np.real(value * phase) if state.representation == "real" else np.imag(value * phase)
            values = np.linalg.norm(snapshot, axis=-1) if component is None else snapshot
        scale = amp  # fixed across a phase sweep; no pumping of colour/arrow lengths
        representation_name = {"real": "instantaneous", "imag": "imaginary quadrature", "magnitude": "phasor amplitude", "phase": "phase"}[state.representation]
        label = f"{state.field}{'' if component is None else state.component} {representation_name}"
        if component is None and state.representation in {"real", "imag"}:
            label += " vector magnitude"
    return dict(values=values, vectors=vector, scale=scale, signed=signed,
                label=f"{label} [{unit}]{suffix}", unit=unit, warnings=warnings)


def colour_limits(result, state):
    values = result["values"]
    finite = values[np.isfinite(values)]
    true = [float(finite.min()), float(finite.max())] if finite.size else [0., 0.]
    warnings = list(result["warnings"])
    is_phase = state.representation == "phase" and state.field != "Savg"
    if state.range_mode == "manual" and state.clim is not None:
        lo, hi = map(float, state.clim)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("Colour maximum must exceed minimum, and both must be finite")
    elif is_phase:
        lo, hi = -180., 180.
    else:
        scale = result["scale"]
        scale = scale[np.isfinite(scale)]
        if not scale.size:
            lo, hi = 0., 1.
        elif result["signed"]:
            peak = np.percentile(np.abs(scale), 99.5) if state.range_mode == "robust" else np.max(np.abs(scale))
            lo, hi = -float(peak), float(peak)
        else:
            lo, hi = (np.percentile(scale, [0.5, 99.5]) if state.range_mode == "robust"
                      else [0., float(np.max(scale))])
    if hi <= lo:
        hi = lo + max(abs(lo) * .01, 1e-15)
    log = bool(state.log_scale)
    if log and (result["signed"] or is_phase):
        log = False
        warnings.append("Log scaling is disabled for signed values and phase.")
    elif log:
        envelope = np.asarray(result["scale"])
        positive = envelope[np.isfinite(envelope) & (envelope > 0)]
        if not positive.size or hi <= 0:
            log = False
            warnings.append("Log scaling requires positive values.")
        else:
            lo = max(float(lo), float(positive.min()), float(hi) * 1e-8)
            if hi <= lo:
                hi = lo * 1.01
    cmap = state.cmap
    if cmap in {"auto", "Auto", ""}:
        cmap = "twilight_shifted" if is_phase else ("coolwarm" if result["signed"] else FIELD_COLORMAPS[state.field])
    return [float(lo), float(hi)], true, log, cmap, warnings


class RFScene:
    def __init__(self, plotter):
        self.plotter = plotter
        self.frame = None
        self.state = None
        self.info = {}
        self._grid = None
        self._mesh_cache = {}
        self._initialized = False
        self._visual_frame = None
        self._visual_sampler = None
        self._visual_sections = {}
        self._visual_arrows = {}

    def _grid_for(self, frame):
        import pyvista as pv
        if frame is not self.frame or self._grid is None:
            self._grid = pv.StructuredGrid(*np.meshgrid(*frame.axes, indexing="ij"))
        return self._grid

    def _sampler_for(self, frame):
        from .rf_display_geometry import VisualFieldSampler
        if self._visual_frame is not frame:
            self._visual_sampler = VisualFieldSampler(frame)
            self._visual_sections.clear()
            self._visual_arrows.clear()
            self._visual_frame = frame
        return self._visual_sampler

    def _arrow_section(self, frame, axis, origin, bounds, density):
        from .rf_display_geometry import cartesian_plane_points
        sampler = self._sampler_for(frame)
        key = (axis, tuple(origin), tuple(bounds), density)
        if key not in self._visual_arrows:
            geometry = getattr(frame, "cylinder_geometry", None)
            points = cartesian_plane_points(bounds, axis, origin, density, geometry)
            if geometry is None and len(points):
                points = points[sampler.valid_at(points)]
            samples = sampler(points) if len(points) else None
            if len(self._visual_arrows) >= 6:
                self._visual_arrows.pop(next(iter(self._visual_arrows)))
            self._visual_arrows[key] = points, samples
        return self._visual_arrows[key]

    def _smooth_section(self, frame, axis, origin):
        from .rf_display_geometry import cylinder_section
        sampler = self._sampler_for(frame)
        key = (axis, tuple(origin))
        if key not in self._visual_sections:
            section = cylinder_section(frame.cylinder_geometry, axis, origin)
            samples = sampler(section.points) if section.n_points else None
            # Bound memory while dragging a plane.
            if len(self._visual_sections) >= 6:
                self._visual_sections.pop(next(iter(self._visual_sections)))
            self._visual_sections[key] = section, samples
        section, samples = self._visual_sections[key]
        return section.copy(), samples

    def render(self, frame, state):
        import pyvista as pv
        result = field_values(frame, state)
        clim, true, log, cmap, warnings = colour_limits(result, state)
        p = self.plotter
        old_camera = p.camera_position if self._initialized else None
        is_current = state.field == "Js"
        smooth = bool(state.smooth_boundary and getattr(frame, "cylinder_geometry", None)
                      and state.display in {"slice", "slices"} and not is_current)
        if is_current:
            surface = frame.surface_current
            faces = np.column_stack((np.full(len(surface["triangles"]), 3), surface["triangles"])).ravel()
            domain = pv.PolyData(surface["points"], faces)
            domain.cell_data["RF scalar"] = result["values"]
            domain.cell_data["RF vectors"] = result["vectors"]
            domain.cell_data["RF normals"] = surface["normals"]
            bounds = np.asarray(domain.bounds)
        else:
            grid = self._grid_for(frame)
            grid["RF scalar"] = np.asarray(result["values"]).ravel(order="F")
            grid["RF vectors"] = np.asarray(result["vectors"]).reshape((-1, 3), order="F")
            valid = np.isfinite(frame.E).all(axis=-1) & np.isfinite(frame.H).all(axis=-1)
            grid["inside"] = valid.ravel(order="F").astype(float)
            domain = grid if np.all(valid) else grid.threshold(.5, scalars="inside", preference="point", all_scalars=True)
            bounds = np.asarray(grid.bounds)
            if getattr(frame, "cylinder_geometry", None) and state.display in {"slice", "slices"}:
                geometry = frame.cylinder_geometry
                r = geometry["radius"]
                bounds = np.array([-r, r, -r, r, geometry["zmin"], geometry["zmax"]])
        center = bounds.reshape(3, 2).mean(axis=1)
        axis = "xyz".index(state.axis)
        origin = center.copy()
        origin[axis] = bounds[2 * axis] + np.clip(state.position, .0001, .9999) * (bounds[2 * axis + 1] - bounds[2 * axis])
        if is_current:
            sections = [domain.clip(normal=state.axis, origin=origin, invert=state.invert)] if state.display == "clip" else [domain]
            section_names = ["wall"]
        elif smooth:
            section_names = list("xyz") if state.display == "slices" else [state.axis]
            sections = []
            for section_axis in section_names:
                section, samples = self._smooth_section(frame, section_axis, origin if section_axis == state.axis else center)
                if samples is not None:
                    displayed = field_values(frame, state, samples=samples)
                    section["RF scalar"] = displayed["values"]
                    section["RF vectors"] = displayed["vectors"]
                sections.append(section)
            warnings.append("Circular-edge smoothing is display-only; near-wall colors use interpolated interior samples. Numerical probes are unchanged.")
        elif state.display == "slices":
            sections = [domain.slice(normal=a, origin=origin if a == state.axis else center) for a in "xyz"]
            section_names = list("xyz")
        elif state.display == "slice":
            sections = [domain.slice(normal=state.axis, origin=origin)]
            section_names = [state.axis]
        elif state.display == "clip":
            sections = [domain.clip(normal=state.axis, origin=origin, invert=state.invert)]
            section_names = ["clip"]
        else:
            sections = [domain.extract_surface(algorithm="dataset_surface")]
            section_names = ["surface"]
        p.clear()
        p.set_background("#f3f6fa")
        p.add_mesh(pv.Box(bounds=bounds).outline(), color="#8394a7", line_width=1, pickable=False)
        bar = False
        for section_name, section in zip(section_names, sections):
            if not section.n_points:
                continue
            actor = p.add_mesh(section, scalars="RF scalar", cmap=cmap, clim=clim,
                       log_scale=log, nan_opacity=1.0 if is_current else 0.0, opacity=1.0 if is_current else float(state.opacity),
                       name=f"rf-field-{section_name}",
                       lighting=False, show_scalar_bar=not bar,
                       scalar_bar_args=dict(title=result["label"], color="#20354a", vertical=False,
                                            position_x=.18, position_y=.035, width=.68, height=.065,
                                            title_font_size=12, label_font_size=11, fmt="%.3g"))
            if is_current:
                # A transparent NaN lookup entry can put even finite, opaque
                # wall triangles into VTK's translucent render pass. Keep
                # conductor surfaces in the depth-writing opaque pass.
                actor.ForceOpaqueOn()
            bar = True
        if state.mesh:
            mesh_path = getattr(frame, "fem_mesh_path", None)
            if mesh_path and Path(mesh_path).is_file():
                key = str(mesh_path)
                try:
                    if key not in self._mesh_cache:
                        self._mesh_cache[key] = pv.read(key).extract_surface(algorithm="dataset_surface")
                    p.add_mesh(self._mesh_cache[key], style="wireframe", color="#54728e", opacity=.25,
                               line_width=.5, pickable=False)
                except Exception as exc:
                    warnings.append(f"FEM mesh overlay unavailable: {exc}")
            else:
                warnings.append("No actual FEM mesh available; the sample grid is not shown as a solver mesh.")
        vector_count = 0
        vectors_by_section = {}
        vector_candidates_by_section = {}
        density = int(np.clip(state.vector_density, 5, 41))
        if state.vectors:
            # Every visible section gets glyphs. Their lengths share one phasor
            # scale across all planes and the whole phase cycle.
            reference = (average_poynting(frame.E, frame.H) if state.field == "Savg"
                         else frame.surface_current["Js"] if is_current
                         else getattr(frame, state.field))
            norm = np.linalg.norm(reference, axis=-1)
            peak = float(np.nanmax(norm)) if np.any(np.isfinite(norm)) else 0.
            for name, section in zip(section_names, sections):
                if not section.n_points:
                    vectors_by_section[name] = 0
                    vector_candidates_by_section[name] = 0
                    continue
                if name in {"x", "y", "z"} and not is_current:
                    coords, samples = self._arrow_section(frame, name, origin if name == state.axis else center,
                                                         bounds, density)
                    vec = (field_values(frame, state, samples=samples)["vectors"]
                           if samples is not None else np.empty((0, 3)))
                elif is_current:
                    coords = section.cell_centers().points
                    # Lift tangent glyphs a fraction of the display size so the
                    # solid wall does not hide half of each arrow.
                    coords = coords + section.cell_data["RF normals"]*np.linalg.norm(bounds[1::2]-bounds[::2])*.0015
                    vec = np.asarray(section.cell_data["RF vectors"])
                else:
                    coords = section.points
                    vec = np.asarray(section.point_data["RF vectors"])
                if name not in {"x", "y", "z"}:
                    # Curved surfaces keep native/sample locations; density
                    # controls their budget instead of imposing a planar grid.
                    budget = max(1, round((350 if is_current else 150)*(density/15)**2))
                    stride = max(1, int(np.ceil(len(coords)/budget)))
                    coords, vec = coords[::stride], vec[::stride]
                vector_candidates_by_section[name] = len(coords)
                keep = np.isfinite(vec).all(axis=1) & (np.linalg.norm(vec, axis=1) > peak*.015)
                count = int(keep.sum()) if peak > 0 else 0
                vectors_by_section[name] = count
                if count:
                    cloud = pv.PolyData(coords[keep])
                    cloud["vectors"] = vec[keep]/peak
                    diag = np.linalg.norm(bounds[1::2]-bounds[::2])
                    arrows = cloud.glyph(orient="vectors", scale="vectors", factor=diag*.032*state.vector_scale)
                    p.add_mesh(arrows, color="#183f53", show_scalar_bar=False, pickable=False,
                               name=f"rf-vectors-{name}")
                    vector_count += count
        label = getattr(frame, "label", "") or f"Result {frame.index + 1}"
        norm_note = getattr(frame, "normalization", "")
        if not norm_note:
            norm_note = ("Arbitrary eigenmode amplitude" if getattr(frame, "kind", "") == "eigenmode"
                         else f"Peak phasors; 1 W incident at port {frame.port}")
        heading = label if getattr(frame, "kind", "") == "eigenmode" else f"Driven {frame.frequency_hz / 1e9:.3f} GHz / port {frame.port}"
        title = f"{heading}\n{result['label']} / phase {state.phase_deg:g} deg"
        if getattr(frame, "q_text", ""):
            title += "\n" + frame.q_text
        p.add_text(title, position="upper_left", font_size=9, color="#20354a", name="rf-title")
        p.add_axes(xlabel="x", ylabel="y", zlabel="z", viewport=(.01, .15, .13, .35))
        p.show_bounds(bounds=bounds, xtitle="x (m)", ytitle="y (m)", ztitle="z (m)",
                      color="#546a7d", font_size=9, n_xlabels=3, n_ylabels=3, n_zlabels=4,
                      location="outer", ticks="outside", all_edges=False, fmt="%.3f")
        if state.parallel:
            p.enable_parallel_projection()
        else:
            p.disable_parallel_projection()
        if state.camera:
            p.camera_position = state.camera
            if state.parallel_scale is not None and state.parallel_scale > 0:
                p.camera.parallel_scale = state.parallel_scale
            if state.view_angle is not None and 0 < state.view_angle < 180:
                p.camera.view_angle = state.view_angle
            state.camera = None  # apply a loaded camera once; subsequent mouse input owns it
        elif old_camera is not None:
            p.camera_position = old_camera
        else:
            extents = bounds[1::2] - bounds[::2]
            if getattr(frame, "kind", "") == "driven" and extents[2] > 2 * max(extents[:2]):
                # Long guides read naturally across a wide viewport with y up.
                p.camera_position = [center + np.array([-3 * extents[0], 9 * extents[1], .15 * extents[2]]), center, (0, 1, 0)]
                zoom = 1.2
            else:
                p.view_isometric()
                zoom = .82
            p.reset_camera()
            p.camera.zoom(zoom)
        p.render()
        self.frame, self.state = frame, state
        self._initialized = True
        self.info = dict(label=result["label"], unit=result["unit"], clim=clim, true_range=true,
                         log_scale=log, cmap=cmap, warnings=warnings, vectors=vector_count,
                         vectors_by_section=vectors_by_section, display_smoothing=smooth,
                         vector_candidates_by_section=vector_candidates_by_section,
                         vector_density=density,
                         field_opacity=1.0 if is_current else float(state.opacity),
                         vector_layout="uniform Cartesian cut-plane grid" if state.display in {"slice", "slices"} and not is_current else "surface samples",
                         surface_current_method="Js = H_air cross n_air-to-metal, native FEM exterior faces" if is_current else None,
                         coordinates="m", source=str(frame.source), plane_origin_m=origin.tolist(),
                         sampling="EMerge-interpolated visualization grid; not FEM nodes",
                           normalization=norm_note,
                           unloaded_q=getattr(frame, "unloaded_q", None),
                           q_text=getattr(frame, "q_text", ""),
                           loss_model=getattr(frame, "loss_model", "unknown"))
        return self.info

    def set_view(self, name):
        p = self.plotter
        n = name.lower()
        if n in {"iso", "isometric"}:
            p.view_isometric()
        elif n == "fit":
            p.reset_camera()
        elif "x" in n:
            p.view_yz(negative=n.startswith("-"))
        elif "y" in n:
            p.view_xz(negative=n.startswith("-"))
        elif "z" in n:
            p.view_xy(negative=n.startswith("-"))
        p.render()

    def probe(self, point):
        """Report raw complex phasors at the nearest valid sample, with the offset."""
        if self.frame is None:
            return {}
        frame = self.frame
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Probe requires three finite coordinates in meters")
        bounds = [(float(a[0]), float(a[-1])) for a in frame.axes]
        if any(q < lo or q > hi for q, (lo, hi) in zip(point, bounds)):
            return {"error": "Point is outside the sampled domain", "requested_m": point.tolist()}
        index = tuple(int(np.argmin(abs(a - q))) for a, q in zip(frame.axes, point))
        E, H = frame.E[index], frame.H[index]
        if not np.isfinite(E).all() or not np.isfinite(H).all():
            return {"error": "No interior field sample here (outside geometry)", "requested_m": point.tolist()}
        sampled = np.array([a[i] for a, i in zip(frame.axes, index)])
        def complex_values(arr):
            return {axis: {"real": float(v.real), "imag": float(v.imag), "magnitude": float(abs(v)),
                           "phase_deg": float(np.angle(v, deg=True))} for axis, v in zip("xyz", arr)}
        return {"method": "Nearest visualization sample; raw saved complex phasors",
                "requested_m": point.tolist(), "sample_m": sampled.tolist(),
                "distance_m": float(np.linalg.norm(sampled - point)),
                "frequency_hz": float(frame.frequency_hz), "port": frame.port,
                "normalization": self.info.get("normalization", ""),
                "display_common_gauge_deg": float(np.angle(eigen_gauge(frame.E), deg=True)) if getattr(frame, "kind", "") == "eigenmode" else 0.,
                "display_phase_deg": float(self.state.phase_deg) if self.state else 0.,
                "phase_note": "Probe phase is the raw saved phasor phase. The display adds the common gauge and RF phase to both E and H.",
                "E_V_per_m": complex_values(E), "H_A_per_m": complex_values(H),
                "Savg_W_per_m2": average_poynting(E, H).tolist()}
