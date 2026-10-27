"""Offscreen figures and VTU export from the same scene code the GUI uses.

``fempost render <run> --out fig.png [--scene spec.json] [--field von_mises --deform 10 ...]`` and
``fempost export <run> --vtu out.vtu``. Options override the scene spec; the resulting state is
printed as JSON next to the render information so a report can quote exactly what was drawn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .bundle import RunBundle, _json_default
from .fields import FIELD_SPECS
from .scene import OVERLAY_KEYS, Scene, SceneState, apply_plot_group, offscreen_plotter


def _plane(text: str | None) -> dict | None:
    """``nx,ny,nz@ox,oy,oz`` (origin optional) -> plane dict."""
    if not text:
        return None
    normal_text, _, origin_text = text.partition("@")
    normal = [float(v) for v in normal_text.split(",")]
    origin = [float(v) for v in origin_text.split(",")] if origin_text.strip() else None
    return {"normal": normal, "origin": origin}


def state_from_args(args, bundle=None) -> SceneState:
    state = SceneState.load(args.scene) if getattr(args, "scene", None) else SceneState()
    if getattr(args, "plot_group", None):
        state = apply_plot_group(state, args.plot_group, bundle)
    if getattr(args, "field", None):
        if args.field not in FIELD_SPECS:
            raise SystemExit(f"unknown field {args.field!r}; choose from {', '.join(FIELD_SPECS)}")
        state.field = args.field
        state.mesh_quality = False
    if getattr(args, "location", None):
        state.location = "cell" if args.location == "element" else "point"
    if getattr(args, "frame", None) is not None:
        state.frame = int(args.frame)
    if getattr(args, "deform", None) is not None:
        text = str(args.deform).strip().lower()
        state.deform = None if text == "auto" else float(text)
    if getattr(args, "view", None):
        state.view = args.view
        state.camera = None
    if getattr(args, "clip", None):
        clip = _plane(args.clip)
        clip.update({"invert": False, "ghost": True})
        state.clip = clip
    if getattr(args, "slice", None):
        cut = _plane(args.slice)
        cut.update({"contours": 10, "only": False})
        state.slice = cut
    if getattr(args, "cmap", None):
        state.cmap = args.cmap
    if getattr(args, "clim", None):
        state.clim = [float(v) for v in args.clim.split(",")]
    if getattr(args, "log", False):
        state.log_scale = True
    if getattr(args, "limit", None) is not None:
        state.limit = float(args.limit)
    if getattr(args, "overlay", None):
        for name in args.overlay:
            for key in name.split(","):
                key = key.strip()
                if key == "none":                     # restraints and loads are on by default; 'none' clears them
                    state.overlays = {k: False for k in OVERLAY_KEYS}
                elif key in OVERLAY_KEYS:
                    state.overlays[key] = True
                elif key == "mesh":
                    state.mesh_quality = True
                elif key == "ghost":
                    state.ghost = True
                elif key == "edges":
                    state.edges = True
    if getattr(args, "no_overlay", None):
        for name in args.no_overlay:
            for key in name.split(","):
                key = key.strip()
                if key in OVERLAY_KEYS:
                    state.overlays[key] = False
                elif key == "ghost":
                    state.ghost = False
                elif key == "edges":
                    state.edges = False
    if getattr(args, "bodies", None):
        state.bodies = [b.strip() for b in args.bodies.split(",") if b.strip()]
    if getattr(args, "title", None):
        state.title = args.title
    if getattr(args, "range", None):
        state.range_mode = args.range
    if getattr(args, "text_scale", None) is not None:
        state.text_scale = float(args.text_scale)
    if getattr(args, "minus_frame", None) is not None:
        state.minus_frame = int(args.minus_frame)
    return state


def convergence_command(bundle: RunBundle, path: Path, size) -> dict:
    """Write the convergence figure of a run and return the record printed by ``render --convergence``."""
    from . import convergence as convergence_module
    summary = bundle.convergence
    acceptance = bundle.acceptance()
    width, height = size
    convergence_module.convergence_figure(summary, path, size=(max(width, 400) / 110.0, max(height, 400) / 110.0), dpi=110,
                                          title=f"{bundle.name}: Newton convergence", evidence=acceptance.get("equilibrium"))
    return {"convergence": str(path), "run": str(bundle.folder), "case": bundle.name, "label": acceptance["label"],
            "kind": acceptance.get("kind"), "converged": summary.get("converged"), "reasons": summary.get("reasons"),
            "totals": summary.get("totals"), "steps": convergence_module.table_rows(summary),
            "sources": summary.get("sources"), "total_time_s": summary.get("total_time_s"),
            "final_iteration": summary.get("final_iteration"), "equilibrium": acceptance.get("equilibrium")}


def render_command(args) -> int:
    bundle = RunBundle(args.run)
    size = tuple(int(v) for v in str(getattr(args, "size", "1600,1000")).split(","))
    convergence_path = getattr(args, "convergence", None)
    if convergence_path and not getattr(args, "out", None) and not getattr(args, "animate", None):
        out = Path(convergence_path if convergence_path is not True else "convergence.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(convergence_command(bundle, out, size), indent=2, default=_json_default))
        return 0
    if not getattr(args, "out", None) and not getattr(args, "animate", None):
        raise SystemExit("render: --out is required (or --convergence <png> / --animate <gif>)")
    scene = Scene(bundle)
    state = state_from_args(args, bundle)
    if getattr(args, "step", None) is not None:
        frame = bundle.last_frame_of_step(int(args.step))
        if frame is None:
            raise SystemExit(f"no frame for step {args.step}; frames: {[f.label for f in bundle.frames]}")
        state.frame = frame.index
    other = Scene(RunBundle(args.minus)) if getattr(args, "minus", None) else None
    plotter = offscreen_plotter(size)
    record: dict = {"run": str(bundle.folder), "case": bundle.name}
    try:
        if getattr(args, "animate", None):
            from .scene import animate
            animation = animate(scene, plotter, state, Path(args.animate), frames=int(getattr(args, "frames", 24) or 24),
                                fps=int(getattr(args, "fps", 10) or 10), other=other)
            info = animation.pop("last")
            record["animation"] = animation
        else:
            info = scene.render(plotter, state, other)
        if getattr(args, "out", None):
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if getattr(args, "animate", None):
                info = scene.render(plotter, state, other)
            plotter.screenshot(str(out))
            record["out"] = str(out)
    finally:
        plotter.close()
    if convergence_path:
        out = Path(convergence_path) if convergence_path is not True else Path(args.out).with_name(Path(args.out).stem + "-convergence.png")
        record["convergence"] = convergence_command(bundle, out, size)
    record.update({"state": state.to_dict(), "info": {k: v for k, v in info.items() if k != "overlays"},
                   "overlays": [{k: v for k, v in o.items() if k in ("kind", "label", "count", "step", "magnitude", "resultant",
                                                                     "glyph_radius_mm", "extent_mm")}
                                for o in info.get("overlays", [])]})
    print(json.dumps(record, indent=2, default=_json_default))
    return 0


def export_command(args) -> int:
    """Write the grid with the frame's nodal fields (and DAT element means when present) as VTU."""
    from . import fields as fields_module
    bundle = RunBundle(args.run)
    scene = Scene(bundle)
    state = SceneState()
    if getattr(args, "frame", None) is not None:
        state.frame = int(args.frame)
    if getattr(args, "step", None) is not None:
        frame = bundle.last_frame_of_step(int(args.step))
        if frame is None:
            raise SystemExit(f"no frame for step {args.step}")
        state.frame = frame.index
    frame = bundle.frames[scene.frame_index(state)]
    grid = scene.grid.copy()
    grid.point_data["U"] = scene.displacement(state)
    for key in fields_module.available(frame):
        result = fields_module.compute(key, frame)
        spec = FIELD_SPECS[key]
        grid.point_data[f"{key}_{spec.unit.replace('%', 'pct')}_nodal_display" if spec.source == "STRESS" else f"{key}_{spec.unit.replace('%', 'pct')}"] = result.values
    for name, raw in frame.fields.items():
        grid.point_data[f"FRD_{name}"] = raw
    record = scene.element_stress(frame)
    if record is not None:
        for key in ("von_mises", "s1", "s3"):
            cell = fields_module.compute_element(key, bundle.mesh, record, frame.derived)
            grid.cell_data[f"{key}_MPa_element_mean_ip"] = cell.values[grid.cell_data["frd_row"]]
    grid.field_data["fempost_frame"] = np.array([frame.label.replace("·", "-").encode("ascii", "replace").decode()])
    grid.field_data["fempost_run"] = np.array([str(bundle.folder).encode("ascii", "replace").decode()])
    grid.field_data["stress_location"] = np.array(["point arrays: FRD extrapolated nodal display; cell arrays: DAT integration-point element mean"])
    out = Path(args.vtu)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out))
    print(json.dumps({"vtu": str(out), "frame": frame.label, "points": grid.n_points, "cells": grid.n_cells,
                      "point_arrays": list(grid.point_data.keys()), "cell_arrays": list(grid.cell_data.keys())}, indent=2))
    return 0
