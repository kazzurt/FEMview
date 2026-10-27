"""Command line entry: ``python -m fempost {info,gui,render,export,scene} ...``.

``info`` prints a JSON summary of a run (no VTK, no Qt). ``gui`` opens the desktop viewer.
``render``/``export``/``scene`` use the same scene code path as the GUI, offscreen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ensure_root():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _load_gmsh_first():
    # Gmsh's bundled OpenCascade DLLs must load before Qt/VTK in this pinned Windows environment
    # (see tests/structural/conftest.py); harmless when gmsh is unavailable.
    try:
        import gmsh  # noqa: F401
    except Exception:
        pass


def cmd_info(args) -> int:
    from .bundle import RunBundle, _json_default
    bundle = RunBundle(args.run, use_cache=not args.no_cache, log=(lambda m: print(m, file=sys.stderr)) if args.verbose else None)
    summary = bundle.summary(include_frames=not args.no_frames)
    text = json.dumps(summary, indent=None if args.compact else 2, default=_json_default)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_gui(args) -> int:
    _load_gmsh_first()
    from .app import run_app
    return run_app(args.runs, offscreen=args.offscreen, screenshot=args.screenshot, scene=args.scene,
                   settings_path=getattr(args, "settings", None))


def cmd_workbench(args) -> int:
    _load_gmsh_first()
    from .workbench import run_workbench
    return run_workbench(args.paths, offscreen=args.offscreen, screenshot=args.screenshot)


def cmd_render(args) -> int:
    _load_gmsh_first()
    from .render import render_command
    return render_command(args)


def cmd_export(args) -> int:
    _load_gmsh_first()
    from .render import export_command
    return export_command(args)


def cmd_scene(args) -> int:
    _load_gmsh_first()
    from .app import run_app
    return run_app([], offscreen=args.offscreen, screenshot=args.screenshot, scene=args.scene)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fempost", description="Read-only structural and RF results post-processing.")
    sub = parser.add_subparsers(dest="command", required=True)

    workbench = sub.add_parser("workbench", help="open the shared structural / RF desktop workbench")
    workbench.add_argument("paths", nargs="*", help="structural run folders, RF studies, or saved RF views")
    workbench.add_argument("--offscreen", action="store_true", help="construct without showing a desktop window")
    workbench.add_argument("--screenshot", help="write a window PNG and exit (implies offscreen)")
    workbench.set_defaults(func=cmd_workbench)

    info = sub.add_parser("info", help="print a JSON summary of a run")
    info.add_argument("run", help="run folder, or a .frd/.inp/.dat, results.json, fields.npz or .assembly.json in it")
    info.add_argument("--out", help="write the JSON here instead of stdout")
    info.add_argument("--compact", action="store_true", help="single-line JSON")
    info.add_argument("--no-frames", action="store_true", help="omit the per-frame field listing")
    info.add_argument("--no-cache", action="store_true", help="ignore and do not write the FRD cache")
    info.add_argument("--verbose", action="store_true", help="progress on stderr")
    info.set_defaults(func=cmd_info)

    gui = sub.add_parser("gui", help="open the desktop viewer with 0..n cases")
    gui.add_argument("runs", nargs="*", help="run folders or triple members")
    gui.add_argument("--scene", help="scene spec JSON to restore on start")
    gui.add_argument("--offscreen", action="store_true", help="never show a window (tests)")
    gui.add_argument("--screenshot", help="offscreen: write this PNG and exit")
    gui.add_argument("--settings", default=None, help="INI file for the window layout (default: the user's registry settings)")
    gui.set_defaults(func=cmd_gui)

    render = sub.add_parser("render", help="render a figure offscreen from the same scene code as the GUI")
    render.add_argument("run")
    render.add_argument("--out", default=None, help="PNG of the 3D view (required unless --convergence/--animate is given)")
    render.add_argument("--convergence", nargs="?", const=True, default=None, metavar="PNG",
                        help="write the solver-progress figure (.sta/.cvg/stdout log); alone: only that figure")
    render.add_argument("--animate", default=None, metavar="GIF",
                        help="deformation sweep 0 -> factor -> 0 as a GIF (or a PNG sequence for another suffix)")
    render.add_argument("--frames", type=int, default=24, help="frames of --animate (default 24)")
    render.add_argument("--fps", type=int, default=10, help="frame rate of the GIF (default 10)")
    render.add_argument("--range", choices=["robust", "full"], default=None,
                        help="automatic colour range: robust 0.5-99.5 %% of the visible bodies (default) or full min..max")
    render.add_argument("--text-scale", type=float, default=None, help="multiply every 3D text size (default 1)")
    render.add_argument("--minus-frame", type=int, default=None, help="A - B within the run: subtract this frame index")
    render.add_argument("--no-overlay", action="append", default=None,
                        help="switch overlays off (restraints and loads are on by default); repeatable")
    render.add_argument("--scene", help="scene spec JSON (other options override it)")
    render.add_argument("--field", default=None, help="field key, e.g. von_mises, u_mag, s1, cpress (see README)")
    render.add_argument("--location", choices=["nodal", "element"], default=None)
    render.add_argument("--frame", type=int, default=None, help="frame index (default: last)")
    render.add_argument("--step", type=int, default=None, help="use the last frame of this step")
    render.add_argument("--deform", default=None, help="deformation factor, a number or 'auto' or 0")
    render.add_argument("--view", default=None, help="iso, +x, -x, +y, -y, +z, -z")
    render.add_argument("--clip", default=None, help="nx,ny,nz@ox,oy,oz")
    render.add_argument("--slice", default=None, help="nx,ny,nz@ox,oy,oz")
    render.add_argument("--cmap", default=None)
    render.add_argument("--clim", default=None, help="min,max")
    render.add_argument("--log", action="store_true")
    render.add_argument("--limit", type=float, default=None, help="clamp colour above this value and report volume fraction")
    render.add_argument("--overlay", action="append", default=None,
                        help="restraints, loads, contact, ties, pretension, reactions, mesh, ghost, edges, or 'none' (repeatable)")
    render.add_argument("--bodies", default=None, help="comma separated body names to show")
    render.add_argument("--size", default="1600,1000")
    render.add_argument("--title", default=None)
    render.add_argument("--plot-group", default=None, help="apply a standard plot group first (names in README)")
    render.add_argument("--minus", default=None, help="second run on the same solver mesh: draw A minus B")
    render.set_defaults(func=cmd_render)

    export = sub.add_parser("export", help="write the grid with nodal fields as VTU (ParaView escape hatch)")
    export.add_argument("run")
    export.add_argument("--vtu", required=True)
    export.add_argument("--frame", type=int, default=None)
    export.add_argument("--step", type=int, default=None)
    export.set_defaults(func=cmd_export)

    scene = sub.add_parser("scene", help="reopen a saved scene spec in the GUI")
    scene.add_argument("scene")
    scene.add_argument("--offscreen", action="store_true")
    scene.add_argument("--screenshot")
    scene.set_defaults(func=cmd_scene)
    return parser


def main(argv=None) -> int:
    _ensure_root()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"fempost: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
