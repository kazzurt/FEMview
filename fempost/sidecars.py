"""Accelerators and labels beside a run, all read-only and all optional.

Codex study runs: ``model.json`` (pairs with THREAD/BITE/BEARING types, load parts, fixed sets,
materials), ``model-arrays.npz`` (nodal ``forces``, ``fix`` node ids, element ``labels``),
``fields.npz`` (``U``, ``S6``, ``vm``, boundary ``tri``), ``results.json`` (residuals, region peaks)
and the mesher's ``meshes/<level>/mesh.npz`` + ``mesh.json`` (named ``<SURFACE>_tri``/``_faces``,
counts, minDetJac). KurtsFEM runs: ``<job>.assembly.json`` (supports, contact pairs, pretension,
applied nodal forces) and ``structural-results/results.json`` (stages, metrics, acceptance gates).
Every run family: ``<job>.run.json`` from the pinned runner. Nothing here is required; the FRD and
INP remain the ground truth and sidecar arrays are only accepted when they match the FRD mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_npz(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError):
        return None


def points_match(a: np.ndarray | None, b: np.ndarray | None, tolerance: float = 2e-3) -> bool:
    """True when two point arrays describe the same nodes in the same order (FRD is rounded)."""
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return False
    return bool(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= tolerance)


def run_json_summary(run: dict | None) -> dict:
    if not run:
        return {}
    keys = ("status", "returncode", "elapsed_s", "expected_steps", "completed_steps", "solver_version", "threads",
            "accepted", "peak_rss_bytes", "recorded_utc", "units", "errors", "warnings", "critical_warnings")
    return {k: run[k] for k in keys if k in run}


def codex_model_summary(model: dict | None) -> dict:
    if not model:
        return {}
    out = {k: model[k] for k in ("tag", "level", "mesh_sha256", "input_sha256", "body_names", "fix", "material_E_MPa",
                                  "preload_N", "contact_note", "applied_force_N", "applied_moment_vertex_Nmm",
                                  "bite_scale", "thread_scale", "radial_gap_mm", "bite_fraction") if k in model}
    pairs = []
    for entry in model.get("pairs", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            pairs.append({"slave": entry[0], "master": entry[1], "type": entry[2],
                          "label": entry[3] if len(entry) > 3 else ""})
    out["pairs"] = pairs
    out["load"] = model.get("load")
    out["loadparts"] = model.get("loadparts")
    return out


def codex_results_summary(results: dict | None) -> dict:
    if not results:
        return {}
    out = {k: results[k] for k in ("tag", "force_residual_N", "moment_residual_Nmm", "max_displacement_mm",
                                    "frd_complete", "hashes") if k in results}
    regions = results.get("regions")
    if isinstance(regions, dict):
        out["regions"] = {name: {k: v for k, v in values.items() if k != "elements"} for name, values in regions.items()
                          if isinstance(values, dict)}
    interfaces = results.get("interfaces")
    if isinstance(interfaces, list):
        out["interfaces"] = interfaces
    out["body_balance"] = results.get("body_balance")
    return out


def kurtsfem_results_summary(report: dict | None) -> dict:
    if not report:
        return {}
    out = {k: report[k] for k in ("schema_version", "solver_status", "diagnostic_only", "accepted", "source_run",
                                   "comparison_classification", "limitations", "metrics", "reader_errors",
                                   "critical_solver_warnings") if k in report}
    assessment = report.get("assessment") or report.get("acceptance") or {}
    out["assessment"] = {k: assessment[k] for k in ("status", "solver_status", "design_adequacy", "scope") if k in assessment}
    out["gates"] = assessment.get("gates", [])
    stages = {}
    for name, stage in (report.get("stages") or {}).items():
        if not isinstance(stage, dict):
            continue
        components = {}
        for component, values in (stage.get("components") or {}).items():
            if isinstance(values, dict):
                components[component] = {k: values.get(k) for k in ("vtu", "status", "max_displacement_mm",
                                                                      "raw_nodal_peak_von_mises_MPa", "stress_location")}
        stages[name] = {"time": stage.get("time"), "equilibrium": stage.get("equilibrium"),
                        "contact_status": (stage.get("contact") or {}).get("status"), "components": components}
    out["stages"] = stages
    pretension = report.get("pretension")
    if isinstance(pretension, dict):
        out["pretension"] = {"max_preload_relative_error": pretension.get("max_preload_relative_error"),
                             "max_tightening_drift_mm": pretension.get("max_tightening_drift_mm"),
                             "screws": {name: {k: v for k, v in values.items() if k != "history"}
                                        for name, values in (pretension.get("screws") or {}).items() if isinstance(values, dict)}}
    return out


def kurtsfem_assembly_summary(assembly: dict | None) -> dict:
    if not assembly:
        return {}
    out = {k: assembly[k] for k in ("project_fingerprint", "coordinate_frame", "pretension", "contact_pairs",
                                     "pretension_formulation", "anchorage_enforcement", "anchorage_pair_mapping",
                                     "hub_node_set", "midpoint_node_set", "fixed_hub_stress", "stages",
                                     "expected_step_times", "model_preparation_verified", "contact_initial_adjustment",
                                     "contact_clearances_verified", "comparison_classification",
                                     "screw_forces_free_to_redistribute", "field_history") if k in assembly}
    load = assembly.get("load")
    if isinstance(load, dict):
        out["load"] = {k: load.get(k) for k in ("area_mm2", "centroid_mm", "requested_resultant_n",
                                                "integrated_resultant_n", "integrated_moment_nmm", "moment_origin_mm",
                                                "integration")}
    out["hub_fixed_node_count"] = len(assembly.get("hub_fixed_nodes") or [])
    out["midpoint_node_count"] = len(assembly.get("midpoint_nodes") or [])
    out["applied_nodal_force_count"] = len(assembly.get("applied_nodal_forces") or {})
    out["final_artificial_supports"] = assembly.get("final_artificial_supports")
    counts = assembly.get("rigid_anchorage_nodes")
    if isinstance(counts, dict):
        out["rigid_anchorage_node_counts"] = {k: len(v) for k, v in counts.items()}
    return out


def mesh_json_summary(mesh_json: dict | None) -> dict:
    if not mesh_json:
        return {}
    return {k: mesh_json[k] for k in ("level", "nodes", "tet10", "body_names", "minDetJac_mm3", "elapsed_s",
                                      "sizes_mm", "surface_counts", "gmsh_version", "mesh_sha256") if k in mesh_json}
