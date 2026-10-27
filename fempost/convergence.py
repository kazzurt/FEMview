"""Solver progress (COMSOL-style convergence view) from the files every CalculiX run keeps.

Sources, all read-only and all optional:

- ``<job>.sta`` — one row per increment attempt: step, increment, attempt (``U`` suffix for an
  unconverged attempt), iterations, total/step/increment time.
- ``<job>.cvg`` — one row per Newton iteration: step, increment, attempt, iteration, active contact
  elements, residual force %, displacement correction %, flux %, temperature correction %.
- ``<job>.stdout.log`` — per iteration ``average force``, ``largest residual force`` (node, dof),
  ``largest increment of disp``, ``largest correction to disp`` (node, dof), ``convergence`` /
  ``no convergence``; per increment ``increment N attempt M``, ``increment size=``; increment-size
  changes and cutbacks; ``Total CalculiX Time``. The pinned build flushes Fortran and C output
  separately, so the ``STEP n`` banners are not in sequence with the iterations: step boundaries
  come from the increment numbering (an increment number that does not grow starts a new step)
  and are cross-checked against the STA.
- ``results.json`` (Codex study runs: force/moment residuals, per-body balance, frd_complete),
  ``model.json`` (applied force and moment), ``structural-results/results.json`` (KurtsFEM stage
  equilibrium) for the final equilibrium residuals shown with the solver progress.

Nothing here claims acceptance; every number is reported with the file it came from.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_FLOAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_RE_INCREMENT = re.compile(r"^\s*increment\s+(\d+)\s+attempt\s+(\d+)", re.I)
_RE_SIZE = re.compile(r"^\s*increment size=\s*(" + _FLOAT + ")", re.I)
_RE_STEP_TIME = re.compile(r"^\s*actual step time=\s*(" + _FLOAT + ")", re.I)
_RE_TOTAL_TIME = re.compile(r"^\s*actual total time=\s*(" + _FLOAT + ")", re.I)
_RE_ITERATION = re.compile(r"^\s*iteration\s+(\d+)\s*$", re.I)
_RE_CONTACT = re.compile(r"Number of contact spring elements=\s*(\d+)", re.I)
_RE_AVG = re.compile(r"^\s*average force=\s*(" + _FLOAT + ")", re.I)
_RE_RESIDUAL = re.compile(r"^\s*largest residual force=\s*(" + _FLOAT + r")(?:\s+in node\s+(\d+)\s+and dof\s+(\d+))?", re.I)
_RE_DINC = re.compile(r"^\s*largest increment of disp=\s*(" + _FLOAT + ")", re.I)
_RE_DCOR = re.compile(r"^\s*largest correction to disp=\s*(" + _FLOAT + r")(?:\s+in node\s+(\d+)\s+and dof\s+(\d+))?", re.I)
_RE_LINE_SEARCH = re.compile(r"line search factor=\s*(" + _FLOAT + ")", re.I)
_RE_INCREASED = re.compile(r"increment size is increased to\s*(" + _FLOAT + ")", re.I)
_RE_DECREASED = re.compile(r"decreased to\s*(" + _FLOAT + ")", re.I)
_RE_TOTAL_CCX = re.compile(r"Total CalculiX Time:\s*(" + _FLOAT + ")", re.I)
_RE_EQUATIONS = re.compile(r"^\s*number of equations\s*$", re.I)
_RE_INT = re.compile(r"^\s*(\d+)\s*$")


def _num(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


# ---- file parsers ----------------------------------------------------------------------------
def parse_sta(path: str | Path) -> list[dict]:
    """STA increment table including unconverged attempts (ATT column ``2U`` → attempt 2, converged False)."""
    rows: list[dict] = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 7:
            continue
        step, inc, att, itrs = tokens[:4]
        unconverged = att[-1:].upper() == "U"
        att = att.rstrip("Uu")
        if not (step.isdigit() and inc.isdigit() and att.isdigit() and itrs.isdigit()):
            continue
        try:
            times = [_num(t) for t in tokens[4:7]]
        except ValueError:
            continue
        rows.append({"step": int(step), "increment": int(inc), "attempt": int(att), "iterations": int(itrs),
                     "total_time": times[0], "step_time": times[1], "increment_time": times[2],
                     "converged": not unconverged})
    return rows


def parse_cvg(path: str | Path) -> list[dict]:
    """CVG table: one row per Newton iteration."""
    rows: list[dict] = []
    path = Path(path)
    if not path.is_file():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        tokens = line.split()
        if len(tokens) < 9:
            continue
        head = [t.rstrip("Uu") for t in tokens[:5]]
        if not all(t.isdigit() for t in head):
            continue
        try:
            values = [_num(t) for t in tokens[5:9]]
        except ValueError:
            continue
        rows.append({"step": int(head[0]), "increment": int(head[1]), "attempt": int(head[2]), "iteration": int(head[3]),
                     "contact_elements": int(head[4]), "residual_force_pct": values[0], "disp_correction_pct": values[1],
                     "residual_flux_pct": values[2], "temp_correction_pct": values[3]})
    return rows


def parse_log(path: str | Path) -> dict:
    """Solver stdout: increments (size, times, attempts), iterations with the Newton numbers, events."""
    out = {"increments": [], "iterations": [], "events": [], "total_time_s": None, "job_finished": False,
           "errors": [], "equations": None, "steps_seen": 0}
    path = Path(path)
    if not path.is_file():
        return out
    increment = None
    iteration = None
    step = 0
    last_inc_number = 0
    expect_equations = False
    with path.open("r", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if expect_equations:
                expect_equations = False
                found = _RE_INT.match(stripped)
                if found and out["equations"] is None:
                    out["equations"] = int(found.group(1))
                continue
            if _RE_EQUATIONS.match(stripped):
                expect_equations = True
                continue
            found = _RE_INCREMENT.match(line)
            if found:
                inc_number, attempt = int(found.group(1)), int(found.group(2))
                if inc_number <= last_inc_number and attempt == 1 or step == 0:
                    step += 1
                last_inc_number = inc_number
                increment = {"step": step, "increment": inc_number, "attempt": attempt, "size": None, "step_time": None,
                             "total_time": None, "line": number, "iterations": 0, "converged": None}
                out["increments"].append(increment)
                if attempt > 1:
                    out["events"].append({"line": number, "kind": "reattempt", "step": step, "increment": inc_number,
                                          "attempt": attempt, "text": stripped})
                iteration = None
                continue
            found = _RE_SIZE.match(line)
            if found and increment is not None:
                increment["size"] = _num(found.group(1))
                continue
            found = _RE_STEP_TIME.match(line)
            if found and increment is not None:
                increment["step_time"] = _num(found.group(1))
                continue
            found = _RE_TOTAL_TIME.match(line)
            if found and increment is not None:
                increment["total_time"] = _num(found.group(1))
                continue
            found = _RE_ITERATION.match(line)
            if found:
                iteration = {"step": step, "increment": increment["increment"] if increment else None,
                             "attempt": increment["attempt"] if increment else None, "iteration": int(found.group(1)),
                             "line": number, "contact_elements": None, "average_force": None, "largest_residual_force": None,
                             "residual_node": None, "residual_dof": None, "largest_disp_increment": None,
                             "largest_disp_correction": None, "correction_node": None, "correction_dof": None,
                             "line_search": None, "converged": None, "messages": []}
                out["iterations"].append(iteration)
                if increment is not None:
                    increment["iterations"] += 1
                continue
            if iteration is not None:
                found = _RE_CONTACT.search(line)
                if found:
                    iteration["contact_elements"] = int(found.group(1))
                    continue
                found = _RE_AVG.match(line)
                if found:
                    iteration["average_force"] = _num(found.group(1))
                    continue
                found = _RE_RESIDUAL.match(line)
                if found:
                    iteration["largest_residual_force"] = _num(found.group(1))
                    iteration["residual_node"] = int(found.group(2)) if found.group(2) else None
                    iteration["residual_dof"] = int(found.group(3)) if found.group(3) else None
                    continue
                found = _RE_DINC.match(line)
                if found:
                    iteration["largest_disp_increment"] = _num(found.group(1))
                    continue
                found = _RE_DCOR.match(line)
                if found:
                    iteration["largest_disp_correction"] = _num(found.group(1))
                    iteration["correction_node"] = int(found.group(2)) if found.group(2) else None
                    iteration["correction_dof"] = int(found.group(3)) if found.group(3) else None
                    continue
                found = _RE_LINE_SEARCH.search(line)
                if found:
                    iteration["line_search"] = _num(found.group(1))
                    continue
            lowered = stripped.lower()
            if lowered.startswith("no convergence"):
                if iteration is not None:
                    iteration["converged"] = False
                continue
            if lowered.startswith("convergence"):
                if iteration is not None:
                    iteration["converged"] = True
                if increment is not None:
                    increment["converged"] = True
                found = _RE_INCREASED.search(line)
                if found:
                    out["events"].append({"line": number, "kind": "size_increase", "step": step, "value": _num(found.group(1)),
                                          "text": stripped})
                continue
            if lowered.startswith("divergence"):
                if iteration is not None:
                    iteration["messages"].append(stripped)
                kind = "divergence_allowed" if "allowed" in lowered else "divergence"
                out["events"].append({"line": number, "kind": kind, "step": step, "text": stripped})
                if kind == "divergence" and increment is not None:
                    increment["converged"] = False
                continue
            found = _RE_DECREASED.search(line)
            if found:
                if "exceeds the maximum allowed" in lowered:
                    kind = "size_capped"
                elif "exceeds the remainder" in lowered:
                    kind = "size_fit_to_step"
                else:
                    kind = "cutback"
                    if increment is not None and increment["converged"] is None:
                        increment["converged"] = False
                out["events"].append({"line": number, "kind": kind, "step": step, "value": _num(found.group(1)), "text": stripped})
                continue
            if stripped.startswith("*ERROR"):
                out["errors"].append(stripped)
                out["events"].append({"line": number, "kind": "error", "step": step, "text": stripped})
                continue
            if lowered.startswith("job finished"):
                out["job_finished"] = True
                continue
            found = _RE_TOTAL_CCX.search(line)
            if found:
                out["total_time_s"] = _num(found.group(1))
    out["steps_seen"] = step
    return out


# ---- equilibrium evidence --------------------------------------------------------------------
def _vec(value) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size == 3 and np.isfinite(array).all() else None


def equilibrium_evidence(results: dict | None, model: dict | None, report: dict | None, diagonal_mm: float | None) -> list[dict]:
    """Final equilibrium residuals as recorded by the pipeline, each with its source and the applied load."""
    out: list[dict] = []
    if results and (_vec(results.get("force_residual_N")) is not None):
        force = _vec(results.get("force_residual_N"))
        moment = _vec(results.get("moment_residual_Nmm"))
        applied = _vec((model or {}).get("applied_force_N")) if model else None
        applied_moment = _vec((model or {}).get("applied_moment_vertex_Nmm")) if model else None
        entry = {"stage": "final", "time": None, "source": "results.json", "force_residual_N": force.tolist(),
                 "force_residual_norm_N": float(np.linalg.norm(force)),
                 "moment_residual_Nmm": moment.tolist() if moment is not None else None,
                 "moment_residual_norm_Nmm": float(np.linalg.norm(moment)) if moment is not None else None,
                 "applied_force_N": applied.tolist() if applied is not None else None,
                 "applied_force_norm_N": float(np.linalg.norm(applied)) if applied is not None else None,
                 "applied_source": "model.json" if applied is not None else None,
                 "frd_complete": results.get("frd_complete"), "max_displacement_mm": results.get("max_displacement_mm")}
        if applied is not None and np.linalg.norm(applied) > 0:
            entry["force_fraction"] = entry["force_residual_norm_N"] / float(np.linalg.norm(applied))
            if moment is not None and diagonal_mm:
                entry["moment_fraction"] = entry["moment_residual_norm_Nmm"] / (float(np.linalg.norm(applied)) * float(diagonal_mm))
                entry["moment_fraction_basis"] = f"|M| / (|F applied| x model diagonal {diagonal_mm:.1f} mm)"
            if moment is not None and applied_moment is not None and np.linalg.norm(applied_moment) > 0:
                entry["moment_fraction_of_applied"] = entry["moment_residual_norm_Nmm"] / float(np.linalg.norm(applied_moment))
        balance = results.get("body_balance")
        if isinstance(balance, dict):
            entry["body_balance"] = {name: {"force_residual_norm_N": float(np.linalg.norm(_vec(v.get("force_residual_N")))) if _vec(v.get("force_residual_N")) is not None else None,
                                            "moment_residual_norm_Nmm": float(np.linalg.norm(_vec(v.get("moment_residual_Nmm")))) if _vec(v.get("moment_residual_Nmm")) is not None else None}
                                     for name, v in balance.items() if isinstance(v, dict)}
        out.append(entry)
    stages = (report or {}).get("stages") if report else None
    if isinstance(stages, dict):
        for name, stage in stages.items():
            if not isinstance(stage, dict):
                continue
            eq = stage.get("equilibrium") or {}
            force = _vec(eq.get("force_residual_N"))
            if force is None:
                continue
            moment = _vec(eq.get("moment_residual_current_Nmm"))
            if moment is None:
                moment = _vec(eq.get("moment_residual_undeformed_Nmm"))
            applied = _vec(eq.get("applied_resultant_N"))
            entry = {"stage": name, "time": stage.get("time"), "source": "structural-results/results.json",
                     "force_residual_N": force.tolist(), "force_residual_norm_N": float(np.linalg.norm(force)),
                     "moment_residual_Nmm": moment.tolist() if moment is not None else None,
                     "moment_residual_norm_Nmm": float(np.linalg.norm(moment)) if moment is not None else None,
                     "applied_force_N": applied.tolist() if applied is not None else None,
                     "applied_force_norm_N": float(np.linalg.norm(applied)) if applied is not None else None,
                     "applied_source": "structural-results/results.json",
                     "force_fraction": eq.get("force_imbalance_fraction"),
                     "moment_fraction": eq.get("moment_imbalance_fraction"),
                     "moment_fraction_basis": "pipeline moment_imbalance_fraction (F_characteristic x lever arm, see gates)"}
            out.append(entry)
    return out


# ---- summary -----------------------------------------------------------------------------------
def summarize(sta_path, cvg_path, log_path, *, expected_steps: int | None = None, expected_step_times=None) -> dict:
    """Combine the three files into per-step rows, a merged iteration series and a converged flag."""
    sta = parse_sta(sta_path) if sta_path else []
    cvg = parse_cvg(cvg_path) if cvg_path else []
    log = parse_log(log_path) if log_path else parse_log("")
    steps: dict[int, dict] = {}
    for row in sta:
        entry = steps.setdefault(row["step"], {"step": row["step"], "increments": 0, "attempts": 0, "unconverged_attempts": 0,
                                               "iterations": 0, "end_step_time": None, "end_total_time": None,
                                               "min_increment": None, "max_increment": None})
        entry["attempts"] += 1
        entry["iterations"] += row["iterations"]
        if row["converged"]:
            entry["increments"] += 1
            entry["end_step_time"] = row["step_time"]
            entry["end_total_time"] = row["total_time"]
            size = row["increment_time"]
            entry["min_increment"] = size if entry["min_increment"] is None else min(entry["min_increment"], size)
            entry["max_increment"] = size if entry["max_increment"] is None else max(entry["max_increment"], size)
        else:
            entry["unconverged_attempts"] += 1
    if not steps and cvg:
        for row in cvg:
            entry = steps.setdefault(row["step"], {"step": row["step"], "increments": 0, "attempts": 0, "unconverged_attempts": 0,
                                                   "iterations": 0, "end_step_time": None, "end_total_time": None,
                                                   "min_increment": None, "max_increment": None, "from": "cvg"})
            entry["iterations"] += 1
    cutbacks = [e for e in log["events"] if e["kind"] in ("cutback", "reattempt", "divergence")]
    for event in cutbacks:
        entry = steps.get(event.get("step"))
        if entry is not None:
            entry.setdefault("cutbacks", 0)
            entry["cutbacks"] += 1
    # merged iteration series: CVG rows first (authoritative count), log details attached by order
    iterations: list[dict] = []
    if cvg:
        for k, row in enumerate(cvg):
            merged = dict(row)
            merged["index"] = k
            if k < len(log["iterations"]):
                detail = log["iterations"][k]
                same = (detail["iteration"] == row["iteration"]) and (detail["increment"] in (None, row["increment"]))
                if same:
                    for key in ("average_force", "largest_residual_force", "residual_node", "residual_dof",
                                "largest_disp_increment", "largest_disp_correction", "correction_node", "correction_dof",
                                "line_search", "converged", "messages"):
                        merged[key] = detail.get(key)
                    if merged.get("contact_elements") in (None, 0) and detail.get("contact_elements"):
                        merged["contact_elements"] = detail["contact_elements"]
            iterations.append(merged)
    else:
        for k, detail in enumerate(log["iterations"]):
            merged = dict(detail)
            merged["index"] = k
            iterations.append(merged)
    # increments (sizes) from the STA, with the log's attempt sizes when the STA is absent
    increments = [dict(row) for row in sta]
    if not increments:
        for inc in log["increments"]:
            increments.append({"step": inc["step"], "increment": inc["increment"], "attempt": inc["attempt"],
                               "iterations": inc["iterations"], "total_time": inc.get("total_time"),
                               "step_time": inc.get("step_time"), "increment_time": inc.get("size"),
                               "converged": bool(inc.get("converged"))})
    last = sta[-1] if sta else None
    converged = None
    reasons: list[str] = []
    if last is not None:
        converged = bool(last["converged"])
        if not converged:
            reasons.append(f"last STA attempt (step {last['step']} inc {last['increment']} attempt {last['attempt']}) is unconverged")
        if expected_steps is not None and max(steps) < int(expected_steps):
            converged = False
            reasons.append(f"STA ends in step {max(steps)} of {expected_steps}")
        if expected_step_times:
            try:
                final = float(expected_step_times[-1])
                last_step = max(steps)
                end = steps[last_step]["end_step_time"]
                if end is not None and abs(end - final) > 1e-6 * max(1.0, final) and end < final:
                    converged = False
                    reasons.append(f"step {last_step} ends at step time {end:g} of {final:g}")
            except (TypeError, ValueError, KeyError):
                pass
    elif log["iterations"]:
        flags = [it["converged"] for it in log["iterations"] if it["converged"] is not None]
        converged = bool(flags and flags[-1]) and not log["errors"]
        if not converged:
            reasons.append("stdout log ends without a converged iteration")
    if log["errors"]:
        converged = False
        reasons.append(log["errors"][0][:120])
    total_iterations = sum(s["iterations"] for s in steps.values())
    total_increments = sum(s["increments"] for s in steps.values())
    total_cutbacks = sum(s.get("cutbacks", 0) + s.get("unconverged_attempts", 0) for s in steps.values())
    return {"sources": {"sta": str(sta_path) if sta_path else None, "cvg": str(cvg_path) if cvg_path else None,
                        "log": str(log_path) if log_path else None},
            "steps": [steps[k] for k in sorted(steps)], "increments": increments, "iterations": iterations,
            "events": log["events"], "errors": log["errors"], "total_time_s": log["total_time_s"],
            "job_finished": log["job_finished"], "equations": log["equations"],
            "totals": {"increments": total_increments, "iterations": total_iterations, "cutbacks": total_cutbacks,
                       "steps": len(steps)},
            "converged": converged, "reasons": reasons,
            "final_iteration": iterations[-1] if iterations else None,
            "first_iteration": iterations[0] if iterations else None}


def short_text(summary: dict) -> str:
    """``converged (1 increment, 9 iterations)`` / ``NOT converged (…)`` for status lines."""
    totals = summary.get("totals") or {}
    if not summary.get("steps"):
        return "no convergence record"
    inc = totals.get("increments", 0)
    its = totals.get("iterations", 0)
    parts = [f"{inc} increment{'s' if inc != 1 else ''}", f"{its} iteration{'s' if its != 1 else ''}"]
    if totals.get("cutbacks"):
        parts.append(f"{totals['cutbacks']} cutback{'s' if totals['cutbacks'] != 1 else ''}")
    if totals.get("steps", 0) > 1:
        parts.insert(0, f"{totals['steps']} steps")
    body = ", ".join(parts)
    if summary.get("converged"):
        return f"converged ({body})"
    if summary.get("converged") is False:
        return f"NOT converged ({body}; {'; '.join(summary.get('reasons') or [])})"
    return f"solver progress ({body})"


# ---- figure -----------------------------------------------------------------------------------
SERIES = {"residual": "#2a78d6", "correction": "#eb6834", "contact": "#1baf7a", "size": "#eda100",
          "cutback": "#d03b3b", "text": "#52514e", "surface": "#fcfcfb", "grid": "#e4e3df"}


def convergence_figure(summary: dict, path: str | Path, *, size=(9.0, 8.4), dpi: int = 110,
                       title: str | None = None, evidence: list[dict] | None = None) -> Path:
    """Three stacked panels (one y axis each): Newton residual/correction per iteration on a log
    axis with increment and step boundaries, active contact elements per iteration, increment size
    against step time with attempts and cutbacks marked. Agg backend; nothing on screen."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    iterations = summary.get("iterations") or []
    increments = summary.get("increments") or []
    figure, axes = plt.subplots(3, 1, figsize=size, dpi=dpi, gridspec_kw={"height_ratios": [3, 1.6, 1.8]})
    figure.patch.set_facecolor(SERIES["surface"])
    for axis in axes:
        axis.set_facecolor(SERIES["surface"])
        axis.grid(True, color=SERIES["grid"], linewidth=0.6)
        axis.tick_params(labelsize=8, colors=SERIES["text"])
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(SERIES["grid"])
    floor = 1e-9
    if iterations:
        x = np.arange(len(iterations))
        resid = np.array([it.get("residual_force_pct", np.nan) if it.get("residual_force_pct") is not None else np.nan for it in iterations], dtype=float)
        corr = np.array([it.get("disp_correction_pct", np.nan) if it.get("disp_correction_pct") is not None else np.nan for it in iterations], dtype=float)
        contact = np.array([it.get("contact_elements") if it.get("contact_elements") is not None else np.nan for it in iterations], dtype=float)
        axis = axes[0]
        zero_resid = np.isfinite(resid) & (resid <= 0)
        zero_corr = np.isfinite(corr) & (corr <= 0)
        positive = np.concatenate([resid[resid > 0], corr[corr > 0]])
        floor = float(positive.min()) / 30.0 if positive.size else 1e-9
        axis.plot(x, np.where(resid > 0, resid, np.nan), color=SERIES["residual"], linewidth=2, marker="o", markersize=4,
                  label="residual force (% of average force, .cvg)")
        axis.plot(x, np.where(corr > 0, corr, np.nan), color=SERIES["correction"], linewidth=2, marker="s", markersize=4,
                  label="displacement correction (% of largest increment, .cvg)")
        if zero_resid.any():
            axis.plot(x[zero_resid], np.full(zero_resid.sum(), floor), linestyle="none", marker="o", markersize=5,
                      markerfacecolor="white", markeredgecolor=SERIES["residual"], label="residual force = 0 (drawn at the floor)")
        if zero_corr.any():
            axis.plot(x[zero_corr], np.full(zero_corr.sum(), floor), linestyle="none", marker="s", markersize=5,
                      markerfacecolor="white", markeredgecolor=SERIES["correction"])
        axis.set_yscale("log")
        finite = np.concatenate([resid[resid > 0], corr[corr > 0]])
        top = float(np.nanmax(finite)) * 3 if finite.size else 1e3
        axis.set_ylim(floor / 3, max(top, 1e3))
        axis.set_ylabel("%", fontsize=8, color=SERIES["text"])
        axis.set_title(title or "Newton convergence per iteration", fontsize=10, color="#0b0b0b", loc="left")
        # increment and step boundaries; cutbacks and unconverged iterations
        previous = None
        labels_done = set()
        for k, it in enumerate(iterations):
            key = (it.get("step"), it.get("increment"), it.get("attempt"))
            if previous is not None and key != previous:
                step_change = key[0] != previous[0]
                axis.axvline(k - 0.5, color="#9a9890" if step_change else SERIES["grid"], linewidth=1.6 if step_change else 0.8,
                             linestyle="-" if step_change else ":")
                if step_change:
                    axis.text(k - 0.3, top, f"step {key[0]}", fontsize=8, color=SERIES["text"], va="top")
                elif key[2] != previous[2] and (key[2] or 0) > 1:
                    axis.plot([k], [top / 2], marker="v", color=SERIES["cutback"], markersize=7, linestyle="none",
                              label=None if "cutback" in labels_done else "cutback / re-attempt")
                    labels_done.add("cutback")
            previous = key
        for k, it in enumerate(iterations):
            if it.get("messages"):
                axis.plot([k], [top / 2], marker="^", color=SERIES["cutback"], markersize=6, linestyle="none",
                          label=None if "divergence" in labels_done else "divergence allowed (contact stabilised)")
                labels_done.add("divergence")
        if iterations and iterations[0].get("step") is not None:
            axis.text(0.2, top, f"step {iterations[0]['step']}", fontsize=8, color=SERIES["text"], va="top")
        axis.legend(fontsize=7.5, loc="upper right", ncol=2, frameon=False)
        axis.set_xlim(-0.5, len(iterations) - 0.5)
        axis.set_xlabel("cumulative Newton iteration", fontsize=8, color=SERIES["text"])
        axis = axes[1]
        if np.isfinite(contact).any():
            axis.plot(x, contact, color=SERIES["contact"], linewidth=2, marker="o", markersize=3)
            lo, hi = np.nanmin(contact), np.nanmax(contact)
            axis.annotate(f"{int(contact[np.isfinite(contact)][0]):,}", (x[np.isfinite(contact)][0], contact[np.isfinite(contact)][0]),
                          fontsize=7.5, color=SERIES["text"], xytext=(3, 4), textcoords="offset points")
            axis.annotate(f"{int(contact[np.isfinite(contact)][-1]):,}", (x[np.isfinite(contact)][-1], contact[np.isfinite(contact)][-1]),
                          fontsize=7.5, color=SERIES["text"], xytext=(-6, 6), textcoords="offset points", ha="right")
            if hi > lo:
                axis.set_ylim(lo - 0.08 * (hi - lo), hi + 0.18 * (hi - lo))
        else:
            axis.text(0.5, 0.5, "no contact element count in the .cvg / log", transform=axis.transAxes, ha="center",
                      fontsize=8, color=SERIES["text"])
        axis.set_ylabel("active contact elements", fontsize=8, color=SERIES["text"])
        axis.set_xlim(-0.5, len(iterations) - 0.5)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        for axis in axes[:2]:
            axis.text(0.5, 0.5, "no .cvg / stdout iteration record", transform=axis.transAxes, ha="center", fontsize=9,
                      color=SERIES["text"])
    axis = axes[2]
    if increments:
        steps_present = sorted({row["step"] for row in increments})
        offset = 0.0
        done_label = set()
        for step in steps_present:
            rows = [r for r in increments if r["step"] == step]
            ok = [r for r in rows if r.get("converged") and r.get("step_time") is not None and r.get("increment_time") is not None]
            bad = [r for r in rows if not r.get("converged")]
            if ok:
                for r in ok:
                    x_end = offset + r["step_time"]
                    y = r["increment_time"]
                    axis.plot([x_end - y, x_end], [y, y], color=SERIES["size"], linewidth=2.5, solid_capstyle="butt",
                              label=None if "size" in done_label else "converged increment: size over its step-time span (.sta)")
                    done_label.add("size")
                    axis.plot([x_end], [y], marker="o", markersize=4, color=SERIES["size"], linestyle="none")
                    axis.annotate(f"{r['iterations']} it", (x_end - 0.5 * y, y), fontsize=7, color=SERIES["text"], xytext=(0, 4),
                                  textcoords="offset points", ha="center")
            if bad:
                xs = [offset + (r.get("step_time") or 0.0) - (r.get("increment_time") or 0.0) for r in bad]
                ys = [r.get("increment_time") or 0.0 for r in bad]
                axis.plot(xs, ys, linestyle="none", marker="x", markersize=8, color=SERIES["cutback"],
                          label=None if "bad" in done_label else "unconverged attempt, cut back (.sta)")
                done_label.add("bad")
            span = max((r.get("step_time") or 0.0) for r in rows) if rows else 1.0
            axis.axvline(offset, color="#9a9890", linewidth=1.2)
            axis.text(offset + 0.01 * max(span, 1e-9), axis.get_ylim()[1] if axis.lines else 1.0, f"step {step}", fontsize=8,
                      color=SERIES["text"], va="top")
            offset += span
        axis.set_xlabel("step time (steps laid end to end)", fontsize=8, color=SERIES["text"])
        axis.set_ylabel("increment size", fontsize=8, color=SERIES["text"])
        sizes = [r.get("increment_time") or 0.0 for r in increments]
        axis.set_ylim(0.0, max(sizes) * 1.35 if sizes and max(sizes) > 0 else 1.0)
        axis.set_xlim(-0.02 * offset, offset * 1.02 if offset > 0 else 1.0)
        axis.legend(fontsize=7.5, loc="upper right", frameon=False)
        y_top = axis.get_ylim()[1]
        for text in axis.texts:
            if text.get_text().startswith("step "):
                text.set_y(y_top)
    else:
        axis.text(0.5, 0.5, "no .sta increment table", transform=axis.transAxes, ha="center", fontsize=9, color=SERIES["text"])
    lines = []
    for step in summary.get("steps") or []:
        lines.append(f"step {step['step']}: {step['increments']} increments, {step['attempts']} attempts, {step['iterations']} iterations"
                     + (f", {step.get('cutbacks', 0)} cutbacks" if step.get("cutbacks") else "")
                     + (f", ends at t={step['end_total_time']:g}" if step.get("end_total_time") is not None else ""))
    if summary.get("total_time_s") is not None:
        lines.append(f"Total CalculiX time {summary['total_time_s']:.0f} s (stdout log; per-step wall time is not written by CalculiX)")
    for entry in evidence or []:
        text = f"{entry['source']} {entry['stage']}: |F residual| = {entry['force_residual_norm_N']:.3g} N"
        if entry.get("moment_residual_norm_Nmm") is not None:
            text += f", |M residual| = {entry['moment_residual_norm_Nmm']:.3g} N·mm"
        if entry.get("force_fraction") is not None:
            text += f" ({entry['force_fraction']:.2e} of applied |F|)"
        lines.append(text)
    if lines:
        figure.text(0.01, 0.005, "\n".join(lines[:6]), fontsize=7.5, color=SERIES["text"], va="bottom")
    figure.tight_layout(rect=(0, 0.02 + 0.018 * min(len(lines), 6), 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def table_rows(summary: dict) -> list[dict]:
    """Per-step table rows for the Details dock and the CLI record."""
    rows = []
    for step in summary.get("steps") or []:
        rows.append({"step": step["step"], "increments": step["increments"], "attempts": step["attempts"],
                     "unconverged_attempts": step.get("unconverged_attempts", 0), "cutbacks": step.get("cutbacks", 0),
                     "iterations": step["iterations"], "end_step_time": step.get("end_step_time"),
                     "end_total_time": step.get("end_total_time"), "min_increment": step.get("min_increment"),
                     "max_increment": step.get("max_increment")})
    return rows
