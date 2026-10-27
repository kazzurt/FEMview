"""Derived scalar fields from FRD frames (nodal display data) and DAT element means.

Every field is a plain numpy array; ``compute`` returns a ``FieldResult`` with the label and unit
the GUI and the CLI print. FRD STRESS is labelled "nodal display" (extrapolated and averaged by
CalculiX); the DAT element mean is labelled "integration-point element mean" and lives on cells.
Tensor helpers come from ``structural.comparison`` (read-only import). Units N, mm, MPa.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frd import Frame, FrdMesh

ROOT = Path(__file__).resolve().parents[1]
STRESS_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
_FRD_STRESS_NAMES = {"SXX": "xx", "SYY": "yy", "SZZ": "zz", "SXY": "xy", "SZX": "xz", "SXZ": "xz", "SYZ": "yz", "SZY": "yz"}


def _comparison():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from structural import comparison
    return comparison


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    unit: str
    source: str          # DISP | STRESS | CONTACT | ERROR | FORC
    signed: bool = False  # diverging colormap centred on zero by default
    group: str = ""
    short: str = ""       # scalar-bar title

    @property
    def bar_title(self) -> str:
        return f"{self.short or self.label} [{self.unit}]"


FIELD_SPECS: dict[str, FieldSpec] = {s.key: s for s in [
    FieldSpec("u_mag", "Displacement magnitude", "mm", "DISP", group="Displacement", short="|U|"),
    FieldSpec("ux", "Displacement Ux", "mm", "DISP", True, "Displacement", "Ux"),
    FieldSpec("uy", "Displacement Uy", "mm", "DISP", True, "Displacement", "Uy"),
    FieldSpec("uz", "Displacement Uz", "mm", "DISP", True, "Displacement", "Uz"),
    FieldSpec("von_mises", "von Mises stress", "MPa", "STRESS", group="Stress", short="von Mises"),
    FieldSpec("tresca", "Tresca stress", "MPa", "STRESS", group="Stress", short="Tresca"),
    # 3D text goes through VTK's Latin-1 font: keep the short names ASCII (S1 for sigma 1, Txy for tau xy).
    FieldSpec("s1", "Maximum principal stress σ1", "MPa", "STRESS", True, "Stress", "S1 max principal"),
    FieldSpec("s2", "Middle principal stress σ2", "MPa", "STRESS", True, "Stress", "S2 mid principal"),
    FieldSpec("s3", "Minimum principal stress σ3", "MPa", "STRESS", True, "Stress", "S3 min principal"),
    FieldSpec("hydrostatic", "Hydrostatic (mean) stress", "MPa", "STRESS", True, "Stress", "S mean"),
    FieldSpec("sxx", "Stress σxx", "MPa", "STRESS", True, "Stress components", "Sxx"),
    FieldSpec("syy", "Stress σyy", "MPa", "STRESS", True, "Stress components", "Syy"),
    FieldSpec("szz", "Stress σzz", "MPa", "STRESS", True, "Stress components", "Szz"),
    FieldSpec("sxy", "Stress τxy", "MPa", "STRESS", True, "Stress components", "Sxy"),
    FieldSpec("sxz", "Stress τxz", "MPa", "STRESS", True, "Stress components", "Sxz"),
    FieldSpec("syz", "Stress τyz", "MPa", "STRESS", True, "Stress components", "Syz"),
    FieldSpec("cpress", "Contact pressure CPRESS (+ = compression)", "MPa", "CONTACT", group="Contact", short="CPRESS (+ compression)"),
    FieldSpec("gap", "Contact opening COPEN (+ gap, - penetration)", "mm", "CONTACT", True, "Contact", "COPEN (+ gap)"),
    FieldSpec("cslip", "Contact slip magnitude", "mm", "CONTACT", group="Contact", short="|slip|"),
    FieldSpec("cshear", "Contact shear magnitude", "MPa", "CONTACT", group="Contact", short="|CSHEAR|"),
    # *CONTACT PRINT CSTR/CDIS tables of the DAT, one value per slave face (mean of its contact points);
    # listed only when the DAT carries them at the frame time (Scene.available_fields).
    FieldSpec("dat_cpress", "DAT contact pressure per slave face (+ = compression)", "MPa", "DATCONTACT", group="Contact (DAT prints)", short="DAT CPRESS"),
    FieldSpec("dat_gap", "DAT contact opening per slave face (+ gap, - penetration)", "mm", "DATCONTACT", True, "Contact (DAT prints)", "DAT COPEN"),
    FieldSpec("dat_slip", "DAT contact slip magnitude per slave face", "mm", "DATCONTACT", group="Contact (DAT prints)", short="DAT |slip|"),
    FieldSpec("dat_cshear", "DAT contact shear magnitude per slave face", "MPa", "DATCONTACT", group="Contact (DAT prints)", short="DAT |shear|"),
    # Beam runs written with *EL FILE, SECTION FORCES: the FRD STRESS block holds section forces and
    # moments (CalculiX's own column order SXX SYY SZZ = force components, SXY SYZ SZX = moment components,
    # in the beam's local system). Listed instead of the stress fields for such frames.
    FieldSpec("sf_f1", "Section force component 1 (FRD SXX column)", "N", "SECTION", True, "Section forces (beam)", "F1 (SXX col)"),
    FieldSpec("sf_f2", "Section force component 2 (FRD SYY column)", "N", "SECTION", True, "Section forces (beam)", "F2 (SYY col)"),
    FieldSpec("sf_f3", "Section force component 3 (FRD SZZ column)", "N", "SECTION", True, "Section forces (beam)", "F3 (SZZ col)"),
    FieldSpec("sf_m1", "Section moment component 1 (FRD SXY column)", "N·mm", "SECTION", True, "Section forces (beam)", "M1 (SXY col)"),
    FieldSpec("sf_m2", "Section moment component 2 (FRD SYZ column)", "N·mm", "SECTION", True, "Section forces (beam)", "M2 (SYZ col)"),
    FieldSpec("sf_m3", "Section moment component 3 (FRD SZX column)", "N·mm", "SECTION", True, "Section forces (beam)", "M3 (SZX col)"),
    FieldSpec("sf_fmag", "Section force magnitude", "N", "SECTION", group="Section forces (beam)", short="|F section|"),
    FieldSpec("sf_mmag", "Section moment magnitude", "N·mm", "SECTION", group="Section forces (beam)", short="|M section|"),
    FieldSpec("error", "ZZ stress error estimate STR(%)", "%", "ERROR", group="Error", short="STR error"),
    FieldSpec("f_mag", "Nodal force magnitude FORC", "N", "FORC", group="Force", short="|F|"),
    FieldSpec("fx", "Nodal force Fx", "N", "FORC", True, "Force", "Fx"),
    FieldSpec("fy", "Nodal force Fy", "N", "FORC", True, "Force", "Fy"),
    FieldSpec("fz", "Nodal force Fz", "N", "FORC", True, "Force", "Fz"),
]}
STRESS_KEYS = tuple(k for k, s in FIELD_SPECS.items() if s.source == "STRESS")
DAT_CONTACT_KEYS = tuple(k for k, s in FIELD_SPECS.items() if s.source == "DATCONTACT")
DAT_CONTACT_NOTE = ("*CONTACT PRINT tables of the DAT (CSTR pressure/shear, CDIS opening/slip), mean over each slave face's "
                    "contact integration points; faces without a print are not drawn; CalculiX signs: pressure > 0 compression, "
                    "opening > 0 gap, < 0 penetration")
NODAL_STRESS_NOTE = "FRD nodal display stress (extrapolated, averaged by CalculiX); not integration-point data"
ELEMENT_STRESS_NOTE = "DAT integration-point stress, mean over each element's integration points"
CONTACT_SIGN_NOTE = ("CalculiX sign convention: CPRESS > 0 is compression on the slave surface; COPEN > 0 is an open gap, "
                     "COPEN < 0 is penetration; values exist on slave-surface nodes only")


@dataclass
class FieldResult:
    key: str
    label: str
    unit: str
    values: np.ndarray          # (n_points,) or (n_cells,)
    location: str               # "point" or "cell"
    signed: bool
    note: str = ""

    @property
    def title(self) -> str:
        return f"{self.label} [{self.unit}]"

    @property
    def ascii_title(self) -> str:
        """Latin-1 safe title for VTK text actors."""
        return ascii_text(self.title)

    @property
    def bar_title(self) -> str:
        spec = FIELD_SPECS.get(self.key)
        return ascii_text(spec.bar_title if spec else self.title)


_ASCII = {"σ": "S", "τ": "T", "Σ": "Sum ", "μ": "mu", "→": "->", "×": "x", "≈": "~", "≤": "<=", "≥": ">=", "–": "-"}


def ascii_text(text: str) -> str:
    """Replace the few non-Latin-1 glyphs VTK's built-in font drops."""
    for source, target in _ASCII.items():
        text = text.replace(source, target)
    return text


def _finite_range(self) -> tuple[float, float]:
    finite = self.values[np.isfinite(self.values)]
    if finite.size == 0:
        return (0.0, 1.0)
    return float(finite.min()), float(finite.max())


FieldResult.finite_range = _finite_range


# ---- tensor helpers -----------------------------------------------------------------------
def stress6(frame: Frame) -> np.ndarray | None:
    """(n, 6) FRD stress in xx, yy, zz, xy, xz, yz order, or None when the frame has no STRESS."""
    if "STRESS" not in frame.fields:
        return None
    if "stress6" in frame.derived:
        return frame.derived["stress6"]
    raw = frame.fields["STRESS"]
    names = [_FRD_STRESS_NAMES.get(c.upper(), c.lower()) for c in frame.components.get("STRESS", [])]
    out = np.full((raw.shape[0], 6), np.nan)
    for column, name in enumerate(STRESS_ORDER):
        if name in names:
            out[:, column] = raw[:, names.index(name)]
    frame.derived["stress6"] = out
    return out


def principal(s6: np.ndarray) -> np.ndarray:
    """(n, 3) principal stresses sorted ascending (σ3, σ2, σ1); NaN rows stay NaN."""
    tensor = _comparison().symmetric_tensor(s6)
    out = np.full((len(s6), 3), np.nan)
    finite = np.isfinite(tensor).all(axis=(1, 2))
    if finite.any():
        out[finite] = np.linalg.eigvalsh(tensor[finite])
    return out


def von_mises6(s6: np.ndarray) -> np.ndarray:
    s = np.asarray(s6, dtype=float)
    return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                   + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))


def stress_scalar(key: str, s6: np.ndarray, cache: dict | None = None) -> np.ndarray:
    """One stress scalar from (n, 6) components; ``cache`` memoises principal values."""
    if key == "von_mises":
        return von_mises6(s6)
    if key in ("s1", "s2", "s3", "tresca"):
        if cache is not None and "principal" in cache:
            p = cache["principal"]
        else:
            p = principal(s6)
            if cache is not None:
                cache["principal"] = p
        if key == "tresca":
            return p[:, 2] - p[:, 0]
        return p[:, {"s1": 2, "s2": 1, "s3": 0}[key]]
    if key == "hydrostatic":
        return s6[:, :3].mean(axis=1)
    if key in ("sxx", "syy", "szz", "sxy", "sxz", "syz"):
        return s6[:, STRESS_ORDER.index(key[1:])]
    raise KeyError(key)


# ---- availability and computation ---------------------------------------------------------
SECTION_FORCE_NOTE = ("beam run written with *EL FILE, SECTION FORCES: the FRD STRESS block holds section forces (N) and "
                      "moments (N·mm) in the beam's local system, not stresses; column order as CalculiX writes it")


def available(frame: Frame) -> list[str]:
    if frame.derived.get("section_forces") and "STRESS" in frame.fields:
        return [key for key, spec in FIELD_SPECS.items()
                if (spec.source in frame.fields and spec.source != "STRESS") or spec.source == "SECTION"]
    return [key for key, spec in FIELD_SPECS.items() if spec.source in frame.fields and spec.source != "SECTION"]


def compute(key: str, frame: Frame) -> FieldResult:
    """Nodal (point) field ``key`` from one frame."""
    spec = FIELD_SPECS[key]
    if spec.source == "SECTION":
        if "STRESS" not in frame.fields or not frame.derived.get("section_forces"):
            raise KeyError(f"frame has no section-force block for field {key}")
        raw = frame.fields["STRESS"][:, :6]
        if key == "sf_fmag":
            values = np.linalg.norm(raw[:, :3], axis=1)
        elif key == "sf_mmag":
            values = np.linalg.norm(raw[:, 3:6], axis=1)
        else:
            values = raw[:, {"sf_f1": 0, "sf_f2": 1, "sf_f3": 2, "sf_m1": 3, "sf_m2": 4, "sf_m3": 5}[key]]
        return FieldResult(key, spec.label, spec.unit, np.asarray(values, dtype=float), "point", spec.signed, SECTION_FORCE_NOTE)
    if spec.source not in frame.fields:
        raise KeyError(f"frame has no {spec.source} block for field {key}")
    note = ""
    if spec.source == "DISP":
        u = frame.fields["DISP"][:, :3]
        values = np.linalg.norm(u, axis=1) if key == "u_mag" else u[:, "xyz".index(key[1])]
    elif spec.source == "FORC":
        f = frame.fields["FORC"][:, :3]
        values = np.linalg.norm(f, axis=1) if key == "f_mag" else f[:, "xyz".index(key[1])]
    elif spec.source == "STRESS":
        values = stress_scalar(key, stress6(frame), frame.derived)
        note = NODAL_STRESS_NOTE
    elif spec.source == "CONTACT":
        contact = frame.fields["CONTACT"]
        names = [c.upper() for c in frame.components.get("CONTACT", [])]

        def column(name):
            return contact[:, names.index(name)] if name in names else np.full(len(contact), np.nan)

        if key == "cpress":
            values = column("CPRESS")
        elif key == "gap":
            values = column("COPEN")
        elif key == "cslip":
            values = np.hypot(column("CSLIP1"), column("CSLIP2"))
        else:
            values = np.hypot(column("CSHEAR1"), column("CSHEAR2"))
        note = CONTACT_SIGN_NOTE
    else:
        values = frame.fields["ERROR"][:, 0]
        note = "Zienkiewicz-Zhu stress error estimate written by CalculiX"
    return FieldResult(key, spec.label, spec.unit, np.asarray(values, dtype=float), "point", spec.signed, note)


def compute_element(key: str, mesh: FrdMesh, element_stress: dict, cache: dict | None = None) -> FieldResult:
    """Cell field ``key`` from a DAT element-mean stress record (``DatReader.element_mean_stress``)."""
    spec = FIELD_SPECS[key]
    if spec.source != "STRESS":
        raise KeyError(f"{key} has no element (DAT) form")
    rows = mesh.element_index(element_stress["element_ids"])
    s6 = np.full((mesh.n_elements, 6), np.nan)
    ok = rows >= 0
    s6[rows[ok]] = element_stress["stress"][ok]
    cache_key = f"element_{element_stress.get('time')}"
    local = cache.setdefault(cache_key, {}) if cache is not None else None
    values = stress_scalar(key, s6, local)
    missing = int((~np.isfinite(s6[:, 0])).sum())
    note = ELEMENT_STRESS_NOTE + (f"; {missing} elements without DAT stress shown blank" if missing else "")
    return FieldResult(key, spec.label, spec.unit, values, "cell", spec.signed, note)


def auto_deformation_factor(frame: Frame, diagonal: float, fraction: float = 0.1) -> float:
    """Factor that scales the peak displacement to ``fraction`` of the model diagonal (10 % default)."""
    if "DISP" not in frame.fields:
        return 1.0
    peak = np.nanmax(np.linalg.norm(frame.fields["DISP"][:, :3], axis=1))
    if not np.isfinite(peak) or peak <= 0 or diagonal <= 0:
        return 1.0
    return float(fraction * diagonal / peak)


def volume_fraction_above(values: np.ndarray, limit: float, volumes: np.ndarray, connectivity: np.ndarray,
                          location: str) -> dict:
    """Volume fraction with value above ``limit``: cell values directly, point values by cell mean."""
    if location == "cell":
        cell_values = values
    else:
        nodes = connectivity.copy()
        nodes[nodes < 0] = 0
        sampled = values[nodes]
        sampled[connectivity < 0] = np.nan
        cell_values = np.nanmean(sampled, axis=1)
    finite = np.isfinite(cell_values) & np.isfinite(volumes)
    total = float(volumes[finite].sum())
    above = finite & (cell_values > limit)
    return {"limit": float(limit), "volume_above_mm3": float(volumes[above].sum()), "volume_total_mm3": total,
            "fraction": float(volumes[above].sum() / total) if total > 0 else float("nan"),
            "cells_above": int(above.sum())}
