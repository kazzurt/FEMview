"""Targeted, streaming reads of CalculiX DAT tables.

Only the table kinds the viewer uses are read, one block at a time, so a 130 MB DAT is never
held in memory as Python lists: integration-point element stress (``*EL PRINT, S``), total
forces (``*NODE PRINT, RF, TOTALS=ONLY``) and contact print statistics. The structural reader
``parse_dat`` returns every table and is used by the pipeline for acceptance; this module is a
display accelerator only. No averaging across elements, no sign changes.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .frd import CACHE_DIR, sha256_file

_FLOAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_TIME = re.compile(r"time\s+(" + _FLOAT + ")", re.I)
_SET = re.compile(r"for set\s+(\S+)", re.I)
_EXPANDED = re.compile(r"(\d)([+-]\d{3})(?!\d)")   # 1.2345-100 -> 1.2345E-100 (MinGW three-digit exponents)
STRESS_COLUMNS = ("xx", "yy", "zz", "xy", "xz", "yz")


@dataclass
class DatTable:
    kind: str            # stress | total_force | displacement | force | contact_statistics | other
    header: str
    set: str | None
    time: float | None
    line: int            # 1-based line number of the header
    rows: int = 0        # numeric rows that follow
    extra: dict = field(default_factory=dict)


def _classify(header: str) -> str:
    text = header.lower()
    if text.startswith("stresses (elem, integ.pnt."):
        return "stress"
    if text.startswith("total force"):
        return "total_force"
    if text.startswith("displacements"):
        return "displacement"
    if text.startswith("forces (fx"):
        return "force"
    if text.startswith("statistics for slave set"):
        return "contact_statistics"
    if text.startswith("contact stress") or text.startswith("relative contact displacement"):
        return "contact"
    return "other"


def _is_numeric(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    head = stripped[0]
    return head.isdigit() or head in "+-." or stripped[:3].lower() in ("nan", "inf")


_PAIR = re.compile(r"statistics for slave set\s+(\S+),\s*master set\s+(\S+)(?:\s+and time\s+(" + _FLOAT + "))?", re.I)
_BANNER = re.compile(r"^(?:S\s*T\s*E\s*P\s+\d+|INCREMENT\s+\d+)$", re.I)


def scan_tables(path: str | Path) -> list[DatTable]:
    """Index every table (header text, time, set, line, row count) without keeping any row.

    A contact print block starts with one ``statistics for slave set A, master set B and time t``
    line followed by several titled sub-tables; the pair and time stay attached to every sub-table
    until the next block of another kind begins.
    """
    tables: list[DatTable] = []
    current: DatTable | None = None
    pending: list[str] = []
    pair: dict | None = None
    with Path(path).open("r", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if _is_numeric(line):
                if current is None:
                    titles = [p for p in pending if not _BANNER.match(p)]
                    for text in titles:
                        found = _PAIR.search(text)
                        if found:
                            pair = {"slave": found.group(1), "master": found.group(2),
                                    "time": float(found.group(3).replace("D", "E")) if found.group(3) else None}
                    header = " ".join(titles)
                    title = titles[-1] if titles else ""
                    kind = _classify(title)
                    time_match = _TIME.search(header)
                    set_match = _SET.search(header)
                    time_value = float(time_match.group(1).replace("D", "E")) if time_match else None
                    if kind == "other" and pair is not None:
                        kind = "contact_statistics"
                    elif kind in ("stress", "total_force", "displacement", "force"):
                        pair = None
                    current = DatTable(kind, header, set_match.group(1) if set_match else None, time_value, number - 1)
                    if kind == "contact_statistics" and pair is not None:
                        current.extra = {"slave": pair["slave"], "master": pair["master"], "table": title}
                        if current.time is None:
                            current.time = pair["time"]
                    tables.append(current)
                    pending = []
                current.rows += 1
            else:
                current = None
                pending.append(line)
    return tables


def _number(token: str) -> float | None:
    try:
        return float(token.replace("D", "E"))
    except ValueError:
        try:
            return float(_EXPANDED.sub(r"\1E\2", token).replace("D", "E"))
        except ValueError:
            return None


def _parse_rows(lines: list[str]) -> np.ndarray:
    text = " ".join(lines)
    width = len(lines[0].split()) if lines else 1
    try:
        values = np.array(text.replace("D", "E").split(), dtype=np.float64)
        return values.reshape(-1, width)
    except ValueError:
        pass
    try:
        values = np.array(_EXPANDED.sub(r"\1E\2", text).replace("D", "E").split(), dtype=np.float64)
        return values.reshape(-1, width)
    except ValueError:
        pass
    # text columns (beam stress rows end with the *ORIENTATION name): keep the numeric tokens only
    rows = [[v for v in (_number(t) for t in line.split()) if v is not None] for line in lines]
    width = max((len(r) for r in rows), default=1)
    out = np.full((len(rows), width), np.nan)
    for k, row in enumerate(rows):
        out[k, :len(row)] = row
    return out


def read_table(path: str | Path, table: DatTable) -> np.ndarray:
    """Rows of one indexed table as a float array (ids and integration-point columns retained)."""
    lines: list[str] = []
    with Path(path).open("r", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            if number <= table.line:
                continue
            line = raw.strip()
            if not line:
                continue
            if not _is_numeric(line):
                break
            lines.append(line)
            if len(lines) >= table.rows:
                break
    return _parse_rows(lines) if lines else np.zeros((0, 0))


class DatReader:
    """Lazy index of one DAT file with cached element-mean stress per time."""

    def __init__(self, path: str | Path, *, use_cache: bool = True):
        self.path = Path(path)
        self.use_cache = use_cache
        self._tables: list[DatTable] | None = None
        self._digest: str | None = None

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = sha256_file(self.path)
        return self._digest

    @property
    def tables(self) -> list[DatTable]:
        if self._tables is None:
            index = CACHE_DIR / f"{self.digest}.dat-index.json"
            if self.use_cache and index.exists():
                try:
                    self._tables = [DatTable(**entry) for entry in json.loads(index.read_text(encoding="utf-8"))]
                except Exception:
                    self._tables = None
            if self._tables is None:
                self._tables = scan_tables(self.path)
                if self.use_cache:
                    try:
                        index.parent.mkdir(parents=True, exist_ok=True)
                        index.write_text(json.dumps([t.__dict__ for t in self._tables]), encoding="utf-8")
                    except OSError:
                        pass
        return self._tables

    def stress_times(self) -> list[float]:
        return sorted({t.time for t in self.tables if t.kind == "stress" and t.time is not None})

    def _match(self, kind: str, time: float | None, tolerance: float = 1e-6) -> list[DatTable]:
        found = [t for t in self.tables if t.kind == kind]
        if time is not None:
            found = [t for t in found if t.time is not None and abs(t.time - time) <= tolerance * max(1.0, abs(time))]
        return found

    def element_mean_stress(self, time: float) -> dict | None:
        """Per-element mean of the integration-point stress tensor at ``time``.

        Returns ``{"element_ids", "stress" (m, 6 in xx yy zz xy xz yz order), "points_per_element",
        "sets", "time"}`` or ``None`` when the DAT has no stress table at that time. The mean over
        integration points is the only reduction; it is labelled as such in the viewer.
        """
        tables = self._match("stress", time)
        if not tables:
            return None
        cache = CACHE_DIR / f"{self.digest}.dat-stress-{time:.9g}.npz"
        if self.use_cache and cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as data:
                    return {"element_ids": data["element_ids"], "stress": data["stress"],
                            "points_per_element": data["points_per_element"], "sets": list(data["sets"]), "time": time}
            except Exception:
                pass
        rows = [read_table(self.path, t) for t in tables]
        rows = [r for r in rows if r.size]
        if not rows:
            return None
        data = np.vstack(rows)
        element_ids = data[:, 0].astype(np.int64)
        unique, inverse, counts = np.unique(element_ids, return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 6))
        for column in range(6):
            sums[:, column] = np.bincount(inverse, weights=data[:, 2 + column], minlength=len(unique))
        mean = sums / counts[:, None]
        result = {"element_ids": unique, "stress": mean, "points_per_element": counts,
                  "sets": [t.set or "" for t in tables], "time": time}
        if self.use_cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(cache.name + ".tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, element_ids=unique, stress=mean, points_per_element=counts,
                             sets=np.array(result["sets"]))
                os.replace(temporary, cache)
            except OSError:
                pass
        return result

    def total_forces(self, time: float | None = None) -> list[dict]:
        out = []
        for table in self._match("total_force", time):
            values = read_table(self.path, table)
            if values.size:
                out.append({"set": table.set, "time": table.time, "force_N": values[-1, :3].tolist()})
        return out

    def contact_prints(self, time: float | None = None) -> dict | None:
        """``*CONTACT PRINT`` CSTR / CDIS tables at ``time`` reduced to one row per slave face.

        CalculiX prints one row per contact integration point: ``slave element, face, press, tang1,
        tang2`` (contact stress) and ``slave element, face, normal, tang1, tang2`` (relative contact
        displacement; normal > 0 is a gap, < 0 a penetration). The mean over each face's points is
        the only reduction; the point count per face is kept. Returns ``None`` without such tables.
        """
        tables = [t for t in self._match("contact", time)]
        if not tables:
            return None
        out: dict = {"time": time, "stress": None, "displacement": None, "tables": len(tables)}
        for table in tables:
            rows = read_table(self.path, table)
            if rows.size == 0 or rows.shape[1] < 5:
                continue
            key = "stress" if "contact stress" in table.header.lower() else "displacement"
            if out[key] is not None:
                continue                         # one table of each kind per time
            elements = rows[:, 0].astype(np.int64)
            faces = rows[:, 1].astype(np.int64)
            pair = elements * 100 + faces
            unique, inverse, counts = np.unique(pair, return_inverse=True, return_counts=True)
            mean = np.zeros((len(unique), 3))
            for column in range(3):
                mean[:, column] = np.bincount(inverse, weights=rows[:, 2 + column], minlength=len(unique)) / counts
            out[key] = {"element_ids": unique // 100, "faces": unique % 100, "mean": mean, "points": counts,
                        "header": table.header, "rows": int(rows.shape[0])}
        if out["stress"] is None and out["displacement"] is None:
            return None
        return out

    def contact_statistics(self, time: float | None = None) -> list[dict]:
        """Contact print statistics blocks (area/normal/shear force etc.) as raw rows with headers."""
        out = []
        for table in self.tables:
            if table.kind != "contact_statistics":
                continue
            if time is not None and table.time is not None and abs(table.time - time) > 1e-6 * max(1.0, abs(time)):
                continue
            out.append({"slave": table.extra.get("slave"), "master": table.extra.get("master"), "time": table.time,
                        "table": table.extra.get("table", ""), "header": table.header,
                        "rows": read_table(self.path, table).tolist()})
        return out
