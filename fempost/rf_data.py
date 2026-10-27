"""RF result adapters, independent of Qt, VTK and the EMerge runtime.

Stored result trees are read-only. Native interpolation runs in a persistent
subprocess using the existing RF Python and caches samples separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np


# Keep the solver worker separate from the Qt/VTK environment.
RF_RUNTIME = Path(os.environ.get("FEMPOST_RF_PYTHON", str(
    Path(__file__).resolve().parents[2] / "EMFEA" / ".venv" / "Scripts" / "python.exe")))
CACHE_VERSION = "edge-samples-v1"


@dataclass
class RFFrame:
    E: np.ndarray
    H: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    frequency_hz: float
    index: int
    port: int
    source: str
    kind: str = "driven"
    normalization: str = ""
    label: str = ""
    name: str = ""
    fem_mesh_path: Optional[Path] = None
    unloaded_q: Optional[float] = None
    q_text: str = ""
    loss_model: str = "unknown"
    cylinder_geometry: Optional[dict] = None
    surface_current: Optional[dict] = None


def eigenmode_quality(metadata, index):
    """Read explicitly characterized eigenmode Q; never interpret PEC roundoff as loss.

    A finite Q requires a positive saved unloaded_q and a documented method.
    Unknown or invalid losses stay unavailable. Explicit ideal PEC studies report
    infinity regardless of the sign/magnitude of numerical imaginary frequency.
    """
    if metadata.get("ideal_q") == "infinite":
        return dict(value=None, text="Unloaded Q0 = infinity (ideal lossless model)",
                    method="Explicit ideal lossless model", loss_model="lossless")
    modes = metadata.get("modes", [])
    row = modes[index] if 0 <= index < len(modes) else {}
    method = row.get("q_method") or metadata.get("q_method")
    raw = row.get("unloaded_q")
    try:
        value = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        value = None
    if value is not None and np.isfinite(value) and value > 0 and method:
        return dict(value=value, text=f"Unloaded Q0 = {value:,.0f}",
                    method=str(method), loss_model="lossy")
    return dict(value=None, text="Unloaded Q0: not characterized",
                method="No valid, documented unloaded-Q result is saved", loss_model="unknown")


def _fingerprint(paths):
    digest = hashlib.sha256(CACHE_VERSION.encode())
    for path in paths:
        if path and path.exists():
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:24]


def _case_path(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"RF result does not exist: {path}")
    native = path if path.is_file() and path.suffix.lower() == ".emerge" else None
    folder = path.parent if path.is_file() else path
    if folder.name.lower().endswith(".emresults"):
        native = native or folder / "simdata.emerge"
        folder = folder.parent
    if (folder / "results").is_dir():
        results = folder / "results"
        for preferred in ("refined", "sweep", "coarse", "smoke"):
            candidate = results / preferred
            if candidate.is_dir() and ((candidate / "summary.json").exists() or list(candidate.glob("*.EMResults/simdata.emerge"))):
                folder = candidate
                break
    if native is None:
        matches = sorted(folder.glob("*.EMResults/simdata.emerge"))
        if len(matches) > 1:
            raise ValueError("Multiple native results in this folder; open the intended .emerge file")
        native = matches[0] if matches else None
    return folder, native


class _NativeBridge:
    def __init__(self, source: Path, cache: Path):
        self.source, self.cache = source, cache
        self.process = None
        self._messages = queue.Queue()
        self._lock = threading.RLock()
        self._log = None

    def _start(self):
        # A previous native reader can have exited after reporting an error.
        # Release its pipes and log before replacing the process references.
        if self.process is not None or self._log is not None:
            self.close()
        python = Path(os.environ.get("FEMPOST_RF_PYTHON", str(RF_RUNTIME)))
        if not python.is_file():
            raise FileNotFoundError(f"RF Python was not found: {python}. Set FEMPOST_RF_PYTHON to the existing EMerge Python executable.")
        self.cache.mkdir(parents=True, exist_ok=True)
        self._log = (self.cache / "native-reader.log").open("a", encoding="utf-8")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        self._messages = queue.Queue()
        try:
            self.process = subprocess.Popen([str(python), "-B", str(Path(__file__).with_name("rf_worker.py")),
                                             "--source", str(self.source)], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self._log, text=True,
                                            encoding="utf-8", env=env, cwd=str(self.cache),
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self._log.close()
            self._log = None
            raise
        process, messages = self.process, self._messages
        def collect():
            for line in process.stdout:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and "ok" in value:
                        messages.put(value)
                except (ValueError, TypeError):
                    pass
            messages.put({"ok": False, "error": "Native RF reader stopped. See " + str(self.cache / "native-reader.log")})
        threading.Thread(target=collect, daemon=True, name="fempost-rf-reader").start()

    def request(self, **payload):
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                self._start()
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                response = self._messages.get(timeout=240)
            except queue.Empty:
                self.close()
                raise TimeoutError("EMerge field reader took more than four minutes; no solve was requested")
            except (OSError, ValueError) as exc:
                self.close()
                raise RuntimeError("Native RF reader connection closed; retry the request") from exc
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response["result"]

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                # Terminate rather than waiting behind an active interpolation request.
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        if self._log:
            self._log.close()
            self._log = None


class RFStudy:
    """One saved eigenmode or driven case, retaining complex SI-coordinate fields."""
    def __init__(self, path):
        self.path, self.native_path = _case_path(Path(path))
        summary = self.path / "summary.json"
        self.metadata = json.loads(summary.read_text(encoding="utf-8-sig")) if summary.exists() else {}
        self.name = self.path.parent.parent.name + " / " + self.path.name if self.path.parent.name == "results" else self.path.name
        self.fem_mesh_path = next(iter(sorted(self.path.glob("*.msh"))), None)
        self._sample_path = next((self.path / p for p in ("sampled_fields.npz", "fields_2p45GHz.npz") if (self.path / p).exists()), None)
        s_path = self.path / "sparameters.npz"
        base_cache = Path(os.environ.get("FEMPOST_RF_CACHE", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "FEMview" / "fempost_rf_cache"))).expanduser().resolve()
        # Guard the whole saved study, including sibling cases and documentation.
        source_root = self.path.parent.parent if self.path.parent.name == "results" else self.path
        if base_cache == source_root or base_cache.is_relative_to(source_root):
            raise ValueError("FEMPOST_RF_CACHE must be outside the original RF study")
        self.cache_path = base_cache / _fingerprint((self.native_path, self._sample_path, summary, s_path))
        self._bridge = _NativeBridge(self.native_path, self.cache_path) if self.native_path else None
        self._arrays = None
        self._memory = {}
        self._lock = threading.RLock()
        self.sparameters = None
        self.port_ids = [1, 2]
        self._sample_frequency = None
        if self._sample_path:
            with np.load(self._sample_path, allow_pickle=False) as saved:
                self.axes = tuple(np.array(saved[k], dtype=float) for k in ("x", "y", "z"))
                self.kind = "eigenmode" if "frequencies" in saved else "driven"
                if self.kind == "eigenmode":
                    self.frequencies_hz = np.asarray(saved["frequencies"].real, dtype=float)
                    self.metadata.setdefault("imag_frequencies_hz", saved["frequencies"].imag.tolist())
                else:
                    self._sample_frequency = float(saved["frequency_hz"])
                    self.frequencies_hz = np.array(self.metadata.get("frequencies_hz", [self._sample_frequency]), dtype=float)
            if s_path.exists():
                with np.load(s_path, allow_pickle=False) as saved:
                    self.frequencies_hz = np.asarray(saved["frequencies_hz"], dtype=float)
                    self.sparameters = np.asarray(saved["S"], dtype=complex)
        elif self._bridge:
            native = self._bridge.request(command="inspect")
            self.kind = native["kind"]
            self.frequencies_hz = np.asarray(native["frequencies_hz"], dtype=float)
            self.port_ids = native["port_ids"]
            bounds = np.asarray(native["bounds_m"])
            extents = bounds[:, 1] - bounds[:, 0]
            counts = np.clip(np.round(65 * extents / extents.max()).astype(int), 17, 65)
            self.axes = tuple(np.linspace(lo + span * 1e-6, hi - span * 1e-6, int(n))
                              for (lo, hi), span, n in zip(bounds, extents, counts))
            if native["S"]:
                pairs = np.asarray(native["S"])
                self.sparameters = pairs[..., 0] + 1j * pairs[..., 1]
            self.metadata.update(native)
        else:
            raise ValueError("No saved RF samples or native .emerge result found in this folder")
        if not np.isfinite(self.frequencies_hz).all() or not len(self.frequencies_hz):
            raise ValueError("Saved frequency axis is empty or invalid")
        if self.sparameters is not None and self.sparameters.shape[0] != len(self.frequencies_hz):
            raise ValueError("S-parameter and field frequency axes do not match")
        modes = self.metadata.get("modes", [])
        self.labels = [f"Mode {i + 1}: {modes[i].get('analytic_family', 'eigenmode')} · {f / 1e9:.3f} GHz"
                       if self.kind == "eigenmode" and i < len(modes) else
                       (f"Mode {i + 1} · {f / 1e9:.3f} GHz" if self.kind == "eigenmode" else f"{f / 1e9:.3f} GHz")
                       for i, f in enumerate(self.frequencies_hz)]
        self.normalization = ("Arbitrary eigenvector amplitude; E and H retain their relative phase"
                              if self.kind == "eigenmode" else
                              "1 W incident at selected port; SI peak phasors" if self.metadata.get("incident_power_w") == 1.0 else
                              "Stored EMerge port normalization; SI peak phasors (incident power unverified)")
        self.metadata.setdefault("coordinate_units", "m")
        self.metadata.setdefault("sample_grid_notice", "Interpolated visualization grid, not the FEM mesh")

    def _check_index(self, index, port):
        index = int(index)
        if not 0 <= index < len(self.frequencies_hz):
            raise IndexError("Select a saved mode or solved frequency")
        if self.kind == "driven" and port not in self.port_ids:
            raise ValueError(f"Port {port} is not available")
        return index

    def _frame(self, electric, magnetic, index, port, source):
        expected = tuple(len(a) for a in self.axes) + (3,)
        if electric.shape != expected or magnetic.shape != expected:
            raise ValueError(f"RF samples have an unexpected shape; expected {expected}")
        electric.setflags(write=False)
        magnetic.setflags(write=False)
        normalization = (f"1 W incident at port {port}; SI peak phasors"
                         if self.kind == "driven" and self.metadata.get("incident_power_w") == 1.0
                         else self.normalization)
        quality = eigenmode_quality(self.metadata, index) if self.kind == "eigenmode" else {}
        frame = RFFrame(electric, magnetic, self.axes, float(self.frequencies_hz[index]), index, port,
                       source, self.kind, normalization, self.labels[index], self.name, self.fem_mesh_path,
                       quality.get("value"), quality.get("text", ""), quality.get("loss_model", "unknown"))
        config = self.metadata.get("configuration", {})
        diameter, length = config.get("diameter_mm"), config.get("length_mm")
        walls = str(self.metadata.get("walls", "")).lower()
        closed = (self.kind == "eigenmode" and str(self.metadata.get("ports", "")).lower().startswith("none")
                  and ("every exterior face" in walls or "sidewall and both endcaps" in walls))
        if closed and diameter and length and float(diameter) > 0 and float(length) > 0:
            frame.cylinder_geometry = dict(radius=float(diameter)/2000, zmin=0., zmax=float(length)/1000)
        if closed and self._bridge is not None:
            frame.surface_current = self._surface_current(index, port)
        return frame

    def _surface_current(self, index, port):
        destination = self.cache_path / f"wall-current-v1-{index:04d}-port-{port}.npz"
        if not destination.exists():
            self._bridge.request(command="surface_current", index=index, port=port,
                                 closed_conductor_walls=True, output=str(destination), cache_root=str(self.cache_path))
        for attempt in range(12):
            try:
                with np.load(destination, allow_pickle=False) as saved:
                    if not np.isclose(float(saved["frequency_hz"]), self.frequencies_hz[index], rtol=0, atol=1):
                        raise ValueError("Surface-current frequency differs from selected mode")
                    surface = {name: np.array(saved[name]) for name in ("points", "triangles", "centers", "normals", "areas", "H", "Js")}
                for array in surface.values():
                    array.setflags(write=False)
                return surface
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(.1)

    def sample(self, index, port=1):
        """Return raw joint complex E/H; no per-field normalization or phase change."""
        index = self._check_index(index, port)
        key = (index, port)
        with self._lock:
            if key in self._memory:
                return self._memory[key]
            saved_available = self._sample_path is not None and (self.kind == "eigenmode" or
                (port == 1 and np.isclose(self.frequencies_hz[index], self._sample_frequency, rtol=0, atol=1)))
            if saved_available:
                if self._arrays is None:
                    with np.load(self._sample_path, allow_pickle=False) as data:
                        self._arrays = (np.array(data["E"], dtype=complex), np.array(data["H"], dtype=complex))
                e, h = self._arrays
                if self.kind == "eigenmode":
                    e, h = e[index], h[index]
                frame = self._frame(e, h, index, port, "Saved EMerge field samples")
            else:
                if self._bridge is None:
                    raise FileNotFoundError("This frequency/port needs the original native .emerge solution")
                destination = self.cache_path / f"field-{index:04d}-port-{port}.npz"
                if not destination.exists():
                    self._bridge.request(command="sample", index=index, port=port,
                                         axes=[a.tolist() for a in self.axes], output=str(destination),
                                         cache_root=str(self.cache_path))
                for attempt in range(12):
                    try:
                        with np.load(destination, allow_pickle=False) as data:
                            if not np.isclose(float(data["frequency_hz"]), self.frequencies_hz[index], rtol=0, atol=1):
                                raise ValueError("Cached/native frequency differs from the study frequency")
                            frame = self._frame(np.array(data["E"]), np.array(data["H"]), index, port,
                                                "Native EMerge edge-element interpolation (cached)")
                        break
                    except PermissionError:
                        # A transient Windows handle is not evidence of corruption.
                        # Keep the immutable published entry for concurrent readers.
                        if attempt == 11:
                            raise
                        time.sleep(.1)
                    except Exception:
                        # Only this invalid derived cache file is removed; saved
                        # results stay read-only. Never delete on sharing failures.
                        destination.unlink(missing_ok=True)
                        raise
            self._memory[key] = frame
            while len(self._memory) > 4:
                self._memory.pop(next(iter(self._memory)))
            return frame

    def probe(self, index, point_m, port=1):
        """Evaluate native edge fields at an exact SI point, without solving."""
        index = self._check_index(index, port)
        if self._bridge is None:
            raise FileNotFoundError("Exact point evaluation needs the original native .emerge solution")
        result = self._bridge.request(command="probe", index=index, port=port,
                                      point_m=np.asarray(point_m, dtype=float).tolist())
        for name in ("E", "H"):
            if result[name] is not None:
                pairs = np.asarray(result[name])
                result[name] = pairs[:, 0] + 1j * pairs[:, 1]
        if result["valid"]:
            result["Savg"] = 0.5 * np.real(np.cross(result["E"], np.conjugate(result["H"])))
        normalization = (f"1 W incident at port {port}; SI peak phasors"
                         if self.kind == "driven" and self.metadata.get("incident_power_w") == 1.0
                         else self.normalization)
        result.update(source="Native EMerge edge-element interpolation", normalization=normalization)
        return result

    def close(self):
        if self._bridge:
            self._bridge.close()
        self._memory.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
