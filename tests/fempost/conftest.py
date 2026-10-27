"""fempost test harness: Gmsh first (native DLL order), offscreen Qt, read-only fixture paths."""
import os
import sys
from pathlib import Path

import gmsh  # noqa: F401 -- must load before OCP/Qt/VTK in this pinned Windows environment
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

FIXTURES = ROOT / "Dodec FEA/structural-fea-benchmark-runs/fixtures"
SHORT_SCREW = ROOT / "tests/fixtures/explicit-bar"
FRICTION = FIXTURES / "friction-native-tetra10-vector-v1/vector-mu-010"
KURTSFEM_RUN = ROOT / "Dodec FEA/structural-fea-benchmark-runs/assembly-35mm-two-layers-adaptive"
CODEX_RUN = Path(os.environ.get("FEMVIEW_STRUCTURAL_FIXTURE", str(ROOT / "tests/external/structural-run")))
REVIEW_DIR = ROOT / "structural-fea-environment/temporary/fempost-review"


def _snapshot(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*")}


@pytest.fixture(scope="session", autouse=True)
def read_only_guard():
    """Fail the session if any test wrote beside a fixture or a Dodec run."""
    folders = [SHORT_SCREW, FRICTION, KURTSFEM_RUN, CODEX_RUN]
    before = {f: _snapshot(f) for f in folders}
    yield
    for folder in folders:
        assert _snapshot(folder) == before[folder], f"read-only folder changed: {folder}"


@pytest.fixture
def short_screw():
    if not (SHORT_SCREW / "short_screw.frd").is_file():
        pytest.skip("short_screw fixture missing")
    return SHORT_SCREW


@pytest.fixture
def friction():
    if not (FRICTION / "friction.frd").is_file():
        pytest.skip("friction fixture missing")
    return FRICTION


@pytest.fixture
def kurtsfem_run():
    if not (KURTSFEM_RUN / "dodec_assembly.frd").is_file():
        pytest.skip("KurtsFEM benchmark run missing")
    return KURTSFEM_RUN


@pytest.fixture
def codex_run():
    if not (CODEX_RUN / "job.frd").is_file():
        pytest.skip("Codex study run missing")
    return CODEX_RUN


def pytest_configure(config):
    config.addinivalue_line("filterwarnings",
                            "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore::pyvista.PyVistaFutureWarning")
