"""Stable Qt/VTK lifetime for RF and mixed structural/RF pytest sessions.

Keep one QApplication alive throughout collection and all test modules. Creating
module-local application wrappers allowed native graphics cleanup to outlive Qt
and crash Windows after otherwise passing assertions. Original structural test
fixtures are retained; they reuse this application through QApplication.instance().
"""
import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import gmsh  # noqa: E402,F401 -- verified native DLL load order
import pytest  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402
import shiboken6  # noqa: E402


_SESSION_APPLICATION = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="session", autouse=True)
def graphics_session_lifetime():
    """Dispose graphics explicitly while the process-global QApplication is alive."""
    yield _SESSION_APPLICATION
    import pyvista
    from fempost.rf_app import _FINISHING_WORKERS
    for worker in list(_FINISHING_WORKERS):
        if shiboken6.isValid(worker) and not worker.wait(30000):
            raise RuntimeError("An RF test worker did not finish during session teardown")
    pyvista.close_all()
    for widget in list(_SESSION_APPLICATION.topLevelWidgets()):
        if shiboken6.isValid(widget):
            widget.close()
            widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    _SESSION_APPLICATION.processEvents()
    gc.collect()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    _SESSION_APPLICATION.processEvents()
