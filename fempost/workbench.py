"""Shared desktop shell; each physics workspace keeps its own units and result model.

The existing structural ViewerWindow is embedded intact. RF native loading happens in
its dedicated worker interpreter, so this GUI never imports EMerge or changes a venv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


def configure_workbench_application(app):
    """Use one explicit light palette; desktop dark mode must not hide RF labels."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    colors = {
        role.Window: "#f4f6f8", role.WindowText: "#20354a",
        role.Base: "#ffffff", role.AlternateBase: "#f1f5f9",
        role.Text: "#20354a", role.Button: "#e9eef3", role.ButtonText: "#20354a",
        role.Highlight: "#2c6a9f", role.HighlightedText: "#ffffff",
        role.ToolTipBase: "#ffffff", role.ToolTipText: "#20354a",
        role.PlaceholderText: "#718292", role.Link: "#2265a5", role.LinkVisited: "#705598",
        role.Light: "#ffffff", role.Midlight: "#e2e8ee", role.Mid: "#bac6d1",
        role.Dark: "#8c9baa", role.Shadow: "#677789", role.BrightText: "#b52b37",
    }
    for color_role, color in colors.items():
        palette.setColor(color_role, QtGui.QColor(color))
    disabled = QtGui.QPalette.ColorGroup.Disabled
    for color_role in (role.WindowText, role.Text, role.ButtonText, role.PlaceholderText):
        palette.setColor(disabled, color_role, QtGui.QColor("#8896a3"))
    palette.setColor(disabled, role.Base, QtGui.QColor("#f4f6f8"))
    palette.setColor(disabled, role.Button, QtGui.QColor("#edf0f3"))
    palette.setColor(disabled, role.Highlight, QtGui.QColor("#b4c5d4"))
    app.setPalette(palette)
    app.setFont(QtGui.QFont("Segoe UI", 10))


def target_kind(target: str | Path) -> str:
    """Classify documented result targets without importing any solver or loading arrays."""
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".frd", ".inp", ".dat", ".sta"}:
        return "structural"
    if path.suffix.lower() == ".emerge":
        return "rf"
    if path.is_file() and path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            data = {}
        if isinstance(data, dict):
            # Native view schemas are explicitly tagged. The fallback supports
            # project-local early view records without confusing structural scenes.
            tag = str(data.get("schema", data.get("kind", data.get("format", "")))).lower()
            if "rf" in tag and ("view" in tag or "scene" in tag):
                return "rf_view"
            if "study" in data and "state" in data:
                return "rf_view"
    folder = path if path.is_dir() else path.parent
    # Structural evidence wins over generic names such as fields.npz/results.json.
    if any(folder.glob("*.frd")):
        return "structural"
    if any(folder.glob("*.emerge")) or any(folder.glob("**/*.emerge")):
        return "rf"
    if any(folder.glob("*complex*field*.npz")) or any(folder.glob("*eigenmode*.npz")):
        return "rf"
    if path.suffix.lower() == ".npz" and path.name != "fields.npz":
        return "rf"
    # Existing discovery handles structural results.json/source_run and reports
    # a precise diagnostic for unrecognized files rather than guessing an RF model.
    return "structural"


class WorkbenchWindow(QtWidgets.QMainWindow):
    """One application with independently configurable structural and RF tabs."""

    def __init__(self, paths=(), *, offscreen=False, parent=None):
        super().__init__(parent)
        self.offscreen = bool(offscreen)
        self.setObjectName("fempostWorkbench")
        self.setWindowTitle("fempost · Structural & RF results")
        self.resize(1600, 1000)
        self.setAcceptDrops(True)
        self.settings = None if offscreen else QtCore.QSettings("KurtsFEM", "fempostWorkbench")
        self.stack = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self.stack)
        self.welcome = self._welcome()
        self.stack.addWidget(self.welcome)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.stack.addWidget(self.tabs)
        self._menus()
        self.statusBar().showMessage("Read-only results · Structural: N, mm, MPa · RF: m, V/m, A/m, W/m²")
        if self.settings is not None:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        for path in paths:
            self.open_target(path)

    def _welcome(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.addStretch()
        title = QtWidgets.QLabel("fempost")
        title.setStyleSheet("font-size: 36px; font-weight: 600; color: #15374c")
        layout.addWidget(title)
        subtitle = QtWidgets.QLabel("Structural & RF results")
        subtitle.setStyleSheet("font-size: 23px; color: #3d596c")
        layout.addWidget(subtitle)
        intro = QtWidgets.QLabel("Open saved studies, explore fields, and compare views in separate tabs.\n"
                                "Original solver results remain read-only.")
        intro.setStyleSheet("font-size: 16px; margin-top: 16px; margin-bottom: 24px")
        layout.addWidget(intro)
        for title, detail, callback in (
            ("Open RF study folder…", "EMerge eigenmodes and driven frequencies · meters", self.open_rf_dialog),
            ("Open structural run…", "Existing CalculiX workflow · N, mm, MPa", self.open_structural_dialog),
            ("Open result file or saved RF view…", ".emerge, .npz, .frd, or view .json", self.open_file_dialog),
        ):
            button = QtWidgets.QPushButton(title)
            button.setMinimumHeight(44)
            button.setMaximumWidth(440)
            button.setToolTip(detail)
            button.clicked.connect(callback)
            layout.addWidget(button)
            note = QtWidgets.QLabel(detail)
            note.setStyleSheet("color: #526678; margin-bottom: 14px")
            layout.addWidget(note)
        layout.addStretch(2)
        return page

    def _menus(self):
        file_menu = self.menuBar().addMenu("&File")
        self.open_rf_action = file_menu.addAction("Open RF study folder…", self.open_rf_dialog)
        self.open_rf_action.setShortcut("Ctrl+O")
        file_menu.addAction("Open structural run…", self.open_structural_dialog)
        file_menu.addAction("Open result file or saved RF view…", self.open_file_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Duplicate current RF study tab", self.duplicate_rf_tab)
        file_menu.addAction("Close tab", lambda: self.close_tab(self.tabs.currentIndex()), "Ctrl+W")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("RF usage guide", self.open_guide)
        help_menu.addAction("About fempost", self.about)

    def _try_open(self, path, kind=None):
        if not path:
            return None
        try:
            return self.open_target(path, kind=kind)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open results", str(exc))
            return None

    def open_rf_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open saved EMerge RF study")
        return self._try_open(path, "rf")

    def open_structural_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open CalculiX run folder")
        return self._try_open(path, "structural")

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open results or saved RF view", "",
            "Results / views (*.emerge *.npz *.json *.frd *.inp *.dat);;All files (*)")
        return self._try_open(path)

    def open_target(self, target, *, kind=None):
        path = Path(target).expanduser().resolve()
        kind = kind or target_kind(path)
        if kind in {"rf", "rf_view"}:
            from .rf_app import RFWorkspace
            workspace = RFWorkspace(path=None if kind == "rf_view" else path, parent=self.tabs)
            if kind == "rf_view":
                workspace.load_view(path)
            label = "RF · " + (path.stem if path.is_file() else path.name)
        elif kind == "structural":
            from .app import ViewerWindow
            workspace = ViewerWindow([str(path)], offscreen=self.offscreen)
            workspace.setParent(self.tabs)
            workspace.setWindowFlags(QtCore.Qt.WindowType.Widget)
            # Embedded geometry must not overwrite the standalone viewer's
            # persistent layout. The legacy ViewerWindow itself stays unchanged.
            workspace.settings = None
            label = "Structural · " + (path.parent.name if path.is_file() else path.name)
        else:
            raise ValueError(f"Unknown workspace kind: {kind}")
        workspace.setProperty("fempost_kind", kind)
        workspace.setProperty("fempost_source", str(path))
        index = self.tabs.addTab(workspace, label)
        self.tabs.setTabToolTip(index, str(path))
        self.tabs.setCurrentIndex(index)
        if kind in {"rf", "rf_view"}:
            workspace.title_changed.connect(lambda title, widget=workspace: self._update_rf_title(widget, title))
        self.stack.setCurrentWidget(self.tabs)
        return workspace

    def _update_rf_title(self, workspace, title):
        index = self.tabs.indexOf(workspace)
        if index >= 0:
            self.tabs.setTabText(index, "RF · " + title)
            study = getattr(workspace, "study", None)
            if study is not None:
                self.tabs.setTabToolTip(index, str(study.path))

    def duplicate_rf_tab(self):
        workspace = self.tabs.currentWidget()
        if workspace is None or workspace.property("fempost_kind") not in {"rf", "rf_view"}:
            self.statusBar().showMessage("Select an RF tab to open another view of its study.", 5000)
            return
        # Loading another saved view inside a tab may change its study. Duplicate
        # the current study, not the path with which the tab originally opened.
        study = getattr(workspace, "study", None)
        if study is not None:
            self._try_open(study.path, "rf")
        else:
            self._try_open(workspace.property("fempost_source"), workspace.property("fempost_kind"))

    def close_tab(self, index):
        if index < 0:
            return
        workspace = self.tabs.widget(index)
        if workspace is None:
            return
        if hasattr(workspace, "shutdown"):
            workspace.shutdown()
        workspace.close()
        self.tabs.removeTab(index)
        workspace.deleteLater()
        if self.tabs.count() == 0:
            self.stack.setCurrentWidget(self.welcome)

    def open_guide(self):
        candidates = (Path(__file__).with_name("RF_USAGE.md"), Path(__file__).with_name("RF_README.md"),
                      Path(__file__).with_name("README.md"))
        guide = next((p for p in candidates if p.is_file()), candidates[-1])
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(guide)))

    def about(self):
        QtWidgets.QMessageBox.information(self, "fempost",
            "Structural & RF results workbench\n\n"
            "Structural fields retain their N–mm–MPa workflow.\n"
            "RF complex fields use SI coordinates and explicit normalization.\n"
            "EMerge interpolation runs in the existing RF Python environment.\n"
            "Original results are never modified.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and all(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self._try_open(url.toLocalFile())
        event.acceptProposedAction()

    def closeEvent(self, event):
        if self.settings is not None:
            self.settings.setValue("geometry", self.saveGeometry())
        while self.tabs.count():
            self.close_tab(self.tabs.count() - 1)
        super().closeEvent(event)


def run_workbench(paths=(), *, offscreen=False, screenshot=None):
    if offscreen or screenshot:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        offscreen = True
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    configure_workbench_application(app)
    icon = Path(__file__).with_name("resources") / "fempost.ico"
    if icon.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    window = WorkbenchWindow(paths, offscreen=offscreen)
    if offscreen:
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    if not offscreen:
        return app.exec()
    # Let asynchronous RF loads finish when the shell is exercised by a harness.
    deadline = time.monotonic() + 180
    while True:
        app.processEvents()
        busy = any(bool(getattr(window.tabs.widget(i), "is_busy", False)) for i in range(window.tabs.count()))
        if not busy:
            break
        if time.monotonic() >= deadline:
            window.close()
            raise TimeoutError("RF workspace did not finish loading within 180 seconds")
        time.sleep(0.02)
    if screenshot:
        target = Path(screenshot).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(target))
    window.close()
    return 0
