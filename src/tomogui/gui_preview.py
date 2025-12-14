# gui_preview.py - layout-only preview of TomoGUI
import sys
import types
from PyQt5.QtWidgets import QApplication, QPushButton

print(">> gui_preview.py starting (layout preview)…")

# ------------------------------------------------
# 1) Stub h5py ONLY if it is missing
# ------------------------------------------------
try:
    import h5py  # noqa: F401
    print(">> h5py found, using real h5py")
except ImportError:
    print(">> h5py not found, using dummy stub")
    sys.modules["h5py"] = types.ModuleType("h5py")


# ------------------------------------------------
# 2) Import TomoGUI from the *package*, not from gui.py
#    This uses your __init__.py:
#        from .gui import TomoGUI
# ------------------------------------------------
try:
    from tomogui import TomoGUI
    print(">> Imported TomoGUI via 'from tomogui import TomoGUI'")
except ImportError as e:
    print("!! ERROR importing TomoGUI from package 'tomogui':", e)
    print("   Make sure you are running this from the 'src' folder with:")
    print("       python -m tomogui.gui_preview")
    raise


class PreviewGUI(TomoGUI):
    """
    Layout-only version of TomoGUI.

    Uses the real layout from gui.py,
    but disconnects all button actions so nothing heavy runs.
    """
    def __init__(self):
        super().__init__()

        # Disconnect every button so no reconstruction / I/O is triggered
        for btn in self.findChildren(QPushButton):
            try:
                btn.clicked.disconnect()
            except Exception:
                pass

        print(">> Preview mode: GUI layout loaded, all functionality disabled.")


def main():
    app = QApplication(sys.argv)
    win = PreviewGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
