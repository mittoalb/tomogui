"""
3D Volume Viewer for TIFF Stacks
Fast GPU-accelerated volume rendering using VisPy
"""

import sys
import os
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import MatrixTransform
from PIL import Image
import glob


class VolumeViewer(QMainWindow):
    """Fast 3D volume rendering viewer for TIFF stacks"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Volume Viewer - GPU Accelerated")
        self.setGeometry(100, 100, 1200, 800)

        # Data
        self.volume_data = None
        self.current_folder = None

        # Create UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel - controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_panel.setMinimumWidth(250)
        control_panel.setMaximumWidth(250)
        control_panel.setStyleSheet("background-color: #2b2b2b;")

        # Title
        title = QLabel("3D Volume Viewer")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: white; padding: 5px;")
        control_layout.addWidget(title)

        # Instructions
        instructions = QLabel("Left: Rotate\nRight: Zoom\nMiddle: Pan")
        instructions.setStyleSheet("color: #888; font-size: 9pt; padding: 5px; background: #1e1e1e; border-radius: 3px;")
        control_layout.addWidget(instructions)

        control_layout.addSpacing(10)

        # Load button
        load_btn = QPushButton("📁 Load TIFF Stack")
        load_btn.setStyleSheet("padding: 8px; font-size: 10pt;")
        load_btn.clicked.connect(self.load_volume)
        control_layout.addWidget(load_btn)

        control_layout.addSpacing(5)

        # File info
        self.info_label = QLabel("No volume loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #bbb; font-size: 9pt; padding: 5px; background: #1e1e1e; border-radius: 3px;")
        control_layout.addWidget(self.info_label)

        # Section header helper
        def add_section_header(text):
            label = QLabel(text)
            label.setStyleSheet("color: #aaa; font-size: 9pt; font-weight: bold; margin-top: 5px;")
            control_layout.addWidget(label)

        # Rendering section
        add_section_header("RENDERING")
        self.method_combo = QComboBox()
        self.method_combo.addItems(['mip', 'translucent', 'additive', 'iso', 'attenuated_mip'])
        self.method_combo.setCurrentText('mip')
        self.method_combo.currentTextChanged.connect(self.update_rendering)
        control_layout.addWidget(self.method_combo)

        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.threshold_label = QLabel("50%")
        self.threshold_label.setStyleSheet("color: #4A9EFF; font-size: 9pt;")
        threshold_layout.addWidget(self.threshold_label)
        control_layout.addLayout(threshold_layout)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        control_layout.addWidget(self.threshold_slider)

        # Step size
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.step_label = QLabel("0.20")
        self.step_label.setStyleSheet("color: #4A9EFF; font-size: 9pt;")
        step_layout.addWidget(self.step_label)
        control_layout.addLayout(step_layout)
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setMinimum(1)
        self.step_slider.setMaximum(100)
        self.step_slider.setValue(20)
        self.step_slider.valueChanged.connect(self.update_step_size)
        control_layout.addWidget(self.step_slider)

        control_layout.addSpacing(10)

        # Appearance section
        add_section_header("APPEARANCE")
        control_layout.addWidget(QLabel("Colormap:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['grays', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'])
        self.cmap_combo.currentTextChanged.connect(self.update_colormap)
        control_layout.addWidget(self.cmap_combo)

        control_layout.addSpacing(10)

        # Quality section
        add_section_header("QUALITY")
        control_layout.addWidget(QLabel("Downsample:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setMinimum(1)
        self.downsample_spin.setMaximum(8)
        self.downsample_spin.setValue(1)
        control_layout.addWidget(self.downsample_spin)

        control_layout.addSpacing(10)

        # Contrast section
        add_section_header("CONTRAST")
        control_layout.addWidget(QLabel("Min %:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setMinimum(0)
        self.vmin_slider.setMaximum(100)
        self.vmin_slider.setValue(1)
        self.vmin_slider.valueChanged.connect(self.update_clim)
        control_layout.addWidget(self.vmin_slider)

        control_layout.addWidget(QLabel("Max %:", styleSheet="color: #ccc; font-size: 9pt;"))
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setMinimum(0)
        self.vmax_slider.setMaximum(100)
        self.vmax_slider.setValue(99)
        self.vmax_slider.valueChanged.connect(self.update_clim)
        control_layout.addWidget(self.vmax_slider)

        control_layout.addSpacing(10)

        # Action buttons
        add_section_header("ACTIONS")
        reset_btn = QPushButton("🔄 Reset Camera")
        reset_btn.setStyleSheet("padding: 6px;")
        reset_btn.clicked.connect(self.reset_camera)
        control_layout.addWidget(reset_btn)

        screenshot_btn = QPushButton("📷 Save Screenshot")
        screenshot_btn.setStyleSheet("padding: 6px;")
        screenshot_btn.clicked.connect(self.save_screenshot)
        control_layout.addWidget(screenshot_btn)

        control_layout.addStretch()

        # Right panel - 3D canvas (full screen)
        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=60, azimuth=45, elevation=30, distance=500)

        # Add volume visual
        self.volume_visual = None

        # Add panels to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas.native, stretch=1)

    def load_volume(self):
        """Load TIFF stack from a folder via dialog"""
        folder = QFileDialog.getExistingDirectory(self, "Select TIFF Stack Folder",
                                                   self.current_folder or os.path.expanduser("~"))
        if not folder:
            return

        self.load_volume_from_folder(folder)

    def load_volume_from_folder(self, folder):
        """Load TIFF stack from a specific folder path"""
        if not folder or not os.path.exists(folder):
            self.info_label.setText(f"Folder not found: {folder}")
            return

        self.current_folder = folder

        # Find all TIFF files
        tiff_files = sorted(glob.glob(os.path.join(folder, "*.tiff")) +
                           glob.glob(os.path.join(folder, "*.tif")))

        if not tiff_files:
            self.info_label.setText("No TIFF files found in folder")
            return

        self.info_label.setText(f"Loading {len(tiff_files)} slices...")
        QApplication.processEvents()

        try:
            # Load first image to get dimensions
            first_img = np.array(Image.open(tiff_files[0]))
            h, w = first_img.shape[:2]

            # Check if downsampling is needed
            downsample = self.downsample_spin.value()
            if downsample > 1:
                h = h // downsample
                w = w // downsample

            # Allocate volume
            volume = np.zeros((len(tiff_files), h, w), dtype=np.float32)

            # Load all slices
            for i, tiff_file in enumerate(tiff_files):
                img = np.array(Image.open(tiff_file))
                if img.ndim == 3:
                    img = img[..., 0]  # Take first channel if RGB

                if downsample > 1:
                    img = img[::downsample, ::downsample]

                volume[i] = img.astype(np.float32)

                if i % 50 == 0:
                    self.info_label.setText(f"Loading {i+1}/{len(tiff_files)}...")
                    QApplication.processEvents()

            self.volume_data = volume

            # Update info
            mem_mb = volume.nbytes / (1024 * 1024)
            self.info_label.setText(
                f"Loaded: {len(tiff_files)} slices\n"
                f"Size: {volume.shape}\n"
                f"Memory: {mem_mb:.1f} MB\n"
                f"Range: [{volume.min():.1f}, {volume.max():.1f}]"
            )

            # Create/update volume visual
            self.create_volume_visual()

        except Exception as e:
            self.info_label.setText(f"Error loading: {str(e)}")

    def create_volume_visual(self):
        """Create or update the 3D volume visualization"""
        if self.volume_data is None:
            return

        # Remove old visual if exists
        if self.volume_visual is not None:
            self.volume_visual.parent = None

        # Normalize data to 0-1 range for better visualization
        data_min = np.percentile(self.volume_data, self.vmin_slider.value())
        data_max = np.percentile(self.volume_data, self.vmax_slider.value())
        normalized = np.clip((self.volume_data - data_min) / (data_max - data_min + 1e-6), 0, 1)

        # Create volume visual
        self.volume_visual = visuals.Volume(
            normalized,
            parent=self.view.scene,
            threshold=self.threshold_slider.value() / 100.0,
            method=self.method_combo.currentText(),
            cmap=self.cmap_combo.currentText(),
            relative_step_size=self.step_slider.value() / 100.0
        )

        # Center the volume
        self.volume_visual.transform = MatrixTransform()
        self.volume_visual.transform.translate(
            (-self.volume_data.shape[2] / 2,
             -self.volume_data.shape[1] / 2,
             -self.volume_data.shape[0] / 2)
        )

        # Reset camera
        self.reset_camera()

    def update_rendering(self):
        """Update rendering method"""
        if self.volume_visual is not None:
            self.volume_visual.method = self.method_combo.currentText()
            self.canvas.update()

    def update_threshold(self):
        """Update threshold for iso rendering"""
        value = self.threshold_slider.value()
        self.threshold_label.setText(f"{value}%")
        if self.volume_visual is not None:
            self.volume_visual.threshold = value / 100.0
            self.canvas.update()

    def update_step_size(self):
        """Update step size (quality vs speed)"""
        value = self.step_slider.value()
        self.step_label.setText(f"{value/100:.2f}")
        if self.volume_visual is not None:
            self.volume_visual.relative_step_size = value / 100.0
            self.canvas.update()

    def update_colormap(self):
        """Update colormap"""
        if self.volume_visual is not None:
            self.volume_visual.cmap = self.cmap_combo.currentText()
            self.canvas.update()

    def update_clim(self):
        """Update contrast limits"""
        if self.volume_data is not None:
            self.create_volume_visual()

    def reset_camera(self):
        """Reset camera to default view"""
        if self.volume_data is not None:
            max_dim = max(self.volume_data.shape)
            self.view.camera.distance = max_dim * 2
            self.view.camera.azimuth = 45
            self.view.camera.elevation = 30
            self.canvas.update()

    def save_screenshot(self):
        """Save current view as image"""
        if self.volume_data is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot",
            os.path.join(self.current_folder or "", "volume_screenshot.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )

        if filename:
            img = self.canvas.render()
            from PIL import Image
            Image.fromarray(img).save(filename)
            self.info_label.setText(f"Screenshot saved:\n{os.path.basename(filename)}")


def main():
    """Standalone main function"""
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle('Fusion')
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    viewer = VolumeViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
