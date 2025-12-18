# 3D Volume Viewer

Fast GPU-accelerated 3D volume rendering for TIFF stacks using VisPy.

## Features

- **Fast GPU rendering** - Hardware-accelerated visualization using OpenGL
- **Multiple rendering modes**:
  - `mip` - Maximum Intensity Projection (default, fastest)
  - `translucent` - Semi-transparent rendering
  - `additive` - Additive blending
  - `iso` - Isosurface rendering
  - `attenuated_mip` - Attenuated MIP
- **Interactive controls**:
  - Real-time rotation, zoom, pan
  - Adjustable threshold for isosurface
  - Quality vs speed control (step size)
  - Contrast adjustment
  - Multiple colormaps
- **Downsampling** - Reduce memory for large datasets
- **Screenshot export** - Save current view as PNG/JPG

## Usage

### Standalone

```bash
# After installation
tomogui-volume

# Or run directly
python -m tomogui.volume_viewer
```

### From Python

```python
from tomogui.volume_viewer import VolumeViewer
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
viewer = VolumeViewer()
viewer.show()
sys.exit(app.exec_())
```

## Controls

- **Mouse**:
  - Left-drag: Rotate view
  - Right-drag: Zoom in/out
  - Middle-drag: Pan

- **UI Controls**:
  - Rendering Method: Choose visualization technique
  - Threshold: Adjust isosurface level (for 'iso' mode)
  - Step Size: Trade quality for speed (lower = better quality, slower)
  - Colormap: Color scheme for data
  - Downsample: Reduce dataset size for faster loading
  - Contrast: Min/max percentile for intensity range

## Performance Tips

1. **For large datasets**: Use downsampling (factor 2-4)
2. **For speed**:
   - Use 'mip' rendering method
   - Increase step size (0.5-1.0)
3. **For quality**:
   - Use 'translucent' or 'iso' methods
   - Decrease step size (0.1-0.3)
4. **Memory**: Each 1000x1000x1000 volume uses ~4GB RAM

## Example Datasets

Works with any folder containing sequentially numbered TIFF files:
```
/path/to/reconstruction/
  ├── slice_0000.tiff
  ├── slice_0001.tiff
  ├── slice_0002.tiff
  └── ...
```

## Requirements

- Python 3.8+
- VisPy >= 0.11.0
- PyQt5 >= 5.15.0
- NumPy >= 1.20.0
- Pillow >= 8.0.0
- OpenGL-capable GPU (recommended)
