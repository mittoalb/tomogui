"""Apply tomocupy's Paganin phase retrieval to a single 2D image.

The input image should be the flat-field-normalized projection
(I = projection / flat), NOT the -log of it. The output is the
phase-retrieved projection in the same convention used by tomocupy
(i.e. -log is applied AFTER paganin in the normal pipeline).

Example
-------
    python paganin_2d.py input.tif output.tif \
        --pixel-size 1.17 --distance 50 --energy 25 \
        --alpha 1e-3 --method paganin

    # Generalized Paganin
    python paganin_2d.py input.tif output.tif \
        --pixel-size 1.17 --distance 50 --energy 25 \
        --method Gpaganin --delta-beta 1000 --W 2.0
"""

import argparse
import os
import sys

import numpy as np
import cupy as cp
import tifffile

from tomocupy.processing.retrieve_phase import paganin_filter


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path)
    if ext == ".npy":
        return np.load(path)
    try:
        from PIL import Image
        return np.array(Image.open(path))
    except Exception as e:
        raise ValueError(f"Unsupported input format '{ext}': {e}")


def save_image(path, arr):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        tifffile.imwrite(path, arr.astype(np.float32))
    elif ext == ".npy":
        np.save(path, arr)
    else:
        raise ValueError(f"Unsupported output format '{ext}'")


def paganin_2d(
    image,
    pixel_size_um=1.17,
    distance_mm=50.0,
    energy_kev=25.0,
    alpha=1e-3,
    method="paganin",
    delta_beta=1000.0,
    W_um=2.0,
    pad=True,
):
    """Apply Paganin phase retrieval to a 2D numpy image.

    Parameters
    ----------
    image : (H, W) ndarray
        Flat-field-normalized projection (intensity ratio, not -log).
    pixel_size_um : float
        Detector pixel size in micrometers.
    distance_mm : float
        Sample-to-detector distance in millimeters.
    energy_kev : float
        X-ray energy in keV.
    alpha : float
        Regularization (standard Paganin).
    method : {"paganin", "Gpaganin"}
        Standard or generalized Paganin.
    delta_beta : float
        delta/beta ratio (generalized Paganin only).
    W_um : float
        Characteristic transverse length in micrometers (generalized only).
    pad : bool
        Zero-pad before FFT to suppress wrap-around.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    # tomocupy expects float32 cupy arrays of shape (nproj, H, W)
    data = cp.asarray(image, dtype=cp.float32)[None, :, :]

    # Match the unit conversions used in tomocupy/processing/proc_functions.py:
    #   pixel_size: um -> cm  (*1e-4)
    #   distance:   mm -> cm  (/10)
    #   W:          um -> cm  (*1e-4)
    out = paganin_filter(
        data,
        pixel_size=pixel_size_um * 1e-4,
        dist=distance_mm / 10.0,
        energy=energy_kev,
        alpha=alpha,
        method=method,
        db=delta_beta,
        W=W_um * 1e-4,
        pad=pad,
    )
    return cp.asnumpy(out[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Input image (.tif/.tiff/.npy/.png/...)")
    p.add_argument("output", help="Output image (.tif/.tiff/.npy)")
    p.add_argument("--pixel-size", type=float, default=1.17,
                   help="Detector pixel size in micrometers (default: 1.17)")
    p.add_argument("--distance", type=float, default=50.0,
                   help="Sample-detector distance in mm (default: 50)")
    p.add_argument("--energy", type=float, default=25.0,
                   help="X-ray energy in keV (default: 25)")
    p.add_argument("--alpha", type=float, default=1e-3,
                   help="Regularization for standard Paganin (default: 1e-3)")
    p.add_argument("--method", choices=["paganin", "Gpaganin"],
                   default="paganin", help="Phase retrieval method")
    p.add_argument("--delta-beta", type=float, default=1000.0,
                   help="delta/beta for Gpaganin (default: 1000)")
    p.add_argument("--W", type=float, default=2.0,
                   help="Characteristic transverse length in um, Gpaganin (default: 2.0)")
    p.add_argument("--no-pad", action="store_true",
                   help="Disable zero-padding")
    p.add_argument("--minus-log", action="store_true",
                   help="Apply -log after phase retrieval (typical for tomography)")
    args = p.parse_args()

    img = load_image(args.input)
    print(f"Loaded {args.input}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min():.4g}, max={img.max():.4g}")

    if img.ndim != 2:
        sys.exit(f"Expected 2D image, got shape {img.shape}")

    out = paganin_2d(
        img,
        pixel_size_um=args.pixel_size,
        distance_mm=args.distance,
        energy_kev=args.energy,
        alpha=args.alpha,
        method=args.method,
        delta_beta=args.delta_beta,
        W_um=args.W,
        pad=not args.no_pad,
    )

    if args.minus_log:
        out = -np.log(np.clip(out, 1e-6, None))

    save_image(args.output, out)
    print(f"Saved {args.output}: shape={out.shape}, dtype={out.dtype}, "
          f"min={out.min():.4g}, max={out.max():.4g}")


if __name__ == "__main__":
    main()
