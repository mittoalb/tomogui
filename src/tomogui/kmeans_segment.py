"""K-means segmentation of an electrode tomography volume into 4 classes:
pore, C, S, Ti — applied to each input file SEPARATELY (single-volume,
1D feature = voxel intensity).

Ti is rare (<5%) and brightest in the volume, so cluster centers are
initialized from intensity percentiles to keep Ti from being swallowed by S.
Fitting is done on a stratified random subsample (with oversampling of bright
voxels) for speed and to give the rare Ti class a fair shot. Prediction is
then applied to the full volume.

Usage
-----
    # Segment both files independently with defaults
    python kmeans_segment.py

    # Segment a specific file
    python kmeans_segment.py --input /path/to/volume.tif --out /path/to/labels.tif

Output (per input)
------------------
    Labeled uint8 volume, same shape as input, named <input>_kmeans_labels.tif:
        0 = pore, 1 = C, 2 = S, 3 = Ti
"""

import argparse
import os
import sys

import numpy as np
import tifffile
from sklearn.cluster import KMeans


LABEL_NAMES = {0: "pore", 1: "C", 2: "S", 3: "Ti"}


def load_volume(path):
    vol = tifffile.imread(path)
    if vol.ndim == 2:
        vol = vol[None, :, :]
    return vol.astype(np.float32)


def robust_normalize(vol, lo_pct=0.5, hi_pct=99.5):
    """Map [lo_pct, hi_pct] percentile range to [0, 1], clip outside."""
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = (vol - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def init_centers_from_percentiles(values, percentiles):
    """1D initial cluster centers placed at the requested intensity
    percentiles of the voxel distribution."""
    centers = np.percentile(values, percentiles).reshape(-1, 1)
    return centers.astype(np.float32)


def stratified_subsample(values, n_total, bright_frac=0.3, bright_pct=95.0,
                         rng=None):
    """Random subsample, with a fraction drawn from the brightest tail so the
    rare Ti class is represented. `bright_frac` of the samples come from
    voxels above `bright_pct` percentile."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = values.size
    if n_total >= n:
        return values

    n_bright = int(n_total * bright_frac)
    n_rest = n_total - n_bright

    thr = np.percentile(values, bright_pct)
    bright_mask = values >= thr
    bright_idx = np.flatnonzero(bright_mask)
    rest_idx = np.flatnonzero(~bright_mask)

    n_bright = min(n_bright, bright_idx.size)
    n_rest = min(n_rest, rest_idx.size)

    pick_bright = rng.choice(bright_idx, size=n_bright, replace=False)
    pick_rest = rng.choice(rest_idx, size=n_rest, replace=False)
    pick = np.concatenate([pick_bright, pick_rest])
    return values[pick]


def relabel_by_brightness(labels, centers):
    """Reorder labels so they go pore=0, C=1, S=2, Ti=3 by ascending center
    intensity."""
    order = np.argsort(centers[:, 0])  # darkest -> brightest
    remap = np.zeros_like(order)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label
    return remap[labels], centers[order]


def segment_one(input_path, out_path, n_clusters=4, sample_size=500_000,
                seed=0, save_masks=False):
    print(f"\n=== {input_path} ===")
    vol = load_volume(input_path)
    print(f"Loaded: shape={vol.shape}, dtype={vol.dtype}, "
          f"min={vol.min():.4g}, max={vol.max():.4g}")

    print("Normalizing (robust percentile scaling)...")
    vol_n = robust_normalize(vol)
    values = vol_n.ravel()
    print(f"Total voxels: {values.size:,}")

    rng = np.random.default_rng(seed)
    print(f"Drawing stratified subsample (n={sample_size:,}, "
          "bright_frac=0.30 above 95th pct)...")
    sub = stratified_subsample(values, sample_size, bright_frac=0.30,
                               bright_pct=95.0, rng=rng).reshape(-1, 1)

    # Percentile-based init: pore (1%), C (40%), S (85%), Ti (99.5%)
    init_pcts = [1.0, 40.0, 85.0, 99.5][:n_clusters]
    init_centers = init_centers_from_percentiles(sub.ravel(), init_pcts)
    print(f"Initial centers: {init_centers.ravel()}")

    print(f"Fitting KMeans (k={n_clusters}) on subsample...")
    km = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1,
                max_iter=300, random_state=seed)
    km.fit(sub)
    print(f"Fitted centers: {km.cluster_centers_.ravel()}")

    print("Predicting labels on full volume...")
    labels_flat = km.predict(values.reshape(-1, 1)).astype(np.uint8)
    labels_flat, ordered_centers = relabel_by_brightness(
        labels_flat, km.cluster_centers_)
    labels = labels_flat.reshape(vol.shape)

    total = labels.size
    print("Class fractions:")
    for k in range(n_clusters):
        frac = (labels == k).sum() / total * 100
        print(f"  {k} {LABEL_NAMES.get(k, '?'):>5s}: {frac:6.2f}%  "
              f"center = {ordered_centers[k, 0]:.3f}")

    print(f"Saving labels: {out_path}")
    tifffile.imwrite(out_path, labels)

    if save_masks:
        base, ext = os.path.splitext(out_path)
        for k, name in LABEL_NAMES.items():
            if k >= n_clusters:
                continue
            mask_path = f"{base}_{name}{ext}"
            tifffile.imwrite(mask_path, (labels == k).astype(np.uint8) * 255)
            print(f"  wrote mask: {mask_path}")

    return labels, ordered_centers


def default_out_path(in_path):
    base, ext = os.path.splitext(in_path)
    return f"{base}_kmeans_labels{ext}"


def main():
    base = "/data/Xiaoyang/phaseimg/20260612"
    default_inputs = [
        os.path.join(base, "electrode_10x_phase_rec_crop_rot_crop.tif"),
        os.path.join(base, "electrode_10x_tomo_rec_0106_crop_rot_aligned.tif"),
    ]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", default=None,
                   help="Input volume (tif). Repeat to segment multiple files. "
                        "Defaults to the two electrode volumes in "
                        f"{base}.")
    p.add_argument("--out", default=None,
                   help="Output labeled volume. Only valid with a single "
                        "--input. Defaults to <input>_kmeans_labels.tif.")
    p.add_argument("--k", type=int, default=4, help="Number of clusters")
    p.add_argument("--sample-size", type=int, default=500_000,
                   help="Voxels used to fit KMeans (stratified)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-masks", action="store_true",
                   help="Also write per-class binary masks")
    args = p.parse_args()

    inputs = args.input if args.input else default_inputs

    if args.out is not None and len(inputs) != 1:
        sys.exit("--out can only be used with a single --input")

    for in_path in inputs:
        out_path = args.out if args.out else default_out_path(in_path)
        segment_one(in_path, out_path, n_clusters=args.k,
                    sample_size=args.sample_size, seed=args.seed,
                    save_masks=args.save_masks)


if __name__ == "__main__":
    main()
