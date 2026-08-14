"""k-means (k=3) segmentation on the two electrode volumes, side-by-side.

For each volume:
  1. Fit 1D k-means with k=3 on a large intensity subsample.
  2. Apply the fitted centers to a representative mid-slice and to a sampled
     subset of slices to estimate per-phase volume fractions.
  3. Save: original slice | k-means labels (top: phase; bottom: tomo).

Output: kmeans_segment_compare.png in the same directory.
"""

import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DIR = "/data/Xiaoyang/phaseimg/20260612/final_compare"
FILES = {
    "phase":  "Reslice of electrode_10x_phase_rec_crop_rot_crop_z125-300.tif",
    "tomo":   "Reslice of electrode_10x_tomo_rec_0106_crop_rot_aligned_125-300.tif",
}
N_FIT_SLICES = 40           # sampled slices for fitting centers + volume fractions
FIT_SAMPLE = 500_000        # voxel subsample for the actual k-means iterations
K = 3
N_RESTARTS = 5              # k-means is sensitive to init; pick best inertia
SEG_CMAP = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])  # phase 1/2/3


def sample_stack(path, n=N_FIT_SLICES):
    with tifffile.TiffFile(path) as tf:
        n_pages = len(tf.pages)
        idx = np.linspace(0, n_pages - 1, min(n, n_pages)).astype(int)
        arr = np.stack([tf.pages[i].asarray() for i in idx])
        mid = tf.pages[n_pages // 2].asarray()
    return arr, mid


def kmeans_1d(x, k=K, n_iter=80, n_restarts=N_RESTARTS, seed=0):
    """Lloyd's algorithm in 1D with k-means++ initialization."""
    x = np.ascontiguousarray(x.ravel(), dtype=np.float32)
    best_centers, best_inertia = None, np.inf
    rng = np.random.default_rng(seed)
    for trial in range(n_restarts):
        # k-means++ init
        centers = [float(rng.choice(x))]
        for _ in range(k - 1):
            d2 = np.min((x[:, None] - np.array(centers)[None, :]) ** 2, axis=1)
            probs = d2 / (d2.sum() + 1e-12)
            centers.append(float(rng.choice(x, p=probs)))
        c = np.sort(np.array(centers))
        for _ in range(n_iter):
            labels = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
            new_c = np.array([x[labels == i].mean() if (labels == i).any() else c[i]
                              for i in range(k)])
            new_c = np.sort(new_c)
            if np.allclose(new_c, c):
                break
            c = new_c
        inertia = float(((x - c[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_centers = inertia, c
    return best_centers


def apply_centers(volume, centers):
    """Assign each voxel to its nearest center → labels in [0, K-1]."""
    v = volume[..., None]
    d = np.abs(v - centers[None, ...])
    return d.argmin(axis=-1).astype(np.uint8)


def segment_one(name, path):
    print(f"\n[{name}] loading {N_FIT_SLICES} slices ...")
    vol, mid = sample_stack(path)
    # clip extreme outliers so a few stray voxels don't drag a centroid
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol_c = np.clip(vol, lo, hi)
    mid_c = np.clip(mid, lo, hi)

    print(f"[{name}] fitting k=3 k-means on {FIT_SAMPLE} voxels ...")
    rng = np.random.default_rng(7)
    flat = vol_c.ravel()
    x = rng.choice(flat, size=min(FIT_SAMPLE, flat.size), replace=False)
    centers = kmeans_1d(x, k=K)
    print(f"[{name}] centers = {centers}")

    labels_mid = apply_centers(mid_c, centers)
    labels_vol = apply_centers(vol_c, centers)
    fractions = np.bincount(labels_vol.ravel(), minlength=K) / labels_vol.size
    print(f"[{name}] volume fractions = "
          + ", ".join(f"phase{i}={fractions[i]*100:.2f}%" for i in range(K)))

    return {
        "name": name, "centers": centers, "fractions": fractions,
        "mid_slice": mid_c, "mid_labels": labels_mid,
        "vmin": float(lo), "vmax": float(hi),
    }


def main():
    results = [segment_one(n, os.path.join(DIR, f)) for n, f in FILES.items()]

    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))
    for row, r in enumerate(results):
        # column 0: original mid-slice
        ax = axes[row, 0]
        im = ax.imshow(r["mid_slice"], cmap="gray", vmin=r["vmin"], vmax=r["vmax"])
        ax.set_title(f"{r['name']} — original mid-slice")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # column 1: k-means labels
        ax = axes[row, 1]
        ax.imshow(r["mid_labels"], cmap=SEG_CMAP, vmin=0, vmax=K - 1,
                  interpolation="nearest")
        title = (f"{r['name']} — k-means(k=3) labels\n"
                 f"centers={np.array2string(r['centers'], precision=4)}\n"
                 + " | ".join(f"p{i}={r['fractions'][i]*100:.1f}%" for i in range(K)))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # column 2: histogram with centers + decision boundaries
        ax = axes[row, 2]
        vals = r["mid_slice"].ravel()
        ax.hist(vals, bins=256, color="gray", alpha=0.7)
        for i, c in enumerate(r["centers"]):
            ax.axvline(c, color=SEG_CMAP(i), lw=2, label=f"center {i}={c:.3g}")
        # decision boundaries are midpoints between adjacent centers
        for c1, c2 in zip(r["centers"][:-1], r["centers"][1:]):
            ax.axvline(0.5 * (c1 + c2), color="k", ls="--", lw=1)
        ax.set_title(f"{r['name']} — histogram + centers + boundaries")
        ax.set_xlabel("intensity"); ax.set_ylabel("count")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(DIR, "kmeans_segment_compare.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
