"""Quantify per-pair region contrast in the two electrode volumes.

For each volume:
  1. Fit k-means(k=3) on an intensity subsample to identify three regions.
  2. Estimate per-class mean (signal) and within-class std (noise) on a large
     sampled volume.
  3. Compute three standard contrast metrics for every pair of classes:
       - CNR (contrast-to-noise ratio) = |mu_A - mu_B| / sqrt((s_A^2 + s_B^2)/2)
           Higher = better; UNITLESS, so directly comparable across images.
           This is the *standard* radiology / X-ray imaging definition.
       - Weber contrast       = |mu_A - mu_B| / max(|mu_A|, |mu_B|)
       - Michelson contrast   = |mu_A - mu_B| / (|mu_A| + |mu_B|)
           These are scale-relative; only meaningful when intensities are
           strictly positive. For signed phase data they can be misleading and
           are reported only for reference.

Output: contrast_compare.png with bar chart + table printed.
"""

import os
import itertools
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "/data/Xiaoyang/phaseimg/20260612/final_compare"
FILES = {
    "phase":  "Reslice of electrode_10x_phase_rec_crop_rot_crop_z125-300.tif",
    "tomo":   "Reslice of electrode_10x_tomo_rec_0106_crop_rot_aligned_125-300.tif",
}
N_FIT_SLICES = 40
FIT_SAMPLE = 500_000
K = 3


def sample_stack(path, n=N_FIT_SLICES):
    with tifffile.TiffFile(path) as tf:
        n_pages = len(tf.pages)
        idx = np.linspace(0, n_pages - 1, min(n, n_pages)).astype(int)
        return np.stack([tf.pages[i].asarray() for i in idx])


def kmeans_1d(x, k=K, n_iter=80, n_restarts=5, seed=0):
    x = np.ascontiguousarray(x.ravel(), dtype=np.float32)
    best_c, best_inertia = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
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
            best_inertia, best_c = inertia, c
    return best_c


def analyse(name, path):
    print(f"\n[{name}] loading {N_FIT_SLICES} slices ...")
    vol = sample_stack(path)
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol_c = np.clip(vol, lo, hi).astype(np.float32)
    flat = vol_c.ravel()
    rng = np.random.default_rng(7)
    fit_x = rng.choice(flat, size=min(FIT_SAMPLE, flat.size), replace=False)
    centers = kmeans_1d(fit_x, k=K)
    print(f"[{name}] centers = {centers}")

    # assign every voxel and compute per-class signal+noise on the full sample
    labels = np.abs(flat[:, None] - centers[None, :]).argmin(axis=1)
    mus = np.array([flat[labels == i].mean() for i in range(K)])
    sigs = np.array([flat[labels == i].std()  for i in range(K)])
    fracs = np.bincount(labels, minlength=K) / labels.size

    # Re-label so that phase-0 is the matrix (largest volume fraction) and
    # phases 1,2 are the minority classes. Makes "matrix vs feature" pairs
    # comparable across both volumes (phase has matrix bright, tomo has matrix
    # dark — the absolute intensity ordering differs but the *roles* don't).
    order = np.argsort(-fracs)         # descending fraction
    mus = mus[order]; sigs = sigs[order]; fracs = fracs[order]
    centers_ordered = centers[order]

    # Pairwise contrast metrics
    pairs = list(itertools.combinations(range(K), 2))
    metrics = {}
    for a, b in pairs:
        diff = abs(mus[a] - mus[b])
        cnr = diff / np.sqrt((sigs[a] ** 2 + sigs[b] ** 2) / 2 + 1e-12)
        max_abs = max(abs(mus[a]), abs(mus[b])) + 1e-12
        weber = diff / max_abs
        michel = diff / (abs(mus[a]) + abs(mus[b]) + 1e-12)
        metrics[(a, b)] = dict(diff=float(diff), cnr=float(cnr),
                               weber=float(weber), michel=float(michel))
    return dict(name=name, centers=centers_ordered, mus=mus, sigs=sigs,
                fracs=fracs, metrics=metrics)


def main():
    results = [analyse(n, os.path.join(DIR, f)) for n, f in FILES.items()]

    # --- Print per-class signal+noise ---
    print("\n=== Per-class signal/noise (class 0 = matrix, 1+2 = features) ===")
    print(f"{'class':<8}{'metric':<12}" + "".join(f"{r['name']:>14}" for r in results))
    for i in range(K):
        for label, key in [("μ (mean)", "mus"), ("σ (noise)", "sigs"),
                           ("frac %", "fracs")]:
            vals = [r[key][i] for r in results]
            fmt = "%14.5g" if "frac" not in label else "%14.2f"
            print(f"  c{i:<5}{label:<12}" + "".join(fmt % (v * 100 if "frac" in label else v) for v in vals))

    # --- Pairwise contrast table ---
    print("\n=== Pairwise contrast (matrix=0, mid=1, rare=2 by volume fraction) ===")
    print(f"{'pair':<8}{'metric':<10}" + "".join(f"{r['name']:>14}" for r in results))
    pairs = list(itertools.combinations(range(K), 2))
    for p in pairs:
        for label, key in [("CNR (↑)", "cnr"), ("Weber (↑)", "weber"),
                           ("Michelson", "michel")]:
            vals = [r["metrics"][p][key] for r in results]
            print(f"  {p[0]}-{p[1]:<5}{label:<10}" + "".join(f"{v:14.4f}" for v in vals))
        print()

    # CNR-based winner per pair
    print("=== CNR winner per pair (which volume better separates this pair) ===")
    for p in pairs:
        winners = sorted(results, key=lambda r: -r["metrics"][p]["cnr"])
        margin = winners[0]["metrics"][p]["cnr"] - winners[1]["metrics"][p]["cnr"]
        print(f"  pair {p}: {winners[0]['name']} wins  (Δ CNR = {margin:+.3f})")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    pair_labels = [f"matrix vs c{p[1]}" if p[0] == 0 else f"c{p[0]} vs c{p[1]}" for p in pairs]
    x = np.arange(len(pairs))
    for i, r in enumerate(results):
        cnrs = [r["metrics"][p]["cnr"] for p in pairs]
        ax.bar(x + i * width, cnrs, width, label=r["name"])
        for j, v in enumerate(cnrs):
            ax.text(x[j] + i * width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(pair_labels)
    ax.set_ylabel("CNR (higher = more distinction)")
    ax.set_title("Pairwise contrast-to-noise ratio between the three k-means phases")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(DIR, "contrast_compare.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
