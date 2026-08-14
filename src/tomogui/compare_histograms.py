"""Compare two 3D TIF stacks for 3-phase segmentation suitability.

Metrics (THREE classes assumed):
- Multi-Otsu separability: between-class variance / total variance using 2 thresholds
- 3-peak / 2-valley contrast: average dip depth between adjacent peaks
- 3-component GMM: pairwise Bayes error (adjacent classes) + mean Mahalanobis separation
- Multi-threshold consensus: agreement between multi-Otsu, k-means (k=3), and Kapur entropy
- Multi-Otsu stability under additive noise
- Per-class spatial fragmentation after multi-Otsu segmentation

Saves: histogram_compare.png plus prints a small table.
"""

import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm
from skimage.filters import threshold_multiotsu
from skimage.measure import label

N_CLASSES = 3

DIR = "/data/Xiaoyang/phaseimg/20260612/final_compare"
FILES = {
    "phase":  "Reslice of electrode_10x_phase_rec_crop_rot_crop_z125-300.tif",
    "tomo":   "Reslice of electrode_10x_tomo_rec_0106_crop_rot_aligned_125-300.tif",
}
N_SAMPLE_SLICES = 40          # subsample to keep memory bounded
BINS = 512


def sample_stack(path, n=N_SAMPLE_SLICES):
    with tifffile.TiffFile(path) as tf:
        n_pages = len(tf.pages)
        idx = np.linspace(0, n_pages - 1, min(n, n_pages)).astype(int)
        arr = np.stack([tf.pages[i].asarray() for i in idx])
    return arr


def multiotsu_separability(values, n_classes=N_CLASSES, bins=BINS, max_samples=500_000):
    # Multi-Otsu separability: between-class variance / total variance after
    # cutting the histogram into n_classes via skimage's multi-Otsu.
    x = values.ravel()
    if x.size > max_samples:
        x = np.random.default_rng(3).choice(x, max_samples, replace=False)
    thresholds = threshold_multiotsu(x, classes=n_classes, nbins=bins)
    edges = np.concatenate([[x.min()], thresholds, [x.max()]])
    mu_total = x.mean()
    var_total = x.var()
    sigma_b2 = 0.0
    for i in range(n_classes):
        mask = (x >= edges[i]) & (x <= edges[i + 1])
        w = mask.mean()
        if w > 0:
            sigma_b2 += w * (x[mask].mean() - mu_total) ** 2
    return float(sigma_b2 / (var_total + 1e-12)), thresholds


def kapur_multilevel(values, n_classes=N_CLASSES, bins=BINS, max_samples=300_000):
    # Kapur's maximum-entropy multi-level thresholding (alternative to Otsu).
    # Pick (n_classes-1) thresholds that maximize sum of class entropies.
    x = values.ravel()
    if x.size > max_samples:
        x = np.random.default_rng(4).choice(x, max_samples, replace=False)
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    if n_classes == 3:
        # exhaustive 2D search on coarse grid for tractability
        best_H, best = -np.inf, (bins // 3, 2 * bins // 3)
        step = max(1, bins // 64)
        for i in range(step, bins - 2 * step, step):
            for j in range(i + step, bins - step, step):
                slices = [p[:i], p[i:j], p[j:]]
                H = 0.0
                ok = True
                for s in slices:
                    w = s.sum()
                    if w < 1e-9:
                        ok = False; break
                    q = s[s > 0] / w
                    H += -(q * np.log(q)).sum()
                if ok and H > best_H:
                    best_H, best = H, (i, j)
        return np.array([centers[best[0]], centers[best[1]]])
    raise NotImplementedError("Kapur only wired for 3 classes here")


def kmeans_1d(values, k=N_CLASSES, n_iter=50, max_samples=200_000):
    # 1D k-means (Lloyd's). Returns sorted (k-1) midpoints between centroids
    # as thresholds.
    x = values.ravel()
    if x.size > max_samples:
        x = np.random.default_rng(5).choice(x, max_samples, replace=False)
    c = np.quantile(x, np.linspace(1 / (2 * k), 1 - 1 / (2 * k), k))
    for _ in range(n_iter):
        d = np.abs(x[:, None] - c[None, :])
        labels = d.argmin(axis=1)
        new_c = np.array([x[labels == i].mean() if (labels == i).any() else c[i]
                          for i in range(k)])
        if np.allclose(new_c, c):
            break
        c = new_c
    c = np.sort(c)
    return 0.5 * (c[:-1] + c[1:])


def peak_valley_contrast_3(values, n_peaks=N_CLASSES, bins=BINS):
    # Detect up to n_peaks dominant peaks, return mean (peak,valley) contrast
    # across adjacent pairs.
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(hist, np.ones(7) / 7, mode="same")
    peaks, _ = find_peaks(smooth, prominence=smooth.max() * 0.01, distance=8)
    if len(peaks) < 2:
        return 0.0, peaks, centers, smooth
    # keep the n_peaks most prominent, in left-to-right order
    keep = np.sort(peaks[np.argsort(smooth[peaks])[::-1][:n_peaks]])
    contrasts = []
    for a, b in zip(keep[:-1], keep[1:]):
        valley = smooth[a:b].min()
        mp = min(smooth[a], smooth[b])
        contrasts.append((mp - valley) / (mp + 1e-12))
    return float(np.mean(contrasts)), keep, centers, smooth


def fit_k_gaussians_em(x, k=N_CLASSES, n_iter=120, tol=1e-6):
    # 1D k-component Gaussian mixture via EM. Init means at evenly-spaced quantiles.
    mu = np.percentile(x, np.linspace(100 / (2 * k), 100 - 100 / (2 * k), k))
    s = np.full(k, x.std() + 1e-9)
    w = np.full(k, 1.0 / k)
    prev_ll = -np.inf
    for _ in range(n_iter):
        p = np.stack([w[i] * norm.pdf(x, mu[i], s[i]) for i in range(k)], axis=1)
        p_sum = p.sum(axis=1, keepdims=True) + 1e-300
        r = p / p_sum
        Nk = r.sum(axis=0) + 1e-9
        w = Nk / x.size
        mu = (r * x[:, None]).sum(axis=0) / Nk
        s = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk) + 1e-9
        order = np.argsort(mu)
        mu, s, w = mu[order], s[order], w[order]
        ll = np.log(p_sum).sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return mu, s, w


def gmm_bayes_error_k(values, k=N_CLASSES, max_samples=200_000):
    # Fit a k-component GMM; estimate the Bayes error by Monte-Carlo: for each
    # sample, optimal class = argmax(posterior); error = mass whose argmax
    # disagrees with the ground-truth component it was drawn from. With a
    # fitted model on real data we instead use: 1 - mean(max(posterior)),
    # which equals the expected misclassification rate under that model.
    x = values.ravel()
    if x.size > max_samples:
        x = np.random.default_rng(0).choice(x, max_samples, replace=False)
    x = x.astype(np.float64)
    mu, s, w = fit_k_gaussians_em(x, k=k)
    p = np.stack([w[i] * norm.pdf(x, mu[i], s[i]) for i in range(k)], axis=1)
    post = p / (p.sum(axis=1, keepdims=True) + 1e-300)
    err = float(1.0 - post.max(axis=1).mean())
    # Mean pairwise Mahalanobis separation between adjacent components.
    pairs = [abs(mu[i + 1] - mu[i]) / np.sqrt((s[i] ** 2 + s[i + 1] ** 2) / 2 + 1e-12)
             for i in range(k - 1)]
    return err, float(np.mean(pairs)), (mu, s, w)


def multi_threshold_consensus_k(values, k=N_CLASSES):
    # Compare three principled multi-level threshold finders. CV across them is
    # the consensus measure (low = robust).
    methods = {
        "multi-Otsu": multiotsu_separability(values, n_classes=k)[1],
        "k-means":    kmeans_1d(values, k=k),
        "Kapur":      kapur_multilevel(values, n_classes=k),
    }
    # CV per threshold position (low / mid)
    mat = np.stack([np.asarray(t) for t in methods.values()])     # (n_methods, k-1)
    cvs = mat.std(axis=0) / (np.abs(mat.mean(axis=0)) + 1e-12)
    return methods, float(cvs.mean())


def multiclass_spatial_coherence(vol, thresholds, min_frag_voxels=50):
    # For each class produced by multi-Otsu, what fraction of its voxels live
    # in tiny disconnected fragments? Returns the class-averaged fragmentation.
    edges = np.concatenate([[-np.inf], np.asarray(thresholds), [np.inf]])
    frags = []
    for i in range(len(edges) - 1):
        mask = (vol >= edges[i]) & (vol < edges[i + 1])
        if mask.sum() < min_frag_voxels:
            frags.append(np.nan); continue
        lbl = label(mask, connectivity=1)
        counts = np.bincount(lbl.ravel())[1:]
        if counts.size == 0:
            frags.append(np.nan); continue
        big = counts[counts >= min_frag_voxels].sum()
        frags.append(1.0 - big / counts.sum())
    return float(np.nanmean(frags)), frags


def multiotsu_stability(values, k=N_CLASSES, n_trials=5, noise_frac=0.05):
    # How much do the two multi-Otsu thresholds drift when 5% Gaussian noise
    # is added? Returns mean CV across the (k-1) thresholds.
    rng = np.random.default_rng(2)
    base = rng.choice(values.ravel(), min(values.size, 200_000), replace=False)
    sigma = noise_frac * (values.max() - values.min())
    ts = np.array([threshold_multiotsu(base + rng.normal(0, sigma, base.shape),
                                       classes=k) for _ in range(n_trials)])
    cvs = ts.std(axis=0) / (np.abs(ts.mean(axis=0)) + 1e-12)
    return float(cvs.mean())


def analyze(name, path):
    print(f"\n[{name}] loading {N_SAMPLE_SLICES} slices from {os.path.basename(path)} ...")
    vol = sample_stack(path)
    # clip extreme outliers so the histogram isn't dominated by tails
    lo, hi = np.percentile(vol, [0.5, 99.5])
    v = vol[(vol >= lo) & (vol <= hi)].astype(np.float32)
    sep_otsu, t_otsu = multiotsu_separability(v, n_classes=N_CLASSES)
    contrast, peaks, centers, smooth = peak_valley_contrast_3(v, n_peaks=N_CLASSES)
    bayes_err, mahal, gmm_params = gmm_bayes_error_k(v, k=N_CLASSES)
    thr_dict, thr_cv = multi_threshold_consensus_k(v, k=N_CLASSES)
    frag, frags_per_class = multiclass_spatial_coherence(vol, t_otsu)
    stability = multiotsu_stability(v, k=N_CLASSES)
    return {
        "name": name, "values": v, "otsu_sep": sep_otsu, "t_otsu": t_otsu,
        "valley_contrast": contrast,
        "peaks": peaks, "centers": centers, "smooth": smooth,
        "shape": vol.shape, "range": (float(vol.min()), float(vol.max())),
        "bayes_err": bayes_err, "mahal": mahal, "gmm_params": gmm_params,
        "thresholds": thr_dict, "thr_cv": thr_cv,
        "fragmentation": frag, "frags_per_class": frags_per_class,
        "stability": stability,
    }


def main():
    results = [analyze(n, os.path.join(DIR, f)) for n, f in FILES.items()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # Top row: histogram + 3-component GMM fit overlay + multi-Otsu thresholds.
    for ax, r in zip(axes[0], results):
        ax.fill_between(r["centers"], r["smooth"], alpha=0.35)
        ax.plot(r["centers"], r["smooth"], lw=1)
        mu, s, w = r["gmm_params"]
        bin_w = r["centers"][1] - r["centers"][0]
        n_eff = r["smooth"].sum() * bin_w
        pdf = sum(w[i] * norm.pdf(r["centers"], mu[i], s[i]) for i in range(len(mu))) * n_eff
        ax.plot(r["centers"], pdf, "k-", lw=1.2, label=f"{len(mu)}-comp GMM")
        for i, t in enumerate(r["t_otsu"]):
            ax.axvline(t, color="tab:red", ls="--", lw=1,
                       label=f"multi-Otsu t{i+1}={t:.2g}")
        ax.set_title(
            f"{r['name']}\n"
            f"Bayes err={r['bayes_err']:.3f}  Mahal={r['mahal']:.2f}  "
            f"thr-CV={r['thr_cv']:.3f}\n"
            f"multi-Otsu sep={r['otsu_sep']:.3f}  valley={r['valley_contrast']:.3f}  "
            f"frag={r['fragmentation']:.3f}  stab={r['stability']:.3f}"
        )
        ax.set_xlabel("intensity"); ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper left")

    # Bottom row: grouped barplot — each method gives (k-1) thresholds.
    for ax, r in zip(axes[1], results):
        method_names = list(r["thresholds"].keys())
        n_methods = len(method_names)
        n_thr = len(r["t_otsu"])
        width = 0.8 / n_methods
        x_pos = np.arange(n_thr)
        cmap = plt.get_cmap("tab10")
        for i, m in enumerate(method_names):
            ax.bar(x_pos + i * width, r["thresholds"][m], width,
                   label=m, color=cmap(i))
        ax.set_xticks(x_pos + width * (n_methods - 1) / 2)
        ax.set_xticklabels([f"t{i+1}" for i in range(n_thr)])
        ax.set_title(f"{r['name']} — thresholds by method (tight cluster = robust)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(DIR, "histogram_compare.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")

    print(f"\n=== Summary ({N_CLASSES}-phase segmentation) ===")
    print(f"{'metric':<38}" + "".join(f"{r['name']:>14}" for r in results))
    rows = [
        ("multi-Otsu separability  (higher better)", "otsu_sep", "%.4f"),
        ("Mean adjacent-peak valley contrast  (higher)", "valley_contrast", "%.4f"),
        (f"{N_CLASSES}-comp GMM Bayes error  (LOWER better)", "bayes_err", "%.4f"),
        ("Mean GMM Mahalanobis sep  (higher)", "mahal", "%.4f"),
        ("Threshold CV across methods  (LOWER)", "thr_cv", "%.4f"),
        ("Multi-Otsu stability under noise  (LOWER)", "stability", "%.4f"),
        ("Mean per-class fragmentation  (LOWER)", "fragmentation", "%.4f"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<38}" + "".join(f"{(fmt % r[key]):>14}" for r in results))

    print("\nMulti-Otsu thresholds:")
    for r in results:
        print(f"  {r['name']:<8} {np.array2string(r['t_otsu'], precision=4)}")

    def score(r):
        return (r["otsu_sep"] + r["valley_contrast"] + 0.2 * r["mahal"]
                + (1 - r["bayes_err"]) + (1 - r["fragmentation"])
                + 1 / (1 + r["thr_cv"]) + 1 / (1 + r["stability"]))
    winner = max(results, key=score)
    print("\nComposite scores:")
    for r in results:
        print(f"  {r['name']:<8} {score(r):.3f}")
    print(f"\n>>> Recommended for {N_CLASSES}-phase segmentation: {winner['name']}")


if __name__ == "__main__":
    main()
