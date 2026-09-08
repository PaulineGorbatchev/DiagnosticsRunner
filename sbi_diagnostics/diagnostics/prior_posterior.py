"""Diagnostic A: Prior vs Posterior shrinkage and distance metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from ..utils import summarize_1d, sample_posterior, sample_prior_resample_thetas, mmd_rbf
from ..config import DiagnosticsConfig


def run(
    posterior,
    obs: np.ndarray,
    thetas: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    print("\n================ DIAG A: PRIOR vs POSTERIOR (obs) ================")

    post_obs = sample_posterior(posterior, obs, cfg.num_samples_posterior, cfg.seed)
    prior_s = sample_prior_resample_thetas(thetas, cfg.num_samples_posterior, cfg.seed + 1)

    n_dim = thetas.shape[1]
    results = {"shrinkage_std": [], "shrinkage_iqr": [], "wasserstein": []}

    for p in range(n_dim):
        pm, ps, pi = summarize_1d(prior_s[:, p])
        qm, qs, qi = summarize_1d(post_obs[:, p])

        std_ratio = qs / (ps + 1e-12)
        iqr_ratio = qi / (pi + 1e-12)
        results["shrinkage_std"].append(std_ratio)
        results["shrinkage_iqr"].append(iqr_ratio)

        print(f"\nParameter {p} ({cfg.parameter_names[p]}):")
        print(f"  prior   mean={pm:.6g}  std={ps:.6g}  IQR={pi:.6g}")
        print(f"  post    mean={qm:.6g}  std={qs:.6g}  IQR={qi:.6g}")
        print(
            f"  shrinkage: std_post/std_prior={std_ratio:.3f}, "
            f"IQR_post/IQR_prior={iqr_ratio:.3f}"
        )

    mmd2, gamma = mmd_rbf(post_obs, prior_s, seed=cfg.seed)
    results["mmd2"] = mmd2
    print(f"\nMMD^2(post, prior) = {mmd2:.6g} (RBF gamma={gamma:.3e})")

    for p in range(n_dim):
        w1 = wasserstein_distance(post_obs[:, p], prior_s[:, p])
        results["wasserstein"].append(w1)
        print(f"Wasserstein-1({cfg.parameter_names[p]}) = {w1:.6g}")

    # Plot
    fig, axes = plt.subplots(1, n_dim, figsize=(11, 4))
    if n_dim == 1:
        axes = [axes]
    for p in range(n_dim):
        axes[p].hist(prior_s[:, p], bins=60, alpha=0.5, density=True, label="Prior (resampled)")
        axes[p].hist(post_obs[:, p], bins=60, alpha=0.5, density=True, label="Posterior (x=obs)")
        axes[p].set_title(cfg.parameter_names[p])
        axes[p].legend()
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"prior_vs_posterior_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

    return results
