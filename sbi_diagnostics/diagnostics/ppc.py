"""Approximate posterior predictive check.

No new simulations are run. For each posterior sample, the predicted summary
is the mean of the summaries of the ``ppc_knn`` nearest simulations in
parameter space, plus Gaussian noise of standard deviation ``ppc_noise_std``.
The check compares the mean over the summary vector of these predictions
with that of the observed summary.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from ..config import DiagnosticsConfig


def run(
    samples_np: np.ndarray,
    obs: np.ndarray,
    x: np.ndarray,
    thetas: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    print("\n================ POSTERIOR PREDICTIVE CHECK ================")

    tree = cKDTree(thetas)
    rng = np.random.default_rng(cfg.seed)

    def predict_summary(theta):
        _, idxs = tree.query(theta, k=cfg.ppc_knn)
        mean_summary = np.mean(x[idxs], axis=0)
        return mean_summary + cfg.ppc_noise_std * rng.standard_normal(x.shape[1])

    chosen = rng.choice(samples_np.shape[0], cfg.ppc_num_samples, replace=False)
    ppc_samples = np.array([predict_summary(samples_np[i]) for i in chosen])

    plt.figure(figsize=(9, 6))
    plt.hist(np.mean(ppc_samples, axis=1), bins=30, alpha=0.75,
             edgecolor="black", label="Posterior predictive")
    plt.axvline(np.mean(obs), linestyle="--", linewidth=2, label="Observed")
    plt.xlabel("Mean of the summary vector")
    plt.ylabel("Frequency")
    plt.title(r"Posterior Predictive Check")
    plt.legend()
    plt.grid(alpha=0.7, linestyle=":")
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"posterior_predictive_check_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    return {
        "ppc_mean": float(np.mean(ppc_samples)),
        "obs_mean": float(np.mean(obs)),
    }
