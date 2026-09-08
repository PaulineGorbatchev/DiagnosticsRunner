"""Diagnostic B: Sensitivity of posterior mean to conditioning x."""

import os
import numpy as np
import matplotlib.pyplot as plt

from ..utils import sample_posterior
from ..config import DiagnosticsConfig


def run(
    posterior,
    obs: np.ndarray,
    x: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    print("\n================ DIAG B: SENSITIVITY to x ================")

    n_dim = obs.shape[0] if obs.ndim == 1 else obs.shape[-1]
    rng = np.random.default_rng(cfg.seed)
    idxs = rng.choice(x.shape[0], size=cfg.num_x_conds - 1, replace=False)

    x_conds = [obs] + [x[i] for i in idxs]
    labels = ["obs"] + [f"sim_{i}" for i in idxs]

    post_means = []
    post_stds = []

    for k, xc in enumerate(x_conds):
        s = sample_posterior(posterior, xc, cfg.num_samples_sensitivity, cfg.seed + 10 + k)
        post_means.append(np.mean(s, axis=0))
        post_stds.append(np.std(s, axis=0))

    post_means = np.asarray(post_means)
    post_stds = np.asarray(post_stds)
    n_dim = post_means.shape[1]

    results = {"span": [], "post_means": post_means.tolist(), "labels": labels}

    for p in range(n_dim):
        span = post_means[:, p].max() - post_means[:, p].min()
        results["span"].append(span)
        print(f"{cfg.parameter_names[p]}: posterior mean span = {span:.6g}")
        print("  " + ", ".join([f"{labels[i]}={post_means[i, p]:.4g}" for i in range(len(labels))]))

    fig, axes = plt.subplots(1, n_dim, figsize=(12, 4))
    if n_dim == 1:
        axes = [axes]
    for p in range(n_dim):
        axes[p].plot(range(len(labels)), post_means[:, p], marker="o")
        axes[p].set_xticks(range(len(labels)))
        axes[p].set_xticklabels(labels, rotation=30, ha="right")
        axes[p].set_title(f"Posterior mean vs x for {cfg.parameter_names[p]}")
        axes[p].set_ylabel("Posterior mean")
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"posterior_mean_sensitivity_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

    return results
