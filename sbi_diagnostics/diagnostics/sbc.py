"""Diagnostic C: Simulation-Based Calibration (SBC) rank histograms."""

import os
import numpy as np
import matplotlib.pyplot as plt

from ..utils import sample_posterior
from ..config import DiagnosticsConfig


def run(
    posterior,
    x: np.ndarray,
    thetas: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    if not cfg.do_sbc:
        print("SBC skipped (do_sbc=False).")
        return {}

    print("\n================ DIAG C: APPROX SBC RANKS ================")

    rng = np.random.default_rng(cfg.seed)
    n_cases = min(cfg.sbc_num_cases, x.shape[0])
    sbc_idxs = rng.choice(x.shape[0], size=n_cases, replace=False)
    n_dim = thetas.shape[1]

    ranks = np.zeros((n_cases, n_dim), dtype=int)

    for i, idx in enumerate(sbc_idxs):
        s = sample_posterior(posterior, x[idx], cfg.sbc_post_samples, cfg.seed + 1000 + i)
        theta_true = thetas[idx]
        for p in range(n_dim):
            ranks[i, p] = int(np.sum(s[:, p] < theta_true[p]))

    fig, axes = plt.subplots(1, n_dim, figsize=(11, 4))
    if n_dim == 1:
        axes = [axes]
    for p in range(n_dim):
        axes[p].hist(ranks[:, p], bins=20, density=True)
        axes[p].set_title(f"SBC rank histogram: {cfg.parameter_names[p]}")
        axes[p].set_xlabel("rank")
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"sbc_ranks_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

    return {"ranks": ranks.tolist()}
