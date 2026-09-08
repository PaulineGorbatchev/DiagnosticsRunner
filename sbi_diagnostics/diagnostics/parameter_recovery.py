"""Parameter recovery: posterior mean vs true theta."""

import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from ..config import DiagnosticsConfig


def run(
    posterior,
    x: np.ndarray,
    thetas: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    print("\n================ PARAMETER RECOVERY ================")

    n_test = min(cfg.recovery_num_test, x.shape[0])
    test_x = x[-n_test:]
    test_thetas = thetas[-n_test:]
    n_dim = thetas.shape[1]

    posterior_means = []
    for i in range(n_test):
        key = jax.random.PRNGKey(i + 999)
        s = posterior.sample(x=test_x[i], num_samples=cfg.recovery_post_samples, key=key)
        posterior_means.append(np.mean(np.asarray(s), axis=0))

    posterior_means = np.array(posterior_means)

    fig, axes = plt.subplots(1, n_dim, figsize=(7 * n_dim, 5))
    if n_dim == 1:
        axes = [axes]

    for p in range(n_dim):
        axes[p].scatter(test_thetas[:, p], posterior_means[:, p], alpha=0.7, edgecolors="k")
        mn = min(test_thetas[:, p].min(), posterior_means[:, p].min())
        mx = max(test_thetas[:, p].max(), posterior_means[:, p].max())
        axes[p].plot([mn, mx], [mn, mx], "r--")
        axes[p].set_xlabel(f"True {cfg.parameter_names[p]}")
        axes[p].set_ylabel(f"Posterior Mean {cfg.parameter_names[p]}")
        axes[p].grid(alpha=0.7, linestyle=":")

    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"parameter_recovery_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

    residuals = posterior_means - test_thetas
    return {
        "rmse": [float(np.sqrt(np.mean(residuals[:, p] ** 2))) for p in range(n_dim)],
        "bias": [float(np.mean(residuals[:, p])) for p in range(n_dim)],
    }
