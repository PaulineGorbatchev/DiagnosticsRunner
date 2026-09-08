"""Coverage probability diagnostic."""

import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from ..config import DiagnosticsConfig


def compute_coverage(
    posterior,
    test_x: np.ndarray,
    test_thetas: np.ndarray,
    credibility_levels,
    num_samples: int = 2000,
) -> np.ndarray:
    n_test = test_x.shape[0]
    n_params = test_thetas.shape[1]
    coverages = np.zeros((len(credibility_levels), n_params))

    for i, cred in enumerate(credibility_levels):
        counts = np.zeros(n_params)
        for j in range(n_test):
            key = jax.random.PRNGKey(j + 1234)
            s = posterior.sample(x=test_x[j], num_samples=num_samples, key=key)
            for p in range(n_params):
                lo = np.percentile(s[:, p], 50 * (1 - cred))
                hi = np.percentile(s[:, p], 50 * (1 + cred))
                if lo <= test_thetas[j, p] <= hi:
                    counts[p] += 1
        coverages[i] = counts / n_test

    return coverages


def run(
    posterior,
    x: np.ndarray,
    thetas: np.ndarray,
    cfg: DiagnosticsConfig,
) -> dict:
    print("\n================ COVERAGE PROBABILITY ================")

    n_test = min(cfg.coverage_num_test, x.shape[0])
    test_x = x[:n_test]
    test_thetas = thetas[:n_test]
    n_dim = thetas.shape[1]
    markers = ["o", "s", "^", "D"]

    coverages = compute_coverage(
        posterior, test_x, test_thetas,
        cfg.credibility_levels, cfg.coverage_post_samples
    )

    plt.figure(figsize=(9, 6))
    for i in range(n_dim):
        plt.plot(
            cfg.credibility_levels, coverages[:, i],
            marker=markers[i % len(markers)],
            linewidth=2, markersize=7,
            label=cfg.parameter_names[i],
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Ideal")
    plt.xlabel("Credibility Level")
    plt.ylabel("Coverage Probability")
    plt.title(r"Coverage Probability Diagnostic")
    plt.legend()
    plt.grid(alpha=0.7, linestyle=":")
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"coverage_probability_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    return {"credibility_levels": cfg.credibility_levels, "coverages": coverages.tolist()}
