"""
Smoke test for sbi_diagnostics using a mock posterior.
Run on the server where jax + jaxili are installed:

    python test_diagnostics.py
"""
import os
import sys
import tempfile
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Minimal mock posterior that mimics the JAXILI posterior API
# ---------------------------------------------------------------------------
class MockPosterior:
    """Returns samples from a 2D Gaussian centred near the conditioning x mean."""

    def sample(self, x, num_samples, key):
        mean = jnp.array([jnp.mean(x) * 0.3 + 0.25, jnp.mean(x) * 0.1 + 0.65])
        std  = jnp.array([0.05, 0.03])
        noise = jax.random.normal(key, shape=(num_samples, 2))
        return mean + std * noise


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
N, D, N_PARAMS = 500, 50, 2
rng = np.random.default_rng(0)

thetas = rng.uniform([0.2, 0.6], [0.4, 0.8], size=(N, N_PARAMS)).astype(np.float32)
x      = rng.normal(size=(N, D)).astype(np.float32)
obs    = rng.normal(size=D).astype(np.float32)

posterior = MockPosterior()

# ---------------------------------------------------------------------------
# Run diagnostics
# ---------------------------------------------------------------------------
from sbi_diagnostics import DiagnosticsConfig, DiagnosticsRunner

with tempfile.TemporaryDirectory() as tmpdir:
    cfg = DiagnosticsConfig(
        output_dir=tmpdir,
        data_variant="smoke_test",
        resolution="test",
        parameter_names=[r"$\Omega_m$", r"$h$"],
        fiducial={"Om": 0.3, "h": 0.7},
        seed=42,
        use_latex=False,           # avoid requiring LaTeX in the test env
        # Speed things up
        num_samples_posterior=500,
        num_samples_sensitivity=200,
        num_x_conds=3,
        do_sbc=True,
        sbc_num_cases=20,
        sbc_post_samples=100,
        credibility_levels=[0.5, 0.9],
        coverage_num_test=20,
        coverage_post_samples=100,
        ppc_num_samples=10,
        recovery_num_test=10,
        recovery_post_samples=100,
        final_num_samples=500,
        save_samples=True,
    )

    runner = DiagnosticsRunner(posterior, x, thetas, obs, cfg)

    print("=== Running all diagnostics ===")
    results = runner.run_all()

    # Check every expected PDF was produced
    expected = [
        "prior_vs_posterior_smoke_test.pdf",
        "posterior_mean_sensitivity_smoke_test.pdf",
        "sbc_ranks_smoke_test.pdf",
        "coverage_probability_smoke_test.pdf",
        "posterior_predictive_check_smoke_test.pdf",
        "parameter_recovery_smoke_test.pdf",
        "triangle_plot_smoke_test.pdf",
        "diagnostics_summary.json",
        "samples_smoke_test.npy",
    ]

    out_dir = cfg.dir_path
    missing = [f for f in expected if not os.path.exists(os.path.join(out_dir, f))]

    if missing:
        print(f"\nFAIL — missing output files: {missing}")
        sys.exit(1)
    else:
        print(f"\nPASS — all {len(expected)} output files produced.")
        print(f"Keys in results: {list(results.keys())}")
