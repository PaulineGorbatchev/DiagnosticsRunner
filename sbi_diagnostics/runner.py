"""DiagnosticsRunner: orchestrates all diagnostics in one call."""

import os
import json
import shutil
import warnings
import jax
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .config import DiagnosticsConfig
from .diagnostics import (
    prior_posterior,
    sensitivity,
    sbc,
    coverage,
    ppc,
    parameter_recovery,
    validation_loss,
    triangle_plot,
)


class DiagnosticsRunner:
    """
    Run all SBI diagnostics for a trained NPE posterior.

    Parameters
    ----------
    posterior   : trained JAXILI posterior object
    x           : simulation summaries, shape (N, d)
    thetas      : simulation parameters, shape (N, n_params)
    obs         : observed data summary, shape (d,)
    cfg         : DiagnosticsConfig (optional — uses defaults if omitted)
    test_x      : held-out simulation summaries, shape (M, d) (optional)
    test_thetas : held-out simulation parameters, shape (M, n_params) (optional)

    SBC, coverage and parameter recovery should be run on simulations that
    were not used to train the posterior. Pass them as ``test_x`` and
    ``test_thetas``. If they are omitted, these diagnostics fall back to
    ``x`` and ``thetas`` and a warning is issued.
    """

    def __init__(self, posterior, x: np.ndarray, thetas: np.ndarray,
                 obs: np.ndarray, cfg: DiagnosticsConfig | None = None,
                 test_x: np.ndarray | None = None,
                 test_thetas: np.ndarray | None = None):
        self.posterior = posterior
        self.x = np.asarray(x, dtype=np.float32)
        self.thetas = np.asarray(thetas, dtype=np.float32)
        self.obs = np.asarray(obs, dtype=np.float32)
        self.cfg = cfg or DiagnosticsConfig()
        if (test_x is None) != (test_thetas is None):
            raise ValueError("Pass both test_x and test_thetas, or neither.")
        if test_x is None:
            warnings.warn(
                "No held-out test set given: SBC, coverage and parameter recovery "
                "will use x and thetas. If these are the training simulations, "
                "the calibration tests are not reliable. Pass test_x and test_thetas.",
                stacklevel=2,
            )
            self.test_x, self.test_thetas = self.x, self.thetas
        else:
            self.test_x = np.asarray(test_x, dtype=np.float32)
            self.test_thetas = np.asarray(test_thetas, dtype=np.float32)
        self._samples_np: np.ndarray | None = None
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Run every diagnostic and return a dict of results."""
        self._setup()
        self._print_interpretation_guide()

        self._results["prior_posterior"] = prior_posterior.run(
            self.posterior, self.obs, self.thetas, self.cfg
        )
        self._results["sensitivity"] = sensitivity.run(
            self.posterior, self.obs, self.x, self.cfg
        )
        self._results["sbc"] = sbc.run(
            self.posterior, self.test_x, self.test_thetas, self.cfg
        )

        samples_np = self._get_or_sample()

        self._results["triangle_plot"] = "saved"
        triangle_plot.run(samples_np, self.cfg)

        self._results["coverage"] = coverage.run(
            self.posterior, self.test_x, self.test_thetas, self.cfg
        )
        self._results["ppc"] = ppc.run(
            samples_np, self.obs, self.x, self.thetas, self.cfg
        )
        self._results["parameter_recovery"] = parameter_recovery.run(
            self.posterior, self.test_x, self.test_thetas, self.cfg
        )
        self._results["validation_loss"] = validation_loss.run(self.cfg)

        self._save_summary()
        return self._results

    # Individual diagnostic methods for selective use --------------------

    def diag_prior_posterior(self) -> dict:
        self._setup()
        return prior_posterior.run(self.posterior, self.obs, self.thetas, self.cfg)

    def diag_sensitivity(self) -> dict:
        self._setup()
        return sensitivity.run(self.posterior, self.obs, self.x, self.cfg)

    def diag_sbc(self) -> dict:
        self._setup()
        return sbc.run(self.posterior, self.test_x, self.test_thetas, self.cfg)

    def diag_coverage(self) -> dict:
        self._setup()
        return coverage.run(self.posterior, self.test_x, self.test_thetas, self.cfg)

    def diag_ppc(self) -> dict:
        self._setup()
        s = self._get_or_sample()
        return ppc.run(s, self.obs, self.x, self.thetas, self.cfg)

    def diag_parameter_recovery(self) -> dict:
        self._setup()
        return parameter_recovery.run(self.posterior, self.test_x, self.test_thetas, self.cfg)

    def diag_validation_loss(self) -> dict:
        return validation_loss.run(self.cfg)

    def diag_triangle_plot(self, extra_samples=None, extra_labels=None,
                           extra_colors=None):
        self._setup()
        s = self._get_or_sample()
        return triangle_plot.run(s, self.cfg, extra_samples, extra_labels, extra_colors)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup(self):
        os.makedirs(self.cfg.dir_path, exist_ok=True)
        if self.cfg.use_latex:
            if shutil.which("latex"):
                plt.rcParams["text.usetex"] = True
            else:
                warnings.warn("use_latex=True but no LaTeX installation was found. "
                              "Plotting without LaTeX.", stacklevel=2)

    def _get_or_sample(self) -> np.ndarray:
        if self._samples_np is not None:
            return self._samples_np

        print("\n================ POSTERIOR SAMPLING (obs) ================")
        key = jax.random.PRNGKey(0)
        samples = self.posterior.sample(
            x=self.obs, num_samples=self.cfg.final_num_samples, key=key
        )
        self._samples_np = np.asarray(samples)

        print(f"Samples shape: {self._samples_np.shape}")
        for p, name in enumerate(self.cfg.parameter_names):
            lo, hi = self._samples_np[:, p].min(), self._samples_np[:, p].max()
            print(f"  {name} range: {lo:.3f} – {hi:.3f}")

        if self.cfg.save_samples:
            path = self.cfg.samples_output_path or os.path.join(
                self.cfg.dir_path, f"samples_{self.cfg.data_variant}.npy"
            )
            np.save(path, self._samples_np)
            print(f"  Samples saved to: {path}")

        return self._samples_np

    def _save_summary(self):
        # Remove non-serialisable entries before dumping
        serialisable = {}
        for k, v in self._results.items():
            try:
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError):
                serialisable[k] = str(v)

        out = os.path.join(self.cfg.dir_path, "diagnostics_summary.json")
        with open(out, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nSummary saved to: {out}")

    @staticmethod
    def _print_interpretation_guide():
        print("\n================ INTERPRETATION GUIDE ================")
        print("If BOTH are true, you are effectively returning the prior:")
        print("  (1) shrinkage ratios ~ 1.0 (std_post/std_prior and IQR_post/IQR_prior)")
        print("  (2) posterior means barely move across conditioning x (small 'span')")
        print()
        print("If EITHER is true, the posterior uses x (not just the prior):")
        print("  (A) shrinkage ratios < 1 (often < 0.7), OR")
        print("  (B) posterior means shift across conditioning x.")
        print()
        print("MMD^2 and Wasserstein near-zero ⟹ posterior ≈ prior.")
