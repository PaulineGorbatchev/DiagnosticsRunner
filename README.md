# sbi_diagnostics

A Python package for validating and diagnosing trained [JAXILI](https://github.com/sachaguer/jaxili) Neural Posterior Estimation (NPE) posteriors in simulation-based inference workflows.

It wraps a suite of standard SBI diagnostics — shrinkage tests, SBC rank histograms, coverage probability, posterior predictive checks, and more — into a single callable object that saves every plot and a numerical summary in one step.

---

## Features

| Diagnostic | What it checks |
|---|---|
| **Prior vs Posterior** (Diag A) | Whether the posterior actually updates from the prior (shrinkage ratios, MMD², Wasserstein-1) |
| **Sensitivity to x** (Diag B) | Whether the posterior mean shifts when you condition on different simulations |
| **SBC rank histograms** (Diag C) | Whether the posterior is calibrated (uniform ranks ⟹ well-calibrated) |
| **Coverage probability** | Empirical vs nominal credibility across multiple levels |
| **Posterior predictive check** | Whether simulations drawn from the posterior match the observed summary |
| **Parameter recovery** | Posterior mean vs true θ scatter (RMSE and bias reported) |
| **Triangle plot** | getdist corner plot at the observed data, with optional overlays |
| **Validation loss** | Training curve from JAXILI's `version_N/metrics/` output |

All plots are saved as PDFs. A `diagnostics_summary.json` with numerical results is written alongside them.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/PaulineGorbatchev/DiagnosticsRunner.git
cd DiagnosticsRunner
pip install -e .
```

Since the repository is private, Git will ask for your GitHub username and a [personal access token](https://github.com/settings/tokens/new) (with the `repo` scope) as the password.

**Dependencies** (installed automatically): `jax`, `jaxili`, `numpy`, `matplotlib`, `scipy`, `getdist`.

---

## Quick start

```python
from sbi_diagnostics import DiagnosticsConfig, DiagnosticsRunner

# posterior  : trained JAXILI NPE posterior
# x          : simulation summaries, shape (N, d)
# thetas     : simulation parameters, shape (N, n_params)
# obs        : observed data summary, shape (d,)

cfg = DiagnosticsConfig(
    output_dir="diagnosis",
    data_variant="my_experiment",
    resolution="256",
    parameter_names=[r"$\Omega_m$", r"$h$"],
    fiducial={"Om": 0.3, "h": 0.7},
)

runner = DiagnosticsRunner(posterior, x, thetas, obs, cfg)
results = runner.run_all()
```

This writes all PDFs to `diagnosis/diagnosis_256/diagnosis_my_experiment/` and prints a numerical summary.

---

## Running individual diagnostics

Every diagnostic is also available as a standalone method:

```python
runner.diag_prior_posterior()     # Diag A — shrinkage, MMD², Wasserstein
runner.diag_sensitivity()         # Diag B — posterior mean span across x
runner.diag_sbc()                 # Diag C — SBC rank histograms
runner.diag_coverage()            # coverage probability curve
runner.diag_ppc()                 # posterior predictive check
runner.diag_parameter_recovery()  # true vs posterior-mean scatter
runner.diag_triangle_plot()       # getdist corner plot
runner.diag_validation_loss()     # val-loss curve (requires validation_loss_dir)
```

---

## Configuration reference

All options live in `DiagnosticsConfig`. Pass only the ones you want to override — everything else has a sensible default.

```python
from sbi_diagnostics import DiagnosticsConfig

cfg = DiagnosticsConfig(
    # --- Output ---
    output_dir          = "diagnosis",      # root output folder
    data_variant        = "my_data",        # label used in filenames
    resolution          = "256",            # subfolder label

    # --- Parameter names / fiducial ---
    parameter_names     = [r"$\Omega_m$", r"$h$"],
    fiducial            = {"Om": 0.3, "h": 0.7},   # marked on triangle plot

    # --- Diag A: prior vs posterior ---
    num_samples_posterior = 50_000,

    # --- Diag B: sensitivity ---
    num_samples_sensitivity = 8_000,
    num_x_conds             = 6,           # obs + N random simulation x's

    # --- Diag C: SBC ---
    do_sbc           = True,
    sbc_num_cases    = 200,
    sbc_post_samples = 2000,

    # --- Coverage ---
    credibility_levels   = [0.5, 0.68, 0.8, 0.9, 0.95, 0.98, 0.99],
    coverage_num_test    = 500,
    coverage_post_samples = 2000,

    # --- PPC ---
    ppc_num_samples = 100,
    ppc_knn         = 5,

    # --- Parameter recovery ---
    recovery_num_test    = 100,
    recovery_post_samples = 1000,

    # --- Posterior samples at obs ---
    final_num_samples   = 100_000,
    save_samples        = True,
    samples_output_path = "",              # auto-generated if empty

    # --- Validation loss ---
    validation_loss_dir = "/path/to/NDE_w_Standardization",

    # --- Misc ---
    seed       = 123,
    use_latex  = True,
)
```

---

## Output files

After `run_all()`, the output directory contains:

```
diagnosis/diagnosis_<resolution>/diagnosis_<data_variant>/
├── prior_vs_posterior_<data_variant>.pdf
├── posterior_mean_sensitivity_<data_variant>.pdf
├── sbc_ranks_<data_variant>.pdf
├── coverage_probability_<data_variant>.pdf
├── posterior_predictive_check_<data_variant>.pdf
├── parameter_recovery_<data_variant>.pdf
├── triangle_plot_<data_variant>.pdf
├── val_loss_<data_variant>.pdf
├── samples_<data_variant>.npy          ← 100k posterior samples at obs
└── diagnostics_summary.json           ← all numerical results
```

---

## Interpreting the results

**The posterior is not using the data if:**
1. Shrinkage ratios (std_post / std_prior and IQR_post / IQR_prior) are both close to 1.0, **and**
2. The posterior mean barely moves when you condition on different x (small "span" in Diag B).

**The posterior is informative if:**
- Shrinkage ratios are noticeably below 1 (often < 0.7), **or**
- The posterior mean shifts clearly across different conditioning x.

**SBC:** Flat rank histograms indicate a calibrated posterior. U-shaped histograms mean the posterior is overconfident; hill-shaped means underconfident.

**Coverage:** Points should lie near the diagonal. Points below the diagonal mean the posterior is overconfident (credible intervals are too narrow).

---

## Overlaying multiple posteriors on the triangle plot

```python
samples_variant2 = np.load("samples_variant2.npy")

runner.diag_triangle_plot(
    extra_samples=[samples_variant2],
    extra_labels=["Variant 1 (obs)", "Variant 2"],
    extra_colors=["blue", "red"],
)
```

---

## Example script

See [example_run.py](example_run.py) for a full end-to-end example including data loading, JAXILI training, and diagnostics.

---

## License

MIT
