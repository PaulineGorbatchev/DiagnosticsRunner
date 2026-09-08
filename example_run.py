"""
Example: replace the training block in your script with this.
All paths are the same as your original notebook.
"""
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import warnings

from jaxili.inference import NPE
from sbi_diagnostics import DiagnosticsConfig, DiagnosticsRunner

warnings.filterwarnings("ignore", category=UserWarning, module="getdist")

# ===================== PARAMETERS =====================
SEED = 1
RNG_KEY = jr.PRNGKey(SEED)
RESOLUTION = "256"
DATA_VARIANT = f"beam_noise_masked_mean_removed_{RESOLUTION}"
BINS_min, BINS_max = 150, 400
plt.rcParams["text.usetex"] = True

# ===================== LOAD DATA =====================
data = np.load(f"l1_SBI/{RESOLUTION}/Colore_l1_{DATA_VARIANT}_8000_Lband_theoritical_noise.npz")
cosmologies = data["cosmologies"]
l1 = data["l1"][:, BINS_min:BINS_max]
thetas = cosmologies[:, [0, 1]].astype(np.float32)

obs = np.load(
    f"True_l1_colore/{RESOLUTION}/Lband_l1_{DATA_VARIANT}_theoritical_noise.npy",
    allow_pickle=True,
).flatten()[BINS_min:BINS_max]

# ===================== REMOVE ZERO BINS =====================
zero_obs = np.where(obs == 0)[0]
zero_l1  = np.where((l1 == 0).any(axis=0))[0]
drop = np.unique(np.concatenate((zero_obs, zero_l1)))
obs = np.delete(obs, drop)
l1  = np.delete(l1, drop, axis=1)
x   = l1.astype(np.float32)

assert not np.isnan(x).any()
assert not np.isnan(thetas).any()

# ===================== SBI TRAINING =====================
inference = NPE()
inference = inference.append_simulations(thetas, x)
metrics, density_estimator = inference.train(
    checkpoint_path=os.path.abspath("."),
    learning_rate=1e-2,
    training_batch_size=128,
    num_epochs=10000,
    rng_key=RNG_KEY,
)
posterior = inference.build_posterior()

# ===================== DIAGNOSTICS =====================
cfg = DiagnosticsConfig(
    output_dir="diagnosis",
    data_variant=f"{DATA_VARIANT}_Lband_theoritical_noise_normal",
    resolution=RESOLUTION,
    parameter_names=[r"$\Omega_m$", r"$h$"],
    fiducial={"Om": 0.3, "h": 0.7},
    seed=123,
    # point at your training metrics for the val-loss plot:
    validation_loss_dir="/nas/homes/pauline/workspace/CODE/Project_FR/NDE_w_Standardization",
    # where to save the final posterior samples:
    samples_output_path=f"SBI_samples/{RESOLUTION}/l1_{DATA_VARIANT}_8000_FINAL_processed_theoritical_noise_normal.npy",
)

runner = DiagnosticsRunner(posterior, x, thetas, obs, cfg)
results = runner.run_all()

print("\nAll diagnostics done. Numerical results:")
import json
print(json.dumps({k: v for k, v in results.items() if k != "sbc"}, indent=2))
