"""Validation loss curve from JAXILI training metrics."""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from ..config import DiagnosticsConfig


def find_latest_version(base_dir: str) -> str:
    version_dirs = [d for d in os.listdir(base_dir) if re.match(r"version_\d+", d)]
    if not version_dirs:
        raise FileNotFoundError(f"No version_N directories found in {base_dir}")
    version_numbers = [int(re.findall(r"\d+", d)[0]) for d in version_dirs]
    latest = max(version_numbers)
    return os.path.join(base_dir, f"version_{latest}", "metrics")


def run(cfg: DiagnosticsConfig) -> dict:
    if not cfg.validation_loss_dir:
        print("Validation loss skipped (validation_loss_dir not set).")
        return {}

    print("\n================ VALIDATION LOSS ================")

    metrics_dir = find_latest_version(cfg.validation_loss_dir)
    print(f"Using metrics directory: {metrics_dir}")

    metrics_files = [
        f for f in os.listdir(metrics_dir)
        if f.endswith(".json") and f.startswith("eval_epoch")
    ]

    epoch_nums, val_losses = [], []
    for f in metrics_files:
        epoch = int(re.findall(r"\d+", f)[0])
        with open(os.path.join(metrics_dir, f)) as fh:
            d = json.load(fh)
            if "val/loss" in d:
                epoch_nums.append(epoch)
                val_losses.append(d["val/loss"])

    if not epoch_nums:
        print("No val/loss entries found.")
        return {}

    epoch_nums, val_losses = zip(*sorted(zip(epoch_nums, val_losses)))

    best_val = min(val_losses)
    best_epoch = epoch_nums[val_losses.index(best_val)]

    plt.figure(figsize=(7, 5))
    plt.plot(epoch_nums, val_losses, marker="o")
    plt.axhline(best_val, linestyle="--",
                label=f"Best = {best_val:.3f} (Epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(r"Validation Loss ($\Omega_m$–$h$)")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.tight_layout()
    out = os.path.join(cfg.dir_path, f"val_loss_{cfg.data_variant}.pdf")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

    return {"best_val_loss": best_val, "best_epoch": best_epoch}
