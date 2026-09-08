"""Triangle (corner) plot using getdist."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="getdist")

from getdist import MCSamples, plots

from ..config import DiagnosticsConfig


def run(
    samples_np: np.ndarray,
    cfg: DiagnosticsConfig,
    extra_samples: list | None = None,
    extra_labels: list | None = None,
    extra_colors: list | None = None,
) -> MCSamples:
    """
    Draw a triangle plot.

    extra_samples: list of additional np.ndarray posteriors to overlay.
    extra_labels / extra_colors: matching labels and colors.
    """
    print("\n================ TRIANGLE PLOT ================")

    # getdist wraps labels in $ itself, so strip any surrounding $ from parameter_names
    labels = [n.strip("$") for n in cfg.parameter_names]
    # names must be plain identifiers (no LaTeX)
    names = [re.sub(r"[^A-Za-z0-9_]", "_", n.strip("$")) for n in cfg.parameter_names]

    gd_samples = [
        MCSamples(
            samples=samples_np,
            names=names,
            labels=labels,
        )
    ]
    colors = ["blue"]

    if extra_samples:
        for i, s in enumerate(extra_samples):
            gd_samples.append(
                MCSamples(
                    samples=s,
                    names=names,
                    labels=labels,
                )
            )
        colors += (extra_colors or ["red", "green", "orange"][: len(extra_samples)])

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4
    g.triangle_plot(
        gd_samples,
        filled=True,
        contour_colors=colors,
        markers=cfg.fiducial,
        legend_labels=extra_labels or None,
    )

    out = os.path.join(cfg.dir_path, f"triangle_plot_{cfg.data_variant}.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    return gd_samples[0]
