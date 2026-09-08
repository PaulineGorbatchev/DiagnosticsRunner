from dataclasses import dataclass, field
from typing import List


@dataclass
class DiagnosticsConfig:
    # Output
    output_dir: str = "diagnosis"
    data_variant: str = "data"
    resolution: str = "256"

    # Sampling
    num_samples_posterior: int = 50_000
    num_samples_sensitivity: int = 8_000
    num_x_conds: int = 6

    # SBC
    do_sbc: bool = True
    sbc_num_cases: int = 200
    sbc_post_samples: int = 2000

    # Coverage
    credibility_levels: List[float] = field(
        default_factory=lambda: [0.5, 0.68, 0.8, 0.9, 0.95, 0.98, 0.99]
    )
    coverage_num_test: int = 500
    coverage_post_samples: int = 2000

    # PPC
    ppc_num_samples: int = 100
    ppc_knn: int = 5

    # Parameter recovery
    recovery_num_test: int = 100
    recovery_post_samples: int = 1000

    # Posterior sampling (obs)
    final_num_samples: int = 100_000
    save_samples: bool = True
    samples_output_path: str = ""  # auto-generated if empty

    # Validation loss
    validation_loss_dir: str = ""  # base dir containing version_N subdirs

    # Misc
    seed: int = 123
    parameter_names: List[str] = field(
        default_factory=lambda: [r"$\Omega_m$", r"$h$"]
    )
    fiducial: dict = field(default_factory=lambda: {"Om": 0.3, "h": 0.7})
    use_latex: bool = True

    @property
    def dir_path(self) -> str:
        return (
            f"{self.output_dir}/diagnosis_{self.resolution}"
            f"/diagnosis_{self.data_variant}"
        )
