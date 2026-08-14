"""
Sweep k_d_ds x k_d_ss together.

Neither parameter is mesh-affecting -- one shared mesh for the whole grid.
Shared model/solver/analysis code lives in Sweep_two_core.py; this file only
says what to vary. k_d_ss is genuinely independent of k_d_ds here -- see the
module docstring in Sweep_two_core.py for the pinning bug this fixed.

Initial test grid: 2 x 2 = 4 simulations.
"""

import numpy as np
from Sweep_two_core import TwoParamSweepConfig, run

cfg = TwoParamSweepConfig(
    parameter_one="k_d_ds",
    values_one=np.linspace(0, 1, 10) * 3e-4,
    parameter_two="k_d_ss",
    values_two=np.linspace(0, 1, 10) * 3e-4
)

if __name__ == "__main__":
    run(cfg)
