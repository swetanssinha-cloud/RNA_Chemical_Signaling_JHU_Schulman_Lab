"""
Sweep k_slow x distance_between together.

distance_between is mesh-affecting -- one mesh per distance_between value,
shared across both k_slow values. Shared model/solver/analysis code lives in
Sweep_two_core.py; this file only says what to vary.

Initial test grid: 2 x 2 = 4 simulations.
"""

import numpy as np
from Sweep_two_core import TwoParamSweepConfig, run

cfg = TwoParamSweepConfig(
    parameter_one="k_slow",
    values_one=np.array([1.0, 5.0]) * 5e4 * 1e-6,
    parameter_two="distance_between",
    values_two=np.array([300.0, 600.0]),
)

if __name__ == "__main__":
    run(cfg)
