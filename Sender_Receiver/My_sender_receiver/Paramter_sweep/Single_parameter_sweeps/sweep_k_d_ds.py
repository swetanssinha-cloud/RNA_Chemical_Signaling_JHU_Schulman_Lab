"""
Sweep k_d_ds (double-stranded degradation rate, 1/s).

Not a mesh-affecting parameter -- one mesh is built and reused by every
value. Shared model/solver/analysis code lives in sweep_core.py; this file
only says what to vary.
"""

import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="k_d_ds",
    sweep_values=np.linspace(0, 1, 50) * 3e-4,
)

if __name__ == "__main__":
    run(cfg)
