"""
Sweep k_d_ss (single-stranded degradation rate, 1/s).

Not a mesh-affecting parameter -- one mesh is built and reused by every
value. Shared model/solver/analysis code lives in sweep_core.py; this file
only says what to vary.

NOTE: sweep_values below were reconstructed from the k_d_ds sweep's range
(same commented-out example in the old unified script) -- double check these
against what you actually want before running.
"""

import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="k_d_ss",
    sweep_values=np.linspace(0, 0.1, 50) * 3e-4,
)
#change 10 to 50 
if __name__ == "__main__":
    run(cfg)
