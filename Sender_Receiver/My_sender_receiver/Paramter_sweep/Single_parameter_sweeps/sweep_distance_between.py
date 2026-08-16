"""
Sweep distance_between (centre-to-centre node separation, um).

This is a mesh-affecting parameter -- one mesh is built per value. Shared
model/solver/analysis code lives in sweep_core.py; this file only says what
to vary.
"""

import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="distance_between",
    sweep_values=np.np.arange(200, 1501, 100),
)
if __name__ == "__main__":
    run(cfg)
