"""
Sweep distance_between (centre-to-centre node separation, um).

This is a mesh-affecting parameter -- one mesh is built per value. Shared
model/solver/analysis code lives in sweep_core.py; this file only says what
to vary.
"""

import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="I1O2_init",
    sweep_values=np.array([0.1,1,2,3,4,5,6,7,8,9,10]) * 0.1
)
if __name__ == "__main__":
    run(cfg)
