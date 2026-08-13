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
    sweep_values=np.array([200.0, 300.0, 400.0,500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]),
)#  removing these for now because of intial testing

if __name__ == "__main__":
    run(cfg)
