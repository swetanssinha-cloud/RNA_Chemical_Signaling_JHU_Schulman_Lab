"""
Sweep distance_between (centre-to-centre node separation, um) -- on/off
variant. I1O2 turns off adaptively once the receiver reaches steady state
(see sweep_core_on_off.py); this file only says what to vary.

This is a mesh-affecting parameter -- one mesh is built per value, shared
with (and reused from) Single_parameter_sweeps' meshes_conformal/ cache.

Same values as Single_parameter_sweeps/sweep_distance_between.py, so the two
sweeps are directly comparable point-for-point.
"""

import numpy as np
from sweep_core_on_off import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="distance_between",
    sweep_values=np.array([200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]),
)

if __name__ == "__main__":
    run(cfg)
