"""
Sweep distance_between x I1O2_init together.

distance_between is mesh-affecting -- one mesh per distance_between value,
shared across all 11 I1O2_init values. Shared model/solver/analysis code
lives in Sweep_two_core.py; this file only says what to vary.

parameter_one=distance_between so it lands on the x-axis of the line-family
panels in map_I2_final.png/map_half_time.png; parameter_two=I1O2_init so it
becomes the per-line color/legend key (see Sweep_two_core.py's
_line_family()).

I1O2 (sender template) drives I2_init (receiver) and Th2_init (threshold)
automatically for every grid point: params_for() in Sweep_two_core.py calls
sweep_core.build_params(), which sets I2_init = I1O2_init and
Th2_init = THRESHOLD_MULTIPLIER * I2_init (currently 4x) for any combination
of overrides -- no extra code needed here for that invariant to hold.

distance_between values: the union of the 80-200 (step 10), 200-1000
(step 100) and 1000-1500 (step 100) ranges already swept individually
elsewhere in Paramter_sweep/ -- some of these have a cached mesh at the
current 5000x5000 domain size already; the rest get built fresh (see
Sweep_two_core.build_all_meshes()).

Grid: 26 distance_between values x 11 I1O2_init values = 286 simulations.
"""

import numpy as np
from Sweep_two_core import TwoParamSweepConfig, run

distance_between_values = (
    list(range(80, 201, 10))        # 80, 90, ..., 200    (13 values)
    + list(range(300, 1001, 100))   # 300, 400, ..., 1000 (8 values)
    + list(range(1100, 1501, 100))  # 1100, ..., 1500     (5 values)
)  # 26 values total

I1O2_init_values = np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1  # 11 values

cfg = TwoParamSweepConfig(
    parameter_one="distance_between",
    values_one=distance_between_values,
    parameter_two="I1O2_init",
    values_two=I1O2_init_values,
)

if __name__ == "__main__":
    run(cfg)
