"""
Re-run of the I1O2_init sweep now that Th2_init tracks I2_init
(Th2_init = THRESHOLD_MULTIPLIER * I2_init, see apply_sweep_value() in
sweep_core.py). Same sweep_values as sweep_I1O2_init.py, but pointed at a
separate output_dir -- the original run's timeseries filenames don't encode
Th2_init, so reusing the same folder would let the skip-if-exists logic in
run_single_simulation() silently keep the old (Th2_init=0.4 fixed) results
instead of re-running with the new physics.
"""

from pathlib import Path

import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="I1O2_init",
    sweep_values=np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 0.1,
    output_dir=Path(__file__).resolve().parent
    / "sweep_I1O2_init_th4x_kp=0.01_ImprovedV4_5mmx5mm",
)
if __name__ == "__main__":
    run(cfg)
