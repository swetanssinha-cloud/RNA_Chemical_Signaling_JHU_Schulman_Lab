"""
Sweep k_slow (S2 + I2 -> S2:I2 rate, 1/(uM s)) -- on/off variant. I1O2 turns
off adaptively once the receiver reaches steady state (see
sweep_core_on_off.py); this file only says what to vary.

Same values as Single_parameter_sweeps/sweep_k_slow.py, so the two sweeps
are directly comparable point-for-point.
"""

from sweep_core_on_off import SweepConfig, run
import numpy as np

cfg = SweepConfig(
    sweep_parameter="k_slow",
    sweep_values=np.array([1, 2, 3, 4,]) * 5e4 * 1e-6,
)

if __name__ == "__main__":
    run(cfg)
