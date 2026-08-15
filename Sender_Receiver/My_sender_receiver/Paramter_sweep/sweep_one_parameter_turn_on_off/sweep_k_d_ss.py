"""
Sweep k_d_ss (single-stranded degradation rate, 1/s) -- on/off variant.
I1O2 turns off adaptively once the receiver reaches steady state (see
sweep_core_on_off.py); this file only says what to vary.

Same values as Single_parameter_sweeps/sweep_k_d_ss.py, so the two sweeps
are directly comparable point-for-point.

NOTE: this range (0 up to 3e-5) is at least 10x slower than the default
k_d_ss=3e-4 the adaptive phase logic's defaults were validated against in
On_then_off.py. Expect the low end of this sweep to hit phase_max_time
(48h, see SweepConfig) rather than genuinely converge -- that's expected,
not a bug; it just means those runs are reporting "state after 48h", not
"state at the true floor".
"""

import numpy as np
from sweep_core_on_off import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="k_d_ss",
    sweep_values=np.linspace(0, 0.1, 50) * 3e-4,
)

if __name__ == "__main__":
    run(cfg)
