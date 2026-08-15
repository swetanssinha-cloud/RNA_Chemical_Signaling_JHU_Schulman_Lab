"""
Sweep Th2_init (receiver threshold initial concentration, uM) -- on/off
variant. I1O2 turns off adaptively once the receiver reaches steady state
(see sweep_core_on_off.py); this file only says what to vary.

Same values as Single_parameter_sweeps/sweep_Th2_init.py, so the two sweeps
are directly comparable point-for-point.
"""

from sweep_core_on_off import SweepConfig, run
import numpy as np

cfg = SweepConfig(
    sweep_parameter="Th2_init",
    sweep_values=np.linspace(0.1, 1.0, 50),
)

if __name__ == "__main__":
    run(cfg)
