"""
Sweep k_slow (S2 + I2 -> S2:I2 rate, 1/(uM s)).

Not a mesh-affecting parameter -- one mesh is built and reused by every
value. Shared model/solver/analysis code lives in sweep_core.py; this file
only says what to vary.

NOTE: sweep_values below were reconstructed from filenames seen in
sweep_k_slow_ImprovedV4_5mmx5mm/ earlier in this conversation, not read
from a live config -- double check these before running.
"""

from sweep_core import SweepConfig, run
import numpy as np
cfg = SweepConfig(
    sweep_parameter="k_slow",
    sweep_values= np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]) * 5e4 * 1e-6
)

if __name__ == "__main__":
    run(cfg)
