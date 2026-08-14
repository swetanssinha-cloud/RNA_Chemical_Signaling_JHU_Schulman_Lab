"""
Sweep Th2_init (receiver threshold initial concentration, uM).

Not a mesh-affecting parameter -- one mesh is built and reused by every
value. Shared model/solver/analysis code lives in sweep_core.py; this file
only says what to vary.

NOTE: sweep_values below were reconstructed from filenames seen in
sweep_Th2_init_ImprovedV4_5mmx5mm/ earlier in this conversation, not read
from a live config -- double check these before running.
"""

from sweep_core import SweepConfig, run
import numpy as np
cfg = SweepConfig(
    sweep_parameter="Th2_init",
    sweep_values= np.linspace(0.1,1.0, 50) #[0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5],
)

if __name__ == "__main__":
    run(cfg)
