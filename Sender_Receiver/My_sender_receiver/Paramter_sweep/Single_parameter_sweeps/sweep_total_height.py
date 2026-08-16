import numpy as np
from sweep_core import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="total_height",
    sweep_values=np.linspace(0.1,1,2) * 5000,
)
if __name__ == "__main__":
    print(cfg.sweep_values)
    run(cfg)