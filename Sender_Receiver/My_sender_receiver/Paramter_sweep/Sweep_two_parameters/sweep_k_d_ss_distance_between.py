import numpy as np
from Sweep_two_core import TwoParamSweepConfig, run

cfg = TwoParamSweepConfig(
    parameter_one="distance_between",
    values_one=np.linspace(100,1000,10) * 3e-4,
    parameter_two="k_d_ss",
    values_two=np.linspace(0, 1, 10) * 3e-4
)

if __name__ == "__main__":
    run(cfg)