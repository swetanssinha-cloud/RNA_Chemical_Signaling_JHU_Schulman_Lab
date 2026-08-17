"""
Sweep k_d_ds (double-stranded degradation rate, 1/s) -- on/off variant.
I1O2 turns off adaptively once the receiver reaches steady state (see
sweep_core_on_off.py); this file only says what to vary.

Same values as Single_parameter_sweeps/sweep_k_d_ds.py, so the two sweeps
are directly comparable point-for-point.

Re-run with an extended OFF-phase budget: the first pass (see
sweep_k_d_ds_on_off/, kept for reference) used the old shared 48h cap for
both phases, which was fine for the ON phase (on_converged=True for all 50
runs, observed 20-39h) but wrong for OFF -- only 1 of 50 runs actually
reached off_converged=True; the other 49 all stopped at exactly
t_shutoff + 48h, i.e. the timeout, not real convergence. sweep_core_on_off.py
now defaults off_phase_max_time to 150h, and this run writes to a new folder
(sweep_k_d_ds_on_off_extended/) rather than overwriting the old one, so nei-
ther copy is lost.

NOTE: this range (0 up to 3e-5) is at least 10x slower than the default
k_d_ds=3e-4 the adaptive phase logic was originally tuned against in
On_then_off.py. Even 150h may not be enough to converge every value in this
range -- check each run's off_converged field in its .meta.json / in
raw_results.csv rather than assuming every point finished for real.
"""

from pathlib import Path

import numpy as np
from sweep_core_on_off import SweepConfig, run

cfg = SweepConfig(
    sweep_parameter="k_d_ds",
    sweep_values=np.linspace(0, 0.1, 50) * 3e-4,
    output_dir=Path(__file__).resolve().parent / "sweep_k_d_ds_on_off_extended",
)

if __name__ == "__main__":
    run(cfg)
