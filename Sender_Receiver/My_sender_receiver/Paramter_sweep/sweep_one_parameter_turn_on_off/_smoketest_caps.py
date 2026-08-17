from sweep_core_on_off import SweepConfig, run
from pathlib import Path

cfg = SweepConfig(
    sweep_parameter="k_slow",
    sweep_values=[5e4 * 1e-6],
    output_dir=Path(__file__).resolve().parent / "_smoketest_caps_output",
    check_interval=2,
    ss_window=2,
    on_phase_max_time=120.0,   # should force ON shutoff at ~2 min
    off_phase_max_time=300.0,  # should force OFF stop at ~5 min after that
)

if __name__ == "__main__":
    run(cfg)
