"""
Standalone plot: swept parameter vs "fraction on" = I2_center_final_nM / I2_init_nM.

I2_init is NOT read from run_config_kp=*.json's "default_params" -- that field
is only the static DEFAULT_PARAMS baseline sweep_core.py started from, not the
per-run value actually used. Because I2_init is synced to I1O2_init inside
apply_sweep_value() for every run, the real per-run I2_init is instead read
straight from each timeseries CSV's own t=0 row (the raw initial condition,
before any reaction/diffusion), which is ground truth for whatever was
actually simulated.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

stats_files = sorted(HERE.glob("summary_stats_kp=*.csv"))
config_files = sorted(HERE.glob("run_config_kp=*.json"))
if not stats_files or not config_files:
    raise SystemExit(f"No summary_stats_kp=*.csv / run_config_kp=*.json found in {HERE}")

stats_file = stats_files[0]
config_file = config_files[0]

stats = pd.read_csv(stats_file).sort_values("param_value").reset_index(drop=True)
config = json.loads(config_file.read_text())
sweep_parameter = config["sweep_parameter"]

pattern = re.compile(
    rf"timeseries_{re.escape(sweep_parameter)}=(?P<value>[-+0-9.eE]+)_rep=(?P<rep>\d+)")

# Real per-run I2_init (nM), from each timeseries file's own t=0 row.
i2_init_by_value = {}
for path in HERE.glob(f"timeseries_{sweep_parameter}=*_rep=*.csv"):
    match = pattern.search(path.stem)
    if match is None:
        continue
    value = float(match.group("value"))
    df = pd.read_csv(path, nrows=1)  # t=0 row only
    i2_init_by_value.setdefault(value, []).append(df["I2_center_nM"].iloc[0])

fraction_on_mean = []
fraction_on_std = []
i2_init_used = []
for _, row in stats.iterrows():
    value = row["param_value"]
    # Match on the %g-formatted string, same reasoning as summarise()'s
    # collect_results_from_disk(): raw floats carry rounding noise that
    # can make an exact dict-key lookup miss.
    key = min(i2_init_by_value, key=lambda v: abs(v - value)) if i2_init_by_value else None
    if key is None or abs(key - value) > 1e-6 * max(abs(value), 1):
        raise SystemExit(f"No timeseries file found for {sweep_parameter}={value:g}")
    i2_init_nM = np.mean(i2_init_by_value[key])
    i2_init_used.append(i2_init_nM)
    fraction_on_mean.append(row["I2_center_final_nM_mean"] / i2_init_nM)
    if "I2_center_final_nM_std" in stats.columns and np.isfinite(row["I2_center_final_nM_std"]):
        fraction_on_std.append(row["I2_center_final_nM_std"] / i2_init_nM)
    else:
        fraction_on_std.append(np.nan)

stats["I2_init_nM_actual"] = i2_init_used
stats["fraction_on_mean"] = fraction_on_mean
fraction_on_std = np.array(fraction_on_std)
yerr = fraction_on_std if np.isfinite(fraction_on_std).any() else None

fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
ax.errorbar(
    stats["param_value"], stats["fraction_on_mean"], yerr=yerr,
    marker="o", capsize=4, linewidth=2, color="tab:green",
)
ax.set_xlabel(f"{sweep_parameter} (uM)")
ax.set_ylabel("Fraction on  (I2_final / I2_init)")
ax.set_title(f"Parameter sweep: {sweep_parameter} vs receiver fraction on")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

fig.tight_layout()
kp_tag = config_file.stem.split("kp=")[-1]
out = HERE / f"sweep_{sweep_parameter}_fraction_on_kp={kp_tag}.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Wrote {out}")

print("\nparam_value -> I2_init_nM_actual -> fraction_on")
for _, row in stats.iterrows():
    print(f"  {row['param_value']:g} -> {row['I2_init_nM_actual']:g} -> {row['fraction_on_mean']:.4f}")
