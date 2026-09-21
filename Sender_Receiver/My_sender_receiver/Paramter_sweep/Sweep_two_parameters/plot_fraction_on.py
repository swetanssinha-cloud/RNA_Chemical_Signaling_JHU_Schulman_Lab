"""
Overlay plot for a two-parameter sweep: parameter_one on the x-axis, one
colored/keyed line per parameter_two value, y = "fraction on" =
I2_center_final_nM / I2_init_nM.

Self-contained -- copy this file into the sweep's own output folder (next to
raw_results.csv and run_config.json, written by Sweep_two_core.summarise())
and run it from there.

I2_init_nM is not re-derived from the timeseries files -- Sweep_two_core's
params_for() calls sweep_core.build_params(), which guarantees
I2_init == I1O2_init for every grid point (see build_params() in
sweep_core.py), so I2_init_nM = I1O2_init_value * 1e3 by construction.
This only holds when I1O2_init is one of the two swept parameters; if
neither is, the script says so and exits instead of dividing by the wrong
thing.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

raw_files = sorted(HERE.glob("raw_results.csv"))
config_files = sorted(HERE.glob("run_config.json"))
if not raw_files or not config_files:
    raise SystemExit(f"No raw_results.csv / run_config.json found in {HERE}")

raw = pd.read_csv(raw_files[0])
config = json.loads(config_files[0].read_text())
parameter_one = config["parameter_one"]
parameter_two = config["parameter_two"]

if parameter_two == "I1O2_init":
    x_param, color_param = parameter_one, parameter_two
    x_col, color_col = "value_one", "value_two"
elif parameter_one == "I1O2_init":
    # I1O2_init would be the x-axis instead of the color axis -- still
    # derivable, just relabel which column plays which role.
    x_param, color_param = parameter_two, parameter_one
    x_col, color_col = "value_two", "value_one"
else:
    raise SystemExit(
        f"Neither swept parameter is 'I1O2_init' (got '{parameter_one}' and "
        f"'{parameter_two}') -- I2_init_nM can't be derived as "
        f"I1O2_init * 1e3 for this sweep. Skipping.")

# Mean over replicates at each grid point (matches Sweep_two_core.summarise()).
stats = raw.groupby(["value_one", "value_two"], as_index=False)["I2_center_final_nM"].mean()

color_values = sorted(stats[color_col].unique())
cmap = cm.viridis
colors = [cmap(i / max(1, len(color_values) - 1)) for i in range(len(color_values))]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for color, cval in zip(colors, color_values):
    sub = stats[stats[color_col] == cval].sort_values(x_col)
    i2_init_nM = cval * 1e3  # cval is I1O2_init (uM); I2_init == I1O2_init by construction
    fraction_on = sub["I2_center_final_nM"] / i2_init_nM
    ax.plot(sub[x_col], fraction_on, marker="o", markersize=4, linewidth=1.8,
            color=color, label=f"{color_param}={cval:g}")

ax.set_xlabel(f"{x_param} (um)" if x_param == "distance_between" else x_param)
ax.set_ylabel("Fraction on  (I2_final / I2_init)")
ax.set_title(f"{x_param} vs receiver fraction on, by {color_param}")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, ncol=2, title=color_param)
fig.tight_layout()

out = HERE / "map_fraction_on.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Wrote {out}")
