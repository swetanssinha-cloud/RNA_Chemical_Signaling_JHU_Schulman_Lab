"""
Re-plot this folder's on/off sweep results in a readable form.

The original timeseries_on_off_<param>.png (written by sweep_core_on_off.py's
plot_timeseries()) puts one legend entry per sweep value -- fine for ~10
values, unreadable at 50, which is what every 50-value sweep in this project
produces. This script makes two figures instead:

  1. The same I2-vs-time overlay, but colored by a colorbar instead of a
     50-entry legend.
  2. Two derived timing metrics per run, plotted against the swept
     parameter: the time for I2 to first drop to the midpoint between its
     run-wide max and min, and the time for it to climb back through that
     same midpoint during the recovery phase (NaN / no marker if it never
     does -- see k_d_ds=0, where unbinding is off entirely so I2 can only
     fall, never recover).

Self-contained: reads run_config.json in this same folder to find out which
parameter was swept, so it works unmodified if copied into any other
sweep_<parameter>_on_off/ folder from this project.

Run from inside this folder:
    python replot_readable.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_run_config():
    cfg = json.loads((HERE / "run_config.json").read_text())
    return cfg["sweep_parameter"], np.array(cfg["sweep_values"], dtype=float)


def load_run(param_name, value):
    """Return (dataframe, meta_dict) for one sweep value, or (None, None)."""
    matches = sorted(HERE.glob(f"timeseries_{param_name}={value:g}_rep=*.csv"))
    if not matches:
        return None, None
    path = matches[0]
    df = pd.read_csv(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta


def crossing_time(t, y, target, from_idx=0, direction="down"):
    """
    First time (linearly interpolated between bracketing samples) that y
    crosses `target`, searching only from index from_idx onward.

    direction="down": first sample where y has fallen to <= target.
    direction="up":    first sample where y has risen to >= target.
    Returns np.nan if the crossing never happens in the searched range --
    e.g. the recovery crossing when the run never actually recovers
    (k_d_ds=0: unbinding is off, so I2 can only fall).
    """
    t, y = t[from_idx:], y[from_idx:]
    crossed = (y <= target) if direction == "down" else (y >= target)
    if not crossed.any():
        return np.nan

    idx = int(np.argmax(crossed))
    if idx == 0:
        return float(t[0])

    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def midpoint_timings(df):
    """
    time to first drop to the midpoint between this run's max and min I2,
    and time to climb back through that same midpoint afterward (NaN if it
    never does). Both are absolute times from t=0, not durations.
    """
    t = df["time_hours"].to_numpy()
    y = df["I2_center_nM"].to_numpy()

    y_max, y_min = y.max(), y.min()
    midpoint = 0.5 * (y_max + y_min)
    min_idx = int(np.argmin(y))

    t_down = crossing_time(t, y, midpoint, from_idx=0, direction="down")
    t_up = crossing_time(t, y, midpoint, from_idx=min_idx, direction="up")
    return t_down, t_up, midpoint, y_max, y_min


def plot_timeseries_colorbar(param_name, sweep_values):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    norm = mcolors.Normalize(vmin=sweep_values.min(), vmax=sweep_values.max())
    cmap = cm.viridis

    plotted_any = False
    for value in sweep_values:
        df, meta = load_run(param_name, value)
        if df is None:
            continue
        plotted_any = True
        color = cmap(norm(value))
        ax.plot(df["time_hours"], df["I2_center_nM"], color=color, lw=1.3)

        t_shut = meta.get("t_shutoff_hr")
        if t_shut is not None:
            idx = (df["time_hours"] - t_shut).abs().idxmin()
            ax.scatter([df["time_hours"][idx]], [df["I2_center_nM"][idx]],
                       color=color, marker="v", s=35, zorder=5,
                       edgecolor="black", linewidth=0.4)

    if not plotted_any:
        print("No timeseries files found -- nothing to plot.")
        return

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("[I2] at receiver (nM)")
    ax.set_title(f"On/off timeseries: {param_name}\n(▼ = I1O2 shutoff)")
    ax.grid(alpha=0.3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(param_name)

    fig.tight_layout()
    out = HERE / f"timeseries_on_off_{param_name}_readable.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_midpoint_timings(param_name, sweep_values):
    rows = []
    for value in sweep_values:
        df, meta = load_run(param_name, value)
        if df is None:
            continue
        t_down, t_up, midpoint, y_max, y_min = midpoint_timings(df)
        rows.append({
            "param_value": value,
            "t_down_hr": t_down,
            "t_up_hr": t_up,
            "midpoint_nM": midpoint,
            "I2_max_nM": y_max,
            "I2_min_nM": y_min,
        })

    if not rows:
        print("No timeseries files found -- nothing to plot.")
        return None

    stats = pd.DataFrame(rows).sort_values("param_value")
    stats.to_csv(HERE / f"midpoint_timings_{param_name}.csv", index=False)
    print(f"Wrote {HERE / f'midpoint_timings_{param_name}.csv'}")

    n_missing = stats["t_up_hr"].isna().sum()
    if n_missing:
        print(f"{n_missing} run(s) never recovered back through their own "
              f"midpoint (e.g. {param_name}=0, where unbinding is off) -- "
              f"plotted as a gap, not zero.")

    # Two stacked panels, not one shared axis: the drop time and the
    # recovery time differ by 30-50x in this data (drop is a near-constant
    # ~1.2h, recovery spans ~27-68h), so a shared y-axis flattens the drop
    # time into an invisible line at the bottom -- the same unreadability
    # problem as the 50-entry legend, just from axis scale instead of clutter.
    fig, (ax_down, ax_up) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)

    ax_down.plot(stats["param_value"], stats["t_down_hr"], "o-", color="tab:red")
    ax_down.set_ylabel("Time to drop\nto midpoint (hr)")
    ax_down.grid(alpha=0.3)

    ax_up.plot(stats["param_value"], stats["t_up_hr"], "o-", color="tab:blue")
    ax_up.set_ylabel("Time to recover\nto midpoint (hr)")
    ax_up.set_xlabel(param_name)
    ax_up.grid(alpha=0.3)

    fig.suptitle(f"Midpoint crossing times: {param_name}\n"
                 f"(midpoint = halfway between each run's own max and min I2)")
    fig.tight_layout()
    out = HERE / f"midpoint_timings_{param_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")
    return stats


if __name__ == "__main__":
    param_name, sweep_values = load_run_config()
    print(f"parameter: {param_name}  ({len(sweep_values)} values)")

    plot_timeseries_colorbar(param_name, sweep_values)
    plot_midpoint_timings(param_name, sweep_values)
