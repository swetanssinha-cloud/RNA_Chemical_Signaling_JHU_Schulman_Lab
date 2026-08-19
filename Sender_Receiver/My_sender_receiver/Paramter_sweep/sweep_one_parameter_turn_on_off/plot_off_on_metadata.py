"""
Compute and plot on/off metadata metrics for a sweep_<parameter>_on_off/
folder written by sweep_core_on_off.py.

For each swept value, reduces the I2(t) timeseries (I1O2 on at t=0, shut off
at t_shutoff_hr, read from the run's .meta.json sidecar) to two numbers, both
durations measured FROM shutoff, not absolute times:

  1. t_half_off_hr  -- time AFTER shutoff for I2 to climb back through the
                       midpoint between I2 at t=0 and I2 at shutoff. NaN if
                       it never recovers within the run (e.g. a rate
                       constant of 0, where unbinding is off entirely so I2
                       can only fall).
  2. t_95_off_hr    -- time AFTER shutoff for I2 to first reach 95 nM. NaN
                       if it never gets there.

Self-contained: reads run_config.json in the target folder to find the swept
parameter and its values, so it works unmodified on any sweep_<parameter>
_on_off/ folder this project produces, now or in the future -- just point it
at the folder.

Usage
-----
    python plot_off_on_metadata.py <folder>
    python plot_off_on_metadata.py sweep_k_d_ds_on_off_zoomed_in

Writes off_on_metadata_<param>.csv (one row per swept value) and
off_on_metadata_<param>.png (2 rows x 1 column: OFF-phase half-time on top,
OFF-phase time-to-95nM on bottom) into the target folder.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_I2_NM = 95.0


def load_run_config(folder):
    # Some folders name this run_config.json, others run_config_kp=<k_p>.json
    # (one config file per (parameter, k_p) combination) -- glob rather than
    # hard-code the bare name so this works on either.
    matches = sorted(folder.glob("run_config*.json"))
    if not matches:
        raise FileNotFoundError(f"No run_config*.json found in {folder}")
    cfg = json.loads(matches[0].read_text())
    return cfg["sweep_parameter"], np.array(cfg["sweep_values"], dtype=float)


def load_run(folder, param_name, value):
    """Return (dataframe, meta_dict) for one sweep value, or (None, None)."""
    matches = sorted(folder.glob(f"timeseries_{param_name}={value:g}_rep=*.csv"))
    if not matches:
        return None, None
    path = matches[0]
    df = pd.read_csv(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta


def nearest_index(time_hours, t_target):
    """Row index whose time_hours is closest to t_target."""
    return int((time_hours - t_target).abs().to_numpy().argmin())


def crossing_time(t, y, target, from_idx=0, to_idx=None, direction="down"):
    """
    First time (linearly interpolated between bracketing samples) that y
    crosses `target`, searching only indices [from_idx, to_idx).

    direction="down": first sample where y has fallen to <= target.
    direction="up":    first sample where y has risen to >= target.
    Returns np.nan if the crossing never happens in the searched range.
    """
    t_seg, y_seg = t[from_idx:to_idx], y[from_idx:to_idx]
    crossed = (y_seg <= target) if direction == "down" else (y_seg >= target)
    if not crossed.any():
        return np.nan

    idx = int(np.argmax(crossed))
    if idx == 0:
        return float(t_seg[0])

    t0, t1 = t_seg[idx - 1], t_seg[idx]
    y0, y1 = y_seg[idx - 1], y_seg[idx]
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def compute_metrics(df, meta):
    t = df["time_hours"].to_numpy()
    y = df["I2_center_nM"].to_numpy()

    t_shutoff = meta.get("t_shutoff_hr")
    shutoff_idx = nearest_index(df["time_hours"], t_shutoff) if t_shutoff is not None else 0

    y0 = y[0]
    y_shutoff = y[shutoff_idx]
    midpoint = 0.5 * (y0 + y_shutoff)

    def duration_after_shutoff(target, direction):
        t_abs = crossing_time(t, y, target, from_idx=shutoff_idx, direction=direction)
        if not np.isfinite(t_abs) or t_shutoff is None:
            return np.nan
        return t_abs - t_shutoff

    t_half_off = duration_after_shutoff(midpoint, "up")
    t_95_off = duration_after_shutoff(TARGET_I2_NM, "up")

    return dict(
        t_half_off_hr=t_half_off,
        t_95_off_hr=t_95_off,
        midpoint_nM=midpoint,
        t_shutoff_hr=t_shutoff,
        on_converged=meta.get("on_converged"),
        off_converged=meta.get("off_converged"),
    )


def build_table(folder, param_name, sweep_values):
    rows = []
    for value in sweep_values:
        df, meta = load_run(folder, param_name, value)
        if df is None:
            continue
        rows.append({"param_value": value, **compute_metrics(df, meta)})

    if not rows:
        print("No timeseries files found -- nothing to compute.")
        return None
    return pd.DataFrame(rows).sort_values("param_value")


def plot_metadata(table, folder, param_name):
    n_missing_half = table["t_half_off_hr"].isna().sum()
    n_missing_95 = table["t_95_off_hr"].isna().sum()
    if n_missing_half:
        print(f"{n_missing_half} run(s) never recovered back through their own "
              f"midpoint after shutoff -- plotted as a gap, not zero.")
    if n_missing_95:
        print(f"{n_missing_95} run(s) never reached {TARGET_I2_NM:g} nM after "
              f"shutoff -- plotted as a gap, not zero.")

    xs = table["param_value"].to_numpy()
    fig, (ax_half, ax_95) = plt.subplots(2, 1, figsize=(7, 9))
    fig.suptitle(f"Off/on metadata: {param_name}", fontsize=15, fontweight="bold")

    def series(ax, col, color, title, ylabel):
        ax.plot(xs, table[col], "o-", color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    series(ax_half, "t_half_off_hr", "tab:blue",
           "Recovery half-time (OFF phase)", "Time to midpoint, from shutoff (hr)")
    series(ax_95, "t_95_off_hr", "tab:purple",
           f"Time to reach {TARGET_I2_NM:g} nM (OFF phase)",
           f"Time to {TARGET_I2_NM:g} nM, from shutoff (hr)")

    for ax in (ax_half, ax_95):
        ax.set_xlabel(param_name)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = folder / f"off_on_metadata_{param_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="A sweep_<parameter>_on_off/ folder "
                         "containing run_config.json and timeseries CSVs.")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    param_name, sweep_values = load_run_config(folder)
    print(f"parameter: {param_name}  ({len(sweep_values)} values)")

    table = build_table(folder, param_name, sweep_values)
    if table is None:
        return

    out_csv = folder / f"off_on_metadata_{param_name}.csv"
    table.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    plot_metadata(table, folder, param_name)


if __name__ == "__main__":
    main()
