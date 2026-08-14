"""
Post-hoc analysis of a parameter sweep's raw timeseries CSVs.

Parameter_sweep_unified.py only plots I2 vs time. This script reads the same
per-value timeseries_*.csv files it wrote and builds one combined figure
(center-point readout only):

  Row 1: S2_free and S2_total dynamics, overlaid across parameter values
  Row 2: I2 dynamics, overlaid across parameter values, and the time for I2
         to fall to the midpoint between its initial and final value, as a
         function of the swept parameter

Only the USER CONFIGURATION block below needs to change between sweeps --
swap SWEEP_PARAMETER, PARAMETER_VALUES and CSV_FILENAME_TEMPLATE and rerun.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# USER CONFIGURATION -- edit this block for a new sweep
# =============================================================================

SWEEP_PARAMETER = "k_slow"   # used for axis labels and output filenames

PARAMETER_VALUES = np.array([1, 2, 3, 4, 5]) * 5e4 * 1e-6

# {value} is replaced with each entry of PARAMETER_VALUES (formatted with
# VALUE_FORMAT_SPEC below). Everything else must be identical across the sweep.
CSV_FILENAME_TEMPLATE = "timeseries_k_slow={value}_rep=0_5mmx5mm.csv"

# How a parameter value is turned into the {value} text above. "g" drops
# trailing zeros (0.1 -> "0.1"), matching how Parameter_sweep_unified.py
# names its files (f"{param_value:g}").
VALUE_FORMAT_SPEC = "g"

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR


# =============================================================================
# LOADING
# =============================================================================

def load_all():
    data = {}
    for value in PARAMETER_VALUES:
        value_str = format(value, VALUE_FORMAT_SPEC)
        csv_file_name = CSV_FILENAME_TEMPLATE.format(value=value_str)
        path = DATA_DIR / csv_file_name
        if not path.exists():
            print(f"  WARNING: missing {csv_file_name} "
                  f"-- skipping {SWEEP_PARAMETER}={value}")
            continue
        data[value] = pd.read_csv(path)

    if not data:
        raise SystemExit(
            f"No CSV files found matching template "
            f"'{CSV_FILENAME_TEMPLATE}' in {DATA_DIR}")
    return data


def half_time(time_hours, signal):
    """
    Time at which the signal crosses halfway between its initial and final
    value, linearly interpolated between the two bracketing samples.
    """
    y0, y1 = signal[0], signal[-1]
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-12:
        return np.nan

    target = 0.5 * (y0 + y1)
    crossed = signal <= target if y1 < y0 else signal >= target
    if not crossed.any():
        return np.nan

    idx = int(np.argmax(crossed))
    if idx == 0:
        return float(time_hours[0])

    t0, t1 = time_hours[idx - 1], time_hours[idx]
    s0, s1 = signal[idx - 1], signal[idx]
    if s1 == s0:
        return float(t1)
    return float(t0 + (target - s0) * (t1 - t0) / (s1 - s0))


def build_summary(data):
    """Center-point final value and half-time for every swept value."""
    rows = []
    for value, df in sorted(data.items()):
        final = df.iloc[-1]
        row = {"param_value": value}
        for sp in ["I2", "S2_free", "S2_total"]:
            col = f"{sp}_center_nM"
            if col in df.columns:
                row[f"{sp}_final_nM"] = final[col]
        row["half_time_hr"] = half_time(
            df["time_hours"].to_numpy(), df["I2_center_nM"].to_numpy())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("param_value").reset_index(drop=True)
    out = OUTPUT_DIR / f"summary_{SWEEP_PARAMETER}.csv"
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")
    return summary


# =============================================================================
# COMBINED FIGURE
#   Row 1: S2_free, S2_total dynamics overlaid across parameter values
#   Row 2: I2 dynamics overlaid across parameter values, half-time vs parameter
# =============================================================================

def plot_combined(data, summary):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def overlay(ax, col_name, title):
        for color, (value, df) in zip(colors, sorted(data.items())):
            ax.plot(df["time_hours"], df[col_name], color=color, lw=2,
                     label=f"{SWEEP_PARAMETER}={value:g}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("nM")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    overlay(axes[0][0], "I2_center_nM", "I2")
    overlay(axes[0][1], "S2_free_center_nM", "S2_free")
    overlay(axes[0][2], "S2_total_center_nM", "S2_total")

    ax = axes[1][0]
    ax.plot(summary["param_value"], summary["half_time_hr"],
             marker="o", color="tab:green")
    ax.set_xlabel(SWEEP_PARAMETER)
    ax.set_ylabel("Time for I2 to reach midpoint (hours)")
    ax.set_title(f"I2 half-time vs {SWEEP_PARAMETER}")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(summary["param_value"], summary["I2_final_nM"],
             marker="o", color="tab:purple")
    ax.set_xlabel(SWEEP_PARAMETER)
    ax.set_ylabel("Final I2 (nM)")
    ax.set_title(f"Final I2 vs {SWEEP_PARAMETER}")
    ax.grid(alpha=0.3)

    axes[1][2].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(data),
               fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{SWEEP_PARAMETER} sweep", y=1.06, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = OUTPUT_DIR / f"combined_{SWEEP_PARAMETER}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    data = load_all()
    print(f"Loaded {len(data)}/{len(PARAMETER_VALUES)} timeseries CSV(s) "
          f"for {SWEEP_PARAMETER}.")

    summary = build_summary(data)
    plot_combined(data, summary)

    print("\nDone.")
