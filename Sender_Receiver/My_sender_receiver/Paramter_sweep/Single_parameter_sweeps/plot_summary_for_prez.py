"""
Presentation-ready 2x2 metadata summary plots for the "*_for_prez" sweep
folders in this directory.

sweep_core.py's plot() already writes a 2x2 summary per sweep -- final [I2],
final total [S2], half-time, and wall-time cost -- straight from
summary_stats*.csv. Wall-time is useful while a sweep is running but not for
a presentation, so this script rebuilds that figure for the folders below
using the same already-written summary_stats*.csv (no simulation rerun),
swapping wall-time for final free [S2] and using this panel layout:

    top row:    half-time to reach midpoint | final [I2] at receiver
    bottom row: final free [S2] at receiver | final total [S2] at receiver

Hard-coded to the 3 folders that exist today (sweep_Th2_init_0_to_1_nM,
sweep_k_d_ds, sweep_k_d_ss); add a new entry to FOLDERS if another
"*_for_prez" folder shows up later.

Run from inside this folder:
    python plot_summary_for_prez.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

FOLDERS = [
    "sweep_Th2_init_0_to_1_nM_for_prez",
    "sweep_k_d_ds_For_prez",
    "sweep_k_d_ss_for_prez",
]


def load_summary(folder):
    matches = sorted(folder.glob("summary_stats*.csv"))
    if not matches:
        raise FileNotFoundError(f"No summary_stats*.csv found in {folder}")
    stats = pd.read_csv(matches[0]).sort_values("param_value")

    config_matches = sorted(folder.glob("run_config*.json"))
    sweep_parameter = (
        json.loads(config_matches[0].read_text())["sweep_parameter"]
        if config_matches else "parameter"
    )
    return stats, sweep_parameter


def plot_summary(folder_name):
    folder = HERE / folder_name
    stats, sweep_parameter = load_summary(folder)
    xs = stats["param_value"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"Parameter sweep: {sweep_parameter}",
                 fontsize=15, fontweight="bold")

    def series(ax, metric, color, title, ylabel):
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        yerr = stats[std_col] if std_col in stats.columns else None
        if yerr is not None and not np.isfinite(yerr).any():
            yerr = None          # single replicate: no error bars to draw
        ax.errorbar(xs, stats[mean_col], yerr=yerr, marker="o", capsize=4,
                    linewidth=2, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    series(axes[0, 0], "half_time_center_hr", "tab:green",
           "Turn-on time", "Half-time of I2 (hours)")
    series(axes[0, 1], "I2_center_final_nM", "tab:blue",
           "Receiver switch", "Final [I2] at receiver (nM)")
    series(axes[1, 0], "S2_free_center_final_nM", "tab:orange",
           "Free RNA remaining", "Final free [S2] at receiver (nM)")
    series(axes[1, 1], "S2_total_center_final_nM", "tab:purple",
           "Total RNA delivered", "Final total [S2] at receiver (nM)")

    for ax in axes.flat:
        ax.set_xlabel(sweep_parameter)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = folder / "summary_for_prez.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    for folder_name in FOLDERS:
        plot_summary(folder_name)
