"""
Slide-friendly re-plot of the timeseries CSVs in each two-parameter
"*_for_prez" folder below.

Differences from the timeseries_{I2,S2_free,S2_total}.png that
Sweep_two_core.py already writes in those folders:
  * centre-point probe only -- the node-edge series are dropped entirely
  * panels are laid out across exactly 2 rows instead of one tall column
    (one panel per value of the second swept parameter, colour = first
    swept parameter), which is what made the originals too tall to put on
    a slide
  * colour is read off a shared colorbar (one per figure) instead of a
    per-value legend

Reads only the CSVs already on disk -- nothing is re-simulated. Writes
slides_timeseries_{I2,S2_free,S2_total}.png into each folder, alongside the
existing timeseries_*.png (which are left untouched).

Hard-coded to the "*_for_prez" folders that exist today; add or remove
entries in FOLDERS as folders are added or renamed (e.g.
Two_parameters_varied_results_distance_between_k_d_ss_for_prez was renamed
to ..._short and dropped from here, since it's no longer a "for_prez"
folder).

Run from inside Sweep_two_parameters/:
    python plot_timeseries_slides.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

FOLDERS = [
    "Two_parameters_varied_results_k_slow_distance_between_for_prez",
    "Two_parameters_varied_results_k_d_ds_k_d_ss_for_prez",
]

SPECIES = [
    ("I2",       "I2_center_nM",       "[I2] (nM)"),
    ("S2_free",  "S2_free_center_nM",  "free [S2] (nM)"),
    ("S2_total", "S2_total_center_nM", "total [S2] (nM)"),
]

N_ROWS = 2


def load_runs(folder):
    cfg = json.loads((folder / "run_config.json").read_text())
    p1, p2 = cfg["parameter_one"], cfg["parameter_two"]

    # Swept values are recovered from the filenames, not from run_config.json
    # -- see Two_parameters_varied_results_Th2_init_k_d_ds_for_testing_rockfish
    # /Replot_timeseries_for_slides.py, which this script is adapted from.
    pattern = re.compile(
        rf"^timeseries_{re.escape(p1)}=([^_]+)_{re.escape(p2)}=([^_]+)_rep=\d+$")

    runs = {}
    for path in folder.glob("timeseries_*_rep=*.csv"):
        match = pattern.match(path.stem)
        if match:
            runs[(float(match.group(1)), float(match.group(2)))] = pd.read_csv(path)
    return p1, p2, cfg, runs


def on_intended_grid(v, intended_values):
    """
    True if v is (close to) one of the values run_config.json says were
    actually swept. Some *_for_prez folders also contain a handful of stray
    off-grid runs (e.g. one-off spot checks) that would otherwise blow up
    the panel count with mostly-empty columns -- see
    Two_parameters_varied_results_k_d_ds_k_d_ss_for_prez, which has 4 extra
    runs at k_d_ds/k_d_ss = 1e-05 and 5e-05 alongside its intended 10x10,
    3.33e-5-step grid.
    """
    return intended_values.size == 0 or np.any(
        np.isclose(v, intended_values, rtol=1e-4, atol=1e-12))


def plot_folder(folder_name):
    folder = HERE / folder_name
    p1, p2, cfg, runs = load_runs(folder)
    if not runs:
        print(f"  SKIP {folder_name}: no matching timeseries CSVs found")
        return

    cfg_values_one = np.array(cfg.get("values_one", []), dtype=float)
    cfg_values_two = np.array(cfg.get("values_two", []), dtype=float)

    values_one = sorted(v for v in {v1 for v1, _ in runs}
                         if on_intended_grid(v, cfg_values_one))
    values_two = sorted(v for v in {v2 for _, v2 in runs}
                         if on_intended_grid(v, cfg_values_two))

    n_dropped = len({v1 for v1, _ in runs} - set(values_one)) + \
        len({v2 for _, v2 in runs} - set(values_two))
    if n_dropped:
        print(f"  ({n_dropped} off-grid value(s) excluded as stray runs)")

    n_cols = int(np.ceil(len(values_two) / N_ROWS))
    values_one_arr = np.asarray(values_one, dtype=float)
    norm = mcolors.Normalize(vmin=values_one_arr.min(), vmax=values_one_arr.max())
    cmap = cm.viridis

    print(f"{folder_name}: {len(runs)} run(s), {len(values_one)} x "
          f"{len(values_two)} grid -> {N_ROWS}x{n_cols} panels")

    for name, column, ylabel in SPECIES:
        fig, axes = plt.subplots(N_ROWS, n_cols, figsize=(3.6 * n_cols, 5.0 * N_ROWS),
                                  squeeze=False, sharey=True)
        panels = list(axes.flat)
        fig.suptitle(f"{name} at receiver centre   (colour = {p1})",
                     fontsize=16, fontweight="bold")

        for ax, v2 in zip(panels, values_two):
            for v1 in values_one:
                df = runs.get((v1, v2))
                if df is not None:
                    ax.plot(df["time_hours"], df[column],
                            color=cmap(norm(v1)), lw=2.0)
            ax.set_title(f"{p2} = {v2:g}", fontsize=12)
            ax.set_xlabel("Time (hours)")
            ax.grid(alpha=0.3)

        for row in range(N_ROWS):                 # y label once per row
            axes[row, 0].set_ylabel(ylabel, fontsize=12)

        # concentration can't go negative -- don't let autoscale suppress 0
        # and exaggerate the spread; sharey=True propagates this to every
        # panel in the figure.
        axes[0, 0].set_ylim(bottom=0)

        for ax in panels[len(values_two):]:        # leftover slots, if any
            ax.axis("off")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=panels, shrink=0.85)
        cbar.set_label(p1)

        out = folder / f"slides_timeseries_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out}")


if __name__ == "__main__":
    for folder_name in FOLDERS:
        plot_folder(folder_name)
