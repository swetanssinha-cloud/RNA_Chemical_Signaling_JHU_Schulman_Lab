"""
Re-plot convergence_summary.csv (already on disk in this folder) as 1 row x
4 columns instead of the original 3 rows x 1 column -- the same three panels
(final I2_center, % diff from the finest mesh tested, mesh cell count) plus
a 4th for wall_time_s, all vs. nominal min_cell_size, laid out side-by-side.
Mirrors the layout mesh_floor_sensitivity_check.py's summarize_convergence()
now writes directly; this script just re-draws it from the CSV without
rerunning anything.

wall_time_s only exists in convergence_summary.csv if the sweep script was
run with --force-rerun (a cached/skipped point's timing is unknown, not
zero -- see that script's docstring). If the column is missing or entirely
NaN, the timing panel is left empty with a note instead of a fabricated
curve.

Reads only the CSV already written by the sweep -- nothing is re-simulated.
Overwrites convergence_plot.png in place.

Run from inside this folder:
    python replot_convergence.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_FINE_DX_UM = 5.0   # current default min_cell_size, marked for reference


def main():
    csv_path = HERE / "convergence_summary.csv"
    df = pd.read_csv(csv_path).sort_values("fine_dx_nominal_um", ascending=False)

    fig, (ax_i2, ax_diff, ax_cells, ax_wall) = plt.subplots(1, 4, figsize=(19, 5), sharex=True)

    ax_i2.plot(df["fine_dx_nominal_um"], df["I2_center_final_nM"], "o-", color="C0")
    ax_i2.axvline(DEFAULT_FINE_DX_UM, color="gray", linestyle=":",
                  label=f"current default ({DEFAULT_FINE_DX_UM:.1f} um)")
    ax_i2.set_ylabel("Final I2_center (nM)")
    ax_i2.set_title("Final receiver-probe readout")
    ax_i2.legend(fontsize=8)

    ax_diff.plot(df["fine_dx_nominal_um"], df["pct_diff_from_finest"], "o-", color="C1")
    ax_diff.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="1% band")
    ax_diff.axvline(DEFAULT_FINE_DX_UM, color="gray", linestyle=":")
    ax_diff.set_ylabel("% diff from finest mesh")
    ax_diff.set_title("Convergence vs. finest mesh tested")
    ax_diff.legend(fontsize=8)

    ax_cells.plot(df["fine_dx_nominal_um"], df["n_cells"], "o-", color="C2")
    ax_cells.axvline(DEFAULT_FINE_DX_UM, color="gray", linestyle=":")
    ax_cells.set_ylabel("mesh cell count")
    ax_cells.set_title("Mesh size")

    has_timing = "wall_time_s" in df.columns and df["wall_time_s"].notna().any()
    if has_timing:
        ax_wall.plot(df["fine_dx_nominal_um"], df["wall_time_s"] / 60.0, "o-", color="C3")
        ax_wall.set_ylabel("wall time (min)")
    else:
        ax_wall.text(0.5, 0.5, "no timing data --\nrerun with --force-rerun",
                     ha="center", va="center", fontsize=10, transform=ax_wall.transAxes)
    ax_wall.axvline(DEFAULT_FINE_DX_UM, color="gray", linestyle=":")
    ax_wall.set_title("Simulation wall time")

    for ax in (ax_i2, ax_diff, ax_cells, ax_wall):
        ax.set_xlabel("nominal min_cell_size (um)   [finer ->]")
        ax.grid(alpha=0.3)
    ax_i2.invert_xaxis()   # shared x -- inverting one inverts all four

    fig.suptitle("Mesh convergence: nominal resolution sweep", fontsize=15, fontweight="bold")
    fig.tight_layout()

    out = HERE / "convergence_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
