"""
Slide-friendly re-plot of the timeseries CSVs sitting in this folder.

Differences from the figures Sweep_two_parameters.py produces:
  * panels are laid out 3-across then 2-across instead of one tall column
  * centre-point probe only -- the node-edge series are ignored
  * the leftover 6th slot carries the legend, so there is no empty frame

Reads only the CSVs already in this directory; nothing is re-simulated.
Writes slides_timeseries_{I2,S2_free,S2_total}.png alongside them, leaving the
original timeseries_*.png untouched.

    python Replot_timeseries_for_slides.py
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
P1 = json.loads((HERE / "run_config.json").read_text())["parameter_one"]
P2 = json.loads((HERE / "run_config.json").read_text())["parameter_two"]

# Swept values are recovered from the filenames, not from run_config.json --
# that file stores them as numpy repr strings ("[0.1 0.2 ...]"), which do not
# parse back cleanly.
PATTERN = re.compile(
    rf"^timeseries_{re.escape(P1)}=([^_]+)_{re.escape(P2)}=([^_]+)_rep=(\d+)$")

runs = {}                                   # (value_one, value_two) -> DataFrame
for path in HERE.glob("timeseries_*_rep=*.csv"):
    match = PATTERN.match(path.stem)
    if match:
        runs[(float(match.group(1)), float(match.group(2)))] = pd.read_csv(path)

VALUES_ONE = sorted({v1 for v1, _ in runs})     # colour within each panel
VALUES_TWO = sorted({v2 for _, v2 in runs})     # one panel each
print(f"{len(runs)} run(s): {len(VALUES_ONE)} x {len(VALUES_TWO)} grid")

SPECIES = [
    ("I2",       "I2_center_nM",       "[I2] (nM)"),
    ("S2_free",  "S2_free_center_nM",  "free [S2] (nM)"),
    ("S2_total", "S2_total_center_nM", "total [S2] (nM)"),
]

N_COLS = 3
colors = plt.cm.viridis(np.linspace(0, 0.9, len(VALUES_ONE)))

for name, column, ylabel in SPECIES:
    n_rows = int(np.ceil(len(VALUES_TWO) / N_COLS))
    # sharey so the panels can be compared by eye; sharex is left off so every
    # panel keeps its own time ticks, including the one with no panel beneath it.
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, 5.0 * n_rows),
                             squeeze=False, sharey=True)
    panels = list(axes.flat)
    fig.suptitle(f"{name} at receiver centre   (colour = {P1})",
                 fontsize=16, fontweight="bold")

    for ax, v2 in zip(panels, VALUES_TWO):
        for color, v1 in zip(colors, VALUES_ONE):
            df = runs.get((v1, v2))
            if df is not None:
                ax.plot(df["time_hours"], df[column],
                        color=color, lw=2.2, label=f"{P1} = {v1:g}")
        ax.set_title(f"{P2} = {v2:g}", fontsize=13)
        ax.set_xlabel("Time (hours)")
        ax.grid(alpha=0.3)

    for row in range(n_rows):                   # y label once per row
        axes[row, 0].set_ylabel(ylabel, fontsize=12)

    # Any slot past the last value becomes the legend rather than a blank frame.
    handles, labels = panels[0].get_legend_handles_labels()
    for ax in panels[len(VALUES_TWO):]:
        ax.axis("off")
        ax.legend(handles, labels, loc="center", fontsize=13,
                  frameon=False, title=P1, title_fontsize=14)

    fig.tight_layout()
    out = HERE / f"slides_timeseries_{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.name}")


# =============================================================================
# FLIPPED VIEW: final [I2] and t-half vs the DEGRADATION RATE, one curve per
# threshold -- the transpose of map_*.png, plus Desmos-ready fitting tables.
# =============================================================================

SCALE = 1e-4          # k_d_ds is ~1e-4, which regresses badly; work in 1e-4 units
K_UNIT = "k_d_ds / 1e-4 s^-1"

raw = pd.read_csv(HERE / "raw_results.csv")

# t-half is the time to reach the midpoint between the INITIAL and FINAL [I2].
# When [I2] barely moves over the whole run, that midpoint lands within the first
# few samples and the number stops meaning "turn-on time" -- it is measuring the
# midpoint of nothing. Those points are drawn hollow and are kept out of the
# fitting tables, since they would otherwise drag any fit badly.
MIN_SWING_NM = 20.0
swing = {key: df["I2_center_nM"].iloc[0] - df["I2_center_nM"].iloc[-1]
         for key, df in runs.items()}
raw["I2_swing_nM"] = [swing[(a, b)] for a, b in zip(raw.value_one, raw.value_two)]
raw["reliable"] = raw["I2_swing_nM"] >= MIN_SWING_NM

QUANTITIES = [
    ("I2_center_final_nM",  "Final [I2] at receiver centre (nM)", False),
    ("half_time_center_hr", "Half-time of [I2] (hours)",          True),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Flipped view: quantities vs {P2}, one curve per {P1}",
             fontsize=16, fontweight="bold")

for col, (metric, ylabel, mark_bad) in enumerate(QUANTITIES):
    for color, v1 in zip(colors, VALUES_ONE):
        s = raw[raw.value_one == v1].sort_values("value_two")
        k, y, ok = s.value_two / SCALE, s[metric], s.reliable
        for row in (0, 1):
            axes[row, col].plot(k, y, "-", color=color, lw=2.2,
                                label=f"{P1} = {v1:g}")
            # filled = trustworthy, hollow = the swing was too small to mean anything
            axes[row, col].plot(k[ok], y[ok], "o", color=color, ms=7)
            if mark_bad and (~ok).any():
                axes[row, col].plot(k[~ok], y[~ok], "o", ms=9,
                                    mfc="none", mec=color, mew=2)

    for row in (0, 1):
        ax = axes[row, col]
        ax.set_xlabel(K_UNIT)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        if row == 1:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("log-log (straight line => power law)", fontsize=11)
        else:
            ax.set_title(ylabel, fontsize=13)
    axes[0, col].legend(fontsize=10, title=P1)

if (~raw.reliable).any():
    fig.text(0.5, -0.01, f"hollow markers: [I2] moved < {MIN_SWING_NM:g} nM over "
             f"the run, so t-half is not a meaningful turn-on time",
             ha="center", fontsize=11, style="italic")

fig.tight_layout()
fig.savefig(HERE / "slides_flipped_vs_k_d_ds.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote slides_flipped_vs_k_d_ds.png")


# ------------------------------------------------------------------ Desmos ---

def desmos_list(values, fmt="{:g}"):
    return "[" + ",".join(fmt.format(v) for v in values) + "]"


lines = [
    "DESMOS PASTE-IN TABLES AND REGRESSIONS",
    "=" * 70,
    f"x = {P2} in units of {SCALE:g} s^-1  (so 3e-4 is entered as 3)",
    f"Curve i corresponds to {P1} = " + ", ".join(f"{v:g}" for v in VALUES_ONE),
    "",
    "Paste one block at a time. Desmos fits the ~ line and reports the",
    "parameters plus R^2. Subscripts: type x_1 and Desmos makes the subscript.",
    "",
    "-" * 70,
    "BLOCK A -- final [I2] vs degradation rate",
    "-" * 70,
    "",
    "Physically motivated shape: I2 and S2:I2 are a closed pair (their sum is",
    "the initial 100 nM), so at steady state k_slow*S2*I2 = k_d_ds*(100 - I2),",
    "which rearranges to a saturating Langmuir curve",
    "",
    "    I2_final = 100 * k / (k + c)",
    "",
    "with c = k_slow*S2_ss. Fit c per threshold and see how it moves.",
    "",
]

for i, v1 in enumerate(VALUES_ONE, start=1):
    s = raw[raw.value_one == v1].sort_values("value_two")
    lines += [
        f"# {P1} = {v1:g}",
        f"x_{i} = {desmos_list(s.value_two / SCALE)}",
        f"y_{i} = {desmos_list(s.I2_center_final_nM, '{:.4f}')}",
        f"y_{i} ~ 100*x_{i}/(x_{i} + c_{i})            # Langmuir / saturating",
        f"y_{i} ~ m_{i}*x_{i}^(n_{i})                  # plain power law",
        "",
    ]

lines += [
    "-" * 70,
    "BLOCK B -- half-time vs degradation rate",
    "-" * 70,
    "",
    f"Only points where [I2] fell by at least {MIN_SWING_NM:g} nM are included;",
    "elsewhere the half-time is the midpoint of a flat line and means nothing.",
    "Note the exponent CHANGES SIGN with threshold, so one power law will not",
    "cover all five curves -- fit them separately.",
    "",
]

for i, v1 in enumerate(VALUES_ONE, start=1):
    s = raw[(raw.value_one == v1) & raw.reliable].sort_values("value_two")
    dropped = (raw.value_one == v1).sum() - len(s)
    note = f"   ({dropped} point(s) dropped as unreliable)" if dropped else ""
    lines += [
        f"# {P1} = {v1:g}{note}",
        f"u_{i} = {desmos_list(s.value_two / SCALE)}",
        f"v_{i} = {desmos_list(s.half_time_center_hr, '{:.4f}')}",
        f"v_{i} ~ g_{i}*u_{i}^(p_{i})                  # power law",
        f"v_{i} ~ s_{i} + t_{i}*u_{i}                  # straight line",
        "",
    ]

lines += [
    "-" * 70,
    "BLOCK C -- how the Langmuir constant c depends on the threshold",
    "-" * 70,
    "",
    "c computed point-by-point as c = k*(100/I2_final - 1). If the Langmuir form",
    "were exact, each row would be constant. It drifts with k because S2_ss",
    "itself depends on k_d_ds -- but at FIXED k it is close to linear in the",
    f"threshold, which is the cleanest relationship in this data set.",
    "",
]

for j, v2 in enumerate(VALUES_TWO, start=1):
    s = raw[raw.value_two == v2].sort_values("value_one")
    c = s.value_two.values / SCALE * (100.0 / s.I2_center_final_nM.values - 1.0)
    lines += [
        f"# {P2} = {v2:g}",
        f"T_{j} = {desmos_list(s.value_one)}",
        f"C_{j} = {desmos_list(c, '{:.4f}')}",
        f"C_{j} ~ A_{j} + B_{j}*T_{j}                  # linear in threshold",
        "",
    ]

text = "\n".join(lines)
(HERE / "desmos_fits.txt").write_text(text)
print("Wrote desmos_fits.txt")
print()
print(text)
