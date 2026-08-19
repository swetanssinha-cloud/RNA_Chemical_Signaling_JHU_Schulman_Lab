"""
Re-plot this folder's on/off sweep results in a readable form.

The original timeseries_on_off_<param>.png (written by sweep_core_on_off.py's
plot_timeseries()) puts one legend entry per sweep value -- fine for ~10
values, unreadable at 50, which is what every 50-value sweep in this project
produces. This script makes two figures instead:

  1. The same I2-vs-time overlay, but colored by a colorbar instead of a
     50-entry legend.
  2. Three derived statistics per run, plotted against the swept parameter:
       - t_down: time for I2 to first drop to the midpoint between its value
         at t=0 and its value at the moment I1O2 was shut off (the ON
         phase's own endpoints -- NOT the run-wide max/min, which can differ
         when I2 keeps drifting down for a bit after shutoff before turning
         around, which does happen in this data).
       - t_up: time for I2 to climb back through that SAME midpoint,
         searching from the shutoff time onward (NaN / gap if it never does
         -- e.g. k_d_ds=0, where unbinding is off entirely so I2 can only
         fall, never recover).
       - I2_min: the true minimum I2 over the whole run (not just the ON
         phase), since the minimum can land slightly into the OFF phase.

Runs whose OFF phase hit off_phase_max_time as a timeout rather than
reaching genuine convergence (off_converged=False in the .meta.json) are
flagged in the printed output -- their t_up / I2_min reflect "state at the
cap", not necessarily the true eventual recovery.

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


def nearest_index(time_hours, t_target):
    """Row index whose time_hours is closest to t_target."""
    return int((time_hours - t_target).abs().to_numpy().argmin())


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


def off_on_timings(df, meta):
    """
    midpoint = halfway between I2 at t=0 and I2 at the moment I1O2 was shut
    off (the ON phase's own start/end, i.e. base.half_time()'s definition
    but scoped to the ON phase instead of the whole run).

    t_down: time to first drop to that midpoint, searched from t=0.
    t_up:   time to climb back through that SAME midpoint, searched from the
            shutoff time onward -- not from the run-wide minimum, since I2
            sometimes keeps drifting down a bit after shutoff before turning
            around, which would make "search from the minimum" and "search
            from shutoff" disagree.
    I2_min: true minimum I2 over the WHOLE run (may land a bit into the OFF
            phase, for the same reason).
    """
    t = df["time_hours"].to_numpy()
    y = df["I2_center_nM"].to_numpy()

    t_shutoff = meta.get("t_shutoff_hr")
    shutoff_idx = nearest_index(df["time_hours"], t_shutoff) if t_shutoff is not None else 0

    y0 = y[0]
    y_shutoff = y[shutoff_idx]
    midpoint = 0.5 * (y0 + y_shutoff)

    t_down = crossing_time(t, y, midpoint, from_idx=0, direction="down")
    t_up = crossing_time(t, y, midpoint, from_idx=shutoff_idx, direction="up")
    i2_min = float(y.min())
    return t_down, t_up, midpoint, y0, y_shutoff, i2_min


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


def plot_off_on_stats(param_name, sweep_values):
    rows = []
    for value in sweep_values:
        df, meta = load_run(param_name, value)
        if df is None:
            continue
        t_down, t_up, midpoint, y0, y_shutoff, i2_min = off_on_timings(df, meta)
        rows.append({
            "param_value": value,
            "t_down_hr": t_down,
            "t_up_hr": t_up,
            "midpoint_nM": midpoint,
            "I2_at_t0_nM": y0,
            "I2_at_shutoff_nM": y_shutoff,
            "I2_min_nM": i2_min,
            "on_converged": meta.get("on_converged"),
            "off_converged": meta.get("off_converged"),
        })

    if not rows:
        print("No timeseries files found -- nothing to plot.")
        return None

    stats = pd.DataFrame(rows).sort_values("param_value")
    stats.to_csv(HERE / f"off_on_stats_{param_name}.csv", index=False)
    print(f"Wrote {HERE / f'off_on_stats_{param_name}.csv'}")

    n_missing = stats["t_up_hr"].isna().sum()
    if n_missing:
        print(f"{n_missing} run(s) never recovered back through their own "
              f"midpoint (e.g. {param_name}=0, where unbinding is off) -- "
              f"plotted as a gap, not zero.")

    n_off_timeout = (stats["off_converged"] == False).sum()  # noqa: E712
    if n_off_timeout:
        print(f"{n_off_timeout} run(s) hit off_phase_max_time as a timeout "
              f"rather than genuinely converging -- their t_up / I2_min "
              f"reflect state at the cap, not necessarily the true eventual "
              f"recovery. See the off_converged column in "
              f"off_on_stats_{param_name}.csv.")

    # Three side-by-side panels, not one shared axis: t_down, t_up and I2_min
    # are on very different scales in this kind of data (e.g. for k_d_ds,
    # t_down sits at a near-constant ~1.2h while t_up spans tens of hours) --
    # a shared axis flattens the smaller series to an invisible line, the
    # same unreadability problem as the 50-entry legend, just from axis
    # scale instead of clutter.
    # Wide, short figsize (not matplotlib's default square-ish 6.4x4.8) and
    # constrained_layout so each panel's own x-axis label and its "1e-5"
    # tick offset text have room -- at the default size (and with a plain
    # fig.tight_layout() call) they overlapped each other and the panels
    # were too narrow to read on a slide. constrained_layout also accounts
    # for the two-line suptitle automatically instead of needing a
    # hand-tuned tight_layout(rect=...).
    fig, (ax_down, ax_up, ax_min) = plt.subplots(
        1, 3, figsize=(15, 5), sharex=True, constrained_layout=True)

    ax_down.plot(stats["param_value"], stats["t_down_hr"], "o-",
                 color="tab:red", ms=5)
    ax_down.set_ylabel("Time to drop\nto midpoint (hr)")
    ax_down.grid(alpha=0.3)

    ax_up.plot(stats["param_value"], stats["t_up_hr"], "o-",
               color="tab:blue", ms=5)
    if n_off_timeout:
        timed_out = stats[stats["off_converged"] == False]  # noqa: E712
        ax_up.scatter(timed_out["param_value"], timed_out["t_up_hr"],
                       marker="x", color="black", s=45, zorder=5,
                       label="off_converged=False (hit timeout)")
        ax_up.legend(fontsize=9, loc="lower left")
    ax_up.set_ylabel("Time to recover\nto midpoint (hr)")
    ax_up.grid(alpha=0.3)

    ax_min.plot(stats["param_value"], stats["I2_min_nM"], "o-",
                color="tab:green", ms=5)
    ax_min.set_ylabel("Minimum I2\n(nM)")
    ax_min.grid(alpha=0.3)

    for ax in (ax_down, ax_up, ax_min):    # sharex=True still needs each
        ax.set_xlabel(param_name)          # panel's own label repeated

    fig.suptitle(f"Off/on statistics: {param_name}\n"
                 f"(midpoint = halfway between I2 at t=0 and I2 at shutoff)",
                 fontsize=15)
    out = HERE / f"off_on_stats_{param_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")
    return stats


if __name__ == "__main__":
    param_name, sweep_values = load_run_config()
    print(f"parameter: {param_name}  ({len(sweep_values)} values)")

    plot_timeseries_colorbar(param_name, sweep_values)
    plot_off_on_stats(param_name, sweep_values)
