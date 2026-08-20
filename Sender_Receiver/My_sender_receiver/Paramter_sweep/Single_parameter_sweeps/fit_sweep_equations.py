"""
Fit closed-form equations to parameter-sweep summary metrics (e.g. how the
I2 half-time depends on the swept parameter).

This mirrors the CSV-discovery logic in sweep_core.py -- same
timeseries_<param>=<value>_rep=<rep>*.csv filename convention, same
half_time() definition (copied verbatim below) -- but does not import
sweep_core itself, so it has no FiPy/gmsh dependency and only needs
numpy/scipy/pandas/matplotlib.

For each results folder, every unique swept value is reduced to one (x, y)
point (x = param_value, y = the chosen metric, averaged over replicates),
then a battery of standard closed-form models is fit to those points with
scipy.optimize.curve_fit. Models are ranked by AICc (corrected Akaike
Information Criterion), which penalizes extra parameters, so a 4-parameter
sigmoid has to earn its keep over a 2-parameter line rather than winning
just by having more knobs to turn.

Usage
-----
    python fit_sweep_equations.py
        Fit every subfolder of this script's directory that contains
        timeseries_*.csv files, independently.

    python fit_sweep_equations.py sweep_k_d_ds_Improved50_values_5mmx5mm
        Fit only that one folder.

    python fit_sweep_equations.py sweep_a sweep_b sweep_c
        Merge all listed folders (e.g. adjacent sub-ranges of the same swept
        parameter) into ONE dataset and fit that. If the same (value,
        replicate) shows up in more than one folder -- e.g. an overlapping
        boundary point -- only one copy is kept (see _select_run()). Results
        are written to a new folder unless --out-dir is given.

    python fit_sweep_equations.py <dir> --metric I2_center_final_nM
        Fit the final-I2-vs-parameter relationship instead of half-time.

    python fit_sweep_equations.py sweep_a sweep_b --timeseries
        Also write a combined I2-vs-time overlay plot across every run in
        the merged dataset.

Run with the same conda env used for the sweeps themselves:
    PATH="$HOME/miniconda/envs/diffusion/bin:$PATH" \\
        ~/miniconda/envs/diffusion/bin/python fit_sweep_equations.py
(gmsh/PATH is irrelevant here since no mesh is touched, but the env is what
has scipy/pandas/matplotlib installed.)
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent

FILENAME_RE = re.compile(
    r"^timeseries_(?P<param>.+?)=(?P<value>[-+0-9.eE]+)_rep=(?P<rep>\d+)")

METRICS = ("half_time_center_hr", "I2_center_final_nM")


# =============================================================================
# half_time() -- copied from sweep_core.py so this script has no FiPy import
# =============================================================================

def half_time(time_hours, signal):
    """Time at which the signal crosses halfway between its initial and
    final value, linearly interpolated between the two bracketing samples."""
    y0, y1 = signal[0], signal[-1]
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-12:
        return np.nan

    target = 0.5 * (y0 + y1)
    below = signal <= target if y1 < y0 else signal >= target
    if not below.any():
        return np.nan

    idx = int(np.argmax(below))
    if idx == 0:
        return float(time_hours[0])

    t0, t1 = time_hours[idx - 1], time_hours[idx]
    s0, s1 = signal[idx - 1], signal[idx]
    if s1 == s0:
        return float(t1)
    return float(t0 + (target - s0) * (t1 - t0) / (s1 - s0))


# =============================================================================
# CSV discovery -- same filename convention as sweep_core.collect_results_from_disk,
# plus dedup for the same (value, replicate) appearing more than once, either
# within one folder (leftover files from an older naming scheme) or across
# folders (overlapping sub-ranges of the same sweep, e.g. distance_between
# swept in 80-200, 200-1000, and 1000-1500 chunks that share their endpoints)
# =============================================================================

def _select_run(paths):
    """
    Pick the most trustworthy file when the same (value, replicate) shows up
    more than once. Preference order:
      1. files that carry the explicit kdss=/kdds= physics tag (added to
         disambiguate physics -- see timeseries_path_for() in sweep_core.py),
      2. among those, the one whose time series actually ran longest
         (closer to steady state -- sub-range sweep folders can differ in
         total_time even when every rate constant matches),
      3. most recently written, as a final tiebreak.

    sweep_core.collect_results_from_disk() would silently average all
    duplicates together via groupby; that is wrong here since duplicates can
    come from genuinely different physics (old vs new rate constants) or
    different simulated durations, not just replicate noise.
    """
    if len(paths) == 1:
        return paths[0], []

    tagged = [p for p in paths if "_kdss=" in p.name]
    candidates = tagged or paths

    def sort_key(p):
        try:
            t_max = pd.read_csv(p, usecols=["time_hours"])["time_hours"].max()
        except Exception:
            t_max = -1.0
        return (t_max, p.stat().st_mtime)

    chosen = max(candidates, key=sort_key)
    skipped = [p.name for p in paths if p != chosen]
    return chosen, skipped


def discover_runs(results_dirs):
    """
    Scan one or more result folders for timeseries_<param>=<value>_rep=<rep>*.csv
    files and return (param_name, dataframe) with one row per (param_value,
    replicate), pooled across all the given folders.
    """
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    results_dirs = [Path(d) for d in results_dirs]

    files = []
    for d in results_dirs:
        files.extend(sorted(d.glob("timeseries_*.csv")))
    if not files:
        return None, pd.DataFrame()

    groups = {}
    param_name = None
    for path in files:
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        this_param = m.group("param")
        param_name = param_name or this_param
        if this_param != param_name:
            # Every folder being merged should share one swept parameter;
            # ignore anything that doesn't match the first one we saw.
            continue
        value = float(m.group("value"))
        rep = int(m.group("rep"))
        groups.setdefault((value, rep), []).append(path)

    if param_name is None:
        return None, pd.DataFrame()

    rows = []
    for (value, rep), paths in sorted(groups.items()):
        chosen, skipped = _select_run(paths)
        if skipped:
            print(f"  note: {param_name}={value:g} rep={rep} has "
                  f"{len(paths)} matching files -> using {chosen.name}, "
                  f"skipping {skipped}")

        try:
            df = pd.read_csv(chosen)
        except Exception as exc:
            print(f"  skipping unreadable {chosen.name}: {exc}")
            continue
        if df.empty:
            continue

        t = df["time_hours"].to_numpy(dtype=float)
        i2 = df["I2_center_nM"].to_numpy(dtype=float)
        rows.append({
            "param_value": value,
            "replicate_id": rep,
            "I2_center_final_nM": float(i2[-1]),
            "half_time_center_hr": half_time(t, i2),
            "source_file": chosen.name,
            "source_path": chosen,
        })

    return param_name, pd.DataFrame(rows)


# =============================================================================
# Candidate equation forms
# =============================================================================

def _scale(arr):
    """A characteristic positive magnitude for arr, used to size p0 guesses."""
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span > 0:
        return span
    m = abs(float(np.nanmean(finite)))
    return m if m > 0 else 1.0


MODELS = [
    dict(
        name="linear",
        func=lambda x, a, b: a * x + b,
        param_names=("a", "b"),
        positive_x=False,
        p0=lambda x, y: [
            (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0,
            y[0],
        ],
        fmt=lambda p, v: f"y = {p[0]:.6g}*{v} + {p[1]:.6g}",
    ),
    dict(
        name="quadratic",
        func=lambda x, a, b, c: a * x**2 + b * x + c,
        param_names=("a", "b", "c"),
        positive_x=False,
        p0=lambda x, y: [
            0.0,
            (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0,
            y[0],
        ],
        fmt=lambda p, v: f"y = {p[0]:.6g}*{v}^2 + {p[1]:.6g}*{v} + {p[2]:.6g}",
    ),
    dict(
        name="exp_decay",
        func=lambda x, a, tau, c: a * np.exp(-x / tau) + c,
        param_names=("a", "tau", "c"),
        positive_x=False,
        p0=lambda x, y: [y[0] - y[-1], max(_scale(x), 1e-12), y[-1]],
        fmt=lambda p, v: f"y = {p[0]:.6g}*exp(-{v}/{p[1]:.6g}) + {p[2]:.6g}",
    ),
    dict(
        name="exp_rise",
        func=lambda x, a, tau, c: a * (1 - np.exp(-x / tau)) + c,
        param_names=("a", "tau", "c"),
        positive_x=False,
        p0=lambda x, y: [y[-1] - y[0], max(_scale(x), 1e-12), y[0]],
        fmt=lambda p, v: f"y = {p[0]:.6g}*(1 - exp(-{v}/{p[1]:.6g})) + {p[2]:.6g}",
    ),
    dict(
        name="power_law",
        func=lambda x, a, b: a * np.power(x, b),
        param_names=("a", "b"),
        positive_x=True,
        p0=lambda x, y: [max(np.median(y), 1e-12), -1.0],
        fmt=lambda p, v: f"y = {p[0]:.6g} * {v}^{p[1]:.6g}",
    ),
    dict(
        name="power_law_offset",
        func=lambda x, a, b, c: a * np.power(x, b) + c,
        param_names=("a", "b", "c"),
        positive_x=True,
        p0=lambda x, y: [y[0] - y[-1] if y[0] != y[-1] else 1.0, -1.0, y[-1]],
        fmt=lambda p, v: f"y = {p[0]:.6g} * {v}^{p[1]:.6g} + {p[2]:.6g}",
    ),
    dict(
        name="logarithmic",
        func=lambda x, a, b: a * np.log(x) + b,
        param_names=("a", "b"),
        positive_x=True,
        p0=lambda x, y: [
            (y[-1] - y[0]) / (np.log(x[-1]) - np.log(x[0]))
            if x[-1] != x[0] else 1.0,
            y[0],
        ],
        fmt=lambda p, v: f"y = {p[0]:.6g}*ln({v}) + {p[1]:.6g}",
    ),
    dict(
        name="rational",
        func=lambda x, a, b, c: a / (x + b) + c,
        param_names=("a", "b", "c"),
        positive_x=False,
        p0=lambda x, y: [
            (y[0] - y[-1]) * (0.1 * _scale(x) + 1e-9),
            0.1 * _scale(x) + 1e-9,
            y[-1],
        ],
        fmt=lambda p, v: f"y = {p[0]:.6g} / ({v} + {p[1]:.6g}) + {p[2]:.6g}",
    ),
    dict(
        name="sigmoid",
        func=lambda x, lo, hi, x0, s: lo + (hi - lo) / (1 + np.exp(-(x - x0) / s)),
        param_names=("lo", "hi", "x0", "s"),
        positive_x=False,
        p0=lambda x, y: [y[0], y[-1], float(np.median(x)), max(_scale(x) / 4, 1e-12)],
        fmt=lambda p, v: (
            f"y = {p[0]:.6g} + ({p[1]:.6g} - {p[0]:.6g}) / "
            f"(1 + exp(-({v} - {p[2]:.6g})/{p[3]:.6g}))"
        ),
    ),
]
MODEL_BY_NAME = {m["name"]: m for m in MODELS}


def fit_model(model, x, y):
    """Fit one candidate model; return a result dict, or None if it fails or
    isn't identifiable from this many points."""
    n = len(x)
    k = len(model["param_names"])
    if n <= k + 2:
        return None  # not enough points to trust this many free parameters

    p0 = model["p0"](x, y)
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(model["func"], x, y, p0=p0, maxfev=20000)
        if not np.all(np.isfinite(popt)):
            return None
        y_pred = model["func"](x, *popt)
        if not np.all(np.isfinite(y_pred)):
            return None
    except Exception:
        return None

    rss = float(np.sum((y - y_pred) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else (1.0 if rss < 1e-12 else 0.0)

    mse = max(rss / n, 1e-300)
    aic = n * np.log(mse) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

    return dict(model=model["name"], params=popt, r2=r2, aicc=aicc, n_params=k)


def equation_string(fit, param_name):
    model = MODEL_BY_NAME[fit["model"]]
    return model["fmt"](fit["params"], param_name)


def fit_and_report(x, y, param_name, metric, out_dir):
    """
    Fit every candidate model to (x, y), rank by AICc, print the ranking,
    and write equation_fit_<metric>.{json,png} into out_dir. Shared by every
    entry point (timeseries-derived datasets and precomputed metadata CSVs
    alike) so there is exactly one place that does the actual fitting.
    """
    results = []
    for model in MODELS:
        if model["positive_x"] and not np.all(x > 0):
            continue
        fit = fit_model(model, x, y)
        if fit is not None:
            results.append(fit)

    if not results:
        print("  no candidate model converged.")
        return None

    results.sort(key=lambda r: (r["aicc"], -r["r2"]))
    best = results[0]

    print(f"  best fit: {best['model']}  (R^2={best['r2']:.4f}, "
          f"AICc={best['aicc']:.2f})")
    print(f"    {metric} = {equation_string(best, param_name)}")
    print("  all candidates, ranked by AICc:")
    for r in results:
        print(f"    {r['model']:<17} R^2={r['r2']:.4f}  AICc={r['aicc']:8.2f}  "
              f"{equation_string(r, param_name)}")

    save_report(out_dir, param_name, metric, x, y, results, best)
    plot_fit(out_dir, param_name, metric, x, y, results, best)

    return dict(out_dir=str(out_dir), param_name=param_name, metric=metric,
                model=best["model"], r2=best["r2"],
                equation=equation_string(best, param_name))


# =============================================================================
# Pipeline -- runs on one folder, or a merged set of folders
# =============================================================================

def run_dataset(dirs, metric, out_dir=None, make_timeseries_plot=False,
                 timeseries_t_max_hours=None, make_summary_plot=False):
    """
    dirs: list of one or more result folders, pooled into a single dataset.
    out_dir: where to write the fit report/plot. Defaults to dirs[0] when
    there is only one input folder (matching the original single-folder
    behaviour); when merging multiple folders, defaults to a new folder
    named after the parameter and the merged value range.
    """
    label = " + ".join(d.name for d in dirs)
    print(f"\n=== {label} ===")
    param_name, raw = discover_runs(dirs)
    if raw.empty:
        print("  no timeseries CSVs found, skipping.")
        return None

    if out_dir is None:
        if len(dirs) == 1:
            out_dir = dirs[0]
        else:
            lo, hi = raw["param_value"].min(), raw["param_value"].max()
            out_dir = HERE / f"sweep_{param_name}_{lo:g}_to_{hi:g}_combined"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_raw = raw.dropna(subset=[metric])
    if metric_raw.empty:
        print(f"  no finite '{metric}' values, skipping.")
        return None

    agg = (metric_raw.groupby("param_value", as_index=False)[metric]
                     .mean()
                     .sort_values("param_value"))
    x = agg["param_value"].to_numpy(dtype=float)
    y = agg[metric].to_numpy(dtype=float)

    print(f"  parameter: {param_name}   distinct values: {len(x)}   "
          f"metric: {metric}   (from {len(metric_raw)} run(s) across "
          f"{len(dirs)} folder(s))")
    print(f"  output   : {out_dir}")

    if len(x) < 4:
        print("  fewer than 4 distinct parameter values -- too few to fit "
              "a trustworthy equation. Skipping.")
        return None

    result = fit_and_report(x, y, param_name, metric, out_dir)
    if result is None:
        return None
    result["folder"] = label

    if make_timeseries_plot:
        plot_combined_timeseries(raw, out_dir, param_name,
                                  t_max_hours=timeseries_t_max_hours)

    if make_summary_plot:
        plot_i2_halftime_summary(raw, out_dir, param_name)

    return result


# =============================================================================
# Pipeline -- fits directly from a precomputed off_on_metadata_<param>.csv
# (written by plot_off_on_metadata.py in sweep_one_parameter_turn_on_off/),
# instead of deriving a metric from timeseries CSVs. That file already
# reduces each on/off run to t_half_off_hr and t_95_off_hr (durations
# measured from shutoff, not from t=0), so this only needs to read it and
# hand its columns to the same fit_and_report() everything else uses.
# =============================================================================

def load_off_on_metadata(source):
    """source may be a folder (its one off_on_metadata_*.csv is used) or a
    CSV path directly. Returns (param_name, dataframe)."""
    source = Path(source)
    if source.is_dir():
        matches = sorted(source.glob("off_on_metadata_*.csv"))
        if not matches:
            raise FileNotFoundError(f"No off_on_metadata_*.csv found in {source}")
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple off_on_metadata_*.csv in {source}: "
                f"{[m.name for m in matches]} -- pass the file directly.")
        source = matches[0]

    m = re.match(r"off_on_metadata_(?P<param>.+)\.csv$", source.name)
    param_name = m.group("param") if m else None
    return param_name, pd.read_csv(source)


def run_metadata_fit(source, metric, out_dir=None):
    source = Path(source)
    print(f"\n=== {source.name} (off/on metadata) ===")
    param_name, table = load_off_on_metadata(source)
    if param_name is None:
        print(f"  could not infer the swept parameter from '{source.name}' "
              f"(expected off_on_metadata_<param>.csv); skipping.")
        return None
    if metric not in table.columns:
        print(f"  '{metric}' is not a column in this file "
              f"(have: {list(table.columns)}); skipping.")
        return None

    if out_dir is None:
        out_dir = source if source.is_dir() else source.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = table.dropna(subset=[metric]).sort_values("param_value")
    x = sub["param_value"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    n_dropped = len(table) - len(sub)
    dropped_note = (f"  ({n_dropped} of {len(table)} dropped as NaN -- never "
                     f"got there within the simulated window)" if n_dropped else "")
    print(f"  parameter: {param_name}   distinct values: {len(x)}{dropped_note}")
    print(f"  metric   : {metric}")
    print(f"  output   : {out_dir}")

    if len(x) < 4:
        print("  fewer than 4 distinct parameter values -- too few to fit "
              "a trustworthy equation. Skipping.")
        return None

    result = fit_and_report(x, y, param_name, metric, out_dir)
    if result is not None:
        result["folder"] = source.name
    return result


def save_report(results_dir, param_name, metric, x, y, results, best):
    payload = {
        "sweep_parameter": param_name,
        "metric": metric,
        "n_points": len(x),
        "param_values": x.tolist(),
        "metric_values": y.tolist(),
        "best_model": best["model"],
        "best_equation": equation_string(best, param_name),
        "best_r2": best["r2"],
        "best_params": {
            n: float(v) for n, v in
            zip(MODEL_BY_NAME[best["model"]]["param_names"], best["params"])
        },
        "all_candidates": [
            {
                "model": r["model"],
                "equation": equation_string(r, param_name),
                "r2": r["r2"],
                "aicc": r["aicc"],
                "params": {
                    n: float(v) for n, v in
                    zip(MODEL_BY_NAME[r["model"]]["param_names"], r["params"])
                },
            }
            for r in results
        ],
    }
    out = results_dir / f"equation_fit_{metric}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")


def plot_fit(results_dir, param_name, metric, x, y, results, best):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(x, y, color="black", zorder=5, label="sweep data")

    x_smooth = np.linspace(x.min(), x.max(), 400)

    with np.errstate(all="ignore"):
        best_func = MODEL_BY_NAME[best["model"]]["func"]
        y_best = best_func(x_smooth, *best["params"])
        ax.plot(x_smooth, y_best, color="tab:red", lw=2,
                label=f"best: {best['model']}  (R^2={best['r2']:.3f})")

        if len(results) > 1:
            second = results[1]
            second_func = MODEL_BY_NAME[second["model"]]["func"]
            y_second = second_func(x_smooth, *second["params"])
            ax.plot(x_smooth, y_second, color="tab:blue", lw=1.3, ls="--",
                    alpha=0.7,
                    label=f"runner-up: {second['model']}  (R^2={second['r2']:.3f})")

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {param_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.text(0.5, -0.03, equation_string(best, param_name),
              ha="center", fontsize=8, wrap=True)

    fig.tight_layout()
    out = results_dir / f"equation_fit_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_combined_timeseries(raw, out_dir, param_name, t_max_hours=None):
    """
    Overlay every run's I2-vs-time curve, like sweep_core.plot_timeseries(),
    but built from the already-deduped `raw` table so a value that exists in
    two merged folders (e.g. distance_between=200 in both an 80-200 and a
    200-1000 sub-range) is drawn once, from the run _select_run() chose.

    Merged folders can differ in how long they simulated (e.g. 8h vs 16h);
    t_max_hours truncates every curve to the same window so runs that only
    went to 8h don't stick out as short lines next to 16h ones.
    """
    rows = raw.sort_values("param_value")
    values = rows["param_value"].to_numpy(dtype=float)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(values)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for color, (_, row) in zip(colors, rows.iterrows()):
        df = pd.read_csv(row["source_path"])
        if t_max_hours is not None:
            df = df[df["time_hours"] <= t_max_hours]
        ax.plot(df["time_hours"], df["I2_center_nM"], color=color, lw=1.5,
                label=f"{param_name}={row['param_value']:g}")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("[I2] at centre point (nM)")
    ax.set_title("[I2] at centre point")
    ax.set_ylim(bottom=0)   # concentration can't go negative -- don't let
                            # autoscale suppress 0 and exaggerate the spread
    ax.grid(alpha=0.3)

    # A per-line legend stops being readable past ~15 curves; fall back to a
    # colorbar keyed to param_value, which scales to however many runs were
    # merged in.
    if len(values) <= 15:
        ax.legend(fontsize=7, ncol=2)
    else:
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(values.min(), values.max()))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name)

    fig.tight_layout()
    out = out_dir / f"timeseries_{param_name}_combined.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_i2_halftime_summary(raw, out_dir, param_name):
    """
    Slide-ready 2-row x 1-column metadata plot: half-time (top) and final
    [I2] (bottom) vs the swept parameter -- the same reduced pair of metrics
    as summary_I2_halftime_for_prez.png in the "*_for_prez" folders (see
    plot_summary_for_prez.py in this same directory), but built from `raw`
    (the deduped, merged-folder table discover_runs() already produced) so
    values that came from more than one merged sub-range are only plotted
    once, consistent with the equation fit and the combined timeseries plot.
    """
    agg = (raw.groupby("param_value", as_index=False)
              [["I2_center_final_nM", "half_time_center_hr"]]
              .mean()
              .sort_values("param_value"))
    xs = agg["param_value"].to_numpy(dtype=float)

    fig, (ax_half, ax_i2) = plt.subplots(2, 1, figsize=(7, 9))
    fig.suptitle(f"Parameter sweep: {param_name}", fontsize=15, fontweight="bold")

    ax_half.plot(xs, agg["half_time_center_hr"], "o-", color="tab:green", lw=2)
    ax_half.set_title("Turn-on time")
    ax_half.set_ylabel("Half-time of I2 (hours)")

    ax_i2.plot(xs, agg["I2_center_final_nM"], "o-", color="tab:blue", lw=2)
    ax_i2.set_title("Receiver switch")
    ax_i2.set_ylabel("Final [I2] at receiver (nM)")

    for ax in (ax_half, ax_i2):
        ax.set_xlabel(param_name)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"summary_I2_halftime_{param_name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dirs", nargs="*", default=None,
                         help="Folder(s) with timeseries_<param>=<value>_rep=<rep>*.csv "
                              "files. Zero: fit every subfolder of this script's "
                              "directory independently. One: fit just that folder. "
                              "Two or more: merge them into one dataset and fit that.")
    parser.add_argument("--metric", default="half_time_center_hr",
                         help="Which metric to fit vs the swept parameter. Without "
                              f"--metadata: one of {METRICS}. With --metadata: any "
                              "column in the off_on_metadata_<param>.csv, e.g. "
                              "t_half_off_hr (recovery to midpoint) or t_95_off_hr "
                              "(recovery to 95 nM).")
    parser.add_argument("--metadata", action="store_true",
                         help="Fit directly from a precomputed "
                              "off_on_metadata_<param>.csv (written by "
                              "plot_off_on_metadata.py in "
                              "sweep_one_parameter_turn_on_off/) instead of scanning "
                              "timeseries CSVs. Each positional argument (a folder or "
                              "the CSV itself) is fit independently -- no merging in "
                              "this mode.")
    parser.add_argument("--out-dir", default=None,
                         help="Where to write the merged fit's report/plot. Only "
                              "meaningful with 2+ input folders; defaults to a new "
                              "'sweep_<param>_<min>_to_<max>_combined' folder.")
    parser.add_argument("--timeseries", action="store_true",
                         help="Also write a combined I2-vs-time overlay plot across "
                              "every run in the dataset.")
    parser.add_argument("--truncate-hours", type=float, default=None,
                         help="Cut every curve in the --timeseries plot off at this "
                              "many simulated hours. Useful when merged folders ran "
                              "for different total_time (e.g. 8h vs 16h) and you want "
                              "every line to stop at the same x position.")
    parser.add_argument("--summary", action="store_true",
                         help="Also write a slide-ready 2-row x 1-column metadata plot "
                              "(half-time on top, final [I2] on bottom) vs the swept "
                              "parameter, as summary_I2_halftime_<param>.png.")
    args = parser.parse_args()

    if not args.metadata and args.metric not in METRICS:
        parser.error(f"--metric must be one of {METRICS} (or pass --metadata "
                      "to fit a column from an off_on_metadata_*.csv).")

    if args.metadata:
        if not args.results_dirs:
            parser.error("--metadata requires at least one folder or CSV path.")
        out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        summary = []
        for d in args.results_dirs:
            result = run_metadata_fit(Path(d).resolve(), args.metric, out_dir=out_dir)
            if result:
                summary.append(result)
        return

    if args.results_dirs:
        input_dirs = [Path(d).resolve() for d in args.results_dirs]
        out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        summary = []
        result = run_dataset(input_dirs, args.metric, out_dir=out_dir,
                              make_timeseries_plot=args.timeseries,
                              timeseries_t_max_hours=args.truncate_hours,
                              make_summary_plot=args.summary)
        if result:
            summary.append(result)
        return

    dirs = sorted(
        d for d in HERE.iterdir()
        if d.is_dir() and any(d.glob("timeseries_*.csv"))
    )
    if not dirs:
        print(f"No subfolders of {HERE} contain timeseries_*.csv files.")
        return

    summary = []
    for d in dirs:
        result = run_dataset([d], args.metric, make_timeseries_plot=args.timeseries,
                              timeseries_t_max_hours=args.truncate_hours,
                              make_summary_plot=args.summary)
        if result:
            summary.append(result)

    if len(dirs) > 1:
        print("\n=== summary across all sweeps ===")
        for r in summary:
            print(f"  {r['folder']:<48} {r['model']:<17} R^2={r['r2']:.4f}   "
                  f"{r['metric']} = f({r['param_name']})")
        if summary:
            out = HERE / f"equation_fit_summary_{args.metric}.json"
            out.write_text(json.dumps(summary, indent=2))
            print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
