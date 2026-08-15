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
        timeseries_*.csv files.

    python fit_sweep_equations.py sweep_k_d_ds_Improved50_values_5mmx5mm
        Fit only that one folder.

    python fit_sweep_equations.py <dir> --metric I2_center_final_nM
        Fit the final-I2-vs-parameter relationship instead of half-time.

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
# plus dedup for folders that hold leftover files from an older naming scheme
# =============================================================================

def discover_runs(results_dir):
    """
    Scan results_dir for timeseries_<param>=<value>_rep=<rep>*.csv files and
    return (param_name, dataframe) with one row per (param_value, replicate).

    Some folders (e.g. sweep_k_slow_ImprovedV4_5mmx5mm) hold leftover files
    from an older filename convention (no kdss=/kdds= physics tag) alongside
    current ones for the same value/replicate. sweep_core's own
    collect_results_from_disk() would silently average both together via
    groupby -- here, when duplicates are found, the file with the explicit
    kdss=/kdds= tag is kept (it is self-describing about which physics were
    used) and the rest are reported as skipped instead.
    """
    files = sorted(results_dir.glob("timeseries_*.csv"))
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
            # A folder should only ever hold one swept parameter; ignore
            # anything that doesn't match the first one we saw.
            continue
        value = float(m.group("value"))
        rep = int(m.group("rep"))
        groups.setdefault((value, rep), []).append(path)

    if param_name is None:
        return None, pd.DataFrame()

    rows = []
    for (value, rep), paths in sorted(groups.items()):
        if len(paths) > 1:
            tagged = [p for p in paths if "_kdss=" in p.name]
            chosen = max(tagged or paths, key=lambda p: p.stat().st_mtime)
            skipped = [p.name for p in paths if p != chosen]
            print(f"  note: {param_name}={value:g} rep={rep} has "
                  f"{len(paths)} matching files -> using {chosen.name}, "
                  f"skipping {skipped}")
        else:
            chosen = paths[0]

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


# =============================================================================
# Per-folder pipeline
# =============================================================================

def process_folder(results_dir, metric):
    print(f"\n=== {results_dir.name} ===")
    param_name, raw = discover_runs(results_dir)
    if raw.empty:
        print("  no timeseries CSVs found, skipping.")
        return None

    raw = raw.dropna(subset=[metric])
    if raw.empty:
        print(f"  no finite '{metric}' values, skipping.")
        return None

    agg = (raw.groupby("param_value", as_index=False)[metric]
              .mean()
              .sort_values("param_value"))
    x = agg["param_value"].to_numpy(dtype=float)
    y = agg[metric].to_numpy(dtype=float)

    print(f"  parameter: {param_name}   distinct values: {len(x)}   "
          f"metric: {metric}   (from {len(raw)} run(s))")

    if len(x) < 4:
        print("  fewer than 4 distinct parameter values -- too few to fit "
              "a trustworthy equation. Skipping.")
        return None

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

    save_report(results_dir, param_name, metric, x, y, results, best)
    plot_fit(results_dir, param_name, metric, x, y, results, best)

    return dict(folder=results_dir.name, param_name=param_name, metric=metric,
                model=best["model"], r2=best["r2"],
                equation=equation_string(best, param_name))


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


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=None,
                         help="Folder with timeseries_<param>=<value>_rep=<rep>*.csv "
                              "files. If omitted, every subfolder of this script's "
                              "directory containing such files is processed.")
    parser.add_argument("--metric", choices=METRICS, default="half_time_center_hr",
                         help="Which summary metric to fit vs the swept parameter "
                              "(default: half_time_center_hr).")
    args = parser.parse_args()

    if args.results_dir:
        dirs = [Path(args.results_dir).resolve()]
    else:
        dirs = sorted(
            d for d in HERE.iterdir()
            if d.is_dir() and any(d.glob("timeseries_*.csv"))
        )
        if not dirs:
            print(f"No subfolders of {HERE} contain timeseries_*.csv files.")
            return

    summary = []
    for d in dirs:
        result = process_folder(d, args.metric)
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
