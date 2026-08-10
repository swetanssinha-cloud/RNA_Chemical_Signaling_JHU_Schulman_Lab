"""
Mesh convergence (verification) study for the conformal triangular mesh.
Title: Convergence_stuidy_Claudes_Triangular_mesh_FIXEDV4.py
PURPOSE
-------
This answers "is my simulation resolved?" WITHOUT reference to COMSOL.

Comparing FiPy against COMSOL at the same nominal element size is a weak test:
if both are under-resolved in similar ways they agree with each other while
both being wrong. COMSOL's element size is also not comparable to Gmsh's,
because COMSOL may use quadratic (P2) elements while FiPy is a cell-centred
finite-volume scheme with piecewise-constant unknowns. A P2 solution at 5 um
can be as accurate as a finite-volume solution at 1 um.

So we refine OUR mesh until OUR answer stops changing, extrapolate to zero
cell size, and put an error bar on it. Only then does a COMSOL comparison
mean anything: if the extrapolated value sits several percent from COMSOL,
then COMSOL is carrying that error, not us.

METHOD
------
The mesh is halved EVERYWHERE, repeatedly. The generator's ring schedule has
a closed form:

    size(d) = min_cell_size + d * (growth_rate - 1) / cells_per_level

capped at max_cell_size, where d is distance from a node surface. Halving
size(d) at EVERY d therefore requires three things to scale together:

    min_cell_size    / 2
    max_cell_size    / 2      (kept at 25 * min_cell_size)
    cells_per_level  * 2      (kept at 12 / min_cell_size)

growth_rate is the only quantity genuinely held fixed.

An earlier version of this study varied min_cell_size alone, holding
max_cell_size and cells_per_level fixed. That freezes both the d-dependent
term and the cap, so only a sliver about 6*min_cell_size wide around each
node actually refines. Measured: the bath the RNA crosses stayed at ~31 um
and the far field at ~64 um on every level, cell counts rose x1.6 instead of
the x4 a 2D halving demands, and the effective refinement ratio was 1.27,
not 2. That sequence measured mesh noise, not resolution.

With all three scaled, r = 2 is real and Richardson extrapolation on any
three consecutive grids (coarse, medium, fine) is valid:

    eps_32 = f_coarse - f_medium
    eps_21 = f_medium - f_fine
    p      = ln|eps_32 / eps_21| / ln(2)            observed order of accuracy
    f_ext  = f_fine + (f_fine - f_medium) / (2^p - 1)   estimate of the exact value

and Roache's Grid Convergence Index, the standard reported error bar:

    GCI = 1.25 * |eps_21 / f_fine| / (2^p - 1)

GCI is read as "the fine-grid answer is within about GCI percent of the
converged answer".

Interpreting p:
    p ~ 1-2   normal, healthy convergence
    p > 3     probably not in the asymptotic range yet; refine further
    p < 0.5   something is limiting accuracy other than cell size
    p is nan  the sequence is oscillating, not converging monotonically

QUANTITIES TRACKED
------------------
The volume-averaged quantities converge more smoothly than the point probe,
because integrals average out cell-to-cell noise while the probe cell is a
different triangle on every mesh. Expect I2_avg to give the cleanest p, and
treat I2_center as the one that matters for the COMSOL comparison (COMSOL's
exports are point evaluations).

PHYSICS
-------
Imported directly from Parameter_sweep_unified so that this study verifies
exactly the code the sweep runs. There is deliberately no second copy of the
equations here.

USAGE
-----
    # see cell counts and time estimates without running anything
    python Convergence_stuidy_Claudes_Triangular_mesh_FIXEDV4.py --estimate

    # run it
    python Convergence_stuidy_Claudes_Triangular_mesh_FIXEDV4.py

    # run it in the background, keeping the Mac awake, logging to a file
    caffeinate -i nohup python Convergence_stuidy_Claudes_Triangular_mesh_FIXEDV4.py \
        > convergence_log.txt 2>&1 &

Finished runs are skipped on restart, so the job is safe to interrupt and
resume.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# One BLAS thread per worker. Without this, every worker process spawns its own
# thread pool, they fight over the same cores, and everything runs slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fipy import Gmsh2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Mesh.New_simple_mesh import create_conformal_radial_mesh
from Paramter_sweep.Parameter_sweep_unified import (
    build_equations,
    initialize_fields,
    volume_average,
    half_time,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

FINE_DX_VALUES = [4.0, 2.0, 1.0]     # constant refinement ratio of 2

# The only quantity genuinely held fixed. Changing it mid-study would make
# the comparison meaningless.
GROWTH_RATE = 1.5

# max_cell_size and cells_per_level are NOT fixed -- they are derived from
# fine_dx so that size(d) = fine_dx + d*(GROWTH_RATE-1)/cells_per_level
# halves at every distance d when fine_dx halves. See the METHOD section.
#
#   fine_dx   coarse_dx   cells_per_level   cells
#      4         100             3           9,122
#      2          50             6          36,092   (x3.96)
#      1          25            12         142,516   (x3.95)
#
# The x4 per level is what a 2D halving requires; the old fixed-parameter
# version managed only x1.6.

COARSE_DX_RATIO = 25.0      # coarse_dx = COARSE_DX_RATIO * fine_dx
CELLS_PER_LEVEL_NUMER = 12.0  # cells_per_level = CELLS_PER_LEVEL_NUMER / fine_dx


def coarse_dx_for(fine_dx):
    """max_cell_size for this level. Scales with fine_dx so the coarse/fine
    ratio stays constant and the far field refines along with everything
    else."""
    return COARSE_DX_RATIO * fine_dx


def cells_per_level_for(fine_dx):
    """Ring width in cells. Scales inversely with fine_dx so the coarsening
    slope (GROWTH_RATE-1)/cells_per_level halves whenever fine_dx halves."""
    n = int(round(CELLS_PER_LEVEL_NUMER / fine_dx))
    if n < 1:
        raise ValueError(
            f"fine_dx={fine_dx:g} gives cells_per_level={n}; it must be >= 1. "
            f"Raise CELLS_PER_LEVEL_NUMER or use a smaller fine_dx range.")
    return n

# Physical setup -- matched to TG_Rmesh_tanh.py at the distance you have
# COMSOL data for.
PARAMS = {
    "D_solution": 150.0,
    "D_gel": 60.0,
    "k_p": 0.2,
    "k_d_ds": 3e-4,
    "k_d_ss": 3e-4,
    "k_slow": 1e5 * 1e-6,
    "k_fast": 1e6 * 1e-6,
    "I1O2_init": 0.1,
    "I2_init": 0.1,
    "Th2_init": 5.0,
    "node_diameter": 75.0,
    "distance_between": 1200.0,
    "total_width": 1e4,
    "total_height": 1e3,
    "dt": 60.0,
    "total_time": 8 * 3600,
    "save_interval_time": 60.0,
}

# Number of simulations to run at once.
#
# Multiprocessing works exactly as it does in the sweep: each level is an
# independent simulation, so they parallelise perfectly. Two caveats:
#
#   memory  -- the 0.25 um mesh has ~86,000 cells and five coupled species,
#              so ~430,000 unknowns. Running every level simultaneously needs
#              several GB. Reduce this number if the machine starts swapping.
#
#   timing  -- with other work running, wall times become meaningless as a
#              performance measure. That does NOT affect the converged values,
#              which is all this study cares about.
#
N_PROCESSES = 2

OUTPUT_DIR = Path(__file__).resolve().parent / "convergence_v4"
MESH_DIR = OUTPUT_DIR / "meshes"

# Which scalars to run the convergence analysis on.
QUANTITIES = {
    "I2_center_final_nM":    "Final [I2], probe cell (COMSOL-comparable)",
    "I2_avg_final_nM":       "Final [I2], node volume average",
    "S2_total_avg_final_nM": "Final total [S2], node volume average",
    "half_time_avg_hr":      "Half-time of node-average I2",
}


# =============================================================================
# ONE REFINEMENT LEVEL
# =============================================================================

def _mesh_tag(fine_dx):
    """Every parameter that changes the mesh, in the filename. Without
    coarse_dx and cells_per_level here, a run from the old fixed-parameter
    version would be silently reused against a different mesh."""
    return (f"ccd={PARAMS['distance_between']:.0f}"
            f"_fine={fine_dx:g}"
            f"_coarse={coarse_dx_for(fine_dx):g}"
            f"_gr={GROWTH_RATE:g}"
            f"_cpl={cells_per_level_for(fine_dx):g}")


def mesh_path_for(fine_dx):
    return MESH_DIR / f"conv_{_mesh_tag(fine_dx)}.msh"


def result_path_for(fine_dx):
    return OUTPUT_DIR / f"timeseries_{_mesh_tag(fine_dx)}.csv"


def run_level(fine_dx):
    """Run one refinement level. Returns a dict of scalar results."""
    tag = f"fine_dx={fine_dx:g}"
    ts_path = result_path_for(fine_dx)

    

    # --- resume: skip anything already finished
    if ts_path.exists():
        df = pd.read_csv(ts_path)
        print(f"  SKIP  {tag:<16} already complete ({len(df)} samples)", flush=True)
        return scalars_from(df, fine_dx, wall_time_s=np.nan, reused=True)

    start = time.perf_counter()

    mesh = Gmsh2D(str(mesh_path_for(fine_dx)))
    x = np.asarray(mesh.cellCenters[0])
    y = np.asarray(mesh.cellCenters[1])

    y_center = PARAMS["total_height"] / 2.0
    sender_x = PARAMS["total_width"] / 2.0 - PARAMS["distance_between"] / 2.0
    receiver_x = PARAMS["total_width"] / 2.0 + PARAMS["distance_between"] / 2.0

    (S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
     sender_mask, receiver_mask) = initialize_fields(
        mesh, x, y, sender_x, receiver_x, y_center, PARAMS)

    eq = build_equations(
        S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
        k_p=PARAMS["k_p"], k_slow=PARAMS["k_slow"], k_fast=PARAMS["k_fast"],
        k_d_ss=PARAMS["k_d_ss"], k_d_ds=PARAMS["k_d_ds"],
    )

    volumes = np.asarray(mesh.cellVolumes)
    probe = int(np.argmin(np.hypot(x - receiver_x, y - y_center)))

    dt = PARAMS["dt"]
    n_steps = int(PARAMS["total_time"] / dt)
    save_every = max(1, int(PARAMS["save_interval_time"] / dt))

    rows = []

    def sample(step):
        rows.append({
            "time_hours": step * dt / 3600.0,
            "I2_center_nM": float(I2.value[probe]) * 1e3,
            "S2_free_center_nM": float(S2.value[probe]) * 1e3,
            "S2_total_center_nM": (float(S2.value[probe])
                                   + float(S2_I2.value[probe])
                                   + float(S2_Th2.value[probe])) * 1e3,
            "I2_avg_nM": volume_average(I2, receiver_mask, volumes) * 1e3,
            "S2_free_avg_nM": volume_average(S2, receiver_mask, volumes) * 1e3,
            "S2_total_avg_nM": (volume_average(S2, receiver_mask, volumes)
                                + volume_average(S2_I2, receiver_mask, volumes)
                                + volume_average(S2_Th2, receiver_mask, volumes)) * 1e3,
        })

    sample(0)

    for step in range(1, n_steps + 1):
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        S2_I2.updateOld()
        S2_Th2.updateOld()

        res, n_sweeps = 1e10, 0
        while res > 1e-6 and n_sweeps < 10:
            res = eq.sweep(dt=dt)
            n_sweeps += 1

        if step % save_every == 0:
            sample(step)

        # progress ping roughly every simulated hour, so a background log
        # shows the job is alive
        if step % max(1, int(3600 / dt)) == 0:
            print(f"        {tag:<16} t={step * dt / 3600:.1f} h "
                  f"({time.perf_counter() - start:.0f} s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ts_path, index=False)

    wall = time.perf_counter() - start
    out = scalars_from(df, fine_dx, wall_time_s=wall, reused=False)
    out["n_cells"] = int(mesh.numberOfCells)

    print(f"  DONE  {tag:<16} cells={mesh.numberOfCells:>7,}  "
          f"I2_center={out['I2_center_final_nM']:7.3f} nM  "
          f"I2_avg={out['I2_avg_final_nM']:7.3f} nM  "
          f"[{wall / 60:.1f} min]", flush=True)
    return out


def scalars_from(df, fine_dx, wall_time_s, reused):
    final = df.iloc[-1]
    return {
        "fine_dx": fine_dx,
        "n_cells": np.nan,
        "I2_center_final_nM": float(final["I2_center_nM"]),
        "I2_avg_final_nM": float(final["I2_avg_nM"]),
        "S2_total_center_final_nM": float(final["S2_total_center_nM"]),
        "S2_total_avg_final_nM": float(final["S2_total_avg_nM"]),
        "half_time_center_hr": half_time(df["time_hours"].values,
                                         df["I2_center_nM"].values),
        "half_time_avg_hr": half_time(df["time_hours"].values,
                                      df["I2_avg_nM"].values),
        "wall_time_s": wall_time_s,
        "reused": reused,
    }


# =============================================================================
# RICHARDSON EXTRAPOLATION
# =============================================================================

def richardson(coarse, medium, fine, ratio=2.0):
    """
    Observed order p, extrapolated value, and Roache GCI for one grid triplet.

    `coarse`, `medium`, `fine` are the same scalar computed on grids whose
    cell sizes differ by `ratio`.
    """
    eps_32 = coarse - medium
    eps_21 = medium - fine

    if not np.isfinite(eps_21) or abs(eps_21) < 1e-14:
        return dict(p=np.nan, extrapolated=fine, gci_percent=0.0,
                    note="already converged to machine precision")

    if eps_32 * eps_21 <= 0:
        return dict(p=np.nan, extrapolated=np.nan, gci_percent=np.nan,
                    note="oscillatory - not monotone convergence")

    p = np.log(abs(eps_32 / eps_21)) / np.log(ratio)
    denom = ratio ** p - 1.0
    if abs(denom) < 1e-12:
        return dict(p=p, extrapolated=np.nan, gci_percent=np.nan,
                    note="p too close to zero to extrapolate")

    extrapolated = fine + (fine - medium) / denom
    gci = 1.25 * abs(eps_21 / fine) / denom * 100.0 if fine != 0 else np.nan

    if p > 3.0:
        note = "p high - probably not yet in the asymptotic range"
    elif p < 0.5:
        note = "p low - something other than cell size is limiting accuracy"
    else:
        note = "healthy"

    return dict(p=p, extrapolated=extrapolated, gci_percent=gci, note=note)


def analyse(results_df):
    """Run Richardson on every consecutive triplet, for every quantity."""
    df = results_df.sort_values("fine_dx", ascending=False).reset_index(drop=True)
    lines = []

    print("\n" + "=" * 78)
    print("CONVERGENCE ANALYSIS")
    print("=" * 78)

    for key, description in QUANTITIES.items():
        if key not in df.columns or df[key].isna().all():
            continue

        print(f"\n{description}")
        print(f"  {'fine_dx':>8} {'cells':>9} {'value':>14} {'change vs prev':>16}")
        prev = None
        for _, row in df.iterrows():
            value = row[key]
            delta = "" if prev is None else f"{abs(value - prev) / abs(prev) * 100:14.3f} %"
            cells = "" if not np.isfinite(row["n_cells"]) else f"{int(row['n_cells']):,}"
            print(f"  {row['fine_dx']:>8g} {cells:>9} {value:>14.6f} {delta:>16}")
            prev = value

        for i in range(len(df) - 2):
            coarse, medium, fine = (df[key].iloc[i], df[key].iloc[i + 1],
                                    df[key].iloc[i + 2])
            sizes = (df["fine_dx"].iloc[i], df["fine_dx"].iloc[i + 1],
                     df["fine_dx"].iloc[i + 2])
            r = richardson(coarse, medium, fine)
            triplet = f"    triplet dx={sizes[0]:g}/{sizes[1]:g}/{sizes[2]:g}: "
            if np.isfinite(r["p"]):
                print(triplet
                      + f"p={r['p']:.2f}  extrapolated={r['extrapolated']:.6f}  "
                        f"GCI={r['gci_percent']:.3f} %   [{r['note']}]")
            else:
                print(triplet + f"[{r['note']}]")
            lines.append(dict(quantity=key, dx_coarse=sizes[0],
                              dx_medium=sizes[1], dx_fine=sizes[2], **r))

    if lines:
        out = OUTPUT_DIR / "richardson_analysis.csv"
        pd.DataFrame(lines).to_csv(out, index=False)
        print(f"\nWrote {out}")

    return pd.DataFrame(lines)


def plot(results_df, analysis_df):
    df = results_df.sort_values("fine_dx", ascending=False)
    keys = [k for k in QUANTITIES if k in df.columns and not df[k].isna().all()]
    if not keys:
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4),
                             squeeze=False)

    for ax, key in zip(axes[0], keys):
        ax.plot(df["fine_dx"], df[key], "o-", lw=2, label="computed")

        if analysis_df is not None and not analysis_df.empty:
            rows = analysis_df[analysis_df["quantity"] == key]
            rows = rows[np.isfinite(rows["extrapolated"])]
            if not rows.empty:
                ax.axhline(rows["extrapolated"].iloc[-1], ls="--", color="crimson",
                           label="Richardson dx -> 0")

        ax.set_xscale("log")
        ax.invert_xaxis()          # refinement runs left to right
        ax.set_xlabel("min_cell_size (um)   [refining ->]")
        ax.set_ylabel(QUANTITIES[key])
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "convergence_v4.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


# =============================================================================
# MAIN
# =============================================================================

def build_meshes():
    """
    Build every refinement level (or reuse a cached file), and report the
    resulting cell counts. Meshes are cheap relative to the simulations --
    the finest takes about 13 s -- so they are always built, even under
    --estimate, so that the cost numbers are real rather than guessed.
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    info = []

    print("Meshes (only growth_rate is held fixed; coarse_dx and "
          "cells_per_level scale with fine_dx):\n")
    print(f"  {'fine_dx':>8} {'coarse_dx':>10} {'cells/lvl':>10} "
          f"{'cells':>9} {'x prev':>7} {'h_global':>9} {'r':>6} {'status':>8}")

    prev_cells = prev_h = None
    for fine_dx in FINE_DX_VALUES:
        path = mesh_path_for(fine_dx)
        status = "cached"

        if not path.exists():
            create_conformal_radial_mesh(
                bath_width=PARAMS["total_width"],
                bath_height=PARAMS["total_height"],
                node_diameter=PARAMS["node_diameter"],
                distance_between_nodes=PARAMS["distance_between"],
                min_cell_size=fine_dx,
                max_cell_size=coarse_dx_for(fine_dx),
                growth_rate=GROWTH_RATE,
                cells_per_level=cells_per_level_for(fine_dx),
                mesh_filename=str(path),
                verbose=False,
            )
            status = "built"

        mesh = Gmsh2D(str(path))
        n_cells = mesh.numberOfCells
        # ASME GCI representative cell size in 2D: h = sqrt(total area / N).
        # This is the h that richardson() assumes halves between levels.
        h_global = float(np.sqrt(np.asarray(mesh.cellVolumes).sum() / n_cells))

        growth = f"{n_cells / prev_cells:.2f}" if prev_cells else ""
        ratio = f"{prev_h / h_global:.3f}" if prev_h else ""
        info.append((fine_dx, n_cells, h_global))
        print(f"  {fine_dx:>8g} {coarse_dx_for(fine_dx):>10g} "
              f"{cells_per_level_for(fine_dx):>10d} {n_cells:>9,} {growth:>7} "
              f"{h_global:>9.3f} {ratio:>6} {status:>8}")
        prev_cells, prev_h = n_cells, h_global

    # richardson() assumes ratio=2.0. If the measured ratio is not close to
    # that, every p and GCI downstream is wrong, so say so loudly.
    ratios = [info[i][2] / info[i + 1][2] for i in range(len(info) - 1)]
    if ratios:
        worst = max(abs(r - 2.0) for r in ratios)
        print(f"\n  effective refinement ratio r = "
              f"{', '.join(f'{r:.3f}' for r in ratios)}  (target 2.000)")
        if worst > 0.15:
            print(f"  *** WARNING: r deviates from 2 by up to {worst:.3f}. "
                  f"richardson() assumes r=2, so p and GCI will be wrong.\n"
                  f"      Check that coarse_dx and cells_per_level are "
                  f"scaling with fine_dx.")
        else:
            print(f"  mesh is refining globally; the r=2 assumption in "
                  f"richardson() holds.")

    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true",
                        help="build/count meshes and print cost estimates, then exit")
    parser.add_argument("--processes", type=int, default=N_PROCESSES,
                        help=f"parallel simulations (default {N_PROCESSES})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MESH CONVERGENCE STUDY  (conformal triangular mesh, stepped x1.5)")
    print("=" * 78)
    print(f"ccd            : {PARAMS['distance_between']:.0f} um")
    print(f"simulated time : {PARAMS['total_time'] / 3600:.0f} h at dt={PARAMS['dt']:.0f} s")
    print(f"growth_rate    : {GROWTH_RATE}   (fixed -- the only one)")
    print(f"coarse_dx      : {COARSE_DX_RATIO:g} * fine_dx   (scales with the mesh)")
    print(f"cells_per_level: {CELLS_PER_LEVEL_NUMER:g} / fine_dx   (scales with the mesh)")
    print(f"refinements    : {FINE_DX_VALUES}")
    print(f"output         : {OUTPUT_DIR}")
    print("=" * 78 + "\n")

    info = build_meshes()

    # Cost estimate, anchored on ~20 min for ~8,000 cells. Sparse solves scale
    # a little worse than linearly, hence the exponent.
    print("\nRough single-run cost (extrapolated from ~20 min at 8,000 cells;")
    print("assumes exclusive use of a core -- sharing the machine will be slower):\n")
    total = 0.0
    for fine_dx, n_cells, _h in info:
        if not np.isfinite(n_cells):
            continue
        minutes = 20.0 * (n_cells / 8000.0) ** 1.2
        total += minutes
        done = " (already done)" if result_path_for(fine_dx).exists() else ""
        print(f"  fine_dx={fine_dx:<6g} ~{minutes:6.0f} min{done}")
    print(f"\n  sequential total ~{total / 60:.1f} h")
    print(f"  with {args.processes} processes, wall time is roughly the sum of the")
    print(f"  largest jobs divided across workers, plus contention.\n")

    if args.estimate:
        print("Estimate only. Re-run without --estimate to execute.\n")
        return

    pending = [dx for dx in FINE_DX_VALUES if not result_path_for(dx).exists()]
    if pending:
        # Longest job first, so the critical path starts immediately.
        pending.sort()
        print(f"Running {len(pending)} level(s) on {args.processes} process(es).\n")
        t0 = time.time()
        if args.processes == 1 or len(pending) == 1:
            for dx in pending:
                run_level(dx)
        else:
            from multiprocessing import Pool
            with Pool(processes=args.processes) as pool:
                pool.map(run_level, pending)
        print(f"\nSimulations finished in {(time.time() - t0) / 60:.1f} min.")
    else:
        print("All levels already complete; re-analysing saved results.\n")

    # Re-read every level from disk so the analysis is identical whether the
    # runs happened now or in an earlier session.
    results = []
    for fine_dx in FINE_DX_VALUES:
        path = result_path_for(fine_dx)
        if not path.exists():
            print(f"  WARNING: {path.name} missing; excluded from analysis")
            continue
        row = scalars_from(pd.read_csv(path), fine_dx, np.nan, True)
        mesh_file = mesh_path_for(fine_dx)
        if mesh_file.exists():
            row["n_cells"] = Gmsh2D(str(mesh_file)).numberOfCells
        results.append(row)

    if len(results) < 3:
        print("\nNeed at least three refinement levels for Richardson "
              "extrapolation. Stopping after saving raw results.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "convergence_results.csv", index=False)
    print(f"\nWrote {OUTPUT_DIR / 'convergence_results.csv'}")

    analysis_df = analyse(results_df) if len(results) >= 3 else None
    plot(results_df, analysis_df)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    finest = min(FINE_DX_VALUES)
    print("  GCI on the finest grid is your discretization error bar.")
    print(f"  If GCI at fine_dx={finest:g} is, say, 0.4 %, then that run is within")
    print("  ~0.4 % of the exact solution of these equations -- and any")
    print("  remaining gap to COMSOL is COMSOL's error, a geometry mismatch,")
    print("  or a difference in the model itself, NOT your mesh.")
    print()
    print("  Two caveats this study does NOT cover:")
    print("    * dt is fixed at "
          f"{PARAMS['dt']:.0f} s, so GCI is the SPATIAL error only.")
    print("    * Independently generated meshes of the same resolution differ")
    print("      by a few percent on their own. If GCI comes out below that,")
    print("      you are reading mesh noise -- repeat a level with a slightly")
    print("      perturbed fine_dx to measure the floor.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
