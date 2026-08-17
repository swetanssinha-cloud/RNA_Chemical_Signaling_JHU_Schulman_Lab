"""
Mesh-resolution convergence sweep for the sender/receiver diffusion model.

Background
----------
create_conformal_radial_mesh(min_cell_size=5.0, ...) does NOT guarantee that
every triangle edge is >= 5 um. Gmsh's Frontal-Delaunay generator treats the
size field as a target, not a hard floor, and the field itself is a Min() of
several Distance sources (node boundary curves + node center points), which
is not smooth. Measured on the default geometry: true minimum edge ~3.1 um
(~62% of the 5 um nominal), even though the mean edge near each node is
~5.0 um, right on target.

Every cheap fix for that gap was tried and empirically failed to close it:
Mesh.MeshSizeMin floor, higher Distance-field Sampling, wider Threshold
risers, dropping the center-point field, switching Mesh.Algorithm (1/5/6).
None moved the achieved minimum by more than a few percent. Getting an exact
hard floor would require a structurally different mesh (e.g. a structured
"collar" mesh around each node) -- out of scope here.

What this script actually checks
---------------------------------
Rather than chase an exact floor, this runs a standard mesh-convergence
sweep: build meshes at a range of min_cell_size (nominal resolution) values,
run the IDENTICAL simulation physics (imported directly from
Single_parameter_sweeps/sweep_core.py, not reimplemented) on each, and see
whether the receiver-probe readout (I2_center_nM) has actually stopped
changing by the resolution you've been using (5.0 um) -- i.e. a real
convergence curve instead of a single two-point comparison.

If the curve flattens at or before your working resolution, that's strong,
standard evidence the mesh is adequate -- independent of the "is 5 really 5"
semantic question that started this.

Usage
-----
    python mesh_floor_sensitivity_check.py
        Runs the default sweep (10, 7, 5, 3.5, 2.5, 1.5, 1.0 um), full 16h
        simulated time each. Wall time scales roughly with cell count --
        budget on the order of an hour or more total on a laptop; longer if
        you add points finer than 1 um.

    python mesh_floor_sensitivity_check.py --fine-dx-values 20,10,5,2.5,1
        Custom set of min_cell_size values, um, comma-separated. Very coarse
        values can trip the "node under-resolved" validation gate (needs
        >=50 cells inside each node) -- that point is skipped, not fatal.

    python mesh_floor_sensitivity_check.py --smoke-test
        Caps total_time at 2 simulated hours, to sanity-check the pipeline
        quickly before committing to a full sweep.

Re-running is cheap: both mesh files and completed time series are cached
by their nominal min_cell_size and skipped if already present, so you can
add more points later without redoing earlier ones.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SWEEP_CORE_DIR = PROJECT_ROOT / "Paramter_sweep" / "Single_parameter_sweeps"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWEEP_CORE_DIR))

from fipy import Gmsh2D, LinearLUSolver  # noqa: E402

from Mesh.New_simple_mesh import create_conformal_radial_mesh  # noqa: E402
import sweep_core as sc  # noqa: E402

OUT_DIR = THIS_DIR / "mesh_floor_check"
OUT_DIR.mkdir(exist_ok=True)

# Brackets the 5.0 um default from both sides. Kept >= 1.0 um by default --
# the node is only 75 um across, so a nominal size below ~1.5 um starts
# pushing past what the "node under-resolved" gate (>=50 cells) tolerates
# once you account for the achieved-vs-nominal undershoot; use
# --fine-dx-values to go further if you want.
DEFAULT_FINE_DX_VALUES = [10.0, 7.0, 5.0, 3.5, 2.5, 1.5, 1.0]


def measure_mesh_edges(msh_path):
    """Parse a Gmsh 2.2 file directly and return (min_edge_um, mean_edge_um)."""
    with open(msh_path) as f:
        text = f.read().split("\n")
    ni = text.index("$Nodes")
    n_nodes = int(text[ni + 1])
    node_xy = {}
    for k in range(n_nodes):
        parts = text[ni + 2 + k].split()
        node_xy[int(parts[0])] = (float(parts[1]), float(parts[2]))
    ei = text.index("$Elements")
    n_elem = int(text[ei + 1])
    lengths = []
    seen = set()
    for k in range(n_elem):
        parts = text[ei + 2 + k].split()
        if int(parts[1]) != 2:  # 3-node triangle
            continue
        n_tags = int(parts[2])
        verts = [int(v) for v in parts[3 + n_tags:]]
        for a, b in [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            xa, ya = node_xy[a]
            xb, yb = node_xy[b]
            lengths.append(np.hypot(xa - xb, ya - yb))
    lengths = np.array(lengths)
    return float(lengths.min()), float(lengths.mean())


def build_mesh(fine_dx, params):
    """Build (or reuse) a conformal mesh at the given nominal min_cell_size."""
    mesh_path = OUT_DIR / f"mesh_fine={fine_dx:.2f}um.msh"
    if mesh_path.exists():
        print(f"  [{fine_dx:5.2f} um] reusing existing mesh: {mesh_path.name}")
        return mesh_path

    print(f"  [{fine_dx:5.2f} um] generating mesh...")
    create_conformal_radial_mesh(
        bath_width=params["total_width"],
        bath_height=params["total_height"],
        node_diameter=params["node_diameter"],
        distance_between_nodes=params["distance_between"],
        min_cell_size=fine_dx,
        max_cell_size=params["coarse_dx"],
        growth_rate=params["growth_rate"],
        cells_per_level=params["cells_per_level"],
        mesh_filename=str(mesh_path),
        verbose=False,
    )
    return mesh_path


def run_simulation(mesh_path, params, fine_dx, total_time_override=None):
    """
    One simulation, physics identical to sweep_core.run_single_simulation --
    imported functions, not reimplemented -- just pointed at an explicit mesh
    file and writing to this script's own output folder instead of the
    sweep's. Skips straight to returning a cached CSV if one already exists
    for this exact resolution (and run length).
    """
    suffix = "_smoke" if total_time_override is not None else ""
    ts_path = OUT_DIR / f"timeseries_fine={fine_dx:.2f}um{suffix}.csv"

    mesh = Gmsh2D(str(mesh_path))
    n_cells = mesh.numberOfCells

    if ts_path.exists():
        print(f"  [{fine_dx:5.2f} um] SKIP -- already have {ts_path.name}")
        return pd.read_csv(ts_path), n_cells

    dt = params["dt"]
    total_time = total_time_override if total_time_override is not None else params["total_time"]
    n_steps = int(total_time / dt)
    save_every = max(1, int(params["save_interval_time"] / dt))

    x, y = np.asarray(mesh.cellCenters[0]), np.asarray(mesh.cellCenters[1])
    y_center = params["total_height"] / 2.0
    sender_x = params["total_width"] / 2.0 - params["distance_between"] / 2.0
    receiver_x = params["total_width"] / 2.0 + params["distance_between"] / 2.0

    (S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
     sender_mask, receiver_mask) = sc.initialize_fields(
        mesh, x, y, sender_x, receiver_x, y_center, params)

    if sender_mask.sum() < 50 or receiver_mask.sum() < 50:
        raise RuntimeError(
            f"[{fine_dx:.2f} um] node under-resolved: "
            f"sender={sender_mask.sum()} receiver={receiver_mask.sum()} cells (need >=50 each)")

    eq_S2 = sc.build_S2_equation(
        S2, I2, Th2, I1O2, D_S2,
        k_p=params["k_p"], k_slow=params["k_slow"], k_fast=params["k_fast"],
        k_d_ss=params["k_d_ss"],
    )
    s2_solver = LinearLUSolver(tolerance=1e-10)
    probe_idx = int(np.argmin(np.hypot(x - receiver_x, y - y_center)))

    rows = []

    def sample(step):
        rows.append({
            "time_hours": step * dt / 3600.0,
            "I2_center_nM": float(I2.value[probe_idx]) * 1e3,
            "S2_free_center_nM": float(S2.value[probe_idx]) * 1e3,
            "S2_total_center_nM": (
                float(S2.value[probe_idx])
                + float(S2_I2.value[probe_idx])
                + float(S2_Th2.value[probe_idx])) * 1e3,
        })

    sample(0)
    recent_I2_values = []
    start = time.perf_counter()

    for step in range(1, n_steps + 1):
        S2.updateOld(); I2.updateOld(); Th2.updateOld()
        S2_I2.updateOld(); S2_Th2.updateOld()

        prev_res = None
        for _ in range(sc.MAX_SWEEPS):
            S2_guess = S2.value
            I2_new, S2_I2_new = sc.reaction_pair_step(
                S2_guess, I2.old.value, S2_I2.old.value,
                params["k_slow"], params["k_d_ds"], dt)
            Th2_new, S2_Th2_new = sc.reaction_pair_step(
                S2_guess, Th2.old.value, S2_Th2.old.value,
                params["k_fast"], params["k_d_ds"], dt)
            I2.setValue(I2_new); S2_I2.setValue(S2_I2_new)
            Th2.setValue(Th2_new); S2_Th2.setValue(S2_Th2_new)

            res = eq_S2.sweep(dt=dt, solver=s2_solver)
            if res < sc.SWEEP_RESIDUAL_TARGET:
                break
            if prev_res is not None and abs(res - prev_res) < sc.SWEEP_PLATEAU_TOL:
                break
            prev_res = res

        if step % save_every == 0:
            sample(step)
            recent_I2_values.append(rows[-1]["I2_center_nM"])
            if len(recent_I2_values) > sc.STEADY_STATE_WINDOW:
                window = recent_I2_values[-sc.STEADY_STATE_WINDOW:]
                mean_I2 = np.mean(window)
                if mean_I2 > 0 and np.std(window) / mean_I2 < sc.STEADY_STATE_THRESHOLD:
                    print(f"  [{fine_dx:5.2f} um] steady state at t={step*dt/3600:.2f} h")
                    break

    wall = time.perf_counter() - start
    df = pd.DataFrame(rows)
    print(f"  [{fine_dx:5.2f} um] done: {n_cells} cells, "
          f"{len(df)} samples, {wall/60:.1f} min wall time")

    df.to_csv(ts_path, index=False)
    return df, n_cells


def summarize_convergence(results, is_smoke=False):
    df = pd.DataFrame(results).sort_values(
        "fine_dx_nominal_um", ascending=False).reset_index(drop=True)

    suffix = "_smoke" if is_smoke else ""
    csv_path = OUT_DIR / f"convergence_summary{suffix}.csv"

    # The finest mesh actually run is the best available reference -- not
    # "truth", but the closest thing on hand to compare every coarser point
    # against.
    finest = df.iloc[-1]
    ref = finest["I2_center_final_nM"]
    df["pct_diff_from_finest"] = 100.0 * (df["I2_center_final_nM"] - ref).abs() / max(abs(ref), 1e-6)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("MESH CONVERGENCE SWEEP")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    converged = df[df["pct_diff_from_finest"] < 1.0]
    if len(converged) > 0:
        converged_at = converged["fine_dx_nominal_um"].max()
        in_range = "" if 5.0 <= converged_at else "NOT "
        print(f"Converged (within 1% of the finest mesh tested, {finest['fine_dx_nominal_um']:.2f} um "
              f"nominal) from min_cell_size <= {converged_at:.2f} um onward.")
        print(f"The current default (fine_dx=5.0 um) is {in_range}within that converged range.")
    else:
        print("No tested resolution came within 1% of the finest mesh -- results are still "
              "changing at every step tested; add finer points with --fine-dx-values.")
    print("=" * 100)

    # ------------------------------------------------------------- plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(df["fine_dx_nominal_um"], df["I2_center_final_nM"], "o-", color="C0")
    axes[0].axvline(5.0, color="gray", linestyle=":", label="current default (5.0 um)")
    axes[0].set_ylabel("Final I2_center (nM)")
    axes[0].set_title("Mesh convergence: final receiver-probe readout vs. nominal resolution")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["fine_dx_nominal_um"], df["pct_diff_from_finest"], "o-", color="C1")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="1% band")
    axes[1].axvline(5.0, color="gray", linestyle=":")
    axes[1].set_ylabel("% diff from finest mesh")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["fine_dx_nominal_um"], df["n_cells"], "o-", color="C2")
    axes[2].axvline(5.0, color="gray", linestyle=":")
    axes[2].set_xlabel("nominal min_cell_size (um)   [finer ->]")
    axes[2].set_ylabel("mesh cell count")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.invert_xaxis()

    plt.tight_layout()
    plot_path = OUT_DIR / f"convergence_plot{suffix}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSummary CSV: {csv_path}")
    print(f"Plot: {plot_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--fine-dx-values", type=str, default=None,
        help="comma-separated min_cell_size values in um, e.g. '20,10,5,2.5,1'. "
             f"Default: {DEFAULT_FINE_DX_VALUES}")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="cap total_time at 2 simulated hours, to sanity-check the pipeline "
             "quickly before committing to a full sweep")
    parser.add_argument(
        "--distance-between", type=float, default=None,
        help="override DEFAULT_PARAMS distance_between (default: sweep_core's default, 200 um)")
    args = parser.parse_args()

    if args.fine_dx_values:
        fine_dx_values = sorted(
            {float(v) for v in args.fine_dx_values.split(",")}, reverse=True)
    else:
        fine_dx_values = sorted(set(DEFAULT_FINE_DX_VALUES), reverse=True)

    params = dict(sc.DEFAULT_PARAMS)
    if args.distance_between is not None:
        params["distance_between"] = args.distance_between
    total_time_override = 2 * 3600 if args.smoke_test else None

    print("Parameters (identical for every sweep point, from sweep_core.DEFAULT_PARAMS):")
    for k, v in params.items():
        print(f"  {k} = {v}")
    print(f"\nSweep points (nominal min_cell_size, um): {fine_dx_values}\n")

    results = []
    for fine_dx in fine_dx_values:
        print(f"--- min_cell_size = {fine_dx:.2f} um ---")
        try:
            mesh_path = build_mesh(fine_dx, params)
            min_edge, mean_edge = measure_mesh_edges(mesh_path)
            df, n_cells = run_simulation(mesh_path, params, fine_dx, total_time_override)
        except Exception as exc:
            print(f"  [{fine_dx:5.2f} um] FAILED: {type(exc).__name__}: {exc}")
            continue

        final = df.iloc[-1]
        results.append({
            "fine_dx_nominal_um": fine_dx,
            "achieved_min_edge_um": min_edge,
            "achieved_mean_edge_um": mean_edge,
            "n_cells": n_cells,
            "final_time_hours": final["time_hours"],
            "I2_center_final_nM": final["I2_center_nM"],
            "S2_free_center_final_nM": final["S2_free_center_nM"],
            "S2_total_center_final_nM": final["S2_total_center_nM"],
            "half_time_center_hr": sc.half_time(
                df["time_hours"].values, df["I2_center_nM"].values),
        })

    if len(results) < 2:
        print("\nFewer than 2 successful sweep points -- nothing to compare. "
              "Check the FAILED messages above.")
        return

    summarize_convergence(results, is_smoke=args.smoke_test)


if __name__ == "__main__":
    main()
