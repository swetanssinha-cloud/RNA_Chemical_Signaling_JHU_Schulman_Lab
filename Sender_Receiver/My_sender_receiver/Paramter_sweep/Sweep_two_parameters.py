"""
TWO-PARAMETER (2-D) sweep for the 2-node tethered-genelet sender/receiver model.

This is the 2-D companion to Parameter_sweep_unified.py, which is left
completely untouched and still handles 1-D sweeps.

Every piece of PHYSICS is imported from Parameter_sweep_unified:

    initialize_fields()   field/mask/diffusivity setup
    build_equations()     the coupled PDE system
    mesh_path_for()       deterministic mesh filenames (so meshes are SHARED
                          with the 1-D sweeps -- nothing is rebuilt needlessly)
    half_time()           half-way-point metric
    DEFAULT_PARAMS        the validated baseline
    MESH_AFFECTING        which parameters force a new mesh

so the equations here are identical by construction. If you change the model,
change it in Parameter_sweep_unified.py and both sweeps follow.


WHY run_single_simulation() IS *NOT* IMPORTED
---------------------------------------------
Parameter_sweep_unified.run_single_simulation() reads the module-level globals
SWEEP_PARAMETER and OUTPUT_DIR rather than taking them as arguments. On macOS
multiprocessing uses the "spawn" start method, which re-imports the module from
scratch in every worker, so those globals would revert to whatever is hard-coded
in that file -- monkey-patching them from here would silently do nothing in the
workers and write results into the wrong folder.

The version below therefore takes an explicit `overrides` dict, which is
pickled and shipped to the worker as ordinary data. Its time-stepping block is a
line-for-line copy of the original; only the parameter plumbing and the file
naming differ.


ONE BUG FIXED RELATIVE TO THE 1-D SCRIPT
----------------------------------------
Parameter_sweep_unified.py line ~329 calls build_equations() with

    k_d_ds=params["k_d_ss"]      # <-- passes k_d_ss into the k_d_ds slot

Because both default to 3e-4 this is invisible in every run that does not sweep
one of them, but it means a k_d_ds sweep in the 1-D script was actually holding
k_d_ds pinned at k_d_ss. This file passes params["k_d_ds"], so k_d_ds is genuinely
sweepable here. With the shipped defaults the two scripts agree exactly.


OUTPUTS  (all inside Two_parameters_varied_results_<P1>_<P2>/)
--------------------------------------------------------------
  timeseries_<P1>=<v1>_<P2>=<v2>_rep=<n>.csv   one per run, same columns as the
                                               1-D script
  timeseries_I2.png                            \
  timeseries_S2_free.png                        > overlays, one row per P2 value
  timeseries_S2_total.png                      /
  map_I2_final.png                             final [I2]: heat map + line family
  map_half_time.png                            half-time: heat map + line family
  raw_results.csv, grid_*.csv, run_config.json
"""

import json
import os
import re
import sys
import time
import warnings
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # workers must not try to open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fipy import Gmsh2D

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from Mesh.New_simple_mesh import create_conformal_radial_mesh
from Parameter_sweep_unified import (
    DEFAULT_PARAMS,
    MESH_AFFECTING,
    MESH_DIR,
    build_equations,
    half_time,
    initialize_fields,
    mesh_path_for,
)

warnings.filterwarnings("ignore")


# =============================================================================
# USER CONFIGURATION  -- this is the only block you normally edit
# =============================================================================

PARAMETER_SWEPT_ONE = "Th2_init"
VALUES_ONE = np.array([0.1, 0.2, 0.4, 0.8, 1.6])

PARAMETER_SWEPT_TWO = "k_d_ds"
VALUES_TWO = np.array([0.20,0.25,(1/3), 0.5, 1]) * 3e-4

# Examples:
#   PARAMETER_SWEPT_ONE = "k_slow";   VALUES_ONE = np.array([1, 2, 3, 4, 5]) * 5e4 * 1e-6
#   PARAMETER_SWEPT_TWO = "k_d_ds";   VALUES_TWO = np.array([0.25, 0.5, 1.0]) * 3e-4
#
#   PARAMETER_SWEPT_ONE = "distance_between"; VALUES_ONE = np.array([150., 300., 600.])
#   PARAMETER_SWEPT_TWO = "k_p";              VALUES_TWO = np.array([0.002, 0.01, 0.05])
#
# Either, both, or neither parameter may be geometric -- the mesh stage works
# out how many distinct meshes the grid needs and builds exactly those.

N_REPLICATES = 1
# The model is deterministic, so replicates only make sense if you later add
# noise. Leave at 1 unless you know you want more.

# Total runs is len(VALUES_ONE) * len(VALUES_TWO) * N_REPLICATES -- this grows
# fast. 5 x 3 = 15 runs at ~8 h simulated each is already a real workload.
N_TASKS = len(VALUES_ONE) * len(VALUES_TWO) * N_REPLICATES
N_PROCESSES = 5 #min(N_TASKS, max(1, cpu_count() - 1))


# =============================================================================
# PATHS
# =============================================================================

OUTPUT_DIR = HERE / f"Two_parameters_varied_results_{PARAMETER_SWEPT_ONE}_{PARAMETER_SWEPT_TWO}"

# Meshes live in the SAME folder the 1-D sweep uses, and mesh_path_for() is the
# same function, so any mesh either script has already built is reused as-is.


def timeseries_path_for(value_one, value_two, replicate_id):
    """The durable record of one completed run."""
    return OUTPUT_DIR / (
        f"timeseries_{PARAMETER_SWEPT_ONE}={value_one:g}"
        f"_{PARAMETER_SWEPT_TWO}={value_two:g}"
        f"_rep={replicate_id}.csv"
    )


# Values formatted with %g never contain "_", so [^_]+ is a safe field matcher
# even though the parameter names themselves are full of underscores.
FILENAME_RE = re.compile(
    rf"^timeseries_{re.escape(PARAMETER_SWEPT_ONE)}=(?P<v1>[^_]+)"
    rf"_{re.escape(PARAMETER_SWEPT_TWO)}=(?P<v2>[^_]+)"
    rf"_rep=(?P<rep>\d+)$"
)


def params_for(value_one, value_two):
    """The full parameter dict for one grid point."""
    params = dict(DEFAULT_PARAMS)
    params[PARAMETER_SWEPT_ONE] = value_one
    params[PARAMETER_SWEPT_TWO] = value_two
    return params


# =============================================================================
# ONE SIMULATION
# =============================================================================

def run_single_simulation(value_one, value_two, replicate_id):
    """
    Run one grid point. Workers only LOAD meshes; Gmsh is never called here.

    Raises on any condition that would previously have produced a silent zero,
    so a broken configuration cannot masquerade as a result.
    """
    label = (f"{PARAMETER_SWEPT_ONE}={value_one:g} "
             f"{PARAMETER_SWEPT_TWO}={value_two:g} rep={replicate_id}")
    start_wall = time.perf_counter()

    # Resume: a completed run already has its time series on disk, and every
    # summary number is recomputed from that file, so there is nothing to gain
    # by running it again.
    existing = timeseries_path_for(value_one, value_two, replicate_id)
    if existing.exists():
        print(f"  SKIP {label:<52} already complete", flush=True)
        return {"value_one": value_one, "value_two": value_two,
                "replicate_id": replicate_id, "success": True, "skipped": True}

    try:
        params = params_for(value_one, value_two)

        dt = params["dt"]
        n_steps = int(params["total_time"] / dt)
        save_every = max(1, int(params["save_interval_time"] / dt))
        node_radius = params["node_diameter"] / 2.0

        # ------------------------------------------------------------- mesh
        mesh_file = mesh_path_for(params)
        if not mesh_file.exists():
            raise FileNotFoundError(
                f"Mesh missing: {mesh_file}. It should have been built before "
                f"the pool started.")

        mesh = Gmsh2D(str(mesh_file))
        x, y = np.asarray(mesh.cellCenters[0]), np.asarray(mesh.cellCenters[1])

        y_center = params["total_height"] / 2.0
        sender_x = params["total_width"] / 2.0 - params["distance_between"] / 2.0
        receiver_x = params["total_width"] / 2.0 + params["distance_between"] / 2.0

        # ------------------------------------------------------------ fields
        (S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
         sender_mask, receiver_mask) = initialize_fields(
            mesh, x, y, sender_x, receiver_x, y_center, params)

        # --- validation gate 1: both nodes must actually contain cells
        if sender_mask.sum() < 50:
            raise RuntimeError(
                f"Only {sender_mask.sum()} cells inside the sender node. "
                f"Mesh is too coarse or the geometry is wrong.")
        if receiver_mask.sum() < 50:
            raise RuntimeError(
                f"Only {receiver_mask.sum()} cells inside the receiver node.")

        # NOTE: k_d_ds is passed as k_d_ds here. The 1-D script passes
        # params["k_d_ss"] into this slot -- see the module docstring.
        eq = build_equations(
            S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2,
            k_p=params["k_p"], k_slow=params["k_slow"], k_fast=params["k_fast"],
            k_d_ss=params["k_d_ds"], k_d_ds=params["k_d_ds"],
        )

        # Probe cell 1: nearest to the receiver centre (COMSOL-comparable).
        probe_idx = int(np.argmin(np.hypot(x - receiver_x, y - y_center)))

        # Probe cell 2: nearest to a point on the receiver node's edge, on the
        # side facing the sender.
        edge_idx = int(np.argmin(np.hypot(x - (receiver_x - node_radius),
                                          y - y_center)))

        # ------------------------------------------------------------ storage
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
                "I2_edge_nM": float(I2.value[edge_idx]) * 1e3,
                "S2_free_edge_nM": float(S2.value[edge_idx]) * 1e3,
                "S2_total_edge_nM": (
                    float(S2.value[edge_idx])
                    + float(S2_I2.value[edge_idx])
                    + float(S2_Th2.value[edge_idx])) * 1e3,
            })

        sample(0)   # t = 0

        # ------------------------------------------------------- time stepping
        for step in range(1, n_steps + 1):
            S2.updateOld()
            I2.updateOld()
            Th2.updateOld()
            S2_I2.updateOld()
            S2_Th2.updateOld()

            res = 1e10
            n_sweeps = 0
            while res > 1e-6 and n_sweeps < 10:
                res = eq.sweep(dt=dt)
                n_sweeps += 1

            if step % save_every == 0:
                sample(step)

            # --- validation gate 2: the sender must be producing S2
            if step == max(1, int(600 / dt)):        # after 10 simulated minutes
                if float(np.max(S2.value)) <= 0.0:
                    raise RuntimeError(
                        "No S2 anywhere in the domain after 10 simulated "
                        "minutes. The sender is not transcribing.")

        df = pd.DataFrame(rows)

        # --- validation gate 3: signal must have reached the receiver at all
        if df["S2_total_center_nM"].max() <= 0.0:
            raise RuntimeError(
                "S2 never reached the receiver node (centre-point value stayed "
                "at exactly zero). This is the signature of a disconnected mesh.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_file = timeseries_path_for(value_one, value_two, replicate_id)

        # Write to a temporary file and rename. os.replace is atomic, so an
        # interruption can never leave a half-written CSV behind that a later
        # run would mistake for a completed simulation.
        tmp_file = ts_file.with_suffix(".csv.partial")
        df.to_csv(tmp_file, index=False)
        os.replace(tmp_file, ts_file)

        wall = time.perf_counter() - start_wall

        # Metadata that cannot be recovered from the time series itself. The
        # swept values are recorded explicitly so analysis never has to trust
        # filename parsing.
        ts_file.with_suffix(".meta.json").write_text(json.dumps({
            "parameter_one": PARAMETER_SWEPT_ONE,
            "value_one": float(value_one),
            "parameter_two": PARAMETER_SWEPT_TWO,
            "value_two": float(value_two),
            "replicate_id": replicate_id,
            "wall_time_s": wall,
            "n_cells": int(mesh.numberOfCells),
            "n_cells_sender_node": int(sender_mask.sum()),
            "n_cells_receiver_node": int(receiver_mask.sum()),
            "mesh_file": mesh_file.name,
        }, indent=2))

        final = df.iloc[-1]
        print(f"  OK   {label:<52} "
              f"I2_center={final['I2_center_nM']:7.2f} nM  "
              f"S2tot_center={final['S2_total_center_nM']:8.2f} nM  "
              f"[{wall/60:.1f} min]", flush=True)

        return {"value_one": value_one, "value_two": value_two,
                "replicate_id": replicate_id, "wall_time_s": wall,
                "success": True}

    except Exception as exc:
        import traceback
        print(f"  FAIL {label:<52} {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "value_one": value_one, "value_two": value_two,
            "replicate_id": replicate_id,
            "wall_time_s": time.perf_counter() - start_wall,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# =============================================================================
# MESH PRE-GENERATION
# =============================================================================

def build_all_meshes():
    """
    Build every mesh the grid needs, sequentially, before any worker starts.

    Works out the distinct geometries by asking mesh_path_for() for every grid
    point and de-duplicating: if neither swept parameter is geometric that is
    one mesh, if one is it is len(that axis), if both are it is the full
    product. Gmsh is never called inside a worker process.
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    geometric = [p for p in (PARAMETER_SWEPT_ONE, PARAMETER_SWEPT_TWO)
                 if p in MESH_AFFECTING]
    if geometric:
        print(f"Geometry-changing parameter(s): {', '.join(geometric)}")
    else:
        print("Neither parameter changes the geometry -> 1 shared mesh.")

    wanted = {}      # path -> params, de-duplicated
    for v1, v2 in product(VALUES_ONE, VALUES_TWO):
        params = params_for(v1, v2)
        wanted.setdefault(mesh_path_for(params), params)

    print(f"{len(wanted)} distinct mesh(es) needed.")

    for path, params in wanted.items():
        if path.exists():
            print(f"  reuse {path.name}")
            continue

        print(f"  build {path.name}")
        t0 = time.time()
        create_conformal_radial_mesh(
            bath_width=params["total_width"],
            bath_height=params["total_height"],
            node_diameter=params["node_diameter"],
            distance_between_nodes=params["distance_between"],
            min_cell_size=params["fine_dx"],
            max_cell_size=params["coarse_dx"],
            growth_rate=params["growth_rate"],
            cells_per_level=params["cells_per_level"],
            mesh_filename=str(path),
            verbose=False,
        )
        print(f"        done in {time.time() - t0:.1f} s")

    # Load each one once here, in the parent, so a corrupt file fails now
    # rather than inside a worker.
    for path in wanted:
        mesh = Gmsh2D(str(path))
        print(f"  verify {path.name}: {mesh.numberOfCells:,} cells")


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_sweep():
    tasks = [(v1, v2, rep)
             for v1 in VALUES_ONE
             for v2 in VALUES_TWO
             for rep in range(N_REPLICATES)]

    print(f"\nRunning {len(tasks)} simulation(s) on {N_PROCESSES} process(es).")
    print(f"Each simulates {DEFAULT_PARAMS['total_time'] / 3600:.0f} h "
          f"at dt={DEFAULT_PARAMS['dt']:.0f} s.\n")

    t0 = time.time()
    if N_PROCESSES == 1:
        results = [run_single_simulation(*task) for task in tasks]
    else:
        with Pool(processes=N_PROCESSES) as pool:
            results = pool.starmap(run_single_simulation, tasks)

    print(f"\nAll simulations finished in {(time.time() - t0) / 60:.1f} min.")
    return results


# =============================================================================
# ANALYSIS
# =============================================================================

METRICS = [
    "I2_center_final_nM", "S2_free_center_final_nM", "S2_total_center_final_nM",
    "I2_edge_final_nM", "S2_free_edge_final_nM", "S2_total_edge_final_nM",
    "half_time_center_hr", "half_time_edge_hr",
]


def scalars_from_timeseries(df, value_one, value_two, replicate_id):
    """
    Derive every summary scalar from a saved time series.

    This is the key to not losing work: the per-run time-series CSV is the
    durable record, written the moment that run finishes. Every summary number
    is recomputed from it, so a crash during analysis can never destroy
    completed simulations.
    """
    final = df.iloc[-1]
    return {
        "value_one": value_one,
        "value_two": value_two,
        "replicate_id": replicate_id,
        "I2_center_final_nM": float(final["I2_center_nM"]),
        "S2_free_center_final_nM": float(final["S2_free_center_nM"]),
        "S2_total_center_final_nM": float(final["S2_total_center_nM"]),
        "I2_edge_final_nM": float(final["I2_edge_nM"]),
        "S2_free_edge_final_nM": float(final["S2_free_edge_nM"]),
        "S2_total_edge_final_nM": float(final["S2_total_edge_nM"]),
        "half_time_center_hr": half_time(df["time_hours"].values,
                                         df["I2_center_nM"].values),
        "half_time_edge_hr": half_time(df["time_hours"].values,
                                       df["I2_edge_nM"].values),
        "n_samples": len(df),
    }


def collect_results_from_disk():
    """
    Rebuild the results table by reading every time-series CSV in OUTPUT_DIR.

    Picks up runs from earlier sessions automatically, so an interrupted sweep
    is recovered simply by running the script again -- no simulation is
    repeated and no completed run is lost.
    """
    rows = []
    for path in sorted(OUTPUT_DIR.glob("timeseries_*_rep=*.csv")):
        match = FILENAME_RE.match(path.stem)
        if not match:
            print(f"  skipping unparseable filename: {path.name}")
            continue

        # Prefer the sidecar's explicit values; fall back to the filename.
        meta_path = path.with_suffix(".meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        try:
            value_one = float(meta.get("value_one", match.group("v1")))
            value_two = float(meta.get("value_two", match.group("v2")))
            replicate_id = int(match.group("rep"))
        except ValueError:
            print(f"  skipping unparseable values: {path.name}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  skipping unreadable {path.name}: {exc}")
            continue
        if df.empty:
            print(f"  skipping empty {path.name}")
            continue

        row = scalars_from_timeseries(df, value_one, value_two, replicate_id)
        for key in ("wall_time_s", "n_cells"):
            if key in meta:
                row[key] = meta[key]
        rows.append(row)

    return pd.DataFrame(rows)


def summarise():
    """Build raw_results.csv, the per-metric grid CSVs, and run_config.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = collect_results_from_disk()
    if raw.empty:
        print("No completed runs found on disk -- nothing to summarise.")
        return None

    (OUTPUT_DIR / "run_config.json").write_text(json.dumps({
        "parameter_one": PARAMETER_SWEPT_ONE,
        "values_one": VALUES_ONE,
        "parameter_two": PARAMETER_SWEPT_TWO,
        "values_two": VALUES_TWO,
        "n_replicates": N_REPLICATES,
        "default_params": DEFAULT_PARAMS,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, default=str))

    raw = raw.sort_values(["value_one", "value_two", "replicate_id"])
    raw.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)

    # Mean over replicates at each grid point (a no-op when N_REPLICATES = 1).
    aggregated = METRICS + [c for c in ("wall_time_s", "n_cells")
                            if c in raw.columns]
    stats = (raw.groupby(["value_one", "value_two"])[aggregated]
                .mean()
                .reset_index())

    # One rectangular CSV per headline metric: rows = P1, columns = P2.
    # Labels are formatted with %g so a value like 0.05 is not written out as
    # 0.049999999999999996 in the header.
    for metric in ("I2_center_final_nM", "I2_edge_final_nM",
                   "half_time_center_hr", "half_time_edge_hr"):
        grid = stats.pivot(index="value_one", columns="value_two", values=metric)
        grid.index = [f"{v:g}" for v in grid.index]
        grid.columns = [f"{v:g}" for v in grid.columns]
        grid.index.name = PARAMETER_SWEPT_ONE
        grid.columns.name = PARAMETER_SWEPT_TWO
        grid.to_csv(OUTPUT_DIR / f"grid_{metric}.csv")

    print(f"\nFound {len(raw)} completed run(s) on disk "
          f"({len(stats)} of {len(VALUES_ONE) * len(VALUES_TWO)} grid points).")
    print(f"Wrote {OUTPUT_DIR / 'raw_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'grid_*.csv'} (4 files)")
    print(f"Wrote {OUTPUT_DIR / 'run_config.json'}")

    # Compare on the %g string, not the float, so 1/3 and friends match the
    # value that actually went into the filename.
    done = {(f"{a:g}", f"{b:g}")
            for a, b in zip(stats["value_one"], stats["value_two"])}
    missing = [(f"{a:g}", f"{b:g}") for a, b in product(VALUES_ONE, VALUES_TWO)
               if (f"{a:g}", f"{b:g}") not in done]
    if missing:
        print(f"\nStill missing {len(missing)} grid point(s) "
              f"(re-run to fill these in): {missing}")

    return stats


# =============================================================================
# PLOTS
# =============================================================================

# (species label, centre column, edge column, axis label)
SPECIES = [
    ("I2", "I2_center_nM", "I2_edge_nM", "[I2] (nM)"),
    ("S2_free", "S2_free_center_nM", "S2_free_edge_nM", "free [S2] (nM)"),
    ("S2_total", "S2_total_center_nM", "S2_total_edge_nM", "total [S2] (nM)"),
]


def _colors():
    """One colour per PARAMETER_SWEPT_ONE value, dark -> light."""
    return plt.cm.viridis(np.linspace(0, 0.9, len(VALUES_ONE)))


def plot_timeseries():
    """
    Overlay every run's time series, one figure per species.

    Layout: one ROW per PARAMETER_SWEPT_TWO value, two COLUMNS (centre point,
    node edge) -- the same two-panel centre/edge format the 1-D script uses,
    stacked so the second parameter is read down the page while colour carries
    the first parameter.
    """
    colors = _colors()

    for name, center_col, edge_col, ylabel in SPECIES:
        n_rows = len(VALUES_TWO)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.6 * n_rows),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"{name}:  colour = {PARAMETER_SWEPT_ONE},  "
                     f"row = {PARAMETER_SWEPT_TWO}",
                     fontsize=14, fontweight="bold")

        any_data = False
        for row, v2 in enumerate(VALUES_TWO):
            for color, v1 in zip(colors, VALUES_ONE):
                matches = sorted(OUTPUT_DIR.glob(
                    f"timeseries_{PARAMETER_SWEPT_ONE}={v1:g}"
                    f"_{PARAMETER_SWEPT_TWO}={v2:g}_rep=*.csv"))
                for path in matches:
                    df = pd.read_csv(path)
                    any_data = True
                    label = f"{PARAMETER_SWEPT_ONE}={v1:g}"
                    axes[row, 0].plot(df["time_hours"], df[center_col],
                                      color=color, lw=2, label=label)
                    axes[row, 1].plot(df["time_hours"], df[edge_col],
                                      color=color, lw=2, label=label)

            axes[row, 0].set_ylabel(f"{PARAMETER_SWEPT_TWO} = {v2:g}\n{ylabel}")
            for col, where in enumerate(["centre point", "node edge"]):
                axes[row, col].grid(alpha=0.3)
                if row == 0:
                    axes[row, col].set_title(f"{name} at {where}")

        if not any_data:
            plt.close(fig)
            continue

        for col in range(2):
            axes[-1, col].set_xlabel("Time (hours)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0, 0].legend(unique.values(), unique.keys(), fontsize=8)

        fig.tight_layout()
        out = OUTPUT_DIR / f"timeseries_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def _grid_array(stats, metric):
    """
    Reshape a metric onto the (PARAMETER_SWEPT_TWO x PARAMETER_SWEPT_ONE) grid.

    Rows are P2 and columns are P1 so the array drops straight into imshow with
    P1 running along x, matching the line plots underneath. Missing grid points
    stay NaN rather than being silently dropped or interpolated.
    """
    z = np.full((len(VALUES_TWO), len(VALUES_ONE)), np.nan)
    lookup = {(f"{a:g}", f"{b:g}"): v
              for a, b, v in zip(stats["value_one"], stats["value_two"],
                                 stats[metric])}
    for j, v1 in enumerate(VALUES_ONE):
        for i, v2 in enumerate(VALUES_TWO):
            value = lookup.get((f"{v1:g}", f"{v2:g}"))
            if value is not None:
                z[i, j] = value
    return z


def _heatmap(ax, z, title, cbar_label, fig):
    """
    Categorical heat map: one cell per grid point, ticks labelled with the
    actual swept values. Index-based rather than value-based axes so unevenly
    spaced sweeps (0.2, 0.25, 1/3, 0.5, 1) are not drawn misleadingly.
    """
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(VALUES_ONE)))
    ax.set_xticklabels([f"{v:g}" for v in VALUES_ONE], rotation=45, ha="right")
    ax.set_yticks(range(len(VALUES_TWO)))
    ax.set_yticklabels([f"{v:g}" for v in VALUES_TWO])
    ax.set_xlabel(PARAMETER_SWEPT_ONE)
    ax.set_ylabel(PARAMETER_SWEPT_TWO)
    ax.set_title(title)

    # Annotate when the grid is small enough for the numbers to be readable.
    if z.size <= 64:
        finite = z[np.isfinite(z)]
        midpoint = (finite.max() + finite.min()) / 2 if finite.size else 0.0
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if not np.isfinite(z[i, j]):
                    continue
                ax.text(j, i, f"{z[i, j]:.3g}", ha="center", va="center",
                        fontsize=8,
                        color="white" if z[i, j] < midpoint else "black")

    fig.colorbar(im, ax=ax, label=cbar_label)


def _line_family(ax, z, title, ylabel):
    """The same data as lines: x = P1, one curve per P2 value."""
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(VALUES_TWO)))
    for i, (color, v2) in enumerate(zip(colors, VALUES_TWO)):
        ax.plot(VALUES_ONE, z[i, :], marker="o", lw=2, color=color,
                label=f"{PARAMETER_SWEPT_TWO}={v2:g}")
    ax.set_xlabel(PARAMETER_SWEPT_ONE)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_maps(stats):
    """
    Two figures over the 2-D grid:

      map_I2_final.png   final [I2] at the receiver
      map_half_time.png  time to the midpoint between initial and final [I2]

    Each shows centre-point and node-edge probes as heat maps (top row) and as
    line families (bottom row) -- the heat map for reading the surface, the
    lines for reading off individual trends.
    """
    if stats is None or stats.empty:
        return

    figures = [
        ("map_I2_final",
         "Final [I2] at receiver",
         [("I2_center_final_nM", "centre point (COMSOL-comparable)"),
          ("I2_edge_final_nM", "node edge")],
         "final [I2] (nM)"),
        ("map_half_time",
         "Half-time of I2  (midpoint of initial -> final)",
         [("half_time_center_hr", "centre point (COMSOL-comparable)"),
          ("half_time_edge_hr", "node edge")],
         "half-time (hours)"),
    ]

    for filename, suptitle, metrics, unit_label in figures:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")

        for col, (metric, where) in enumerate(metrics):
            z = _grid_array(stats, metric)
            _heatmap(axes[0, col], z, where, unit_label, fig)
            _line_family(axes[1, col], z, where, unit_label)

        fig.tight_layout()
        out = OUTPUT_DIR / f"{filename}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("TWO-PARAMETER SWEEP")
    print("=" * 78)
    print(f"parameter 1 : {PARAMETER_SWEPT_ONE}")
    print(f"values 1    : {VALUES_ONE}")
    print(f"parameter 2 : {PARAMETER_SWEPT_TWO}")
    print(f"values 2    : {VALUES_TWO}")
    print(f"grid        : {len(VALUES_ONE)} x {len(VALUES_TWO)} "
          f"= {len(VALUES_ONE) * len(VALUES_TWO)} points "
          f"x {N_REPLICATES} replicate(s) = {N_TASKS} run(s)")
    print(f"output      : {OUTPUT_DIR}")
    print("=" * 78)

    for name in (PARAMETER_SWEPT_ONE, PARAMETER_SWEPT_TWO):
        if name not in DEFAULT_PARAMS:
            raise SystemExit(
                f"'{name}' is not a known parameter. "
                f"Valid names:\n  {sorted(DEFAULT_PARAMS)}")
    if PARAMETER_SWEPT_ONE == PARAMETER_SWEPT_TWO:
        raise SystemExit("The two swept parameters must be different.")

    print("\nSTAGE 1 - meshes\n")
    build_all_meshes()

    print("\nSTAGE 2 - simulations\n")
    results = run_sweep()

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} simulation(s) FAILED this session:")
        for r in failed:
            print(f"  {PARAMETER_SWEPT_ONE}={r['value_one']:g} "
                  f"{PARAMETER_SWEPT_TWO}={r['value_two']:g}: "
                  f"{r.get('error', 'unknown error')}")
        print("Completed runs are unaffected and are summarised below.")

    print("\nSTAGE 3 - analysis\n")
    stats = summarise()
    plot_timeseries()
    plot_maps(stats)

    print("\nDone.\n")
