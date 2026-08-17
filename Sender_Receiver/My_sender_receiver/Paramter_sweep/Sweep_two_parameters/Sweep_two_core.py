"""
Shared orchestration for TWO-PARAMETER (2-D) sweeps of the 2-node
tethered-genelet sender/receiver model.

This is the 2-D companion to Single_parameter_sweeps/sweep_core.py. Every
piece of PHYSICS is imported from there -- not redefined here:

    initialize_fields()   field/mask/diffusivity setup
    build_S2_equation()   S2's diffusion-reaction PDE (the only variable
                          with a DiffusionTerm; see reaction_pair_step())
    reaction_pair_step()  closed-form backward-Euler update for I2/S2_I2
                          and Th2/S2_Th2 -- no sparse solve needed
    mesh_path_for()       deterministic mesh filenames (so meshes are SHARED
                          with the 1-D sweeps -- nothing is rebuilt needlessly)
    half_time()           half-way-point metric
    DEFAULT_PARAMS        the validated baseline
    MESH_AFFECTING        which parameters force a new mesh
    MAX_SWEEPS, SWEEP_RESIDUAL_TARGET, SWEEP_PLATEAU_TOL
                          split-solver sweep control

If you change the model, change it in Single_parameter_sweeps/sweep_core.py
and every 1-D AND 2-D sweep follows -- there is exactly one copy of the
solver in the whole codebase.


k_d_ss WAS PINNED TO k_d_ds -- NOW FIXED
-----------------------------------------
The original Sweep_two_parameters.py always called
build_S2_equation(..., k_d_ss=params["k_d_ds"]) and used params["k_d_ds"] as
k_off for BOTH reaction pairs, inherited from a bug that was later fixed in
the 1-D script but never ported here. Net effect: a swept k_d_ss value never
reached the physics -- every run silently used k_d_ds's value instead. This
version calls the imported build_S2_equation()/reaction_pair_step() exactly
as the (corrected) 1-D script does, so k_d_ss and k_d_ds are genuinely
independent: k_d_ds sets both bound-pair dissociation rates, k_d_ss sets the
free-S2 degradation rate in the diffusion equation itself.


OUTPUTS  (all inside Two_parameters_varied_results_<P1>_<P2>/, in Paramter_sweep/)
------------------------------------------------------------------------------
  timeseries_<P1>=<v1>_<P2>=<v2>_rep=<n>.csv   one per run
  timeseries_I2.png, timeseries_S2_free.png, timeseries_S2_total.png
                                                overlays, one row per P2 value
  map_I2_final.png, map_half_time.png          heat map + line family
  raw_results.csv, grid_*.csv, run_config.json
"""

import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")          # workers must not try to open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fipy import Gmsh2D, LinearLUSolver

# This file lives in .../Paramter_sweep/Sweep_two_parameters/. The physics
# lives one level up and over, in Single_parameter_sweeps/sweep_core.py --
# import it from there rather than redefining anything.
SWEEP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SWEEP_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWEEP_ROOT / "Single_parameter_sweeps"))

from Mesh.New_simple_mesh import create_conformal_radial_mesh
from sweep_core import (
    CHECK_INTERVAL,
    DEFAULT_PARAMS,
    MAX_SWEEPS,
    MESH_AFFECTING,
    MESH_DIR,
    STEADY_STATE_THRESHOLD,
    STEADY_STATE_WINDOW,
    SWEEP_PLATEAU_TOL,
    SWEEP_RESIDUAL_TARGET,
    build_S2_equation,
    half_time,
    initialize_fields,
    mesh_path_for,
    reaction_pair_step,
)

warnings.filterwarnings("ignore")


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

@dataclass
class TwoParamSweepConfig:
    """
    What a two-parameter sweep script needs to specify. Everything else (the
    model, the solver, mesh handling, analysis, plotting) is shared.

    output_dir defaults to
    Paramter_sweep/Two_parameters_varied_results_<P1>_<P2>, matching the
    existing folder from the original script.

    n_processes defaults to (SLURM_CPUS_PER_TASK if set, else cpu_count()),
    capped at the number of grid points -- see the note on os.cpu_count()
    below.
    """
    parameter_one: str
    values_one: Sequence[float]
    parameter_two: str
    values_two: Sequence[float]
    n_replicates: int = 1
    output_dir: Optional[Path] = None
    n_processes: Optional[int] = None

    def __post_init__(self):
        if self.parameter_one == self.parameter_two:
            raise SystemExit("The two swept parameters must be different.")

        self.values_one = list(self.values_one)
        self.values_two = list(self.values_two)

        if self.output_dir is None:
            self.output_dir = SWEEP_ROOT / (
                f"Two_parameters_varied_results_"
                f"{self.parameter_one}_{self.parameter_two}")
        else:
            self.output_dir = Path(self.output_dir)

        if self.n_processes is None:
            # os.cpu_count() reports the physical node's full CPU count on
            # Rockfish, not what Slurm actually allocated to this job -- so
            # it can't be trusted to size the pool. SLURM_CPUS_PER_TASK (set
            # by the --cpus-per-task sbatch directive) is authoritative when
            # present; cpu_count() is only a fallback for runs outside Slurm.
            allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
            n_tasks = len(self.values_one) * len(self.values_two) * self.n_replicates
            self.n_processes = min(n_tasks, max(1, allocated))

    @property
    def n_tasks(self):
        return len(self.values_one) * len(self.values_two) * self.n_replicates


def timeseries_path_for(cfg, value_one, value_two, replicate_id):
    """The durable record of one completed run."""
    return cfg.output_dir / (
        f"timeseries_{cfg.parameter_one}={value_one:g}"
        f"_{cfg.parameter_two}={value_two:g}"
        f"_rep={replicate_id}.csv"
    )


def filename_re_for(cfg):
    """
    Values formatted with %g never contain "_", so [^_]+ is a safe field
    matcher even though the parameter names themselves are full of
    underscores.
    """
    return re.compile(
        rf"^timeseries_{re.escape(cfg.parameter_one)}=(?P<v1>[^_]+)"
        rf"_{re.escape(cfg.parameter_two)}=(?P<v2>[^_]+)"
        rf"_rep=(?P<rep>\d+)$"
    )


def params_for(cfg, value_one, value_two):
    """The full parameter dict for one grid point."""
    params = dict(DEFAULT_PARAMS)
    params[cfg.parameter_one] = value_one
    params[cfg.parameter_two] = value_two
    return params


# =============================================================================
# ONE SIMULATION
# =============================================================================

def run_single_simulation(cfg, value_one, value_two, replicate_id):
    """
    Run one grid point. Workers only LOAD meshes; Gmsh is never called here.

    Raises on any condition that would previously have produced a silent zero,
    so a broken configuration cannot masquerade as a result.
    """
    label = (f"{cfg.parameter_one}={value_one:g} "
             f"{cfg.parameter_two}={value_two:g} rep={replicate_id}")
    start_wall = time.perf_counter()

    # Resume: a completed run already has its time series on disk, and every
    # summary number is recomputed from that file, so there is nothing to gain
    # by running it again.
    existing = timeseries_path_for(cfg, value_one, value_two, replicate_id)
    if existing.exists():
        print(f"  SKIP {label:<52} already complete", flush=True)
        return {"value_one": value_one, "value_two": value_two,
                "replicate_id": replicate_id, "success": True, "skipped": True}

    try:
        params = params_for(cfg, value_one, value_two)

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

        # k_d_ds and k_d_ss are genuinely independent here -- see the module
        # docstring. k_d_ss sets the free-S2 sink in the diffusion equation;
        # k_d_ds sets both bound-pair dissociation rates below.
        eq_S2 = build_S2_equation(
            S2, I2, Th2, I1O2, D_S2,
            k_p=params["k_p"], k_slow=params["k_slow"], k_fast=params["k_fast"],
            k_d_ss=params["k_d_ss"],
        )
        s2_solver = LinearLUSolver(tolerance=1e-10)

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

        recent_I2_values = []
        steady_state_reached = False
        steady_state_time_hr = None

        # ------------------------------------------------------- time stepping
        for step in range(1, n_steps + 1):
            S2.updateOld()
            I2.updateOld()
            Th2.updateOld()
            S2_I2.updateOld()
            S2_Th2.updateOld()

            res = 1e10
            n_sweeps = 0
            prev_res = None
            while n_sweeps < MAX_SWEEPS:
                S2_guess = S2.value

                I2_new, S2_I2_new = reaction_pair_step(
                    S2_guess, I2.old.value, S2_I2.old.value,
                    params["k_slow"], params["k_d_ds"], dt)
                Th2_new, S2_Th2_new = reaction_pair_step(
                    S2_guess, Th2.old.value, S2_Th2.old.value,
                    params["k_fast"], params["k_d_ds"], dt)

                I2.setValue(I2_new)
                S2_I2.setValue(S2_I2_new)
                Th2.setValue(Th2_new)
                S2_Th2.setValue(S2_Th2_new)

                res = eq_S2.sweep(dt=dt, solver=s2_solver)
                n_sweeps += 1

                if res < SWEEP_RESIDUAL_TARGET:
                    break
                if prev_res is not None and abs(res - prev_res) < SWEEP_PLATEAU_TOL:
                    break
                prev_res = res

            if step % save_every == 0:
                sample(step)
                recent_I2_values.append(rows[-1]["I2_center_nM"])

                # Check for steady state every CHECK_INTERVAL saved samples.
                # Only the centre probe (I2_center_nM) drives this -- the
                # edge probe is still recorded in the timeseries whenever the
                # loop stops, just not part of the exit condition.
                if (step % (save_every * CHECK_INTERVAL) == 0
                        and len(recent_I2_values) > STEADY_STATE_WINDOW):
                    recent_window = recent_I2_values[-STEADY_STATE_WINDOW:]
                    mean_I2 = np.mean(recent_window)

                    if mean_I2 > 0:
                        relative_change = np.std(recent_window) / mean_I2

                        if relative_change < STEADY_STATE_THRESHOLD:
                            steady_state_reached = True
                            steady_state_time_hr = step * dt / 3600.0
                            print(f"  → Steady state reached at "
                                  f"t={steady_state_time_hr:.2f} hr "
                                  f"(rel. change={relative_change:.2e}) "
                                  f"[{label}]", flush=True)
                            break

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

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        ts_file = timeseries_path_for(cfg, value_one, value_two, replicate_id)

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
            "parameter_one": cfg.parameter_one,
            "value_one": float(value_one),
            "parameter_two": cfg.parameter_two,
            "value_two": float(value_two),
            "replicate_id": replicate_id,
            "wall_time_s": wall,
            "n_cells": int(mesh.numberOfCells),
            "n_cells_sender_node": int(sender_mask.sum()),
            "n_cells_receiver_node": int(receiver_mask.sum()),
            "mesh_file": mesh_file.name,
            "steady_state_reached": bool(steady_state_reached),
            "steady_state_time_hr": steady_state_time_hr,
        }, indent=2))

        final = df.iloc[-1]
        print(f"  OK   {label:<52} "
              f"I2_center={final['I2_center_nM']:7.2f} nM  "
              f"S2tot_center={final['S2_total_center_nM']:8.2f} nM  "
              f"[{wall/60:.1f} min]"
              f"{'  (steady state)' if steady_state_reached else ''}",
              flush=True)

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

def build_all_meshes(cfg):
    """
    Build every mesh the grid needs, sequentially, before any worker starts.

    Works out the distinct geometries by asking mesh_path_for() for every grid
    point and de-duplicating: if neither swept parameter is geometric that is
    one mesh, if one is it is len(that axis), if both are it is the full
    product. Gmsh is never called inside a worker process.
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    geometric = [p for p in (cfg.parameter_one, cfg.parameter_two)
                 if p in MESH_AFFECTING]
    if geometric:
        print(f"Geometry-changing parameter(s): {', '.join(geometric)}")
    else:
        print("Neither parameter changes the geometry -> 1 shared mesh.")

    wanted = {}      # path -> params, de-duplicated
    for v1, v2 in product(cfg.values_one, cfg.values_two):
        params = params_for(cfg, v1, v2)
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

def run_sweep(cfg):
    tasks = [(cfg, v1, v2, rep)
             for v1 in cfg.values_one
             for v2 in cfg.values_two
             for rep in range(cfg.n_replicates)]

    print(f"\nRunning {len(tasks)} simulation(s) on {cfg.n_processes} process(es).")
    print(f"Each simulates {DEFAULT_PARAMS['total_time'] / 3600:.0f} h "
          f"at dt={DEFAULT_PARAMS['dt']:.0f} s.\n")

    t0 = time.time()
    if cfg.n_processes == 1:
        results = [run_single_simulation(*task) for task in tasks]
    else:
        with Pool(processes=cfg.n_processes) as pool:
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


def collect_results_from_disk(cfg):
    """
    Rebuild the results table by reading every time-series CSV in cfg.output_dir.

    Picks up runs from earlier sessions automatically, so an interrupted sweep
    is recovered simply by running the script again -- no simulation is
    repeated and no completed run is lost.
    """
    pattern = filename_re_for(cfg)
    rows = []
    for path in sorted(cfg.output_dir.glob("timeseries_*_rep=*.csv")):
        match = pattern.match(path.stem)
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
        for key in ("wall_time_s", "n_cells",
                    "steady_state_reached", "steady_state_time_hr"):
            if key in meta:
                row[key] = meta[key]
        rows.append(row)

    return pd.DataFrame(rows)


def check_parameters_had_an_effect(cfg, raw):
    """
    Warn loudly if varying either axis alone never changed the answer.

    This is the exact failure mode that motivated fixing the k_d_ss pinning
    bug in the module docstring above: a swept parameter that never reaches
    the equations gives bit-identical results along its axis, which looks
    perfectly healthy in a CSV. Checked independently per axis, holding the
    other axis's value fixed, so a real interaction effect (where P1 matters
    only at some P2) is not mistaken for a broken parameter.
    """
    key = "I2_center_final_nM"
    if key not in raw.columns:
        return

    for axis_name, group_col, fixed_col in (
        (cfg.parameter_one, "value_one", "value_two"),
        (cfg.parameter_two, "value_two", "value_one"),
    ):
        if raw[group_col].nunique() < 2:
            continue

        moved = False
        for _, sub in raw.groupby(fixed_col):
            values = sub[key].to_numpy(dtype=float)
            if sub[group_col].nunique() < 2:
                continue
            spread = np.nanmax(values) - np.nanmin(values)
            scale = max(abs(np.nanmean(values)), 1e-30)
            if spread / scale >= 1e-12:
                moved = True
                break

        if not moved:
            print("\n" + "!" * 78)
            print(f"WARNING: varying '{axis_name}' never changed {key} "
                  f"at any fixed value of the other parameter.")
            print(f"'{axis_name}' is almost certainly not reaching the "
                  f"equations -- check the build_S2_equation() / "
                  f"reaction_pair_step() call sites.")
            print("!" * 78)


def summarise(cfg):
    """Build raw_results.csv, the per-metric grid CSVs, and run_config.json."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_results_from_disk(cfg)
    if raw.empty:
        print("No completed runs found on disk -- nothing to summarise.")
        return None

    (cfg.output_dir / "run_config.json").write_text(json.dumps({
        "parameter_one": cfg.parameter_one,
        "values_one": cfg.values_one,
        "parameter_two": cfg.parameter_two,
        "values_two": cfg.values_two,
        "n_replicates": cfg.n_replicates,
        "default_params": DEFAULT_PARAMS,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, default=str))

    raw = raw.sort_values(["value_one", "value_two", "replicate_id"])
    raw.to_csv(cfg.output_dir / "raw_results.csv", index=False)

    # Mean over replicates at each grid point (a no-op when n_replicates = 1).
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
        grid.index.name = cfg.parameter_one
        grid.columns.name = cfg.parameter_two
        grid.to_csv(cfg.output_dir / f"grid_{metric}.csv")

    print(f"\nFound {len(raw)} completed run(s) on disk "
          f"({len(stats)} of {len(cfg.values_one) * len(cfg.values_two)} grid points).")
    print(f"Wrote {cfg.output_dir / 'raw_results.csv'}")
    print(f"Wrote {cfg.output_dir / 'grid_*.csv'} (4 files)")
    print(f"Wrote {cfg.output_dir / 'run_config.json'}")

    # Compare on the %g string, not the float, so 1/3 and friends match the
    # value that actually went into the filename.
    done = {(f"{a:g}", f"{b:g}")
            for a, b in zip(stats["value_one"], stats["value_two"])}
    missing = [(f"{a:g}", f"{b:g}") for a, b in product(cfg.values_one, cfg.values_two)
               if (f"{a:g}", f"{b:g}") not in done]
    if missing:
        print(f"\nStill missing {len(missing)} grid point(s) "
              f"(re-run to fill these in): {missing}")

    check_parameters_had_an_effect(cfg, raw)

    return stats


# =============================================================================
# PLOTS
# =============================================================================

SPECIES = [
    ("I2", "I2_center_nM", "I2_edge_nM", "[I2] (nM)"),
    ("S2_free", "S2_free_center_nM", "S2_free_edge_nM", "free [S2] (nM)"),
    ("S2_total", "S2_total_center_nM", "S2_total_edge_nM", "total [S2] (nM)"),
]


def plot_timeseries(cfg):
    """
    Overlay every run's time series, one figure per species.

    Layout: one ROW per parameter_two value, two COLUMNS (centre point, node
    edge) -- the same two-panel centre/edge format the 1-D script uses,
    stacked so the second parameter is read down the page while colour
    carries the first parameter.
    """
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cfg.values_one)))

    for name, center_col, edge_col, ylabel in SPECIES:
        n_rows = len(cfg.values_two)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.6 * n_rows),
                                 squeeze=False, sharex=True)
        fig.suptitle(f"{name}:  colour = {cfg.parameter_one},  "
                     f"row = {cfg.parameter_two}",
                     fontsize=14, fontweight="bold")

        any_data = False
        for row, v2 in enumerate(cfg.values_two):
            for color, v1 in zip(colors, cfg.values_one):
                matches = sorted(cfg.output_dir.glob(
                    f"timeseries_{cfg.parameter_one}={v1:g}"
                    f"_{cfg.parameter_two}={v2:g}_rep=*.csv"))
                for path in matches:
                    df = pd.read_csv(path)
                    any_data = True
                    label = f"{cfg.parameter_one}={v1:g}"
                    axes[row, 0].plot(df["time_hours"], df[center_col],
                                      color=color, lw=2, label=label)
                    axes[row, 1].plot(df["time_hours"], df[edge_col],
                                      color=color, lw=2, label=label)

            axes[row, 0].set_ylabel(f"{cfg.parameter_two} = {v2:g}\n{ylabel}")
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
        out = cfg.output_dir / f"timeseries_{name}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def _grid_array(cfg, stats, metric):
    """
    Reshape a metric onto the (parameter_two x parameter_one) grid.

    Rows are P2 and columns are P1 so the array drops straight into imshow
    with P1 running along x, matching the line plots underneath. Missing
    grid points stay NaN rather than being silently dropped or interpolated.
    """
    z = np.full((len(cfg.values_two), len(cfg.values_one)), np.nan)
    lookup = {(f"{a:g}", f"{b:g}"): v
              for a, b, v in zip(stats["value_one"], stats["value_two"],
                                 stats[metric])}
    for j, v1 in enumerate(cfg.values_one):
        for i, v2 in enumerate(cfg.values_two):
            value = lookup.get((f"{v1:g}", f"{v2:g}"))
            if value is not None:
                z[i, j] = value
    return z


def _heatmap(cfg, ax, z, title, cbar_label, fig):
    """
    Categorical heat map: one cell per grid point, ticks labelled with the
    actual swept values. Index-based rather than value-based axes so unevenly
    spaced sweeps are not drawn misleadingly.
    """
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cfg.values_one)))
    ax.set_xticklabels([f"{v:g}" for v in cfg.values_one], rotation=45, ha="right")
    ax.set_yticks(range(len(cfg.values_two)))
    ax.set_yticklabels([f"{v:g}" for v in cfg.values_two])
    ax.set_xlabel(cfg.parameter_one)
    ax.set_ylabel(cfg.parameter_two)
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


def _line_family(cfg, ax, z, title, ylabel):
    """The same data as lines: x = P1, one curve per P2 value."""
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(cfg.values_two)))
    for i, (color, v2) in enumerate(zip(colors, cfg.values_two)):
        ax.plot(cfg.values_one, z[i, :], marker="o", lw=2, color=color,
                label=f"{cfg.parameter_two}={v2:g}")
    ax.set_xlabel(cfg.parameter_one)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_maps(cfg, stats):
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
            z = _grid_array(cfg, stats, metric)
            _heatmap(cfg, axes[0, col], z, where, unit_label, fig)
            _line_family(cfg, axes[1, col], z, where, unit_label)

        fig.tight_layout()
        out = cfg.output_dir / f"{filename}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


# =============================================================================
# ENTRY POINT  -- call this from each per-pair sweep script
# =============================================================================

def run(cfg):
    print("=" * 78)
    print("TWO-PARAMETER SWEEP")
    print("=" * 78)
    print(f"parameter 1 : {cfg.parameter_one}")
    print(f"values 1    : {cfg.values_one}")
    print(f"parameter 2 : {cfg.parameter_two}")
    print(f"values 2    : {cfg.values_two}")
    print(f"grid        : {len(cfg.values_one)} x {len(cfg.values_two)} "
          f"= {len(cfg.values_one) * len(cfg.values_two)} points "
          f"x {cfg.n_replicates} replicate(s) = {cfg.n_tasks} run(s)")
    print(f"output      : {cfg.output_dir}")
    print("=" * 78)

    for name in (cfg.parameter_one, cfg.parameter_two):
        if name not in DEFAULT_PARAMS:
            raise SystemExit(
                f"'{name}' is not a known parameter. "
                f"Valid names:\n  {sorted(DEFAULT_PARAMS)}")

    print("\nSTAGE 1 - meshes\n")
    build_all_meshes(cfg)

    print("\nSTAGE 2 - simulations\n")
    results = run_sweep(cfg)

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} simulation(s) FAILED this session:")
        for r in failed:
            print(f"  {cfg.parameter_one}={r['value_one']:g} "
                  f"{cfg.parameter_two}={r['value_two']:g}: "
                  f"{r.get('error', 'unknown error')}")
        print("Completed runs are unaffected and are summarised below.")

    print("\nSTAGE 3 - analysis\n")
    stats = summarise(cfg)
    plot_timeseries(cfg)
    plot_maps(cfg, stats)

    print("\nDone.\n")
