"""
Unified parameter sweep for the 2-node tethered-genelet sender/receiver model.

Replaces all six earlier sweep scripts. Handles BOTH kinds of parameter:

  * spatial      (distance_between, node_diameter, fine_dx, ...)
        -> a new mesh is generated for each sweep value
  * non-spatial  (Th2_init, k_p, k_slow, k_fast, k_d_ss, k_d_ds, D_gel, ...)
        -> one mesh is generated and reused by every value

Physics, time step and readout are matched to TG_Rmesh_tanh.py, which is the
validated gold standard (agrees with COMSOL to ~0.6 nM). The equation solve
itself uses the split S2-only formulation from TG_Rmesh_fast.py instead of
the original 5-variable coupled solve -- same physics, ~4.6x faster per step
(see item 6 below).


WHAT WAS WRONG WITH THE OLD SWEEPS, AND WHAT CHANGED HERE
---------------------------------------------------------
1. NON-CONFORMAL MESH  (the "no S2 ever reaches the receiver" bug)
   create_gmsh_radial_mesh() never performed a boolean operation, so the two
   node disks were meshed as disconnected islands floating on top of the bath
   mesh. Diagnostics showed 3 connected components on every mesh at every
   distance. The probe cell -- argmin(distance to receiver centre) -- would
   sometimes land inside a sealed island, which can never receive S2. Result:
   I2 frozen at exactly 0.1 and S2 exactly 0.0 forever.
   FIX: use create_conformal_radial_mesh(), which calls occ.fragment().
        Every mesh is verified to be a single connected region before use.

2. TOO LITTLE SIMULATED TIME
   The failing sweeps ran 2-4 h; the working ones ran 8 h. At larger
   separations the threshold simply has not been titrated yet at 2 h.
   FIX: total_time defaults to 8 h, matching TG_Rmesh_tanh.py.

3. RATE CONSTANTS WERE NOT ACTUALLY SWEEPABLE
   Functions.intialize_equations() read k_p / k_slow / k_fast / k_d_ss /
   k_d_ds from Functions.py module globals, so a worker that set them locally
   changed nothing at all.
   FIX: build_equations() below takes every rate constant as an argument.

4. SILENT FAILURE
   Old scripts wrote 0.0 into a CSV and reported success=True.
   FIX: validation gates raise instead. A broken run fails loudly.

5. BROKEN SAMPLING LOOPS
   Various old files sampled twice per step, never sampled at all, or appended
   one series inside another series' if-block.
   FIX: one sampling block, one place.

6. SLOW COUPLED SOLVE
   build_equations() bundled all 5 variables (S2, I2, Th2, S2_I2, S2_Th2) into
   one FiPy coupled equation ("&"), forcing every sweep to assemble and
   factorize a 5*Ncells x 5*Ncells sparse matrix -- even though only S2 has a
   DiffusionTerm. The other four are purely local per-cell reaction ODEs.
   Profiling also showed the sweep loop's "res > 1e-6" target was unreachable:
   the default scipy LinearLUSolver only refines to 1e-5, so every step
   silently maxed out at 10 sweeps with the residual flat after sweep ~4.
   FIX: build_S2_equation() solves only S2 through FiPy's sparse solver
        (Ncells unknowns). I2/Th2/S2_I2/S2_Th2 are updated with the closed-form
        backward-Euler step in reaction_pair_step() -- an exact solve of the
        per-cell 2x2 linear system, vectorized with numpy, no sparse solve
        involved. The LU tolerance is tightened (1e-10) so the sweep
        residual is a real convergence signal. Verified against the original
        coupled solve in TG_Rmesh_fast.py / TG_Rmesh_tanh.py: <0.0001%
        difference in receiver concentrations, ~4.6x faster per step.


READOUT
-------
One measurement is recorded at every save point:

  *_center : value in the single cell nearest the receiver centre.
             This is what TG_Rmesh_tanh.py does AND what COMSOL does
             (its exports are "Table 1 - Point Evaluation 1"), so this is
             the column to compare against COMSOL.

  (The node-volume-averaged "_avg" readout and the node-edge probe have both
  been removed -- only the centre-point probe is recorded now.)
"""

import json
import os
import re
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # workers must not try to open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fipy import CellVariable, DiffusionTerm, Gmsh2D, ImplicitSourceTerm, LinearLUSolver, TransientTerm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Mesh.New_simple_mesh import create_conformal_radial_mesh

warnings.filterwarnings("ignore")


# =============================================================================
# USER CONFIGURATION  -- this is the only block you normally edit
# =============================================================================

# SWEEP_PARAMETER = "k_slow"

# SWEEP_VALUES = np.array([1,2,3,4,5])
# SWEEP_VALUES = SWEEP_VALUES * 5e4 * 1e-6

# Examples for other sweeps (uncomment one):
#   SWEEP_PARAMETER = "Th2_init";  SWEEP_VALUES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
#   SWEEP_PARAMETER = "k_p";       SWEEP_VALUES = [0.02, 0.2, 2.0]
SWEEP_PARAMETER = "k_d_ds";    SWEEP_VALUES = np.linspace(0, 0.1, 50) * 3e-4

N_REPLICATES = 1

# The model is deterministic, so replicates only make sense if you later add
# noise. Leave at 1 unless you know you want more.

# os.cpu_count() reports the physical node's full CPU count on Rockfish,
# not what Slurm actually allocated to this job -- so it can't be trusted
# to size the pool. SLURM_CPUS_PER_TASK (set by the --cpus-per-task sbatch
# directive) is authoritative when present; cpu_count() is only a fallback
# for runs outside Slurm (e.g. on a laptop).
_allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
N_PROCESSES = min(len(SWEEP_VALUES) * N_REPLICATES, max(1, _allocated_cpus))


# =============================================================================
# DEFAULT PARAMETERS  -- matched to TG_Rmesh_tanh.py
# =============================================================================

DEFAULT_PARAMS = {
    # transport
    "D_solution": 150.0,        # um^2/s
    "D_gel": 60.0,              # um^2/s

    # kinetics  (concentrations in uM, so bimolecular rates are 1/(uM s))
    "k_p": 0.01,                 # 1/s      transcription
    "k_d_ds": 3e-4,             # 1/s      double-stranded degradation
    "k_d_ss": 3e-4,             # 1/s      single-stranded degradation
    "k_slow": 5e4 * 1e-6,       # 1/(uM s) S2 + I2  -> S2:I2
    "k_fast": 1e6 * 1e-6,       # 1/(uM s) S2 + Th2 -> S2:Th2

    # initial conditions
    "I1O2_init": 0.1,           # uM, template in the sender node
    "I2_init": 0.1,             # uM, receiver switch
    "Th2_init": 0.4,            # uM, threshold

    # geometry
    "node_diameter": 75.0,      # um
    "distance_between": 300.0,  # um, centre-to-centre
    "total_width": 5000,         # um (1 cm)
    "total_height": 5000,        # um (1 mm)

    # mesh
    "fine_dx": 5.0,             # um, gold-standard resolution
    "coarse_dx": 100.0,         # um
    "growth_rate": 1.5,         # cell size multiplier between rings
    "cells_per_level": 3,       # ring width, in cells

    # time stepping
    "dt": 60.0,                 # s
    "total_time": 8 * 3600,     # s -- 8 h, matching the validated single run
    "save_interval_time": 60.0, # s
}

# Changing any of these changes the geometry, so the mesh must be rebuilt.
MESH_AFFECTING = {
    "node_diameter", "distance_between", "total_width", "total_height",
    "fine_dx", "coarse_dx", "growth_rate", "cells_per_level",
}

MESH_DIR = Path(__file__).resolve().parent / "meshes_conformal"
OUTPUT_DIR = Path(__file__).resolve().parent / f"sweep_{SWEEP_PARAMETER}_zoomed_in_ImprovedV4_5mmx5mm"

# Sweep control for the split-equation solver (see build_S2_equation() and
# reaction_pair_step() below, and item 6 in the module docstring).
MAX_SWEEPS = 15
SWEEP_RESIDUAL_TARGET = 1e-8
SWEEP_PLATEAU_TOL = 1e-9


# =============================================================================
# MODEL
# =============================================================================

def apply_sweep_value(param_value):
    """Build the parameter dict for one sweep point."""
    params = dict(DEFAULT_PARAMS)
    params[SWEEP_PARAMETER] = param_value
    return params


def timeseries_path_for(param_value, replicate_id, params):
    """
    The durable record of one completed run.

    Encodes the actual k_d_ss and k_d_ds values used, not just the swept
    parameter -- these two used to be forced equal (LINKED_PARAMETERS), so a
    run from back then and a run now can share the same SWEEP_PARAMETER value
    while using different physics (e.g. k_d_ss=k_d_ds=6e-5 vs the current
    k_d_ss=3e-4 default with k_d_ds=6e-5 swept). Without this, the old file
    would be mistaken for a completed run of the new configuration and
    silently skipped.
    """
    return OUTPUT_DIR / (
        f"timeseries_{SWEEP_PARAMETER}={param_value:g}_rep={replicate_id}"
        f"_kdss={params['k_d_ss']:g}_kdds={params['k_d_ds']:g}"
        f"_5mmx5mm_speedup_newparameters.csv")


def mesh_path_for(params):
    """Deterministic mesh filename; identical geometry reuses the same file."""
    return MESH_DIR / (
        f"conformal_ccd={params['distance_between']:.1f}"
        f"_nd={params['node_diameter']:.1f}"
        f"_fine={params['fine_dx']:.2f}"
        f"_coarse={params['coarse_dx']:.1f}"
        f"_gr={params['growth_rate']:.2f}"
        f"_cpl={params['cells_per_level']:g}"
        f"_W={params['total_width']:.0f}"
        f"_H={params['total_height']:.0f}.msh"
    )


def initialize_fields(mesh, x, y, sender_x, receiver_x, y_center, params):
    """
    Set up the five reacting species plus the template and the diffusivity.

    Uses hard boolean masks, matching initalize_variables_speedup() in
    Functions.py, which is what the validated TG_Rmesh_tanh.py run uses.
    Because the mesh is now conformal, the node boundary is an actual mesh
    edge, so a cell is unambiguously inside or outside the gel.
    """
    node_radius = params["node_diameter"] / 2.0

    sender_mask = np.sqrt((x - sender_x) ** 2 + (y - y_center) ** 2) <= node_radius
    receiver_mask = np.sqrt((x - receiver_x) ** 2 + (y - y_center) ** 2) <= node_radius
    gel_mask = sender_mask | receiver_mask

    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    I2 = CellVariable(name="I2", mesh=mesh, value=0.0, hasOld=True)
    Th2 = CellVariable(name="Th2", mesh=mesh, value=0.0, hasOld=True)
    S2_I2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
    S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

    I2.setValue(params["I2_init"] * receiver_mask)
    Th2.setValue(params["Th2_init"] * receiver_mask)

    I1O2 = CellVariable(name="I1O2", mesh=mesh,
                        value=params["I1O2_init"] * sender_mask)

    D_S2 = CellVariable(
        name="D_S2", mesh=mesh,
        value=params["D_gel"] * gel_mask + params["D_solution"] * (~gel_mask),
    )

    return S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2, sender_mask, receiver_mask


def build_S2_equation(S2, I2, Th2, I1O2, D_S2, k_p, k_slow, k_fast, k_d_ss):
    """
    S2 is the only variable with a DiffusionTerm -- I2, Th2, S2_I2, S2_Th2
    are purely local per-cell reaction ODEs with no spatial term at all (see
    reaction_pair_step() below). Solving S2 alone through FiPy's sparse
    solver, instead of bundling all 5 variables into one coupled equation,
    shrinks every linear solve from 5*Ncells to Ncells unknowns.

    Every rate constant is an explicit argument. The old
    Functions.intialize_equations() read them from module globals, which is
    why sweeping k_p or k_slow silently did nothing.
    """
    return (
        TransientTerm(var=S2)
        == DiffusionTerm(coeff=D_S2, var=S2)
        + k_p * I1O2
        + ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2)
    )


def reaction_pair_step(S2_now, X_old, C_old, k_on, k_off, dt):
    """
    Closed-form backward-Euler step for one exchange pair (X <-> C):
        dX/dt = -k_on*S2*X + k_off*C
        dC/dt = +k_on*S2*X - k_off*C
    S2 is held fixed at the current Picard-sweep guess -- the same lagging
    FiPy's old coupled solver did for this bilinear term, since k_on*S2*X is
    nonlinear and must be frozen either way. This is the exact solution of
    the resulting per-cell 2x2 linear system, fully vectorized with numpy, no
    sparse solve needed.

    Use with (k_on, k_off) = (k_slow, k_d_ds) for the (I2, S2_I2) pair, and
    (k_fast, k_d_ds) for the (Th2, S2_Th2) pair.
    """
    a = k_on * S2_now
    d = k_off
    det = 1.0 + dt * (a + d)
    X_new = ((1.0 + dt * d) * X_old + dt * d * C_old) / det
    C_new = (dt * a * X_old + (1.0 + dt * a) * C_old) / det
    return X_new, C_new


# =============================================================================
# ONE SIMULATION
# =============================================================================

def run_single_simulation(param_value, replicate_id):
    """
    Run one simulation. Workers only LOAD meshes; Gmsh is never called here.

    Raises on any condition that would previously have produced a silent
    zero, so a broken configuration cannot masquerade as a result.
    """
    label = f"{SWEEP_PARAMETER}={param_value:g} rep={replicate_id}"
    start_wall = time.perf_counter()

    params = apply_sweep_value(param_value)

    # Resume: a completed run already has its time series on disk, and every
    # summary number is recomputed from that file, so there is nothing to gain
    # by running it again.
    existing = timeseries_path_for(param_value, replicate_id, params)
    if existing.exists():
        print(f"  SKIP {label:<38} already complete ({existing.name})", flush=True)
        return {"param_value": param_value, "replicate_id": replicate_id,
                "success": True, "skipped": True}

    try:
        dt = params["dt"]
        n_steps = int(params["total_time"] / dt)
        save_every = max(1, int(params["save_interval_time"] / dt))
        node_radius = params["node_diameter"] / 2.0



        # ------------------------------------------------------------- mesh
        mesh_file = mesh_path_for(params)
        if not mesh_file.exists():
            raise FileNotFoundError(
                f"Mesh missing: {mesh_file}. It should have been built before "
                f"the pool started."
            )

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

        eq_S2 = build_S2_equation(
            S2, I2, Th2, I1O2, D_S2,
            k_p=params["k_p"], k_slow=params["k_slow"], k_fast=params["k_fast"],
            k_d_ss=params["k_d_ss"],
        )
        s2_solver = LinearLUSolver(tolerance=1e-10)

        # Probe cell: nearest to the receiver centre. Safe now that the mesh
        # is conformal -- it can no longer land in a disconnected island.
        probe_idx = int(np.argmin(np.hypot(x - receiver_x, y - y_center)))

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
        ts_file = timeseries_path_for(param_value, replicate_id, params)

        # Write to a temporary file and rename. os.replace is atomic, so an
        # interruption (or a full disk) can never leave a half-written CSV
        # behind that a later run would mistake for a completed simulation.
        tmp_file = ts_file.with_suffix(".csv.partial")
        df.to_csv(tmp_file, index=False)
        os.replace(tmp_file, ts_file)

        wall = time.perf_counter() - start_wall

        # Metadata that cannot be recovered from the time series itself.
        ts_file.with_suffix(".meta.json").write_text(json.dumps({
            "wall_time_s": wall,
            "n_cells": int(mesh.numberOfCells),
            "n_cells_sender_node": int(sender_mask.sum()),
            "n_cells_receiver_node": int(receiver_mask.sum()),
            "mesh_file": mesh_file.name,
        }, indent=2))

        final = df.iloc[-1]

        print(f"  OK   {label:<38} "
              f"I2_center={final['I2_center_nM']:7.2f} nM  "
              f"S2tot_center={final['S2_total_center_nM']:8.2f} nM  "
              f"[{wall/60:.1f} min]", flush=True)

        return {
            "param_value": param_value,
            "replicate_id": replicate_id,
            "I2_center_final_nM": final["I2_center_nM"],
            "S2_free_center_final_nM": final["S2_free_center_nM"],
            "S2_total_center_final_nM": final["S2_total_center_nM"],
            "half_time_center_hr": half_time(df["time_hours"].values,
                                             df["I2_center_nM"].values),
            "n_cells": int(mesh.numberOfCells),
            "wall_time_s": wall,
            "timeseries_file": str(ts_file),
            "success": True,
        }

    except Exception as exc:
        import traceback
        print(f"  FAIL {label:<38} {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return {
            "param_value": param_value,
            "replicate_id": replicate_id,
            "wall_time_s": time.perf_counter() - start_wall,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def half_time(time_hours, signal):
    """
    Time at which the signal falls halfway from its initial to its final
    value, linearly interpolated between the two bracketing samples.
    """
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
# MESH PRE-GENERATION
# =============================================================================

def build_all_meshes():
    """
    Build every mesh the sweep needs, sequentially, before any worker starts.

    Gmsh is never called inside a worker process. A diagnostic confirmed that
    build order does not affect the result, so sequential generation in this
    process is safe; keeping Gmsh out of the pool simply removes a whole class
    of possible interaction.
    """
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    if SWEEP_PARAMETER in MESH_AFFECTING:
        values = SWEEP_VALUES
        print(f"'{SWEEP_PARAMETER}' changes the geometry -> "
              f"{len(values)} mesh(es) needed.")
    else:
        values = [DEFAULT_PARAMS[SWEEP_PARAMETER]]
        print(f"'{SWEEP_PARAMETER}' does not change the geometry -> "
              f"1 shared mesh.")

    wanted = []
    for value in values:
        params = dict(DEFAULT_PARAMS)
        params[SWEEP_PARAMETER] = value
        wanted.append((value, params, mesh_path_for(params)))

    for value, params, path in wanted:
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
    for _, _, path in wanted:
        mesh = Gmsh2D(str(path))
        print(f"  verify {path.name}: {mesh.numberOfCells:,} cells")


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_sweep():
    tasks = [(value, rep)
             for value in SWEEP_VALUES
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


METRICS = [
    "I2_center_final_nM", "S2_free_center_final_nM", "S2_total_center_final_nM",
    "half_time_center_hr",
]


def scalars_from_timeseries(df, param_value, replicate_id):
    """
    Derive every summary scalar from a saved time series.

    This is the key to not losing work: the per-run time-series CSV is the
    durable record, written by the worker the moment that run finishes. Every
    summary number is recomputed from it, so a crash during analysis (or a
    full disk, or a killed process) can never destroy completed simulations.
    """
    final = df.iloc[-1]
    return {
        "param_value": param_value,
        "replicate_id": replicate_id,
        "I2_center_final_nM": float(final["I2_center_nM"]),
        "S2_free_center_final_nM": float(final["S2_free_center_nM"]),
        "S2_total_center_final_nM": float(final["S2_total_center_nM"]),
        "half_time_center_hr": half_time(df["time_hours"].values,
                                         df["I2_center_nM"].values),
        "n_samples": len(df),
    }


def collect_results_from_disk():
    """
    Rebuild the results table by reading every time-series CSV in OUTPUT_DIR.

    Picks up runs from earlier sessions automatically, so an interrupted sweep
    is recovered simply by running the script again -- no simulation is
    repeated and no completed run is lost.
    """
    # Tolerant of anything appended after the replicate number, so adding a
    # suffix like "_5mmx5mm" to the filename does not make runs invisible.
    pattern = re.compile(
        rf"timeseries_{re.escape(SWEEP_PARAMETER)}="
        rf"(?P<value>[-+0-9.eE]+)_rep=(?P<rep>\d+)")

    rows = []
    for path in sorted(OUTPUT_DIR.glob(f"timeseries_{SWEEP_PARAMETER}=*_rep=*.csv")):
        match = pattern.search(path.stem)
        if match is None:
            print(f"  skipping unparseable filename: {path.name}")
            continue
        try:
            param_value = float(match.group("value"))
            replicate_id = int(match.group("rep"))
        except ValueError:
            print(f"  skipping unparseable filename: {path.name}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  skipping unreadable {path.name}: {exc}")
            continue
        if df.empty:
            print(f"  skipping empty {path.name}")
            continue

        row = scalars_from_timeseries(df, param_value, replicate_id)

        # Optional metadata written alongside the run (wall time, cell count).
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                row.update(json.loads(meta_path.read_text()))
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


def summarise():
    """
    Build raw_results.csv and summary_stats.csv from whatever is on disk.

    Note on standard deviations: this model is deterministic, so with
    N_REPLICATES = 1 there is nothing to take a std of and those columns are
    empty by definition. They only become meaningful if you add replicates
    that actually differ (e.g. randomised initial conditions).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = collect_results_from_disk()
    if raw.empty:
        print("No completed runs found on disk -- nothing to summarise.")
        return None

    # Provenance goes in a JSON sidecar and a column, NOT in the filename.
    # Embedding a Python list in a filename produces names containing spaces,
    # commas and brackets, which quote badly in the shell and broke the
    # earlier run.
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps({
        "sweep_parameter": SWEEP_PARAMETER,
        "sweep_values": SWEEP_VALUES,
        "n_replicates": N_REPLICATES,
        "default_params": DEFAULT_PARAMS,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, default=str))

    raw = raw.sort_values(["param_value", "replicate_id"])
    raw.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)

    # wall_time_s / n_cells only exist when the metadata sidecar was written.
    aggregated = METRICS + [c for c in ("wall_time_s", "n_cells")
                            if c in raw.columns]
    agg = {m: ["mean", "std"] for m in aggregated if m in raw.columns}
    stats = raw.groupby("param_value").agg(agg)
    stats.columns = ["_".join(c) for c in stats.columns]
    stats = stats.reset_index()
    stats.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)

    print(f"\nFound {len(raw)} completed run(s) on disk.")
    print(f"Wrote {OUTPUT_DIR / 'raw_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'summary_stats.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'run_config.json'}")

    missing = sorted(set(np.asarray(SWEEP_VALUES).tolist()) - set(raw["param_value"]))
    if missing:
        print(f"\nStill missing (re-run to fill these in): {missing}")

    check_parameter_had_an_effect(raw)

    return stats


def check_parameter_had_an_effect(raw):
    """
    Warn loudly if every run produced the same answer.

    A swept parameter that never reaches the equations gives bit-identical
    results, which look perfectly healthy in a CSV -- no crash, no zeros, no
    failed gate. The only symptom is that the numbers do not move. This is the
    one failure mode the in-run validation gates cannot see, so it is checked
    here instead.
    """
    if raw["param_value"].nunique() < 2:
        return

    key = "I2_center_final_nM"
    if key not in raw.columns:
        return

    values = raw[key].to_numpy(dtype=float)
    spread = np.nanmax(values) - np.nanmin(values)
    scale = max(abs(np.nanmean(values)), 1e-30)

    if spread / scale < 1e-12:
        print("\n" + "!" * 78)
        print(f"WARNING: every run returned the same {key} "
              f"({values[0]:.12g}).")
        print(f"'{SWEEP_PARAMETER}' varied across {raw['param_value'].nunique()} "
              f"values but changed nothing, so it is almost certainly not")
        print("reaching the equations. Check that build_S2_equation() / "
              "reaction_pair_step() is passed")
        print(f"params['{SWEEP_PARAMETER}'] and not a different key.")
        print("!" * 78)


def plot(stats):
    if stats is None or stats.empty:
        return

    xs = stats["param_value"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Parameter sweep: {SWEEP_PARAMETER}",
                 fontsize=15, fontweight="bold")

    def series(ax, metric, **kw):
        """
        Plot one metric. summarise() emits '<metric>_mean' and '<metric>_std'
        columns, so the suffix is added here rather than at every call site --
        the earlier KeyError came from call sites passing the bare metric name
        while the dataframe held the suffixed one.
        """
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in stats.columns:
            return
        yerr = stats[std_col] if std_col in stats.columns else None
        if yerr is not None and not np.isfinite(yerr).any():
            yerr = None          # single replicate: no error bars to draw
        ax.errorbar(xs, stats[mean_col], yerr=yerr,
                    marker="o", capsize=4, linewidth=2, **kw)

    ax = axes[0, 0]
    series(ax, "I2_center_final_nM", label="centre point (COMSOL-comparable)")
    ax.set_ylabel("Final [I2] at receiver (nM)")
    ax.set_title("Receiver switch")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    series(ax, "S2_total_center_final_nM", label="centre point")
    ax.set_ylabel("Final total [S2] at receiver (nM)")
    ax.set_title("Total RNA delivered")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    series(ax, "half_time_center_hr", label="centre point")
    ax.set_ylabel("Half-time of I2 (hours)")
    ax.set_title("Turn-on time")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    series(ax, "wall_time_s", color="tab:orange")
    ax.set_ylabel("Wall time per run (s)")
    ax.set_title("Cost")

    for ax in axes.flat:
        ax.set_xlabel(SWEEP_PARAMETER)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / f"sweep_{SWEEP_PARAMETER}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_timeseries():
    """Overlay every run's time series so the transients can be compared."""
    files = sorted(OUTPUT_DIR.glob("timeseries_*.csv"))
    if not files:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SWEEP_VALUES)))

    for color, value in zip(colors, SWEEP_VALUES):
        matches = sorted(OUTPUT_DIR.glob(
            f"timeseries_{SWEEP_PARAMETER}={value:g}_rep=*.csv"))
        for path in matches:
            df = pd.read_csv(path)
            label = f"{SWEEP_PARAMETER}={value:g}"
            ax.plot(df["time_hours"], df["I2_center_nM"],
                    color=color, lw=2, label=label)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("nM")
    ax.set_title("[I2] at centre point")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / f"timeseries_{SWEEP_PARAMETER}_speedup.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("UNIFIED PARAMETER SWEEP")
    print("=" * 78)
    print(f"parameter : {SWEEP_PARAMETER}")
    print(f"values    : {SWEEP_VALUES}")
    print(f"replicates: {N_REPLICATES}")
    print(f"output    : {OUTPUT_DIR}")
    print("=" * 78)

    if SWEEP_PARAMETER not in DEFAULT_PARAMS:
        raise SystemExit(
            f"'{SWEEP_PARAMETER}' is not a known parameter. "
            f"Valid names:\n  {sorted(DEFAULT_PARAMS)}")

    print("\nSTAGE 1 - meshes\n")
    build_all_meshes()

    print("\nSTAGE 2 - simulations\n")
    results = run_sweep()

    failed = [r for r in results if not r.get("success")]
    if failed:
        print(f"\n{len(failed)} simulation(s) FAILED this session:")
        for r in failed:
            print(f"  {SWEEP_PARAMETER}={r['param_value']}: "
                  f"{r.get('error', 'unknown error')}")
        print("Completed runs are unaffected and are summarised below.")

    print("\nSTAGE 3 - analysis\n")
    stats = summarise()
    plot(stats)
    plot_timeseries()

    print("\nDone.\n")
