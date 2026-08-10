"""
Parameter sweep with pre-generated Gmsh meshes.

Workflow:
1. Generate all required meshes sequentially.
2. Run FiPy simulations in parallel by loading existing mesh files.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fipy import Gmsh2D
from fipy.tools import numerix

# =============================================================================
# PATH SETUP
# =============================================================================

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

warnings.filterwarnings("ignore")

from Functions_and_system.Functions import (
    calculate_total_amount,
    intialize_equations,
    initalize_variables,
)

from Mesh.New_simple_mesh import create_gmsh_radial_mesh


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SWEEP_PARAMETER = "distance_between"

SWEEP_VALUES = [800.0, 1000.0, 1200.0]

N_REPLICATES = 1

if len(SWEEP_VALUES) < 6:
    N_PROCESSES = len(SWEEP_VALUES)
else:
    N_PROCESSES = 5


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_PARAMS = {
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
    "distance_between": 300.0,
    "total_width": 1e4,
    "total_height": 1e3,
    "dt": 60.0,
    "total_time": 4 * 3600,
    "save_interval_time": 60.0,
    "fine_dx": 0.75,
    "coarse_dx": 100.0,
}

# Important: preserve global constants if your Functions.py expects them globally
k_p = DEFAULT_PARAMS["k_p"]
k_d_ds = DEFAULT_PARAMS["k_d_ds"]
k_d_ss = DEFAULT_PARAMS["k_d_ss"]
k_slow = DEFAULT_PARAMS["k_slow"]
k_fast = DEFAULT_PARAMS["k_fast"]


# =============================================================================
# MESH FOLDER
# =============================================================================

MESH_DIR = Path("pregenerated_meshes")
MESH_DIR.mkdir(exist_ok=True)


def get_mesh_filename(params):
    """
    Deterministic mesh filename based on geometry parameters.
    """
    return MESH_DIR / (
        f"mesh_ccd={params['distance_between']:.1f}_"
        f"nd={params['node_diameter']:.1f}_"
        f"fine={params['fine_dx']:.2f}_"
        f"coarse={params['coarse_dx']:.1f}.msh"
    )


# =============================================================================
# STAGE 1: SEQUENTIAL MESH GENERATION
# =============================================================================

def generate_all_meshes_sequentially():
    """
    Generate all meshes one-by-one before multiprocessing starts.
    This removes Gmsh from the parallel part of the workflow.
    """

    print("\n" + "=" * 80)
    print("STAGE 1: GENERATING MESHES SEQUENTIALLY")
    print("=" * 80)

    for param_value in SWEEP_VALUES:
        params = DEFAULT_PARAMS.copy()
        params[SWEEP_PARAMETER] = param_value

        mesh_filename = get_mesh_filename(params)

        if mesh_filename.exists():
            print(f"✓ Mesh already exists: {mesh_filename}")
            continue

        print(f"\nGenerating mesh for {SWEEP_PARAMETER} = {param_value}")
        print(f"Saving to: {mesh_filename}")

        start = time.time()

        # This assumes your create_gmsh_radial_mesh can accept a filename.
        # If it cannot, modify New_simple_mesh.py so it writes to this exact file.
        create_gmsh_radial_mesh(
            bath_width=params["total_width"],
            bath_height=params["total_height"],
            node_diameter=params["node_diameter"],
            distance_between_nodes=params["distance_between"],
            min_cell_size=params["fine_dx"],
            max_cell_size=params["coarse_dx"],
            growth_rate=1.5,
            mesh_filename=str(mesh_filename),
            verbose=True,
        )

        print(f"Generated in {(time.time() - start):.2f} s")

    print("\nAll required meshes generated.")
    print("=" * 80 + "\n")


# =============================================================================
# OPTIONAL: VERIFY MESHES BEFORE RUNNING SWEEP
# =============================================================================

def verify_meshes():
    """
    Load each mesh with FiPy once before multiprocessing.
    This catches corrupted or missing mesh files before the sweep.
    """

    print("\n" + "=" * 80)
    print("VERIFYING MESHES WITH FIPY")
    print("=" * 80)

    for param_value in SWEEP_VALUES:
        params = DEFAULT_PARAMS.copy()
        params[SWEEP_PARAMETER] = param_value

        mesh_filename = get_mesh_filename(params)

        if not mesh_filename.exists():
            raise FileNotFoundError(f"Missing mesh: {mesh_filename}")

        mesh = Gmsh2D(str(mesh_filename))

        print(
            f"✓ {SWEEP_PARAMETER}={param_value}: "
            f"{mesh.numberOfCells:,} cells loaded from {mesh_filename}"
        )

    print("=" * 80 + "\n")


# =============================================================================
# HELPER: HALF TIME
# =============================================================================

def calculate_half_time(time_array, I2_array, I2_init):
    I2_final = I2_array[-1]
    I2_half = (I2_init + I2_final) / 2.0

    below_half = I2_array < I2_half

    if np.any(below_half):
        idx = np.argmax(below_half)
        return time_array[idx]
    else:
        return np.nan


# =============================================================================
# STAGE 2: PARALLEL SIMULATION
# =============================================================================

def run_single_simulation(param_value, replicate_number):
    """
    One FiPy simulation.
    Important: this function does NOT call Gmsh to generate a mesh.
    It only imports an already-existing .msh file.
    """

    try:
        start_time = time.perf_counter()

        params = DEFAULT_PARAMS.copy()
        params[SWEEP_PARAMETER] = param_value

        # Extract parameters
        distance_between = params["distance_between"]
        node_diameter = params["node_diameter"]
        total_width = params["total_width"]
        total_height = params["total_height"]

        D_solution = params["D_solution"]
        D_gel = params["D_gel"]

        I1O2_init = params["I1O2_init"]
        I2_init = params["I2_init"]
        Th2_init = params["Th2_init"]

        dt = params["dt"]
        total_time = params["total_time"]
        save_interval_time = params["save_interval_time"]

        n_steps = int(total_time / dt)
        save_interval_steps = int(save_interval_time / dt)

        # ---------------------------------------------------------------------
        # LOAD PRE-GENERATED MESH
        # ---------------------------------------------------------------------

        mesh_filename = get_mesh_filename(params)

        if not mesh_filename.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_filename}")

        mesh = Gmsh2D(str(mesh_filename))

        # ---------------------------------------------------------------------
        # GEOMETRY FROM PARAMS
        # ---------------------------------------------------------------------

        node_radius = node_diameter / 2.0
        domain_center_x = total_width / 2.0
        domain_center_y = total_height / 2.0

        sender_center_x = domain_center_x - distance_between / 2.0
        receiver_center_x = domain_center_x + distance_between / 2.0

        sender_center_y = domain_center_y
        receiver_center_y = domain_center_y

        x, y = mesh.cellCenters[0], mesh.cellCenters[1]

        # ---------------------------------------------------------------------
        # INITIALIZE VARIABLES
        # ---------------------------------------------------------------------

        S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(
            mesh,
            x,
            y,
            sender_center_x,
            receiver_center_x,
            receiver_center_y,
            node_radius,
            I2_init,
            Th2_init,
            I1O2_init,
            D_gel,
            D_solution,
        )

        eq = intialize_equations(
            S2,
            D_S2,
            I1O2,
            I2,
            Th2,
            S2_I2,
            S2_Th2,
        )


        # =============================================================================
        # DEBUG: CHECK NODE INITIALIZATION
        # =============================================================================

        dist_to_sender = numerix.sqrt(
            (x - sender_center_x) ** 2 + (y - receiver_center_y) ** 2
        )

        dist_to_receiver = numerix.sqrt(
            (x - receiver_center_x) ** 2 + (y - receiver_center_y) ** 2
        )

        sender_mask = dist_to_sender <= node_radius
        receiver_mask = dist_to_receiver <= node_radius

        print("\n" + "-" * 80)
        print(f"DEBUG for {SWEEP_PARAMETER} = {param_value}")
        print(f"Mesh file: {mesh_filename}")
        print(f"Number of mesh cells: {mesh.numberOfCells}")
        print(f"Sender center:   ({sender_center_x:.2f}, {receiver_center_y:.2f})")
        print(f"Receiver center: ({receiver_center_x:.2f}, {receiver_center_y:.2f})")
        print(f"Node radius: {node_radius:.2f}")

        print(f"Cells inside sender node:   {int(np.sum(sender_mask.value))}")
        print(f"Cells inside receiver node: {int(np.sum(receiver_mask.value))}")

        print(f"Min distance to sender center:   {float(np.min(dist_to_sender.value)):.4f}")
        print(f"Min distance to receiver center: {float(np.min(dist_to_receiver.value)):.4f}")

        print(f"I1O2 max: {float(np.max(I1O2.value)):.6g}")
        print(f"I1O2 sum: {float(np.sum(I1O2.value)):.6g}")

        print(f"I2 max:   {float(np.max(I2.value)):.6g}")
        print(f"I2 sum:   {float(np.sum(I2.value)):.6g}")

        print(f"Th2 max:  {float(np.max(Th2.value)):.6g}")
        print(f"Th2 sum:  {float(np.sum(Th2.value)):.6g}")

        print(f"S2 max initial: {float(np.max(S2.value)):.6g}")
        print(f"D_S2 min: {float(np.min(D_S2.value)):.6g}")
        print(f"D_S2 max: {float(np.max(D_S2.value)):.6g}")
        print("-" * 80 + "\n")
                # Receiver center index
        distances_to_receiver = np.sqrt(
                    (x - receiver_center_x) ** 2 + (y - receiver_center_y) ** 2
                )

        receiver_center_idx = np.argmin(distances_to_receiver)

        # ---------------------------------------------------------------------
        # STORAGE
        # ---------------------------------------------------------------------

        time_points = []
        I2_concentration = []
        S2_free_concentration = []
        S2_total_concentration = []

        # ---------------------------------------------------------------------
        # TIME LOOP
        # ---------------------------------------------------------------------

        for step in range(n_steps + 1):

            current_time = step * dt

            # Save before solving step
            if step % save_interval_steps == 0:
                time_points.append(current_time / 3600.0)

                I2_concentration.append(float(I2.value[receiver_center_idx]))
                S2_free_concentration.append(float(S2.value[receiver_center_idx]))

                S2_total = (
                    float(S2.value[receiver_center_idx])
                    + float(S2_I2.value[receiver_center_idx])
                    + float(S2_Th2.value[receiver_center_idx])
                )
            if step % (10 * save_interval_steps) == 0:
                print(
                    f"{SWEEP_PARAMETER}={param_value}, "
                    f"t={step * dt / 3600:.2f} hr, "
                    f"S2 max={float(np.max(S2.value)) * 1000:.3f} nM, "
                    f"I2 receiver={float(I2[receiver_center_idx]) * 1000:.3f} nM, "
                    f"S2 receiver={float(S2[receiver_center_idx]) * 1000:.3f} nM"
    )
                S2_total_concentration.append(S2_total)

            if step == n_steps:
                break

            S2.updateOld()
            I2.updateOld()
            Th2.updateOld()
            S2_I2.updateOld()
            S2_Th2.updateOld()

            res = 1e10
            sweep = 0
            max_sweeps = 10

            while res > 1e-6 and sweep < max_sweeps:
                res = eq.sweep(dt=dt)
                sweep += 1

        # ---------------------------------------------------------------------
        # FINAL OUTPUTS
        # ---------------------------------------------------------------------

        I2_final = I2_concentration[-1]
        S2_final = S2_free_concentration[-1]
        S2_total_final = S2_total_concentration[-1]

        time_array = np.array(time_points)
        I2_array = np.array(I2_concentration)

        half_time = calculate_half_time(time_array, I2_array, I2_init)

        wall_time = time.perf_counter() - start_time

        print(
            f"✓ {SWEEP_PARAMETER}={param_value:.1f}, Rep={replicate_number}: "
            f"I2_final={I2_final * 1000:.2f} nM, "
            f"S2_total_final={S2_total_final * 1000:.2f} nM, "
            f"wall={wall_time:.1f} s"
        )

        return {
            "param_value": param_value,
            "replicate_id": replicate_number,
            "I2_final": I2_final,
            "S2_final": S2_final,
            "S2_total_final": S2_total_final,
            "half_time": half_time,
            "wall_time": wall_time,
            "success": True,
        }

    except Exception as e:
        print(f"✗ ERROR: {SWEEP_PARAMETER}={param_value}, Rep={replicate_number}: {e}")

        import traceback
        traceback.print_exc()

        return {
            "param_value": param_value,
            "replicate_id": replicate_number,
            "I2_final": np.nan,
            "S2_final": np.nan,
            "S2_total_final": np.nan,
            "half_time": np.nan,
            "wall_time": np.nan,
            "success": False,
        }


# =============================================================================
# RUN PARAMETER SWEEP
# =============================================================================

def run_parameter_sweep():
    print("\n" + "=" * 80)
    print("STAGE 2: RUNNING PARAMETER SWEEP")
    print("=" * 80)

    tasks = []

    for param_value in SWEEP_VALUES:
        for rep in range(N_REPLICATES):
            tasks.append((param_value, rep))

    n_processes = N_PROCESSES if N_PROCESSES else cpu_count()

    print(f"Using {n_processes} worker processes")
    print(f"Total simulations: {len(tasks)}")
    print("Gmsh is NOT used during multiprocessing.")
    print("=" * 80 + "\n")

    start = time.time()

    results = []
    for task in tasks:
        results.append(run_single_simulation(*task))

    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE: {(time.time() - start) / 60:.2f} min")
    print("=" * 80)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results):
    df = pd.DataFrame(results)

    df.to_csv("sweep_raw_results_sequential.csv", index=False)

    successful = df[df["success"] == True]

    stats = successful.groupby("param_value").agg(
        I2_final_mean=("I2_final", "mean"),
        I2_final_std=("I2_final", "std"),
        S2_final_mean=("S2_final", "mean"),
        S2_final_std=("S2_final", "std"),
        S2_total_final_mean=("S2_total_final", "mean"),
        S2_total_final_std=("S2_total_final", "std"),
        half_time_mean=("half_time", "mean"),
        half_time_std=("half_time", "std"),
        wall_time_mean=("wall_time", "mean"),
        wall_time_std=("wall_time", "std"),
    ).reset_index()

    stats.to_csv("sweep_summary_stats_sequential_debug.csv", index=False)

    print("\nSummary:")
    print(stats)

    return df, stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       PARAMETER SWEEP WITH PRE-GENERATED GMSH MESHES         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Stage 1: generate all meshes sequentially
    generate_all_meshes_sequentially()

    # Optional but recommended
    verify_meshes()

    # Stage 2: run solver in parallel
    results = run_parameter_sweep()

    # Analyze
    df, stats = analyze_results(results)

    print("\nDone.\n")