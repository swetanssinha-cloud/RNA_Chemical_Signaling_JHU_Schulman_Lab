"""
Parameter sweep using ONE fixed pre-generated Gmsh mesh.

Use this for NON-SPATIAL parameter sweeps:
    Th2_init, I2_init, I1O2_init, k_p, k_slow, k_fast,
    k_d_ss, k_d_ds, D_gel, D_solution

Workflow:
1. Generate one fixed mesh sequentially.
2. Verify FiPy can load it.
3. Run multiprocessing FiPy simulations.
4. Workers only READ the .msh file.
5. Gmsh is NOT used inside multiprocessing.
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

# Use this script only for NON-MESH parameters
SWEEP_PARAMETER = "Th2_init"

SWEEP_VALUES = [0.1,1,5] #setting three values just to test


#Th2 variance
#SWEEP_PARAMETER = "Th2_init"
#SWEEP_VALUES = [0.1,0.2,0.5,1,2,5] means: [I1O2] 1x, 2x, 5x, 10x, 20x, 50x

#Kd,ds variance
#SWEEP_PARAMETER = "k_d_ds"
#SWEEP_VALUES = [1,2,3,4,5] * 3e-4

#Kd,ss variance
#SWEEP_PARAMETER = "k_d_ss"
#SWEEP_VALUES = [1,2,3,4,5] * 3e-4

#Kp variance
#SWEEP_PARAMETER = "k_p"
#SWEEP_VALUES = [1,10,100,etc] * 0.02

TIMESERIES_DIR = Path(f"Parameter_sweep for {SWEEP_PARAMETER} with triangular mesh")
TIMESERIES_DIR.mkdir(exist_ok=True)

N_REPLICATES = 1

if len(SWEEP_VALUES) < 6:
    N_PROCESSES = len(SWEEP_VALUES)
else:
    N_PROCESSES = 5

N_PROCESSES = 3 # just for the testing


# =============================================================================
# DEFAULT MODEL PARAMETERS
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

    "node_size": 50.0,
    "node_diameter": 75.0,
    "bath_margin": 250.0,
    "distance_between": 200.0,

    "total_width": 1e4,
    "total_height": 1e3,

    "dt": 60.0,
    "total_time": 3 * 3600,
    "save_interval_time": 60.0,

    "fine_dx": 0.75,
    "coarse_dx": 100.0,

    "box_padding": 200.0,
    "mesh_transition_width": 100.0,
    "profile_transition_width_factor": 3.0,
}


# =============================================================================
# FIXED MESH SETTINGS
# =============================================================================

MESH_DIR = Path("pregenerated_meshes")
MESH_DIR.mkdir(exist_ok=True)

FIXED_MESH_PARAMS = {
    "distance_between": DEFAULT_PARAMS["distance_between"],
    "node_diameter": DEFAULT_PARAMS["node_diameter"],
    "total_width": DEFAULT_PARAMS["total_width"],
    "total_height": DEFAULT_PARAMS["total_height"],
    "fine_dx": DEFAULT_PARAMS["fine_dx"],
    "coarse_dx": DEFAULT_PARAMS["coarse_dx"],
}

FIXED_MESH_FILE = MESH_DIR / (
    f"fixed_mesh_"
    f"dist={FIXED_MESH_PARAMS['distance_between']:.1f}_"
    f"diam={FIXED_MESH_PARAMS['node_diameter']:.1f}_"
    f"fine={FIXED_MESH_PARAMS['fine_dx']:.2f}_"
    f"coarse={FIXED_MESH_PARAMS['coarse_dx']:.1f}.msh"
)

MESH_AFFECTING_PARAMETERS = {
    "distance_between",
    "node_diameter",
    "total_width",
    "total_height",
    "fine_dx",
    "coarse_dx",
    "box_padding",
    "mesh_transition_width",
}


# =============================================================================
# GENERATE ONE FIXED MESH
# =============================================================================

def generate_fixed_mesh_once():
    """
    Generate exactly one mesh before multiprocessing.
    Gmsh is used only here, serially.
    """

    if SWEEP_PARAMETER in MESH_AFFECTING_PARAMETERS:
        raise ValueError(
            f"\nERROR: {SWEEP_PARAMETER} affects the mesh.\n"
            f"Do NOT use this fixed-mesh script for {SWEEP_PARAMETER}.\n"
            f"Use the spatial pre-generated-mesh sweep script instead.\n"
        )

    print("\n" + "=" * 80)
    print("STAGE 1: GENERATING ONE FIXED MESH")
    print("=" * 80)

    if FIXED_MESH_FILE.exists():
        print(f"✓ Fixed mesh already exists:")
        print(f"  {FIXED_MESH_FILE}")
        print("=" * 80 + "\n")
        return

    print(f"Generating fixed mesh:")
    print(f"  {FIXED_MESH_FILE}")

    create_gmsh_radial_mesh(
        bath_width=FIXED_MESH_PARAMS["total_width"],
        bath_height=FIXED_MESH_PARAMS["total_height"],
        node_diameter=FIXED_MESH_PARAMS["node_diameter"],
        distance_between_nodes=FIXED_MESH_PARAMS["distance_between"],
        min_cell_size=FIXED_MESH_PARAMS["fine_dx"],
        max_cell_size=FIXED_MESH_PARAMS["coarse_dx"],
        growth_rate=1.5,
        mesh_filename=str(FIXED_MESH_FILE),
        verbose=True,
    )

    print("✓ Fixed mesh generated.")
    print("=" * 80 + "\n")


# =============================================================================
# VERIFY FIXED MESH
# =============================================================================

def verify_fixed_mesh():
    """
    Load the fixed mesh with FiPy before multiprocessing.
    """

    print("\n" + "=" * 80)
    print("VERIFYING FIXED MESH WITH FIPY")
    print("=" * 80)

    if not FIXED_MESH_FILE.exists():
        raise FileNotFoundError(f"Missing fixed mesh: {FIXED_MESH_FILE}")

    mesh = Gmsh2D(str(FIXED_MESH_FILE))

    print(f"✓ Mesh loaded successfully.")
    print(f"  File: {FIXED_MESH_FILE}")
    print(f"  Cells: {mesh.numberOfCells:,}")
    print("=" * 80 + "\n")


# =============================================================================
# HELPER: HALF TIME & SAVING 
# =============================================================================

def calculate_half_time(time_array, I2_array, I2_init):
    I2_final = I2_array[-1]
    I2_half = (I2_init + I2_final) / 2.0

    below_half = I2_array < I2_half

    if not np.any(below_half):
        return np.nan

    idx = np.argmax(below_half)

    if idx == 0:
        return time_array[0] / 3600.0

    t1, t2 = time_array[idx - 1], time_array[idx]
    y1, y2 = I2_array[idx - 1], I2_array[idx]

    if y2 == y1:
        return t2 / 3600.0

    t_half = t1 + (I2_half - y1) * (t2 - t1) / (y2 - y1)

    return t_half / 3600.0

def save_timeseries_csv(
    param_value,
    replicate_number,
    time_points,
    I2_concentration,
    S2_free_concentration,
    S2_total_concentration,
):
    """
    Save individual time-series CSV file for one simulation.
    """

    df_ts = pd.DataFrame({
        "time_seconds": time_points,
        "time_hours": np.array(time_points) / 3600.0,
        "I2_receiver_center_uM": I2_concentration,
        "I2_receiver_center_nM": np.array(I2_concentration) * 1000.0,
        "S2_free_receiver_center_uM": S2_free_concentration,
        "S2_free_receiver_center_nM": np.array(S2_free_concentration) * 1000.0,
        "S2_total_receiver_node_uM": S2_total_concentration,
        "S2_total_receiver_node_nM": np.array(S2_total_concentration) * 1000.0,
    })

    filename = TIMESERIES_DIR / (
        f"timeseries_{SWEEP_PARAMETER}={param_value}_rep={replicate_number}.csv"
    )

    df_ts.to_csv(filename, index=False)

    return filename
# =============================================================================
# SINGLE SIMULATION
# =============================================================================

def run_single_simulation(param_value, replicate_number):
    """
    Run one simulation.
    IMPORTANT:
    - No Gmsh mesh generation here.
    - Each worker only loads the same fixed .msh file using FiPy.
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

        k_p = params["k_p"]
        k_d_ds = params["k_d_ds"]
        k_d_ss = params["k_d_ss"]
        k_slow = params["k_slow"]
        k_fast = params["k_fast"]

        I1O2_init = params["I1O2_init"]
        I2_init = params["I2_init"]
        Th2_init = params["Th2_init"]

        dt = params["dt"]
        total_time = params["total_time"]
        save_interval_time = params["save_interval_time"]

        n_steps = int(total_time / dt)
        save_interval_steps = int(save_interval_time / dt)

        node_radius = node_diameter / 2.0

        # ---------------------------------------------------------------------
        # FIXED MESH LOAD ONLY
        # ---------------------------------------------------------------------
        mesh = Gmsh2D(str(FIXED_MESH_FILE))

        sender_center_x = total_width / 2.0 - distance_between / 2.0
        receiver_center_x = total_width / 2.0 + distance_between / 2.0
        receiver_center_y = total_height / 2.0

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

        # ---------------------------------------------------------------------
        # INITIALIZE EQUATIONS
        # ---------------------------------------------------------------------
        eq = intialize_equations(
            S2,
            D_S2,
            I1O2,
            I2,
            Th2,
            S2_I2,
            S2_Th2,
        )

        # ---------------------------------------------------------------------
        # RECEIVER CENTER INDEX
        # ---------------------------------------------------------------------
        distances_to_receiver = numerix.sqrt(
            (x - receiver_center_x) ** 2 + (y - receiver_center_y) ** 2
        )

        receiver_center_idx = numerix.argmin(distances_to_receiver)

        # ---------------------------------------------------------------------
        # STORAGE
        # ---------------------------------------------------------------------
        time_points = []
        I2_concentration = []
        S2_free_concentration = []
        S2_total_concentration = []



        # ---------------------------------------------------------------------
        # SOLVER LOOP
        # ---------------------------------------------------------------------
        for step in range(n_steps):
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

            if (step + 1) % save_interval_steps == 0:
                current_time = (step + 1) * dt

                I2_val = float(I2[receiver_center_idx])
                S2_free_val = float(S2[receiver_center_idx])

                S2_total_val = (S2.value[receiver_center_idx] + 
                               S2_I2.value[receiver_center_idx] + 
                               S2_Th2.value[receiver_center_idx])

                time_points.append(current_time)
                I2_concentration.append(I2_val)
                S2_free_concentration.append(S2_free_val)
                S2_total_concentration.append(float(S2_total_val))


        timeseries_file = save_timeseries_csv(
        param_value,
        replicate_number,
        time_points,
        I2_concentration,
        S2_free_concentration,
        S2_total_concentration)
        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------
        I2_final = I2_concentration[-1]
        S2_final = S2_free_concentration[-1]
        S2_total_final = S2_total_concentration[-1]

        time_array = np.array(time_points)
        I2_array = np.array(I2_concentration)

        half_time = calculate_half_time(time_array, I2_array, I2_init)

        wall_time = time.perf_counter() - start_time

        print(
            f"✓ {SWEEP_PARAMETER}={param_value:.3g}, Rep={replicate_number}: "
            f"I2_final={I2_final * 1000:.2f} nM, "
            f"S2_total_final={S2_total_final * 1000:.2f} nM, "
            f"t_half={half_time:.2f} hr, "
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
            "timeseries_file": str(timeseries_file),
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
            "timseries_file": None,
            "success": False,
        }


# =============================================================================
# RUN PARAMETER SWEEP
# =============================================================================

def run_parameter_sweep():
    print("\n" + "=" * 80)
    print("STAGE 2: RUNNING FIXED-MESH PARAMETER SWEEP")
    print("=" * 80)
    print(f"Sweeping parameter: {SWEEP_PARAMETER}")
    print(f"Values: {SWEEP_VALUES}")
    print(f"Replicates per value: {N_REPLICATES}")
    print(f"Fixed mesh file: {FIXED_MESH_FILE}")
    print("Gmsh is NOT used during multiprocessing.")
    print("=" * 80)

    tasks = []

    for param_value in SWEEP_VALUES:
        for rep in range(N_REPLICATES):
            tasks.append((param_value, rep))

    n_processes = N_PROCESSES if N_PROCESSES else cpu_count()

    print(f"Using {n_processes} worker processes")
    print(f"Total simulations: {len(tasks)}")
    print("=" * 80 + "\n")

    start = time.time()

    with Pool(processes=n_processes) as pool:
        results = pool.starmap(run_single_simulation, tasks)

    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE: {(time.time() - start) / 60:.2f} min")
    print("=" * 80)

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(results):
    df = pd.DataFrame(results)

    df.to_csv("fixed_mesh_sweep_raw_results.csv", index=False)

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

    stats.to_csv("fixed_mesh_sweep_summary_stats.csv", index=False)

    print("\nSummary:")
    print(stats)

    return df, stats


# =============================================================================
# OPTIONAL PLOTs
# =============================================================================

def plot_results(stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = stats["param_value"]

    axes[0].errorbar(
        x,
        stats["I2_final_mean"] * 1000,
        yerr=stats["I2_final_std"] * 1000,
        marker="o",
        capsize=4,
    )
    axes[0].set_xlabel(SWEEP_PARAMETER)
    axes[0].set_ylabel("Final I2 at receiver center (nM)")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        x,
        stats["S2_final_mean"] * 1000,
        yerr=stats["S2_final_std"] * 1000,
        marker="o",
        capsize=4,
    )
    axes[1].set_xlabel(SWEEP_PARAMETER)
    axes[1].set_ylabel("Final free S2 at receiver center (nM)")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(
        x,
        stats["S2_total_final_mean"] * 1000,
        yerr=stats["S2_total_final_std"] * 1000,
        marker="o",
        capsize=4,
    )
    axes[2].set_xlabel(SWEEP_PARAMETER)
    axes[2].set_ylabel("Final total S2 in receiver node (nM)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fixed_mesh_sweep_results.png", dpi=300)
    plt.show()


def plot_overlaid_timeseries():
    """
    Plot overlaid time-series curves for each parameter value.
    Uses the saved CSV files in TIMESERIES_DIR.
    """

    csv_files = sorted(TIMESERIES_DIR.glob("timeseries_*.csv"))

    if len(csv_files) == 0:
        print(f"No time-series CSV files found in {TIMESERIES_DIR}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(SWEEP_VALUES)))

    for color, param_value in zip(colors, SWEEP_VALUES):

        matching_files = list(
            TIMESERIES_DIR.glob(
                f"timeseries_{SWEEP_PARAMETER}={param_value}_rep=*.csv"
            )
        )

        if len(matching_files) == 0:
            print(f"No CSV found for {SWEEP_PARAMETER}={param_value}")
            continue

        # If there are replicates, plot each replicate lightly
        for file in matching_files:
            df = pd.read_csv(file)

            label = f"{SWEEP_PARAMETER}={param_value}"

            axes[0].plot(
                df["time_hours"],
                df["I2_receiver_center_nM"],
                color=color,
                linewidth=2,
                alpha=0.8,
                label=label,
            )

            axes[1].plot(
                df["time_hours"],
                df["S2_free_receiver_center_nM"],
                color=color,
                linewidth=2,
                alpha=0.8,
                label=label,
            )

            axes[2].plot(
                df["time_hours"],
                df["S2_total_receiver_node_nM"],
                color=color,
                linewidth=2,
                alpha=0.8,
                label=label,
            )

    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("I2 at receiver center (nM)")
    axes[0].set_title("I2 Time Series")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Free S2 at receiver center (nM)")
    axes[1].set_title("Free S2 Time Series")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (hours)")
    axes[2].set_ylabel("Total S2 in receiver node (nM)")
    axes[2].set_title("Total S2 Time Series")
    axes[2].grid(True, alpha=0.3)

    # Remove duplicate legend labels
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            fontsize=9,
            loc="best",
        )

    plt.tight_layout()

    save_path = TIMESERIES_DIR / "overlaid_timeseries_plots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"✓ Overlaid time-series plot saved to:")
    print(f"  {save_path}")

    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        FIXED-MESH NON-SPATIAL PARAMETER SWEEP                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Stage 1: generate one mesh sequentially
    generate_fixed_mesh_once()

    # Optional but recommended
    verify_fixed_mesh()

    # Stage 2: run FiPy solver in parallel
    results = run_parameter_sweep()

    # Analyze
    df, stats = analyze_results(results)

    # Plot
    plot_results(stats)
    plot_overlaid_timeseries()

    print("\nDone.\n")