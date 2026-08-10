"""
Parameter sweep over sender-receiver center-to-center distance ("distance_between").

HOW TO USE
----------
1. Put this file in the SAME folder as `Functions.py` (the import block below
   tries a couple of reasonable locations, but the safest thing is to just put
   it next to Functions.py).
2. Edit DISTANCES_TO_SWEEP near the top with the distances (in um) you want to run.
3. (Optional) set PREVIEW_ONLY = True and run once to sanity-check the new
   "stepped" mesh sizing before committing to a long sweep. See PREVIEW section
   at the bottom.
4. Run: python sweep_center_center_distance.py

WHAT THIS DOES
--------------
For each distance in DISTANCES_TO_SWEEP:
  - builds a fresh triangular Gmsh mesh (unique filename, fresh gmsh session)
  - recomputes cell centers + receiver index from THAT mesh (never reuses
    anything from the previous distance)
  - runs the reaction-diffusion simulation until steady state (or a
    per-distance time cap, see min_ss_check_time_hours below)
  - saves a downsampled timeseries CSV (I2, S2_free, S2_total at receiver)
  - records the final concentration of every species at the receiver

All outputs go into ./Center_center_distance_triangular_mesh_info/:
  - timeseries_ccd=<dist>_triangular_mesh_v3.csv          (one per distance)
  - final_concentrations_vs_distance_v3.csv               (one row per distance)
  - final_I2_vs_distance_v3.png                           (summary plot)
  - mesh_files/mesh_ccd=<dist>_v3.msh                     (kept for later use)
  - sweep_log_v3.txt                                      (progress / errors)

IMPORTANT GOTCHA CARRIED OVER FROM Functions.py
------------------------------------------------
`smooth_circular_profile` and `intialize_equations` in Functions.py reference
module-level globals (node_radius, k_p, k_slow, k_fast, k_d_ss, k_d_ds, ...)
directly rather than as function parameters. That means those values always
come from Functions.py itself, no matter what you set in this script. Since
this sweep uses the same node_diameter (75 um) as Functions.py, this is not a
problem today -- but if you ever change node size or rate constants, you must
change them in Functions.py, not here.

WHY YOU WERE SEEING "NO S2 AT RECEIVER" BEFORE (short version)
----------------------------------------------------------------
Almost certainly one or both of:
  1. The mesh/index from a previous distance was reused (stale receiver
     index into a NEW mesh with a different cell ordering/count) -> garbage.
     Fixed here by rebuilding mesh + receiver index fresh every iteration.
  2. The steady-state check looks at the *relative change across the whole
     domain*, not specifically at the receiver. The region near the sender
     can lock into its own local balance quickly, dragging the *global* max
     relative-change below tolerance while the receiver is still sitting at
     its untouched initial value (nothing has diffused there yet) -- that
     looks numerically "converged" but isn't. Fixed here by refusing to
     even start checking for steady state until a minimum simulated time
     has passed, scaled with distance^2 (diffusion is a d^2/D process).
     See min_ss_check_time_hours() -- it's calibrated off the one data point
     you gave me (800 um ~ 3 hr), so treat it as a starting guess and adjust
     ref_time_hr/ref_distance_um if a run's timeseries CSV shows I2 was
     still visibly moving when it stopped.
"""

import matplotlib
matplotlib.use("Agg")  # headless: never blocks on plt.show() during a long sweep

import os
import sys
import time as timer
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmsh
from fipy import Gmsh2D
from fipy.tools import numerix

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from Functions_and_system.Functions import intialize_equations, initalize_variables, initalize_variables_speedup



# -----------------------------------------------------------------------------
# IMPORT Functions.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent



# =============================================================================
# EDIT ME: distances to sweep (um, center-to-center)
# =============================================================================
DISTANCES_TO_SWEEP = [1200]
    # fill these in yourself, e.g. 300, 600, 900, 1200, ...

PREVIEW_ONLY = False  # set True to just build+report the mesh for the first
                       # distance in the list (no simulation) so you can sanity
                       # check the stepped mesh sizing before a long sweep.

USE_STEPPED_MESH = True  # False = fall back to the original smooth/Sigmoid
                          # Threshold sizing (growth_rate has no effect then,
                          # same behavior as your original New_simple_mesh.py)

# =============================================================================
# FIXED SIMULATION PARAMETERS (mirrors TG_Rmesh_tanh.py)
# =============================================================================
node_diameter = 75.0
node_radius = node_diameter / 2.0          # must match Functions.py's global
total_width = 1e4                           # um
total_height = 1e3                          # um

fine_dx = 5.0          # finest cell size at node surface (um)
coarse_dx_cap = 100.0  # hard cap on coarsest cell size, per your instruction
growth_rate = 1.5      # mesh grows by this factor every `cells_per_level` cells
cells_per_level = 8    # how many cells to hold at one size before growing

dt = 60.0
save_interval_steps = 50  # downsample: save every 50 steps

check_steady_state = True
ss_tolerance = 1e-8
ss_window = 50
check_interval = save_interval_steps  # check SS on the same cadence as saving
verbose = True

# --- steady-state safety floor -----------------------------------------------
REF_DISTANCE_UM = 800.0
REF_TIME_HR = 3.0
MIN_CHECK_FLOOR_HR = 0.5


def min_ss_check_time_hours(distance_um):
    return max(MIN_CHECK_FLOOR_HR, REF_TIME_HR * (distance_um / REF_DISTANCE_UM) ** 2)


def max_total_time_hours(distance_um):
    # generous cap so the loop always terminates even if SS is never detected;
    # steady state should kick in well before this in practice
    return min(48.0, max(8.0, 5.0 * min_ss_check_time_hours(distance_um)))


# =============================================================================
# OUTPUT FOLDERS
# =============================================================================
OUT_DIR = SCRIPT_DIR / "Center_center_distance_triangular_mesh_info"
MESH_DIR = OUT_DIR / "mesh_files"
OUT_DIR.mkdir(exist_ok=True)
MESH_DIR.mkdir(exist_ok=True)
LOG_PATH = OUT_DIR / "sweep_log_v3.txt"


def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# =============================================================================
# MESH GENERATION
# =============================================================================
def create_gmsh_stepped_mesh(
    bath_width, bath_height, node_diameter, distance_between_nodes,
    min_cell_size, max_cell_size, growth_rate, cells_per_level,
    mesh_filename, verbose=True,
):
    """
    Triangular Gmsh mesh, fine only near the two circular nodes, growing in
    discrete steps (geometric "rings") rather than one smooth sigmoid ramp.

    Ring i has a constant cell size size_i = min_cell_size * growth_rate**i,
    held for roughly `cells_per_level` cells' worth of radial distance before
    jumping to size_{i+1}. The last ring is clamped to max_cell_size and
    extends to the domain edge.

    Implementation note: this uses only Gmsh's "Threshold" and "Min" field
    types (already used in your original mesh code), combined so that at any
    point the field value is whichever ring's plateau applies there. See the
    long comment in the sweep script docstring for the reasoning.
    """
    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("stepped_radial_mesh")

        gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
        sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0,
                                                     node_radius, node_radius)
        receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                       node_radius, node_radius)
        gmsh.model.occ.synchronize()

        sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)],
                                                   oriented=False, combined=False, recursive=False)
        receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)],
                                                     oriented=False, combined=False, recursive=False)
        sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
        receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]

        sender_center_point = gmsh.model.occ.addPoint(sender_center_x, y_center, 0)
        receiver_center_point = gmsh.model.occ.addPoint(receiver_center_x, y_center, 0)
        gmsh.model.occ.synchronize()

        dist_sender_bnd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_sender_bnd, "CurvesList", sender_curve_tags)
        gmsh.model.mesh.field.setNumber(dist_sender_bnd, "Sampling", 200)

        dist_receiver_bnd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_receiver_bnd, "CurvesList", receiver_curve_tags)
        gmsh.model.mesh.field.setNumber(dist_receiver_bnd, "Sampling", 200)

        dist_sender_ctr = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_sender_ctr, "PointsList", [sender_center_point])
        gmsh.model.mesh.field.setNumber(dist_sender_ctr, "Sampling", 200)

        dist_receiver_ctr = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_receiver_ctr, "PointsList", [receiver_center_point])
        gmsh.model.mesh.field.setNumber(dist_receiver_ctr, "Sampling", 200)

        min_distance_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            min_distance_field, "FieldsList",
            [dist_sender_bnd, dist_receiver_bnd, dist_sender_ctr, dist_receiver_ctr],
        )

        # --- build the geometric ring schedule ---------------------------------
        levels = [min_cell_size]
        while levels[-1] < max_cell_size:
            levels.append(min(levels[-1] * growth_rate, max_cell_size))

        boundaries = [0.0]
        for size in levels[:-1]:
            boundaries.append(boundaries[-1] + cells_per_level * size)

        if verbose:
            print(f"\nStepped mesh: {len(levels)} levels, growth={growth_rate}, "
                  f"~{cells_per_level} cells held per level")
            for i, size in enumerate(levels):
                lo = boundaries[i]
                hi = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
                print(f"  level {i}: {size:.3f} um, valid for distance in [{lo:.1f}, "
                      f"{hi if hi != float('inf') else 'inf'}) um")

        threshold_ids = []
        n_levels = len(levels)
        for i, size in enumerate(levels):
            tid = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tid, "InField", min_distance_field)
            if i < n_levels - 1:
                trans = boundaries[i + 1]
                eps = max(0.02, 0.02 * trans)
                gmsh.model.mesh.field.setNumber(tid, "SizeMin", size)
                gmsh.model.mesh.field.setNumber(tid, "SizeMax", 1e6)
                gmsh.model.mesh.field.setNumber(tid, "DistMin", max(trans - eps, 0.0))
                gmsh.model.mesh.field.setNumber(tid, "DistMax", trans + eps)
            else:
                # coarsest level: constant size everywhere (acts as the far-field
                # fallback so the Min-combination never exceeds max_cell_size)
                gmsh.model.mesh.field.setNumber(tid, "SizeMin", size)
                gmsh.model.mesh.field.setNumber(tid, "SizeMax", size)
                gmsh.model.mesh.field.setNumber(tid, "DistMin", boundaries[i])
                gmsh.model.mesh.field.setNumber(tid, "DistMax", boundaries[i])
            gmsh.model.mesh.field.setNumber(tid, "Sigmoid", 0)
            threshold_ids.append(tid)

        combined_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined_field, "FieldsList", threshold_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(mesh_filename)
    finally:
        gmsh.finalize()

    return mesh_filename, sender_center_x, receiver_center_x, y_center


def create_gmsh_smooth_mesh(
    bath_width, bath_height, node_diameter, distance_between_nodes,
    min_cell_size, max_cell_size, mesh_filename, refinement_radius=200.0,
    verbose=True,
):
    """Fallback: original smooth/Sigmoid Threshold sizing (growth_rate unused),
    kept in case the stepped mesh above ever needs to be swapped out."""
    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("smooth_radial_mesh")

        gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
        sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0,
                                                     node_radius, node_radius)
        receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                       node_radius, node_radius)
        gmsh.model.occ.synchronize()

        sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)],
                                                   oriented=False, combined=False, recursive=False)
        receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)],
                                                     oriented=False, combined=False, recursive=False)
        sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
        receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]

        sender_center_point = gmsh.model.occ.addPoint(sender_center_x, y_center, 0)
        receiver_center_point = gmsh.model.occ.addPoint(receiver_center_x, y_center, 0)
        gmsh.model.occ.synchronize()

        dist_sender_bnd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_sender_bnd, "CurvesList", sender_curve_tags)
        gmsh.model.mesh.field.setNumber(dist_sender_bnd, "Sampling", 200)

        dist_receiver_bnd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_receiver_bnd, "CurvesList", receiver_curve_tags)
        gmsh.model.mesh.field.setNumber(dist_receiver_bnd, "Sampling", 200)

        dist_sender_ctr = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_sender_ctr, "PointsList", [sender_center_point])
        gmsh.model.mesh.field.setNumber(dist_sender_ctr, "Sampling", 200)

        dist_receiver_ctr = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_receiver_ctr, "PointsList", [receiver_center_point])
        gmsh.model.mesh.field.setNumber(dist_receiver_ctr, "Sampling", 200)

        min_distance_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            min_distance_field, "FieldsList",
            [dist_sender_bnd, dist_receiver_bnd, dist_sender_ctr, dist_receiver_ctr],
        )

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", min_distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_cell_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_cell_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_radius)
        gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 1)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(mesh_filename)
    finally:
        gmsh.finalize()

    return mesh_filename, sender_center_x, receiver_center_x, y_center


# =============================================================================
# ONE SIMULATION AT A GIVEN DISTANCE
# =============================================================================
def run_single_distance(distance_between, run_index, n_total):
    log(f"\n{'='*70}\n[{run_index}/{n_total}] distance_between = {distance_between:.1f} um\n{'='*70}")

    if distance_between > 0.6 * total_width:
        log(f"  WARNING: distance ({distance_between:.0f} um) is more than 60% of "
            f"total_width ({total_width:.0f} um) -- receiver may sit too close to "
            f"the domain boundary. Continuing anyway.")

    wall_start = timer.time()

    mesh_filename = str(MESH_DIR / f"mesh_ccd={distance_between:.0f}_v3.msh")

    if USE_STEPPED_MESH:
        mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_gmsh_stepped_mesh(
            bath_width=total_width,
            bath_height=total_height,
            node_diameter=node_diameter,
            distance_between_nodes=distance_between,
            min_cell_size=fine_dx,
            max_cell_size=coarse_dx_cap,
            growth_rate=growth_rate,
            cells_per_level=cells_per_level,
            mesh_filename=mesh_filename,
            verbose=verbose,
        )
    else:
        mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_gmsh_smooth_mesh(
            bath_width=total_width,
            bath_height=total_height,
            node_diameter=node_diameter,
            distance_between_nodes=distance_between,
            min_cell_size=fine_dx,
            max_cell_size=coarse_dx_cap,
            mesh_filename=mesh_filename,
            verbose=verbose,
        )

    # fresh mesh object + fresh cell centers + fresh receiver index, every time
    mesh = Gmsh2D(mesh_filename)
    receiver_center_y = sender_center_y
    x, y = mesh.cellCenters

    log(f"  mesh: {mesh.numberOfCells} cells | sender=({sender_center_x:.0f},{sender_center_y:.0f}) "
        f"receiver=({receiver_center_x:.0f},{receiver_center_y:.0f})")

    S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(
        mesh, x, y, sender_center_x, receiver_center_x, receiver_center_y,
        node_radius, I2_init=0.1, Th2_init=5.0, I1O2_init=0.1,
        D_gel=60.0, D_solution=150.0,
    )
    eq = intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2)

    distances_to_receiver = numerix.sqrt((x - receiver_center_x) ** 2 +
                                          (y - receiver_center_y) ** 2)
    receiver_center_idx = numerix.argmin(distances_to_receiver)

    total_time = max_total_time_hours(distance_between) * 3600.0
    n_steps = int(total_time / dt)
    min_check_steps = int((min_ss_check_time_hours(distance_between) * 3600.0) / dt)

    time_points, I2_conc, S2_free_conc, S2_total_conc = [], [], [], []
    recent_changes = []
    converged_to_ss = False
    current_time = 0.0

    for step in range(n_steps):
        S2.updateOld(); I2.updateOld(); Th2.updateOld(); S2_I2.updateOld(); S2_Th2.updateOld()
        S2_old_vals = S2.value.copy()
        I2_old_vals = I2.value.copy()
        Th2_old_vals = Th2.value.copy()
        S2_I2_old_vals = S2_I2.value.copy()
        S2_Th2_old_vals = S2_Th2.value.copy()

        res, sweep = 1e10, 0
        while res > 1e-6 and sweep < 10:
            res = eq.sweep(dt=dt)
            sweep += 1

        if step % save_interval_steps == 0:
            current_time = step * dt
            time_points.append(current_time / 3600.0)

            I2_val = I2.value[receiver_center_idx]
            S2_free_val = S2.value[receiver_center_idx]
            S2_total_val = (S2.value[receiver_center_idx] +
                             S2_I2.value[receiver_center_idx] +
                             S2_Th2.value[receiver_center_idx])
            I2_conc.append(I2_val)
            S2_free_conc.append(S2_free_val)
            S2_total_conc.append(S2_total_val)

            if check_steady_state and step >= min_check_steps and step % check_interval == 0:
                epsilon = 1e-10
                changes = [
                    np.max(np.abs(S2.value - S2_old_vals) / (np.abs(S2.value) + epsilon)),
                    np.max(np.abs(I2.value - I2_old_vals) / (np.abs(I2.value) + epsilon)),
                    np.max(np.abs(Th2.value - Th2_old_vals) / (np.abs(Th2.value) + epsilon)),
                    np.max(np.abs(S2_I2.value - S2_I2_old_vals) / (np.abs(S2_I2.value) + epsilon)),
                    np.max(np.abs(S2_Th2.value - S2_Th2_old_vals) / (np.abs(S2_Th2.value) + epsilon)),
                ]
                max_change = np.max(changes)
                recent_changes.append(max_change)
                if len(recent_changes) > ss_window:
                    recent_changes.pop(0)
                if len(recent_changes) >= ss_window and all(c < ss_tolerance for c in recent_changes):
                    converged_to_ss = True
                    log(f"  STEADY STATE at t={current_time/3600:.2f} hr "
                        f"(max change {max_change:.2e} < {ss_tolerance:.2e})")
                    break

            if step % (save_interval_steps * 10) == 0:
                log(f"  t={current_time/3600:.2f} hr: I2={I2_val*1000:.2f} nM, "
                    f"S2_total={S2_total_val*1000:.2f} nM")

    wall_time = timer.time() - wall_start
    log(f"  done in {wall_time:.1f} s wall time, converged_to_ss={converged_to_ss}, "
        f"final sim time={current_time/3600:.2f} hr")

    # --- per-distance timeseries CSV -----------------------------------------
    df = pd.DataFrame({
        "Time (hours)": time_points,
        "I2 (nM)": np.array(I2_conc) * 1000,
        "S2_free (nM)": np.array(S2_free_conc) * 1000,
        "S2_total (nM)": np.array(S2_total_conc) * 1000,
    })
    ts_path = OUT_DIR / f"timeseries_ccd={distance_between:.0f}_triangular_mesh_v3.csv"
    df.to_csv(ts_path, index=False)

    return {
        "distance_um": distance_between,
        "S2_free_nM": S2.value[receiver_center_idx] * 1000,
        "I2_nM": I2.value[receiver_center_idx] * 1000,
        "Th2_nM": Th2.value[receiver_center_idx] * 1000,
        "S2_I2_nM": S2_I2.value[receiver_center_idx] * 1000,
        "S2_Th2_nM": S2_Th2.value[receiver_center_idx] * 1000,
        "S2_total_nM": (S2.value[receiver_center_idx] + S2_I2.value[receiver_center_idx]
                         + S2_Th2.value[receiver_center_idx]) * 1000,
        "converged_to_ss": converged_to_ss,
        "final_sim_time_hr": current_time / 3600.0,
        "mesh_cells": mesh.numberOfCells,
        "wall_time_s": wall_time,
    }


# =============================================================================
# PREVIEW MODE
# =============================================================================
def preview_mesh_only(distance_between):
    mesh_filename = str(MESH_DIR / f"preview_ccd={distance_between:.0f}_v3.msh")
    if USE_STEPPED_MESH:
        mesh_filename, sx, rx, yc = create_gmsh_stepped_mesh(
            bath_width=total_width, bath_height=total_height,
            node_diameter=node_diameter, distance_between_nodes=distance_between,
            min_cell_size=fine_dx, max_cell_size=coarse_dx_cap,
            growth_rate=growth_rate, cells_per_level=cells_per_level,
            mesh_filename=mesh_filename, verbose=True,
        )
    else:
        mesh_filename, sx, rx, yc = create_gmsh_smooth_mesh(
            bath_width=total_width, bath_height=total_height,
            node_diameter=node_diameter, distance_between_nodes=distance_between,
            min_cell_size=fine_dx, max_cell_size=coarse_dx_cap,
            mesh_filename=mesh_filename, verbose=True,
        )
    mesh = Gmsh2D(mesh_filename)
    print(f"\nPREVIEW: distance={distance_between} um -> {mesh.numberOfCells} cells")
    print(f"Mesh file kept at: {mesh_filename}")
    print("Open this .msh in the Gmsh GUI to inspect ring sizes visually before "
          "running the full sweep.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if not DISTANCES_TO_SWEEP:
        raise ValueError(
            "DISTANCES_TO_SWEEP is empty -- fill it in near the top of this file "
            "before running."
        )

    if PREVIEW_ONLY:
        preview_mesh_only(DISTANCES_TO_SWEEP[0])
        sys.exit(0)

    results = []
    failures = []
    n_total = len(DISTANCES_TO_SWEEP)

    for i, d in enumerate(DISTANCES_TO_SWEEP, start=1):
        try:
            results.append(run_single_distance(d, i, n_total))
        except Exception as e:
            log(f"  !! FAILED at distance={d}: {e}")
            log(traceback.format_exc())
            failures.append((d, str(e)))
            continue

    if not results:
        log("\nNo distances completed successfully -- nothing to save/plot.")
        sys.exit(1)

    final_df = pd.DataFrame(results).sort_values("distance_um").reset_index(drop=True)
    final_path = OUT_DIR / "final_concentrations_vs_distance_v3.csv"
    final_df.to_csv(final_path, index=False)
    log(f"\nSaved final concentrations table: {final_path}")

    if failures:
        log(f"\n{len(failures)} distance(s) failed and were skipped:")
        for d, err in failures:
            log(f"  distance={d}: {err}")

    # --- plot: final I2 vs distance, linear-linear ---------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(final_df["distance_um"], final_df["I2_nM"], "o-", color="tab:blue")
    ax.set_xlabel("Center-to-center distance (um)", fontsize=12)
    ax.set_ylabel("Final [I2] at receiver (nM)", fontsize=12)
    ax.set_title("Final I2 concentration vs. center-center distance", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUT_DIR / "final_I2_vs_distance_v3.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    log(f"Saved plot: {plot_path}")

    log("\nSweep complete.")