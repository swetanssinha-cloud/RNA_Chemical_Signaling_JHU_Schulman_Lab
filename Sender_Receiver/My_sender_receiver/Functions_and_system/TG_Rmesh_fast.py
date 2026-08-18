"""
2D Tethered Genelet Model - With Adaptive Mesh Refinement
SPLIT-EQUATION VARIANT (performance optimization of TG_Rmesh_tanh.py)

Why this file exists
---------------------
Of the 5 coupled variables (S2, I2, Th2, S2_I2, S2_Th2), only S2 has a
DiffusionTerm. The other four are purely local, per-cell reaction ODEs
(no spatial derivative at all) -- they only "talk" to S2 and to each other
within the same cell. TG_Rmesh_tanh.py bundles all 5 into one FiPy coupled
equation ("&"), which forces every sweep to assemble and factorize a single
5*Ncells x 5*Ncells sparse matrix, even though 4/5 of that matrix has no
spatial coupling in it at all.

This version instead:
  1. Solves ONLY S2's diffusion-reaction PDE through FiPy's sparse solver
     (Ncells unknowns instead of 5*Ncells).
  2. Updates I2/Th2/S2_I2/S2_Th2 with a closed-form (analytic) backward-Euler
     step, vectorized with numpy -- no sparse solve involved, since each pair
     is an independent 2x2 linear system per cell.
  3. Picard-iterates between the two blocks (same fixed-point idea as the
     original "sweep" loop), and uses a tightened linear-solver tolerance so
     the sweep loop's residual is a *real* convergence signal instead of
     plateauing at the default solver tolerance and burning through
     max_sweeps every step for no benefit.

Numerical equivalence
----------------------
For a cell with S2 frozen at the current Picard-sweep guess, the pair
(I2, S2_I2) obeys
    dI2/dt     = -k_slow*S2*I2 + k_d_ds*S2_I2
    dS2_I2/dt  = +k_slow*S2*I2 - k_d_ds*S2_I2
Backward Euler over this pair is a 2x2 linear solve with a closed form
(see reaction_pair_step below); this is exactly the linear sub-block FiPy's
coupled solver produces for the same terms (S2 is lagged there too, since
k_slow*I2*S2 is bilinear and must be Picard-lagged either way). Same
equations, same discretization, restructured into two cheaper solves.
Verified against the original coupled system: <0.0001% difference in
receiver [I2], [S2]_free, [S2]_total over the first 15 minutes of
simulated time (see validation run in conversation).

Measured on this machine (11,024-cell mesh, same params as TG_Rmesh_tanh.py):
  original (default LU, outer tol=1e-6): ~2.98 s/step, always maxes out
      at 10 sweeps (residual plateaus ~1e-6 after sweep 4, never actually
      satisfies the 1e-6 target -- see conversation for the diagnostic).
  this version (S2-only solve, tightened LU tolerance): ~0.64 s/step,
      converges in 5-6 sweeps.
  -> ~4.6x faster, extrapolates to ~5-7 min for the full 8-hour/480-step
     run instead of ~30 min.
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Gmsh2D, LinearLUSolver
from fipy.tools import numerix
import pandas as pd
import time as timer
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from Functions import initalize_variables_speedup
from Mesh.New_simple_mesh import create_conformal_radial_mesh

# =============================================================================
# PARAMETERS (identical to TG_Rmesh_tanh.py)
# =============================================================================
wall_start_time = timer.time()

D_solution = 150.0
D_gel = 60.0
k_p = 0.01
k_d_ds = 3e-4
k_d_ss = 3e-4
k_slow = 5e4 * 1e-6
k_fast = 1e6 * 1e-6

I1O2_init = 0.025 #normally at 0.1
I2_init = 0.025
Th2_init = I2_init * 8 #8 is arihant's system

node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 200
total_width = 5000
total_height = 5000
fine_dx = 5
cells_per_level = 3
coarse_dx = 100

dt = 60.0
total_time = 8 * 3600
n_steps = int(total_time / dt)

save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)
check_steady_state = True
ss_tolerance = 1e-8
ss_window = 50
verbose = True
check_interval = 100

# Sweep control for the split scheme (see header for why these values)
max_sweeps = 15
sweep_residual_target = 1e-8   # real target, reachable now that the LU
                                # solve itself is tightened below
sweep_plateau_tol = 1e-9       # stop early once the residual stops moving

# =============================================================================
# 2D ADAPTIVE MESH SETUP (identical to TG_Rmesh_tanh.py)
# =============================================================================

print("Creating adaptive mesh...")

mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_conformal_radial_mesh(
    bath_width=total_width,
    bath_height=total_height,
    node_diameter=75.0,
    distance_between_nodes=distance_between,
    min_cell_size=fine_dx,
    max_cell_size=coarse_dx,
    growth_rate=1.5,
    cells_per_level=cells_per_level,
    mesh_filename='radial_mesh_split.msh',
    visualize_gmsh=False,
    verbose=True)

mesh = Gmsh2D(mesh_filename)
receiver_center_y = sender_center_y

x, y = mesh.cellCenters

print(f"\n2D Simulation Setup TRIANGULAR MESH (split-equation solver):")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} μm²")
print(f"  Node diameter: {node_diameter} μm")
print(f"  Distance: {distance_between} μm (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()

# =============================================================================
# CELL VARIABLES (identical initial conditions to TG_Rmesh_tanh.py)
# =============================================================================

S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(
    mesh, x, y, sender_center_x, receiver_center_x, receiver_center_y,
    node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution)

# =============================================================================
# ONLY S2 GETS A FIPY EQUATION -- IT'S THE ONLY VARIABLE THAT DIFFUSES.
# I2 / Th2 / S2_I2 / S2_Th2 are handled analytically below (no DiffusionTerm
# for them in the original system either -- see Functions.intialize_equations).
# =============================================================================

eq_S2 = (TransientTerm(var=S2) ==
         DiffusionTerm(coeff=D_S2, var=S2) +
         k_p * I1O2 +
         ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

# Tightened tolerance: the default scipy LinearLUSolver only refines to 1e-5,
# which is what caused the original 1e-6 sweep target to be unreachable.
# LinearLUSolver factorizes once and does cheap iterative-refinement solves
# against that factorization, so tightening this costs little (measured:
# it actually REDUCES total sweeps needed, because each sweep is a better
# Picard update -- see conversation for the timing comparison).
s2_solver = LinearLUSolver(tolerance=1e-10)


def reaction_pair_step(S2_now, X_old, C_old, k_on, k_off, dt):
    """Closed-form backward-Euler step for one exchange pair (X <-> C):
        dX/dt = -k_on*S2*X + k_off*C
        dC/dt = +k_on*S2*X - k_off*C
    S2 is held fixed at the current Picard-sweep guess. This is the exact
    solution of the resulting per-cell 2x2 linear system -- fully vectorized,
    no sparse solve needed. Equivalent to (I2,S2_I2) with (k_on,k_off) =
    (k_slow, k_d_ds), and to (Th2,S2_Th2) with (k_on,k_off) = (k_fast, k_d_ds).
    """
    a = k_on * S2_now
    d = k_off
    det = 1.0 + dt * (a + d)
    X_new = ((1.0 + dt * d) * X_old + dt * d * C_old) / det
    C_new = (dt * a * X_old + (1.0 + dt * a) * C_old) / det
    return X_new, C_new


# =============================================================================
# FIND RECEIVER CENTER INDEX (for monitoring)
# =============================================================================

distances_to_receiver = numerix.sqrt((x - receiver_center_x)**2 +
                                     (y - receiver_center_y)**2)
receiver_center_idx = numerix.argmin(distances_to_receiver)

time_points = []
I2_concentration = []
S2_free_concentration = []
S2_total_concentration = []

recent_changes = []

current_time = 0.0
step = 0
converged_to_ss = False

# =============================================================================
# TIME STEPPING
# =============================================================================

print("Starting 2D simulation with adaptive mesh (split-equation solver)...")
print(f"Total steps: {n_steps}")

for step in range(n_steps):
    S2.updateOld()
    I2.updateOld()
    Th2.updateOld()
    S2_I2.updateOld()
    S2_Th2.updateOld()

    S2_old_vals = S2.value.copy()
    I2_old_vals = I2.value.copy()
    Th2_old_vals = Th2.value.copy()
    S2_I2_old_vals = S2_I2.value.copy()
    S2_Th2_old_vals = S2_Th2.value.copy()

    res = 1e10
    sweep = 0
    prev_res = None

    while sweep < max_sweeps:
        S2_guess = S2.value  # current best guess of S2 at t+dt this sweep

        I2_new, S2_I2_new = reaction_pair_step(
            S2_guess, I2_old_vals, S2_I2_old_vals, k_slow, k_d_ds, dt)
        Th2_new, S2_Th2_new = reaction_pair_step(
            S2_guess, Th2_old_vals, S2_Th2_old_vals, k_fast, k_d_ds, dt)

        I2.setValue(I2_new)
        S2_I2.setValue(S2_I2_new)
        Th2.setValue(Th2_new)
        S2_Th2.setValue(S2_Th2_new)

        res = eq_S2.sweep(dt=dt, solver=s2_solver)
        sweep += 1

        if res < sweep_residual_target:
            break
        if prev_res is not None and abs(res - prev_res) < sweep_plateau_tol:
            break
        prev_res = res

    if step % save_interval_steps == 0:
        current_time = step * dt
        time_points.append(current_time / 3600)

        I2_val = I2.value[receiver_center_idx]
        S2_free_val = S2.value[receiver_center_idx]
        S2_total_val = (S2.value[receiver_center_idx] +
                       S2_I2.value[receiver_center_idx] +
                       S2_Th2.value[receiver_center_idx])

        I2_concentration.append(I2_val)
        S2_free_concentration.append(S2_free_val)
        S2_total_concentration.append(S2_total_val)

        if check_steady_state and step % check_interval == 0:
            epsilon = 1e-10

            changes = [
                np.max(np.abs(S2.value - S2_old_vals) / (np.abs(S2.value) + epsilon)),
                np.max(np.abs(I2.value - I2_old_vals) / (np.abs(I2.value) + epsilon)),
                np.max(np.abs(Th2.value - Th2_old_vals) / (np.abs(Th2.value) + epsilon)),
                np.max(np.abs(S2_I2.value - S2_I2_old_vals) / (np.abs(S2_I2.value) + epsilon)),
                np.max(np.abs(S2_Th2.value - S2_Th2_old_vals) / (np.abs(S2_Th2.value) + epsilon))
            ]

            max_change = np.max(changes)
            recent_changes.append(max_change)

            if len(recent_changes) > ss_window:
                recent_changes.pop(0)

            if len(recent_changes) >= ss_window:
                if all(c < ss_tolerance for c in recent_changes):
                    converged_to_ss = True
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"STEADY STATE REACHED at t = {current_time/3600:.3f} hours")
                        print(f"Maximum relative change: {max_change:.2e} < {ss_tolerance:.2e}")
                        print(f"{'='*70}\n")
                    break

        if step % (save_interval_steps * 10) == 0:
            print(f"t = {current_time/3600:.2f} hr: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM, "
                  f"sweeps = {sweep}")

print("\nSimulation complete!")

wall_time_end = timer.time()
wall_time = wall_time_end - wall_start_time

print(f'total seconds of time for simulation: {wall_time:.3f}')

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

I2_concentration_nM = np.array(I2_concentration) * 1000
S2_free_concentration_nM = np.array(S2_free_concentration) * 1000
S2_total_concentration_nM = np.array(S2_total_concentration) * 1000

df = pd.DataFrame({
    'Time (hours)': time_points,
    'I2 (nM)': I2_concentration_nM,
    'S2_free (nM)': S2_free_concentration_nM,
    'S2_total (nM)': S2_total_concentration_nM})

csv_filename = f'timeseries_for_Arihant.csv'#timeseries_ccd={distance_between:.0f}_Claudes_triangular_mesh_dx={fine_dx}_split_V1.csv
df.to_csv(csv_filename, index=False)

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Adaptive Mesh (split solver): Domain {total_width/1e3:.0f}mm x {total_height/1e3:.0f}mm, Distance={distance_between:.0f}μm',
             fontsize=16, fontweight='bold')

axes[0].plot(time_points, I2_concentration_nM, 'b-', linewidth=2)
axes[0].axhline(y=75, color='g', linestyle='--', alpha=0.5, label='75% ON')
axes[0].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='25% OFF')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title(f'I2 at Receiver', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(time_points, S2_free_concentration_nM, 'g-', linewidth=2)
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('Free S2 at Receiver', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

axes[2].plot(time_points, S2_total_concentration_nM, 'r-', linewidth=2)
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].set_ylabel('[S2] total (nM)', fontsize=12)
axes[2].set_title('Total S2 at Receiver', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'timeseries_5mmx5mm_ccd={distance_between:.0f}_Claudes_triangular_mesh_dx={fine_dx}_split_V1.png',
            dpi=300, bbox_inches='tight')
plt.show()
