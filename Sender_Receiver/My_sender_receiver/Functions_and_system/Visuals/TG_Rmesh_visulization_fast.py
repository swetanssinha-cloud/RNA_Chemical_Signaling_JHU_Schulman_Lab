"""
2D Tethered Genelet Model - With Adaptive Mesh Refinement
SPATIAL VISUALIZATION VARIANT of TG_Rmesh_fast.py

Why this file exists
---------------------
TG_Rmesh_fast.py only ever records the single mesh cell closest to the
receiver center (receiver_center_idx) at each save interval -- it never
keeps the full spatial field. This file runs the EXACT SAME physics, mesh,
and split-equation solver as TG_Rmesh_fast.py (parameters copied verbatim),
but additionally captures the full S2(x, y) field at the final simulated
time and reproduces the sender/receiver spatial-profile plot that used to
exist in TG_Rmesh_tanh.py (found in git history, commit 56d4fbe) and, in a
different form, in TG_model.py.

TG_model.py could do this trivially with imshow() because it used a
structured Grid2D mesh (S2.value.reshape((ny, nx))). TG_Rmesh_fast.py uses
an unstructured adaptive Gmsh2D triangular mesh instead, so there is no grid
to reshape into. Instead this file builds a Delaunay triangulation of the
mesh cell centers (matplotlib.tri) and uses tripcolor(shading='gouraud') for
a smooth heatmap, and a LinearTriInterpolator to sample exact line slices
through that triangulation -- the same triangulation-based approach already
used successfully in Radial_Visuals_ofNode/TG_Rmesh_radial_visuals.py.

This file does NOT modify TG_Rmesh_fast.py, TG_Rmesh_tanh.py, or any other
existing file -- it is a standalone script.

Produces:
  1. The combined spatial plot: full-domain S2 heatmap (smooth triangulated)
     + S2 profile along the line y = y_center between the two nodes.
  2. That same y = y_center line slice as its own standalone figure
     (x vs [S2], exact interpolation at y = y_center; domain origin (0,0)
     is the bottom-left corner, confirmed from Mesh/New_simple_mesh.py).
  3. A side-by-side comparison of two same-size boxes on a shared color
     scale: one centered on the receiver node, one centered at the mirrored
     point (sender_center_x - distance_between, y_center) -- i.e. the same
     displacement from the sender, but on the side with no absorbing node.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
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
# PARAMETERS (identical to TG_Rmesh_fast.py)
# =============================================================================
wall_start_time = timer.time()

D_solution = 150.0
D_gel = 60.0
k_p = 0.01
k_d_ds = 3e-4
k_d_ss = 3e-4
k_slow = 5e4 * 1e-6
k_fast = 1e6 * 1e-6

I1O2_init = 0.1
I2_init = 0.1
Th2_init = I2_init * 4

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

max_sweeps = 15
sweep_residual_target = 1e-8
sweep_plateau_tol = 1e-9

# ---- spatial-plot settings ---------------------------------------------------
# Comparison box half-width for plot 3 (receiver vs. mirrored point),
# following the "2x node_diameter" choice: box spans +/- box_half_width.
box_half_width = 2 * node_diameter

# =============================================================================
# 2D ADAPTIVE MESH SETUP (identical to TG_Rmesh_fast.py, distinct mesh file
# so this run never clobbers/races the .msh used by TG_Rmesh_fast.py)
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
    mesh_filename='radial_mesh_visualization_fast.msh',
    visualize_gmsh=False,
    verbose=True)

mesh = Gmsh2D(mesh_filename)
receiver_center_y = sender_center_y
y_center = sender_center_y

x, y = mesh.cellCenters
x = np.asarray(x)
y = np.asarray(y)

print(f"\n2D Simulation Setup TRIANGULAR MESH (split-equation solver):")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} um^2")
print(f"  Node diameter: {node_diameter} um")
print(f"  Distance: {distance_between} um (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()

# =============================================================================
# CELL VARIABLES (identical initial conditions to TG_Rmesh_fast.py)
# =============================================================================

S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(
    mesh, x, y, sender_center_x, receiver_center_x, receiver_center_y,
    node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution)

# =============================================================================
# ONLY S2 GETS A FIPY EQUATION (identical to TG_Rmesh_fast.py)
# =============================================================================

eq_S2 = (TransientTerm(var=S2) ==
         DiffusionTerm(coeff=D_S2, var=S2) +
         k_p * I1O2 +
         ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

s2_solver = LinearLUSolver(tolerance=1e-10)


def reaction_pair_step(S2_now, X_old, C_old, k_on, k_off, dt):
    """Closed-form backward-Euler step for one exchange pair (X <-> C).
    Identical to TG_Rmesh_fast.py -- see that file's docstring for the
    derivation.
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
# TIME STEPPING (identical to TG_Rmesh_fast.py)
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
        S2_guess = S2.value

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

final_time_hr = current_time / 3600

# =============================================================================
# SAVE RESULTS TO CSV (identical time series to TG_Rmesh_fast.py)
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

csv_filename = f'Fast_Visualization_ccd={distance_between}.csv'
df.to_csv(csv_filename, index=False)

# =============================================================================
# SAVE THE FULL SPATIAL FIELD (final snapshot only -- one row per mesh cell)
# =============================================================================

print("Saving full spatial field (final snapshot)...")

S2_free_nM = np.asarray(S2.value) * 1000.0
I2_nM = np.asarray(I2.value) * 1000.0
Th2_nM = np.asarray(Th2.value) * 1000.0
S2_I2_nM = np.asarray(S2_I2.value) * 1000.0
S2_Th2_nM = np.asarray(S2_Th2.value) * 1000.0
S2_total_nM = S2_free_nM + S2_I2_nM + S2_Th2_nM

spatial_df = pd.DataFrame({
    'x_um': x,
    'y_um': y,
    'S2_free_nM': S2_free_nM,
    'I2_nM': I2_nM,
    'Th2_nM': Th2_nM,
    'S2_I2_nM': S2_I2_nM,
    'S2_Th2_nM': S2_Th2_nM,
    'S2_total_nM': S2_total_nM,
})
spatial_csv_filename = f'Fast_Visualization_spatial_field_ccd={distance_between}_t={final_time_hr:.1f}hr.csv'
spatial_df.to_csv(spatial_csv_filename, index=False)
print(f"  {spatial_csv_filename}  ({len(spatial_df)} cells)")

# =============================================================================
# BUILD THE TRIANGULATION ONCE -- reused for all three plots below.
# Delaunay triangulation of the mesh cell centers (same approach as
# Radial_Visuals_ofNode/TG_Rmesh_radial_visuals.py's panel_heatmap).
# =============================================================================

triangulation = mtri.Triangulation(x, y)
S2_interp = mtri.LinearTriInterpolator(triangulation, S2_free_nM)

# =============================================================================
# PLOT 1: full-domain heatmap + S2 profile along y = y_center
# (reproduces the two-panel spatial plot format from TG_model.py /
#  the pre-deletion TG_Rmesh_tanh.py spatial section)
# =============================================================================

print("\nGenerating spatial plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Adaptive Mesh (split solver): S2 Spatial Distribution at t = {final_time_hr:.1f} hr, '
             f'Distance={distance_between:.0f}um',
             fontsize=15, fontweight='bold')

# --- Panel A: full-domain smooth heatmap ---
tpc = axes[0].tripcolor(triangulation, S2_free_nM, shading='gouraud',
                        cmap='viridis', vmin=0)
axes[0].set_xlabel('X position (um)', fontsize=12)
axes[0].set_ylabel('Y position (um)', fontsize=12)
axes[0].set_title(f'S2 Concentration (nM) at t = {final_time_hr:.1f} hr',
                  fontsize=13, fontweight='bold')
axes[0].set_aspect('equal')
axes[0].set_xlim(0, total_width)
axes[0].set_ylim(0, total_height)

sender_circle = Circle((sender_center_x, sender_center_y), node_radius,
                       fill=False, edgecolor='red', linewidth=2, label='Sender')
receiver_circle = Circle((receiver_center_x, receiver_center_y), node_radius,
                         fill=False, edgecolor='blue', linewidth=2, label='Receiver')
axes[0].add_patch(sender_circle)
axes[0].add_patch(receiver_circle)
axes[0].plot(sender_center_x, sender_center_y, 'r.', markersize=6)
axes[0].plot(receiver_center_x, receiver_center_y, 'b.', markersize=6)
axes[0].legend(fontsize=10, loc='upper right')

cbar1 = plt.colorbar(tpc, ax=axes[0])
cbar1.set_label('[S2] (nM)', fontsize=11)

# --- Panel B: S2 profile along the line y = y_center between the nodes ---
x_line = np.linspace(0, total_width, 2000)
S2_line = S2_interp(x_line, np.full_like(x_line, y_center))
S2_line = np.ma.filled(S2_line, np.nan)

axes[1].plot(x_line, S2_line, 'b-', linewidth=2)
axes[1].axvline(x=sender_center_x, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='Sender')
axes[1].axvline(x=receiver_center_x, color='blue', linestyle='--', linewidth=2,
                alpha=0.8, label='Receiver')
axes[1].set_xlabel('X position (um)', fontsize=12)
axes[1].set_ylabel('[S2] (nM)', fontsize=12)
axes[1].set_title(f'S2 Profile Along Line Between Nodes (y = {y_center:.0f} um)',
                  fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_ylim(bottom=0)
axes[1].set_xlim(0, total_width)

plt.tight_layout()
plt.savefig(f'Visualization_spatial_ccd={distance_between:.0f}.png',
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOT 2: standalone y = y_center slice (x vs [S2]).
# Domain origin (0, 0) is the bottom-left corner (confirmed in
# Mesh/New_simple_mesh.py: gmsh.model.occ.addRectangle(0, 0, 0, bath_width,
# bath_height)), so x=0 here really is the left edge of the bath.
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(x_line, S2_line, 'b-', linewidth=2)
ax2.axvline(x=sender_center_x, color='red', linestyle='--', linewidth=2,
           alpha=0.8, label='Sender')
ax2.axvline(x=receiver_center_x, color='blue', linestyle='--', linewidth=2,
           alpha=0.8, label='Receiver')
ax2.set_xlabel('X position (um)  [origin (0,0) = bottom-left corner]', fontsize=12)
ax2.set_ylabel('[S2] (nM)', fontsize=12)
ax2.set_title(f'S2 vs X at y = midpoint ({y_center:.0f} um), t = {final_time_hr:.1f} hr',
             fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim(bottom=0)
ax2.set_xlim(0, total_width)

plt.tight_layout()
plt.savefig(f'Visualization_yslice_ccd={distance_between:.0f}.png',
            dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOT 3: receiver box vs. mirrored box (sender_center_x - distance_between),
# same box size, shared color scale.
# =============================================================================

mirror_center_x = sender_center_x - distance_between
mirror_center_y = y_center

box_centers = [
    (receiver_center_x, receiver_center_y, 'Receiver', 'blue', True),
    (mirror_center_x, mirror_center_y, f'Mirrored point\n(sender - {distance_between:.0f} um)', 'gray', False),
]

# Shared color scale: max S2 across both box windows (falls back to global
# max if a window happens to contain no data, e.g. a degenerate box size).
box_vmax = 0.0
for cx, cy, _, _, _ in box_centers:
    xw = (x >= cx - box_half_width) & (x <= cx + box_half_width)
    yw = (y >= cy - box_half_width) & (y <= cy + box_half_width)
    in_box = xw & yw
    if in_box.any():
        box_vmax = max(box_vmax, S2_free_nM[in_box].max())
if box_vmax <= 0.0:
    box_vmax = S2_free_nM.max()

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 6))
fig3.suptitle(f'Receiver vs. Mirrored Point (same displacement, no node) at t = {final_time_hr:.1f} hr\n'
             f'Box size: +/-{box_half_width:.0f} um (2x node_diameter), shared color scale',
             fontsize=14, fontweight='bold')

for ax, (cx, cy, label, edgecolor, has_node) in zip(axes3, box_centers):
    tpc_box = ax.tripcolor(triangulation, S2_free_nM, shading='gouraud',
                           cmap='viridis', vmin=0, vmax=box_vmax)
    if has_node:
        ax.add_patch(Circle((cx, cy), node_radius, fill=False,
                            edgecolor=edgecolor, linewidth=2.5))
    else:
        # No real node here -- dashed outline just shows the same footprint
        # size as the receiver node, for visual reference.
        ax.add_patch(Circle((cx, cy), node_radius, fill=False,
                            edgecolor=edgecolor, linewidth=2, linestyle='--'))
    ax.plot(cx, cy, '.', color=edgecolor, markersize=8)
    ax.set_xlim(cx - box_half_width, cx + box_half_width)
    ax.set_ylim(cy - box_half_width, cy + box_half_width)
    ax.set_aspect('equal')
    ax.set_xlabel('X position (um)', fontsize=11)
    ax.set_ylabel('Y position (um)', fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')

cbar3 = plt.colorbar(tpc_box, ax=axes3, fraction=0.046, pad=0.04)
cbar3.set_label('[S2] (nM)', fontsize=11)

plt.savefig(f'Visualization_box_comparison_ccd={distance_between:.0f}.png',
            dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDone. Spatial field, CSVs, and all three plots saved "
      f"for ccd={distance_between:.0f} um at t = {final_time_hr:.1f} hr.")
