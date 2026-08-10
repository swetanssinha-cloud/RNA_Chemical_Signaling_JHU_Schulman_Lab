"""
2D Tethered Genelet Model - With Adaptive Mesh Refinement
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Gmsh2D
from fipy.tools import numerix
import csv
import pandas as pd
import time as timer
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from Functions import calculate_total_amount, smooth_circular_profile, intialize_equations, initalize_variables, initalize_variables_speedup
from Mesh.New_simple_mesh import create_gmsh_radial_mesh, create_conformal_radial_mesh

# =============================================================================
# PARAMETERS (same as before)
# =============================================================================
wall_start_time = timer.time()


D_solution = 150.0 
D_gel = 60.0
k_p = 0.01 #0.2 This was the value CHen 25 used #1/s
k_d_ds = 3e-4 #1/s
k_d_ss = 3e-4 #1/s
k_slow = 5e4 * 1e-6#1e5 * 1e-6 # 1/(Ms) * microMolar
k_fast = 1e6 * 1e-6 # 1/(Ms) * microMolar
 
I1O2_init = 0.1 #(in uM) - 100 nM
I2_init = 0.1 #(in uM) - 100 nM
Th2_init = I2_init * 4 #(in uM) - 


node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 300 # Test at large distance
total_width = 5000 #10000 μm = 1 cm
total_height = 5000 #1000 μm = 1 mm
fine_dx = 5 # Speed it up so use 5um for fine mesh
cells_per_level=3
coarse_dx = 100

dt = 60.0
total_time = 8 * 3600
n_steps = int(total_time / dt)

#just for testing of time: 

save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)
check_steady_state= True
ss_tolerance = 1e-8
ss_window = 50
verbose = True
check_interval = 100

# =============================================================================
# 2D ADAPTIVE MESH SETUP
# =============================================================================

print("Creating adaptive mesh...")

# Create the adaptive mesh
# You can adjust these parameters for mesh refinement:
# - fine_dx: mesh spacing in refined region (smaller = finer, but slower)
# - coarse_dx: mesh spacing in bulk region (larger = coarser, but faster)
# - box_padding: extra padding around nodes for refined region



'''Claude's program - apparently supposed to fix my worries. '''
mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_conformal_radial_mesh(
    bath_width=total_width,           # μm (1 cm)
    bath_height=total_height,           # μm (1 mm)
    node_diameter=75.0,           # μm
    distance_between_nodes=distance_between, # μm (center-to-center)
    min_cell_size=fine_dx,          # μm (finest mesh at node surface)
    max_cell_size=coarse_dx,          # μm (coarsest mesh far from nodes)
    growth_rate=1.5,             # How fast mesh grows with distance
    cells_per_level=cells_per_level,  # Number of cells per refinement level
    mesh_filename='radial_mesh.msh',
    visualize_gmsh=False,        # Show Gmsh GUI
    verbose=True)

mesh = Gmsh2D(mesh_filename)
receiver_center_y = sender_center_y  # Both at same Y position

# Get cell centers
x, y = mesh.cellCenters

print(f"\n2D Simulation Setup TRIANGULAR MESH:")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} μm²")
print(f"  Node diameter: {node_diameter} μm")
print(f"  Distance: {distance_between} μm (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()


# =============================================================================
# CELL VARIABLES WITH SMOOTH INITIAL CONDITIONS
# =============================================================================

S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(mesh, x,y, sender_center_x, 
                                                             receiver_center_x, receiver_center_y, node_radius, I2_init, Th2_init, 
                                                             I1O2_init, D_gel, D_solution)
#Want the values to be true false so this speeds it up

eq = intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2, k_p, k_slow, k_fast, k_d_ss, k_d_ds)

# =============================================================================
# FIND RECEIVER CENTER INDEX (for monitoring)
# =============================================================================

# Find cell closest to receiver center
distances_to_receiver = numerix.sqrt((x - receiver_center_x)**2 + 
                                     (y - receiver_center_y)**2)
receiver_center_idx = numerix.argmin(distances_to_receiver)

# Storage
time_points = []
I2_concentration = []
S2_free_concentration = []
S2_total_concentration = []

# For steady-state detection: store recent changes
recent_changes = []
    
current_time = 0.0
step = 0
converged_to_ss = False

# =============================================================================
# TIME STEPPING
# =============================================================================

print("Starting 2D simulation with adaptive mesh...")
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
    max_sweeps = 10
    
    while res > 1e-6 and sweep < max_sweeps:
        res = eq.sweep(dt=dt)
        sweep += 1
    
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
            # Calculate maximum relative change across all variables
            # Change = |C(t) - C(t-dt)| / (|C(t)| + epsilon)
            epsilon = 1e-10  # Prevent division by zero
            
            changes = [
                np.max(np.abs(S2.value - S2_old_vals) / (np.abs(S2.value) + epsilon)),
                np.max(np.abs(I2.value - I2_old_vals) / (np.abs(I2.value) + epsilon)),
                np.max(np.abs(Th2.value - Th2_old_vals) / (np.abs(Th2.value) + epsilon)),
                np.max(np.abs(S2_I2.value - S2_I2_old_vals) / (np.abs(S2_I2.value) + epsilon)),
                np.max(np.abs(S2_Th2.value - S2_Th2_old_vals) / (np.abs(S2_Th2.value) + epsilon))
            ]
            
            max_change = np.max(changes)
            recent_changes.append(max_change)
            
            # Keep only recent window
            if len(recent_changes) > ss_window:
                recent_changes.pop(0)
            
            # Check if all recent changes are below tolerance
            if len(recent_changes) >= ss_window:
                if all(c < ss_tolerance for c in recent_changes):
                    converged_to_ss = True
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"STEADY STATE REACHED at t = {current_time/3600:.3f} hours")
                        print(f"Maximum relative change: {max_change:.2e} < {ss_tolerance:.2e}")
                        print(f"{'='*70}\n")
                    break
        
        
        if step % (save_interval_steps * 10) == 0:  # Print less frequently
            print(f"t = {current_time/3600:.2f} hr: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM")
            

print("\nSimulation complete!")


wall_time_end = timer.time()
wall_time = wall_time_end - wall_start_time

print(f'total seconds of time for simulation: {wall_time:.3f}')





# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

# Convert to nM for saving
I2_concentration_nM = np.array(I2_concentration) * 1000
S2_free_concentration_nM = np.array(S2_free_concentration) * 1000
S2_total_concentration_nM = np.array(S2_total_concentration) * 1000

# Create DataFrame
df = pd.DataFrame({
    'Time (hours)': time_points,
    'I2 (nM)': I2_concentration_nM,
    'S2_free (nM)': S2_free_concentration_nM,
    'S2_total (nM)': S2_total_concentration_nM})

# Save to CSV
csv_filename = f'timeseries_5mmx5mm_ccd={distance_between:.0f}_Claudes_triangular_mesh_dx={fine_dx}_speedup_V4.csv'
df.to_csv(csv_filename, index=False)
# print(f"Time series data saved to '{csv_filename}'")

# =============================================================================
# PLOTTING
# =============================================================================

#Time series plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Adaptive Mesh: Domain {total_width/1e3:.0f}mm x {total_height/1e3:.0f}mm, Distance={distance_between:.0f}μm', 
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
plt.savefig(f'timeseries_5mmx5mm_ccd={distance_between:.0f}_Claudes_triangular_mesh_dx={fine_dx}_speedup_V4.png', 
            dpi=300, bbox_inches='tight')
plt.show()

