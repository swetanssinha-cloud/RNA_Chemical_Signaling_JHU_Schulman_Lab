"""
2D Tethered Genelet Model - Matches Chloe's COMSOL solver geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import csv
import pandas as pd

# =============================================================================
# PARAMETERS (same as before)
# =============================================================================

D_solution = 150.0 
D_gel = 60.0
k_p = 0.2 #1/s
k_d_ds = 3e-4 #1/s
k_d_ss = 3e-4 #1/s
k_slow = 1e5 * 1e-6 # 1/(Ms) * microMolar
k_fast = 1e6 * 1e-6 # 1/(Ms) * microMolar
 
I1O2_init = 0.1 #(in uM) - 100 nM
I2_init = 0.1 #(in uM) - 100 nM
Th2_init = 5.0 #(in uM) - 5000 nM

node_size = 50.0
node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 1500 # Test at large distance
total_width = 2500.0  # Larger domain for 2D
total_height = node_size + 2 * bath_margin

total_height = 1e3
total_width = 1e4 #first doing Chloe's System

dt = 30.0
total_time = 8 * 3600
n_steps = int(total_time / dt)
save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)

# =============================================================================
# 2D MESH SETUP
# =============================================================================

# Create 2D mesh
# nx = 250  # Resolution in x
# ny = 250  # Resolution in y
# dx = total_width / nx
# dy = total_height / ny

dx = 40
dy = dx

nx = total_width // dx
ny = total_height // dy

nx = int(total_width // dx)  # = 500 (integer)
ny = int(total_height // dy)  # = 50 (integer)

mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# Get cell centers
x, y = mesh.cellCenters

# Center of domain
center_x = total_width / 2
center_y = total_height / 2

# Sender node: square at left
sender_center_x = center_x - distance_between / 2
sender_center_y = center_y


sender_mask = ((x >= sender_center_x - node_size/2) & 
               (x <= sender_center_x + node_size/2) &
               (y >= sender_center_y - node_size/2) & 
               (y <= sender_center_y + node_size/2))

# Receiver node: square at right  
receiver_center_x = center_x + distance_between / 2
receiver_center_y = center_y

receiver_mask = ((x >= receiver_center_x - node_size/2) & 
                 (x <= receiver_center_x + node_size/2) &
                 (y >= receiver_center_y - node_size/2) & 
                 (y <= receiver_center_y + node_size/2))

sender_mask = (np.sqrt((x - sender_center_x)**2 + (y - sender_center_y)**2) <= node_radius)
receiver_mask = (np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2) <= node_radius)

gel_mask = sender_mask | receiver_mask

print(f"2D Simulation Setup:")
print(f"  Mesh: {nx} * {ny} = {nx*ny} cells")
print(f"  Domain: {total_width} * {total_height} μm²")
print(f"  Node diameter: {node_diameter} μm²")
print(f"  Distance: {distance_between} μm (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()

# =============================================================================
# CELL VARIABLES (same as 1D)
# =============================================================================

S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)

I2 = CellVariable(name="I2", mesh=mesh, value=0.0, hasOld=True)
I2.setValue(I2_init, where=receiver_mask)

Th2 = CellVariable(name="Th2", mesh=mesh, value=0.0, hasOld=True)
Th2.setValue(Th2_init, where=receiver_mask)

S2_I2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
S2_Th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

I1O2 = CellVariable(name="I1O2", mesh=mesh, value=0.0)
I1O2.setValue(I1O2_init, where=sender_mask)

# Spatially varying diffusion
D_S2 = CellVariable(name="D_S2", mesh=mesh, value=D_solution)
D_S2.setValue(D_gel, where=gel_mask)

# =============================================================================
# EQUATIONS (same as 1D, but now in 2D!)
# =============================================================================

eq_S2 = (TransientTerm(var=S2) == 
         DiffusionTerm(coeff=D_S2, var=S2) +  
         k_p * I1O2 +
         ImplicitSourceTerm(coeff=-(k_slow * I2 + k_fast * Th2 + k_d_ss), var=S2))

eq_I2 = (TransientTerm(var=I2) == 
         k_d_ds * S2_I2 +
         ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

eq_Th2 = (TransientTerm(var=Th2) == 
          k_d_ds * S2_Th2 +
          ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

eq_S2_I2 = (TransientTerm(var=S2_I2) == 
            k_slow * I2 * S2 +
            ImplicitSourceTerm(coeff=-k_d_ds, var=S2_I2))

eq_S2_Th2 = (TransientTerm(var=S2_Th2) == 
             k_fast * Th2 * S2 +
             ImplicitSourceTerm(coeff=-k_d_ds, var=S2_Th2))

eq = eq_S2 & eq_I2 & eq_Th2 & eq_S2_I2 & eq_S2_Th2

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

# =============================================================================
# TIME STEPPING
# =============================================================================

print("Starting 2D simulation...")
print(f"Total steps: {n_steps}")

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
        
        if step % (save_interval_steps * 10) == 0:  # Print less frequently
            print(f"t = {current_time/3600:.2f} hr: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM")

print("\nSimulation complete!")

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

# Save time series data
df_timeseries = pd.DataFrame({
    'Time_hours': time_points,
    'I2_nM': np.array(I2_concentration) * 1000,
    'S2_free_nM': np.array(S2_free_concentration) * 1000,
    'S2_total_nM': np.array(S2_total_concentration) * 1000
})
df_timeseries.to_csv(f'timeseries_ccd={distance_between:.0f}um.csv', index=False)

print(f"Time series saved to: timeseries_ccd={distance_between:.0f}um.csv")

# =============================================================================
# PLOTTING
# =============================================================================

# Convert to nM
time_points = np.array(time_points)
I2_concentration_nM = np.array(I2_concentration) * 1000
S2_free_concentration_nM = np.array(S2_free_concentration) * 1000
S2_total_concentration_nM = np.array(S2_total_concentration) * 1000

# Time series plots (same as before)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

plt.title('Sender Receiver Chemical Kinetics with domain size of 1cm x 1mm and 200 um ccd')

axes[0].plot(time_points, I2_concentration_nM, 'b-', linewidth=2)
axes[0].axhline(y=75, color='g', linestyle='--', alpha=0.5, label='75% ON')
axes[0].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='25% OFF')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title(f'2D Model: I2 at distance = {distance_between:.0f} μm', 
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(time_points, S2_free_concentration_nM, 'g-', linewidth=2)
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('2D Model: Free S2 at Receiver', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

axes[2].plot(time_points, S2_total_concentration_nM, 'r-', linewidth=2)
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].set_ylabel('[S2] total (nM)', fontsize=12)
axes[2].set_title('2D Model: Total S2 at Receiver', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'2D_model_timeseries_ccd={distance_between:.0f}_dx={dx:.0f}_dt={dt:.0f}.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 2D SPATIAL HEATMAP
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.title('HeatMap of Sender Receiver System 1cm x 1mm and 200um ccd')

# Reshape for plotting
S2_2D = S2.value.reshape((ny, nx)) * 1000  # Convert to nM

# Plot 1: S2 concentration heatmap
im1 = axes[0].imshow(S2_2D, extent=[0, total_width, 0, total_height],
                     origin='lower', cmap='viridis', aspect='equal')
axes[0].set_xlabel('X position (μm)', fontsize=12)
axes[0].set_ylabel('Y position (μm)', fontsize=12)
axes[0].set_title(f'S2 Concentration (nM) at t = {time_points[-1]:.1f} hr', 
                  fontsize=14, fontweight='bold')

# Mark nodes
from matplotlib.patches import Rectangle
sender_rect = Rectangle((sender_center_x - node_size/2, sender_center_y - node_size/2),
                        node_size, node_size, linewidth=2, edgecolor='red',
                        facecolor='none', label='Sender')
receiver_rect = Rectangle((receiver_center_x - node_size/2, receiver_center_y - node_size/2),
                          node_size, node_size, linewidth=2, edgecolor='blue',
                          facecolor='none', label='Receiver')
axes[0].add_patch(sender_rect)
axes[0].add_patch(receiver_rect)
axes[0].legend(fontsize=10)

cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('[S2] (nM)', fontsize=11)

# Plot 2: Cross-section along y = center
y_center_idx = ny // 2
S2_crossection = S2_2D[y_center_idx, :] 

axes[1].plot(np.linspace(0, total_width, nx), S2_crossection, 'b-', linewidth=2)
axes[1].axvline(x=sender_center_x, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Sender')
axes[1].axvline(x=receiver_center_x, color='blue', linestyle='--', linewidth=2,
                alpha=0.7, label='Receiver')
axes[1].set_xlabel('X position (μm)', fontsize=12)
axes[1].set_ylabel('[S2] (nM)', fontsize=12)
axes[1].set_title('S2 Profile Along Line Between Nodes', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_ylim(bottom=0)

plt.tight_layout()
# plt.savefig('2D_spatial_profile.png', dpi=300, bbox_inches='tight')
plt.show()

