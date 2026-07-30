import numpy as np
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm, ImplicitSourceTerm, Viewer
from fipy.tools import numerix
import matplotlib.pyplot as plt
import pandas as pd 

# ============================================================================
# PARAMETERS
# ============================================================================

# Physical parameters
D = 150.0  # Diffusion coefficient [µm²/s]
k = 0.036  # Degradation rate constant [1/s]
r_p_value = 150.0  # Production rate inside source [nM/s]

# Geometric parameters
R = 50.0  # Source radius [µm]
L = 500.0  # Domain length [µm]
W = 500.0  # Domain width [µm]

# Calculated dimensionless parameter
Phi = R * np.sqrt(k / D)  # Thiele modulus
print(f"Thiele modulus Φ = {Phi:.3f}")

# Mesh parameters
nx = 200  # Number of cells in x-direction
ny = 200  # Number of cells in y-direction

# ============================================================================
# CREATE MESH
# ============================================================================

mesh = Grid2D(dx=L/nx, dy=W/ny, nx=nx, ny=ny)

# Get cell centers
x, y = mesh.cellCenters
print(f"Mesh created: {nx} x {ny} = {mesh.numberOfCells} cells")

# ============================================================================
# DEFINE SOURCE GEOMETRY
# ============================================================================

# Source center (center of domain)
sender_center_x = L / 2.0
sender_center_y = W / 2.0

# Create mask for production region (circular source)
distance_from_center = numerix.sqrt((x - sender_center_x)**2 + 
                                     (y - sender_center_y)**2)
production_mask = (distance_from_center <= R)

print(f"Number of cells in source: {numerix.sum(production_mask)}")
print(f"Total cells: {mesh.numberOfCells}")

# ============================================================================
# DEFINE CELL VARIABLES
# ============================================================================

# Concentration field
C = CellVariable(name="Concentration C", 
                 mesh=mesh, 
                 value=0.0, 
                 hasOld=True)

# Production rate field (spatially varying)
r_p = CellVariable(name="Production rate", 
                   mesh=mesh, 
                   value=0.0, 
                   hasOld=False)

# Set production rate inside source region
r_p.setValue(r_p_value, where=production_mask)

# ============================================================================
# DEFINE THE PDE
# ============================================================================

# Equation: ∂C/∂t = D∇²C - kC + r_p
eq_C = (TransientTerm(var=C) == 
        DiffusionTerm(coeff=D, var=C) - 
        ImplicitSourceTerm(coeff=k, var=C) + 
        r_p)

print("\nEquation defined:")
print("∂C/∂t = D∇²C - kC + r_p")
print(f"  D = {D} µm²/s")
print(f"  k = {k} 1/s")
print(f"  r_p = {r_p_value} nM/s (inside source)")
print(f"  r_p = 0 nM/s (outside source)")

'''
# ============================================================================
# VISUALIZE GEOMETRY
# ============================================================================

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Plot 1: Production rate field (shows source location)
# ax1 = axes[0]
# r_p_array = r_p.value.reshape((ny, nx))
# im1 = ax1.imshow(r_p_array, 
#                  extent=[0, L, 0, W], 
#                  origin='lower', 
#                  cmap='Reds',
#                  aspect='equal')
# circle = plt.Circle((sender_center_x, sender_center_y), R, 
#                     color='blue', fill=False, linewidth=2, linestyle='--')
# ax1.add_patch(circle)
# ax1.set_xlabel('x [µm]')
# ax1.set_ylabel('y [µm]')
# ax1.set_title('Production Rate Field $r_p$')
# plt.colorbar(im1, ax=ax1, label='Production rate [nM/s]')

# # Plot 2: Initial concentration (should be zero)
# ax2 = axes[1]
# C_array = C.value.reshape((ny, nx))
# im2 = ax2.imshow(C_array, 
#                  extent=[0, L, 0, W], 
#                  origin='lower', 
#                  cmap='viridis',
#                  aspect='equal')
# circle2 = plt.Circle((sender_center_x, sender_center_y), R, 
#                      color='red', fill=False, linewidth=2, linestyle='--')
# ax2.add_patch(circle2)
# ax2.set_xlabel('x [µm]')
# ax2.set_ylabel('y [µm]')
# ax2.set_title('Initial Concentration C')
# plt.colorbar(im2, ax=ax2, label='Concentration [nM]')

# plt.tight_layout()
# plt.savefig('geometry_setup_2D.png', dpi=150, bbox_inches='tight')
# plt.show()

# print("\nGeometry setup complete!")
# print(f"Source: circle at ({sender_center_x}, {sender_center_y}) with radius {R} µm")
# print(f"Domain: {L} x {W} µm²")
'''

# ============================================================================
# SOLVER SETUP
# ============================================================================

# Time stepping parameters
dt = 1.0  # Time step [s]
t_final = 3600.0  # Final time [s] - 1 hour
n_steps = int(t_final / dt)

# Save parameters
save_interval_time = 60.0  # Save every 60 seconds
save_interval_steps = int(save_interval_time / dt)

# Storage for analysis
time_points = []
C_center_concentration = []  # Concentration at source center
C_edge_concentration = []    # Concentration at source edge
C_far_concentration = []     # Concentration far from source

# Define monitoring points
center_idx = mesh.numberOfCells // 2  # Approximate center
edge_distance = R + 5.0  # Just outside source
far_distance = 3 * R     # 3x source radius away

# Find indices for monitoring points
edge_mask = (numerix.abs(distance_from_center - edge_distance) < L/(2*nx))
far_mask = (numerix.abs(distance_from_center - far_distance) < L/(2*nx))

print("\n" + "="*70)
print("STARTING TIME INTEGRATION")
print("="*70)
print(f"Time step dt = {dt} s")
print(f"Total time = {t_final} s ({t_final/3600:.2f} hours)")
print(f"Number of steps = {n_steps}")
print(f"Saving every {save_interval_time} s")
print("="*70 + "\n")

# ============================================================================
# TIME STEPPING LOOP
# ============================================================================

for step in range(n_steps):
    # Update old values for time stepping
    C.updateOld()
    
    # Solve the PDE with sweeping for convergence
    res = 1e10
    sweep = 0
    max_sweeps = 10
    tolerance = 1e-6
    
    while res > tolerance and sweep < max_sweeps:
        res = eq_C.sweep(dt=dt)
        sweep += 1
    
    # Save data at specified intervals
    if step % save_interval_steps == 0:
        current_time = step * dt
        time_points.append(current_time)
        
        # Get concentrations at monitoring points
        C_center = C.value[center_idx]
        C_edge = numerix.mean(C.value[edge_mask]) if numerix.sum(edge_mask) > 0 else 0
        C_far = numerix.mean(C.value[far_mask]) if numerix.sum(far_mask) > 0 else 0
        
        C_center_concentration.append(C_center)
        C_edge_concentration.append(C_edge)
        C_far_concentration.append(C_far)
        
        # Print progress
        if step % (save_interval_steps * 5) == 0:  # Print every 5 save intervals
            print(f"t = {current_time/60:.1f} min ({current_time/3600:.2f} hr): "
                  f"C_center = {C_center:.2f} nM, "
                  f"C_edge = {C_edge:.2f} nM, "
                  f"C_far = {C_far:.2f} nM, "
                  f"sweeps = {sweep}")
    
    # Check for steady state (optional)
    if step > 0 and step % (save_interval_steps * 10) == 0:
        # Calculate relative change
        if len(C_center_concentration) > 1:
            rel_change = abs(C_center_concentration[-1] - C_center_concentration[-2]) / (C_center_concentration[-1] + 1e-10)
            if rel_change < 1e-4:
                print(f"\n*** Steady state reached at t = {current_time/3600:.2f} hr ***")
                print(f"Relative change < 0.01%\n")

print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70)
print(f"Final time: {time_points[-1]/3600:.2f} hours")
print(f"Final C_center: {C_center_concentration[-1]:.2f} nM")
print(f"Final C_edge: {C_edge_concentration[-1]:.2f} nM")
print(f"Final C_far: {C_far_concentration[-1]:.2f} nM")
print("="*70 + "\n")

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================


print("Saving results to CSV files...")

# Save time series at monitoring points
df = pd.DataFrame({
    'Time_s': time_points,
    'Time_hours': np.array(time_points) / 3600,
    'C_center_nM': C_center_concentration,
    'C_edge_nM': C_edge_concentration,
    'C_far_nM': C_far_concentration,
})

csv_filename = 'concentration_time_series.csv'
df.to_csv(csv_filename, index=False)
print(f"Time series data saved to '{csv_filename}'")

# Save final 2D concentration field
C_final_2D = C.value.reshape((ny, nx))
np.savetxt('final_concentration_field_2D.csv', C_final_2D, delimiter=',')
print(f"Final 2D field saved to 'final_concentration_field_2D.csv'")


# ============================================================================
# PLOT TIME EVOLUTION
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

time_hours = np.array(time_points) / 3600

ax.plot(time_hours, C_center_concentration, 'o-', label='Center (r=0)', linewidth=2)
ax.plot(time_hours, C_edge_concentration, 's-', label=f'Edge (r≈{edge_distance:.0f} µm)', linewidth=2)
ax.plot(time_hours, C_far_concentration, '^-', label=f'Far (r≈{far_distance:.0f} µm)', linewidth=2)

ax.set_xlabel('Time [hours]', fontsize=12)
ax.set_ylabel('Concentration [nM]', fontsize=12)
ax.set_title('Concentration Evolution at Different Locations', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('DWK_BCF_time_evolution_2D.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT STEADY-STATE HEATMAP
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reshape concentration array for plotting
C_array = C.value.reshape((ny, nx))

# Plot 1: Full concentration field
ax1 = axes[0]
im1 = ax1.imshow(C_array, 
                 extent=[0, L, 0, W], 
                 origin='lower', 
                 cmap='viridis',
                 aspect='equal')
circle1 = plt.Circle((sender_center_x, sender_center_y), R, 
                     color='red', fill=False, linewidth=2, linestyle='--', label='Source boundary')
ax1.add_patch(circle1)
ax1.set_xlabel('x [µm]', fontsize=12)
ax1.set_ylabel('y [µm]', fontsize=12)
ax1.set_title('Steady-State Concentration Field', fontsize=14)
cbar1 = plt.colorbar(im1, ax=ax1, label='Concentration [nM]')
ax1.legend(loc='upper right')

# Plot 2: Log scale for better visualization of gradient
ax2 = axes[1]
C_array_log = np.log10(C_array + 1e-3)  # Add small value to avoid log(0)
im2 = ax2.imshow(C_array_log, 
                 extent=[0, L, 0, W], 
                 origin='lower', 
                 cmap='plasma',
                 aspect='equal')
circle2 = plt.Circle((sender_center_x, sender_center_y), R, 
                     color='cyan', fill=False, linewidth=2, linestyle='--')
ax2.add_patch(circle2)
ax2.set_xlabel('x [µm]', fontsize=12)
ax2.set_ylabel('y [µm]', fontsize=12)
ax2.set_title('Log₁₀(Concentration) Field', fontsize=14)
cbar2 = plt.colorbar(im2, ax=ax2, label='Log₁₀(C [nM])')

plt.tight_layout()
plt.savefig('steady_state_heatmap_2D.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plots saved successfully!")




#########################

# ============================================================================
# ANALYTICAL SOLUTION - 2D CASE
# ============================================================================

from scipy.special import i0, i1, k0, k1  # Modified Bessel functions

print("\n" + "="*70)
print("COMPUTING ANALYTICAL SOLUTION")
print("="*70)

# Dimensionless parameters (already calculated)
print(f"Thiele modulus Φ = {Phi:.3f}")
print(f"Characteristic concentration C_char = r_p/k = {r_p_value/k:.2f} nM")

# Create radial coordinate for plotting (from center outward)
r_points = np.linspace(0, L/2, 500)  # Only go to half domain
r_prime = r_points / R  # Dimensionless radius r' = r/R

# Initialize concentration arrays
C_analytical = np.zeros_like(r_points)

# 2D Analytical Solution from Table 1:
# Inside source (r' < 1):  C'(r') = 1 - Φ*K₁(Φ)*I₀(Φ*r')
# Outside source (r' > 1): C'(r') = Φ*I₁(Φ)*K₀(Φ*r')

for i, r_p_val in enumerate(r_prime):
    if r_p_val <= 1.0:  # Inside source
        C_prime = 1.0 - Phi * k1(Phi) * i0(Phi * r_p_val)
    else:  # Outside source
        C_prime = Phi * i1(Phi) * k0(Phi * r_p_val)
    
    # Convert back to dimensional concentration
    C_char = r_p_value / k  # Characteristic concentration
    C_analytical[i] = C_prime * C_char

print(f"\nAnalytical solution computed for {len(r_points)} radial points")
print(f"C at center (r=0): {C_analytical[0]:.2f} nM")
print(f"C at source edge (r=R): {C_analytical[np.argmin(np.abs(r_points - R))]:.2f} nM")
print(f"C at 2R: {C_analytical[np.argmin(np.abs(r_points - 2*R))]:.2f} nM")
print("="*70 + "\n")

# ============================================================================
# EXTRACT NUMERICAL SOLUTION ALONG RADIAL LINE
# ============================================================================

# Extract concentration along horizontal line through center
center_y_idx = ny // 2
C_numerical_line = C_array[center_y_idx, :]

# Get corresponding x-coordinates (distance from center)
x_coords = np.linspace(0, L, nx)
r_numerical = np.abs(x_coords - sender_center_x)  # Distance from center

# Sort by radius for cleaner plotting
sort_idx = np.argsort(r_numerical)
r_numerical_sorted = r_numerical[sort_idx]
C_numerical_sorted = C_numerical_line[sort_idx]

# ============================================================================
# PLOT: ANALYTICAL vs NUMERICAL
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -------------------- Plot 1: Linear Scale --------------------
ax1 = axes[0, 0]
ax1.plot(r_points, C_analytical, 'r-', linewidth=3, label='Analytical', alpha=0.8)
ax1.plot(r_numerical_sorted, C_numerical_sorted, 'b--', linewidth=2, label='Numerical (FiPy)', alpha=0.7)
ax1.axvline(R, color='gray', linestyle=':', linewidth=2, label='Source edge (R)')
ax1.set_xlabel('Distance from center r [µm]', fontsize=11)
ax1.set_ylabel('Concentration C [nM]', fontsize=11)
ax1.set_title('Radial Concentration Profile - Linear Scale', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, L/2)

# -------------------- Plot 2: Semi-log Scale --------------------
ax2 = axes[0, 1]
ax2.semilogy(r_points, C_analytical + 1e-3, 'r-', linewidth=3, label='Analytical', alpha=0.8)
ax2.semilogy(r_numerical_sorted, C_numerical_sorted + 1e-3, 'b--', linewidth=2, label='Numerical (FiPy)', alpha=0.7)
ax2.axvline(R, color='gray', linestyle=':', linewidth=2, label='Source edge (R)')
ax2.set_xlabel('Distance from center r [µm]', fontsize=11)
ax2.set_ylabel('Concentration C [nM]', fontsize=11)
ax2.set_title('Radial Concentration Profile - Log Scale', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, L/2)

# -------------------- Plot 3: Dimensionless Solution --------------------
ax3 = axes[1, 0]
C_analytical_prime = C_analytical / (r_p_value / k)
C_numerical_prime = C_numerical_sorted / (r_p_value / k)
r_numerical_prime = r_numerical_sorted / R
r_plot_prime = r_points / R

ax3.plot(r_plot_prime, C_analytical_prime, 'r-', linewidth=3, label='Analytical', alpha=0.8)
ax3.plot(r_numerical_prime, C_numerical_prime, 'b--', linewidth=2, label='Numerical (FiPy)', alpha=0.7)
ax3.axvline(1.0, color='gray', linestyle=':', linewidth=2, label="Source edge (r'=1)")
ax3.set_xlabel("Dimensionless distance r' = r/R", fontsize=11)
ax3.set_ylabel("Dimensionless concentration C'", fontsize=11)
ax3.set_title(f'Dimensionless Profile (Φ = {Phi:.3f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 5)

# -------------------- Plot 4: Error Analysis --------------------
ax4 = axes[1, 1]

# Interpolate numerical solution to analytical points for error calculation
from scipy.interpolate import interp1d
C_num_interp = interp1d(r_numerical_sorted, C_numerical_sorted, 
                        bounds_error=False, fill_value=0)(r_points)

# Calculate absolute and relative error
abs_error = np.abs(C_analytical - C_num_interp)
rel_error = abs_error / (C_analytical + 1e-6) * 100  # Percentage

ax4.plot(r_points, abs_error, 'g-', linewidth=2, label='Absolute Error')
ax4.axvline(R, color='gray', linestyle=':', linewidth=2)
ax4_twin = ax4.twinx()
ax4_twin.plot(r_points, rel_error, 'orange', linewidth=2, label='Relative Error (%)', linestyle='--')

ax4.set_xlabel('Distance from center r [µm]', fontsize=11)
ax4.set_ylabel('Absolute Error [nM]', fontsize=11, color='g')
ax4_twin.set_ylabel('Relative Error [%]', fontsize=11, color='orange')
ax4.set_title('Error: Numerical vs Analytical', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='g')
ax4_twin.tick_params(axis='y', labelcolor='orange')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, L/2)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('analytical_vs_numerical_2D.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# QUANTITATIVE COMPARISON
# ============================================================================

print("\n" + "="*70)
print("QUANTITATIVE COMPARISON: ANALYTICAL vs NUMERICAL")
print("="*70)

# Calculate errors where both solutions are valid
valid_mask = (r_points < L/2) & (C_analytical > 0.01)
abs_error_valid = abs_error[valid_mask]
rel_error_valid = rel_error[valid_mask]

print(f"Maximum absolute error: {np.max(abs_error_valid):.4f} nM")
print(f"Mean absolute error: {np.mean(abs_error_valid):.4f} nM")
print(f"Maximum relative error: {np.max(rel_error_valid):.2f} %")
print(f"Mean relative error: {np.mean(rel_error_valid):.2f} %")
print(f"\nRMS error: {np.sqrt(np.mean(abs_error_valid**2)):.4f} nM")

# Key point comparisons
comparison_points = [0, R, 2*R, 3*R]
print(f"\n{'Location':<20} {'Analytical':<15} {'Numerical':<15} {'Error':<15}")
print("-"*70)
for r_comp in comparison_points:
    if r_comp < L/2:
        idx_ana = np.argmin(np.abs(r_points - r_comp))
        idx_num = np.argmin(np.abs(r_numerical_sorted - r_comp))
        C_ana = C_analytical[idx_ana]
        C_num = C_numerical_sorted[idx_num]
        error = abs(C_ana - C_num)
        print(f"r = {r_comp:.1f} µm{'':<10} {C_ana:>8.3f} nM     {C_num:>8.3f} nM     {error:>8.3f} nM")

print("="*70 + "\n")