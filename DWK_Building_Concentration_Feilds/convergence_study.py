"""
Convergence Study for 2D Reaction-Diffusion System
Tests spatial (mesh refinement) and temporal (time step) convergence
Compares numerical solutions against analytical steady-state solution
"""

import numpy as np
import pandas as pd
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm, ImplicitSourceTerm
from fipy.tools import numerix
from scipy.special import i0, i1, k0, k1
from scipy.interpolate import interp1d
import time as time_module
import matplotlib.pyplot as plt

# ============================================================================
# PHYSICAL PARAMETERS (FIXED FOR ALL STUDIES)
# ============================================================================

D = 150.0  # Diffusion coefficient [µm²/s]
k = 0.036  # Degradation rate constant [1/s]
r_p_value = 150.0  # Production rate inside source [nM/s]

R = 50.0  # Source radius [µm]
L = 500.0  # Domain length [µm]
W = 500.0  # Domain width [µm]

Phi = R * np.sqrt(k / D)
C_char = r_p_value / k

print("="*70)
print("CONVERGENCE STUDY FOR 2D REACTION-DIFFUSION SYSTEM")
print("="*70)
print(f"Physical Parameters:")
print(f"  D = {D} µm²/s")
print(f"  k = {k} 1/s")
print(f"  r_p = {r_p_value} nM/s")
print(f"  R = {R} µm")
print(f"  Φ = {Phi:.3f}")
print("="*70 + "\n")

# ============================================================================
# ANALYTICAL SOLUTION FUNCTION
# ============================================================================

def analytical_solution_2D(r_points, R, Phi, C_char):
    """
    Compute 2D analytical steady-state solution
    
    Parameters:
    -----------
    r_points : array
        Radial distances from center [µm]
    R : float
        Source radius [µm]
    Phi : float
        Thiele modulus
    C_char : float
        Characteristic concentration [nM]
    
    Returns:
    --------
    C_analytical : array
        Concentration at each radial point [nM]
    """
    r_prime = r_points / R
    C_analytical = np.zeros_like(r_points)
    
    for i, rp in enumerate(r_prime):
        if rp <= 1.0:  # Inside source
            C_prime = 1.0 - Phi * k1(Phi) * i0(Phi * rp)
        else:  # Outside source
            C_prime = Phi * i1(Phi) * k0(Phi * rp)
        
        C_analytical[i] = C_prime * C_char
    
    return C_analytical

# ============================================================================
# SOLVER FUNCTION
# ============================================================================

def run_simulation(nx, ny, dt, t_final, verbose=False):
    """
    Run a single simulation with specified spatial and temporal resolution
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid cells in x and y directions
    dt : float
        Time step [s]
    t_final : float
        Final simulation time [s]
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    dict with keys:
        'C_array': 2D concentration field at steady state
        'C_center': Concentration at center
        'C_edge': Concentration at source edge
        'C_radial': 1D radial profile
        'r_radial': Radial distances
        'mesh_cells': Total number of cells
        'n_steps': Number of time steps
        'wall_time': Computation time [s]
    """
    
    start_time = time_module.time()
    
    # Create mesh
    mesh = Grid2D(dx=L/nx, dy=W/ny, nx=nx, ny=ny)
    x, y = mesh.cellCenters
    
    # Define source
    sender_center_x = L / 2.0
    sender_center_y = W / 2.0
    distance_from_center = numerix.sqrt((x - sender_center_x)**2 + 
                                         (y - sender_center_y)**2)
    production_mask = (distance_from_center <= R)
    
    # Create variables
    C = CellVariable(name="C", mesh=mesh, value=0.0, hasOld=True)
    r_p = CellVariable(name="r_p", mesh=mesh, value=0.0, hasOld=False)
    r_p.setValue(r_p_value, where=production_mask)
    
    # Define equation
    eq_C = (TransientTerm(var=C) == 
            DiffusionTerm(coeff=D, var=C) - 
            ImplicitSourceTerm(coeff=k, var=C) + 
            r_p)
    
    # Time stepping
    n_steps = int(t_final / dt)
    
    if verbose:
        print(f"  Mesh: {nx}×{ny} = {mesh.numberOfCells} cells")
        print(f"  Time: dt={dt}s, {n_steps} steps to t={t_final}s")
    
    # Run simulation
    for step in range(n_steps):
        C.updateOld()
        
        res = 1e10
        sweep = 0
        max_sweeps = 10
        tolerance = 1e-6
        
        while res > tolerance and sweep < max_sweeps:
            res = eq_C.sweep(dt=dt)
            sweep += 1
        
        if verbose and step % (n_steps // 10) == 0:
            print(f"    Step {step}/{n_steps} ({100*step/n_steps:.0f}%)")
    
    # Extract results
    C_array = C.value.reshape((ny, nx))
    
    # Get center concentration
    center_idx = mesh.numberOfCells // 2
    C_center = C.value[center_idx]
    
    # Get edge concentration
    edge_distance = R
    edge_mask = (numerix.abs(distance_from_center - edge_distance) < L/(2*nx))
    C_edge = numerix.mean(C.value[edge_mask]) if numerix.sum(edge_mask) > 0 else 0
    
    # Extract radial profile (horizontal line through center)
    center_y_idx = ny // 2
    C_line = C_array[center_y_idx, :]
    x_coords = np.linspace(0, L, nx)
    r_numerical = np.abs(x_coords - sender_center_x)
    
    # Sort by radius
    sort_idx = np.argsort(r_numerical)
    r_radial = r_numerical[sort_idx]
    C_radial = C_line[sort_idx]
    
    wall_time = time_module.time() - start_time
    
    if verbose:
        print(f"  Completed in {wall_time:.2f}s")
        print(f"  C_center = {C_center:.2f} nM, C_edge = {C_edge:.2f} nM\n")
    
    return {
        'C_array': C_array,
        'C_center': C_center,
        'C_edge': C_edge,
        'C_radial': C_radial,
        'r_radial': r_radial,
        'mesh_cells': mesh.numberOfCells,
        'n_steps': n_steps,
        'wall_time': wall_time
    }

# ============================================================================
# SPATIAL CONVERGENCE STUDY
# ============================================================================

print("\n" + "="*70)
print("SPATIAL CONVERGENCE STUDY")
print("="*70)
print("Testing mesh refinement with fixed time step\n")

# Fixed time parameters
dt_spatial = 1.0  # Fixed time step [s]
t_final_spatial = 3600.0  # 1 hour to ensure steady state

# Mesh resolutions to test
mesh_sizes = [25, 50, 100, 150, 200, 300]  # Number of cells per dimension

spatial_results = []

for nx_test in mesh_sizes:
    ny_test = nx_test  # Square mesh
    
    print(f"Running simulation with {nx_test}×{ny_test} mesh...")
    
    result = run_simulation(nx_test, ny_test, dt_spatial, t_final_spatial, verbose=True)
    
    # Compare with analytical solution
    r_analytical = np.linspace(0, L/2, 500)
    C_analytical = analytical_solution_2D(r_analytical, R, Phi, C_char)
    
    # Interpolate numerical to analytical points
    C_num_interp = interp1d(result['r_radial'], result['C_radial'], 
                            bounds_error=False, fill_value=0)(r_analytical)
    
    # Calculate errors
    abs_error = np.abs(C_analytical - C_num_interp)
    rel_error = abs_error / (C_analytical + 1e-6) * 100
    
    # Only consider valid region (where C > 0.1 nM)
    valid_mask = (r_analytical < L/2) & (C_analytical > 0.1)
    
    L2_error = np.sqrt(np.mean(abs_error[valid_mask]**2))
    Linf_error = np.max(abs_error[valid_mask])
    mean_rel_error = np.mean(rel_error[valid_mask])
    
    spatial_results.append({
        'nx': nx_test,
        'ny': ny_test,
        'total_cells': result['mesh_cells'],
        'dx': L / nx_test,
        'C_center': result['C_center'],
        'C_edge': result['C_edge'],
        'L2_error': L2_error,
        'Linf_error': Linf_error,
        'mean_rel_error': mean_rel_error,
        'wall_time': result['wall_time']
    })
    
    print(f"  L2 error: {L2_error:.4f} nM")
    print(f"  L∞ error: {Linf_error:.4f} nM")
    print(f"  Mean relative error: {mean_rel_error:.2f}%\n")

# Save spatial convergence results
df_spatial = pd.DataFrame(spatial_results)
df_spatial.to_csv('spatial_convergence_study.csv', index=False)
print("Spatial convergence results saved to 'spatial_convergence_study.csv'\n")

# ============================================================================
# TEMPORAL CONVERGENCE STUDY
# ============================================================================

# print("\n" + "="*70)
# print("TEMPORAL CONVERGENCE STUDY")
# print("="*70)
# print("Testing time step refinement with fixed fine mesh\n")

# # Fixed mesh (use fine mesh)
# nx_temporal = 200
# ny_temporal = 200

# # Fixed final time
# t_final_temporal = 3600.0  # 1 hour

# # Time steps to test
# dt_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25]  # [s]

# temporal_results = []

# for dt_test in dt_values:
    
#     print(f"Running simulation with dt = {dt_test} s...")
    
#     result = run_simulation(nx_temporal, ny_temporal, dt_test, t_final_temporal, verbose=True)
    
#     # Compare with analytical solution
#     r_analytical = np.linspace(0, L/2, 500)
#     C_analytical = analytical_solution_2D(r_analytical, R, Phi, C_char)
    
#     # Interpolate numerical to analytical points
#     C_num_interp = interp1d(result['r_radial'], result['C_radial'], 
#                             bounds_error=False, fill_value=0)(r_analytical)
    
#     # Calculate errors
#     abs_error = np.abs(C_analytical - C_num_interp)
#     rel_error = abs_error / (C_analytical + 1e-6) * 100
    
#     valid_mask = (r_analytical < L/2) & (C_analytical > 0.1)
    
#     L2_error = np.sqrt(np.mean(abs_error[valid_mask]**2))
#     Linf_error = np.max(abs_error[valid_mask])
#     mean_rel_error = np.mean(rel_error[valid_mask])
    
#     temporal_results.append({
#         'dt': dt_test,
#         'n_steps': result['n_steps'],
#         'C_center': result['C_center'],
#         'C_edge': result['C_edge'],
#         'L2_error': L2_error,
#         'Linf_error': Linf_error,
#         'mean_rel_error': mean_rel_error,
#         'wall_time': result['wall_time']
#     })
    
#     print(f"  L2 error: {L2_error:.4f} nM")
#     print(f"  L∞ error: {Linf_error:.4f} nM")
#     print(f"  Mean relative error: {mean_rel_error:.2f}%\n")

# # Save temporal convergence results
# df_temporal = pd.DataFrame(temporal_results)
# df_temporal.to_csv('temporal_convergence_study.csv', index=False)
# print("Temporal convergence results saved to 'temporal_convergence_study.csv'\n")

# ============================================================================
# PLOT CONVERGENCE RESULTS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# -------------------- SPATIAL CONVERGENCE PLOTS --------------------

# Plot 1: L2 Error vs Mesh Size
ax1 = axes[0, 0]
ax1.loglog(df_spatial['dx'], df_spatial['L2_error'], 'o-', linewidth=2, markersize=8, label='L2 Error')
# Add reference line for second-order convergence
dx_ref = df_spatial['dx'].values
L2_ref = df_spatial['L2_error'].iloc[0] * (dx_ref / dx_ref[0])**2
ax1.loglog(dx_ref, L2_ref, 'k--', alpha=0.5, label='2nd order reference')
ax1.set_xlabel('Mesh spacing Δx [µm]', fontsize=11)
ax1.set_ylabel('L2 Error [nM]', fontsize=11)
ax1.set_title('Spatial Convergence: L2 Error', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Linf Error vs Mesh Size
ax2 = axes[0, 1]
ax2.loglog(df_spatial['dx'], df_spatial['Linf_error'], 's-', linewidth=2, markersize=8, 
           color='orange', label='L∞ Error')
Linf_ref = df_spatial['Linf_error'].iloc[0] * (dx_ref / dx_ref[0])**2
ax2.loglog(dx_ref, Linf_ref, 'k--', alpha=0.5, label='2nd order reference')
ax2.set_xlabel('Mesh spacing Δx [µm]', fontsize=11)
ax2.set_ylabel('L∞ Error [nM]', fontsize=11)
ax2.set_title('Spatial Convergence: L∞ Error', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Computational Time vs Mesh Size
ax3 = axes[0, 2]
ax3.loglog(df_spatial['total_cells'], df_spatial['wall_time'], '^-', 
           linewidth=2, markersize=8, color='green', label='Wall time')
ax3.set_xlabel('Total cells', fontsize=11)
ax3.set_ylabel('Wall time [s]', fontsize=11)
ax3.set_title('Spatial: Computational Cost', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# -------------------- TEMPORAL CONVERGENCE PLOTS --------------------

# # Plot 4: L2 Error vs Time Step
# ax4 = axes[1, 0]
# ax4.loglog(df_temporal['dt'], df_temporal['L2_error'], 'o-', linewidth=2, markersize=8, label='L2 Error')
# # Add reference line for first-order convergence (typical for implicit methods)
# dt_ref = df_temporal['dt'].values
# L2_temp_ref = df_temporal['L2_error'].iloc[-1] * (dt_ref / dt_ref[-1])**1
# ax4.loglog(dt_ref, L2_temp_ref, 'k--', alpha=0.5, label='1st order reference')
# ax4.set_xlabel('Time step Δt [s]', fontsize=11)
# ax4.set_ylabel('L2 Error [nM]', fontsize=11)
# ax4.set_title('Temporal Convergence: L2 Error', fontsize=12, fontweight='bold')
# ax4.legend()
# ax4.grid(True, alpha=0.3, which='both')

# # Plot 5: Linf Error vs Time Step
# ax5 = axes[1, 1]
# ax5.loglog(df_temporal['dt'], df_temporal['Linf_error'], 's-', linewidth=2, markersize=8, 
#            color='orange', label='L∞ Error')
# Linf_temp_ref = df_temporal['Linf_error'].iloc[-1] * (dt_ref / dt_ref[-1])**1
# ax5.loglog(dt_ref, Linf_temp_ref, 'k--', alpha=0.5, label='1st order reference')
# ax5.set_xlabel('Time step Δt [s]', fontsize=11)
# ax5.set_ylabel('L∞ Error [nM]', fontsize=11)
# ax5.set_title('Temporal Convergence: L∞ Error', fontsize=12, fontweight='bold')
# ax5.legend()
# ax5.grid(True, alpha=0.3, which='both')

# # Plot 6: Computational Time vs Time Step
# ax6 = axes[1, 2]
# ax6.semilogy(df_temporal['dt'], df_temporal['wall_time'], '^-', 
#              linewidth=2, markersize=8, color='green', label='Wall time')
# ax6.set_xlabel('Time step Δt [s]', fontsize=11)
# ax6.set_ylabel('Wall time [s]', fontsize=11)
# ax6.set_title('Temporal: Computational Cost', fontsize=12, fontweight='bold')
# ax6.legend()
# ax6.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('convergence_study_summary.png', dpi=150, bbox_inches='tight')
# plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("CONVERGENCE STUDY SUMMARY")
print("="*70)

print("\nSPATIAL CONVERGENCE:")
print(f"  Finest mesh: {df_spatial.iloc[-1]['nx']}×{df_spatial.iloc[-1]['ny']} cells")
print(f"  L2 error: {df_spatial.iloc[-1]['L2_error']:.4f} nM")
print(f"  L∞ error: {df_spatial.iloc[-1]['Linf_error']:.4f} nM")
print(f"  Mean relative error: {df_spatial.iloc[-1]['mean_rel_error']:.2f}%")

# print("\nTEMPORAL CONVERGENCE:")
# print(f"  Smallest dt: {df_temporal.iloc[-1]['dt']} s")
# print(f"  L2 error: {df_temporal.iloc[-1]['L2_error']:.4f} nM")
# print(f"  L∞ error: {df_temporal.iloc[-1]['Linf_error']:.4f} nM")
# print(f"  Mean relative error: {df_temporal.iloc[-1]['mean_rel_error']:.2f}%")

print("\nRECOMMENDATIONS:")
# Find where error plateaus (within 10% of minimum error)
spatial_plateau_idx = np.where(df_spatial['L2_error'] < 1.1 * df_spatial['L2_error'].min())[0][0]
# temporal_plateau_idx = np.where(df_temporal['L2_error'] < 1.1 * df_temporal['L2_error'].min())[0][0]

print(f"  Recommended mesh: {df_spatial.iloc[spatial_plateau_idx]['nx']}×"
      f"{df_spatial.iloc[spatial_plateau_idx]['ny']} cells")
print(f"    (L2 error: {df_spatial.iloc[spatial_plateau_idx]['L2_error']:.4f} nM, "
      f"time: {df_spatial.iloc[spatial_plateau_idx]['wall_time']:.1f}s)")

# print(f"  Recommended dt: {df_temporal.iloc[temporal_plateau_idx]['dt']} s")
# print(f"    (L2 error: {df_temporal.iloc[temporal_plateau_idx]['L2_error']:.4f} nM, "
#       f"time: {df_temporal.iloc[temporal_plateau_idx]['wall_time']:.1f}s)")

print("="*70 + "\n")

print("✓ All convergence studies complete!")
print("✓ Results saved to CSV files")
print("✓ Summary plot saved as 'convergence_study_summary.png'")