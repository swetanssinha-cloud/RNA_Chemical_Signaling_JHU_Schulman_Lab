import numpy as np
import matplotlib.pyplot as plt
from diff_1D_instantaneous_source import build_problem, run_simulation

'''
Convergence study with instantaneous energy injection.
This eliminates source timing issues and tests pure numerical diffusion.
'''

# Test different grid spacings
dx_values = [2.0, 1.0, 0.5, 0.25]
results = {}

print("="*70)
print("CONVERGENCE STUDY: Instantaneous Source")
print("="*70)

for dx in dx_values:
    print(f"\n{'='*50}")
    print(f"Running with dx = {dx}")
    print(f"{'='*50}")
    
    # Calculate nx to keep domain size constant (Lx = 400)
    Lx = 400.0
    nx = int(Lx / dx)
    
    actual_Lx = nx * dx
    dt = 0.9 * dx**2 / 2.0
    n_timesteps = int(100.0 / dt)
    
    print(f"  nx = {nx}")
    print(f"  Actual domain length = {actual_Lx}")
    print(f"  dt = {dt:.6f} seconds")
    print(f"  Number of timesteps = {n_timesteps}")
    print(f"  Total simulated time = {n_timesteps * dt:.2f} seconds")
    
    mesh, phi, eq, center_source = build_problem(nx=nx, dx=dx, D=1.0)
    
    x_vals = np.array(mesh.cellCenters[0])
    print(f"  Mesh x range: {x_vals[0]:.2f} to {x_vals[-1]:.2f}")
    print(f"  Center cell location: {x_vals[len(x_vals)//2]:.2f}")
    
    # Check source region
    source_indices = np.where(center_source.value)[0]
    print(f"  Source cells: {len(source_indices)}")
    if len(source_indices) > 0:
        print(f"  Source at x = {x_vals[source_indices]}")
    
    phi_final, Q_actual = run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        total_energy=400.0,  # Fixed total energy
        total_time=100.0,
        snapshot_interval_time=20.0,
        show_animation=False,
        save_animation=False
    )
    
    # Store results
    results[dx] = {
        'nx': nx,
        'actual_Lx': actual_Lx,
        'dt': dt,
        'n_timesteps': n_timesteps,
        'Q_actual': Q_actual,
        'peak_at_t100': np.max(phi_final.value),
        'center_value': phi_final.value[len(phi_final.value)//2]
    }

# Print comparison table
print("\n" + "="*70)
print("CONVERGENCE STUDY RESULTS")
print("="*70)
print(f"{'dx':<8} {'nx':<8} {'Actual Lx':<12} {'dt':<10} {'Q_actual':<12} {'Peak T':<12} {'Center T':<12}")
print("-"*70)
for dx in dx_values:
    r = results[dx]
    print(f"{dx:<8.3f} {r['nx']:<8} {r['actual_Lx']:<12.1f} {r['dt']:<10.6f} {r['Q_actual']:<12.6f} "
          f"{r['peak_at_t100']:<12.6f} {r['center_value']:<12.6f}")

# Calculate and print convergence metrics
print("\n" + "="*70)
print("CONVERGENCE METRICS (% change between refinements)")
print("="*70)

for i in range(len(dx_values)-1):
    dx_coarse = dx_values[i]
    dx_fine = dx_values[i+1]
    
    Q_coarse = results[dx_coarse]['Q_actual']
    Q_fine = results[dx_fine]['Q_actual']
    
    peak_coarse = results[dx_coarse]['peak_at_t100']
    peak_fine = results[dx_fine]['peak_at_t100']
    
    center_coarse = results[dx_coarse]['center_value']
    center_fine = results[dx_fine]['center_value']
    
    # Relative change (percentage)
    Q_change = abs(Q_fine - Q_coarse) / Q_coarse * 100
    peak_change = abs(peak_fine - peak_coarse) / abs(peak_coarse) * 100 if peak_coarse != 0 else 0
    center_change = abs(center_fine - center_coarse) / abs(center_coarse) * 100 if center_coarse != 0 else 0
    
    print(f"\nRefining from dx={dx_coarse} to dx={dx_fine}:")
    print(f"  Q_actual change:     {Q_change:.6f}%  (should be ~0% for energy conservation)")
    print(f"  Peak temp change:    {peak_change:.3f}%")
    print(f"  Center temp change:  {center_change:.3f}%")

# Energy conservation check
print("\n" + "="*70)
print("ENERGY CONSERVATION CHECK")
print("="*70)
Q_values = [results[dx]['Q_actual'] for dx in dx_values]
Q_mean = np.mean(Q_values)
Q_std = np.std(Q_values)
Q_variation = Q_std / Q_mean * 100

print(f"Mean Q across all grids: {Q_mean:.6f}")
print(f"Std deviation: {Q_std:.6f}")
print(f"Coefficient of variation: {Q_variation:.4f}%")

if Q_variation < 0.1:
    print("✓ EXCELLENT: Energy conserved to < 0.1%")
elif Q_variation < 1.0:
    print("✓ GOOD: Energy conserved to < 1%")
else:
    print("⚠ WARNING: Energy not well conserved (> 1% variation)")

# Temperature convergence check
print("\n" + "="*70)
print("TEMPERATURE CONVERGENCE CHECK")
print("="*70)

finest_change_peak = abs(results[dx_values[-1]]['peak_at_t100'] - results[dx_values[-2]]['peak_at_t100']) / results[dx_values[-2]]['peak_at_t100'] * 100
finest_change_center = abs(results[dx_values[-1]]['center_value'] - results[dx_values[-2]]['center_value']) / results[dx_values[-2]]['center_value'] * 100

print(f"Change in peak temp (0.5 → 0.25): {finest_change_peak:.3f}%")
print(f"Change in center temp (0.5 → 0.25): {finest_change_center:.3f}%")

if finest_change_peak < 1.0 and finest_change_center < 1.0:
    print("✓ CONVERGED: Changes < 1%")
elif finest_change_peak < 5.0 and finest_change_center < 5.0:
    print("✓ ACCEPTABLE: Changes < 5%")
else:
    print("✗ NOT CONVERGED: Changes > 5%")
    print("  Suggestion: Try smaller dx values (e.g., 0.125, 0.0625)")

print("="*70)

# Plot convergence
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Q_actual convergence (should be flat!)
Q_vals = [results[dx]['Q_actual'] for dx in dx_values]
axes[0, 0].plot(dx_values, Q_vals, 'o-', linewidth=2, markersize=8)
axes[0, 0].axhline(y=400.0, color='r', linestyle='--', label='Target (400)')
axes[0, 0].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[0, 0].set_ylabel('Q_actual', fontsize=12)
axes[0, 0].set_title('Total Energy (Should be Constant)', fontsize=12)
axes[0, 0].grid(True)
axes[0, 0].set_xscale('log')
axes[0, 0].invert_xaxis()
axes[0, 0].legend()

# Peak temperature convergence
peak_vals = [results[dx]['peak_at_t100'] for dx in dx_values]
axes[0, 1].plot(dx_values, peak_vals, 'o-', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[0, 1].set_ylabel('Peak Temperature at t=100s', fontsize=12)
axes[0, 1].set_title('Peak Temperature Convergence', fontsize=12)
axes[0, 1].grid(True)
axes[0, 1].set_xscale('log')
axes[0, 1].invert_xaxis()

# Center temperature convergence
center_vals = [results[dx]['center_value'] for dx in dx_values]
axes[1, 0].plot(dx_values, center_vals, 'o-', linewidth=2, markersize=8, color='green')
axes[1, 0].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[1, 0].set_ylabel('Center Temperature at t=100s', fontsize=12)
axes[1, 0].set_title('Center Temperature Convergence', fontsize=12)
axes[1, 0].grid(True)
axes[1, 0].set_xscale('log')
axes[1, 0].invert_xaxis()

# Convergence rate (log-log plot)
axes[1, 1].loglog(dx_values, peak_vals, 'o-', linewidth=2, markersize=8, label='Peak T')
axes[1, 1].loglog(dx_values, center_vals, 's-', linewidth=2, markersize=8, label='Center T')

# Add reference slopes for convergence order
dx_ref = np.array([0.5, 0.25])
# Second-order convergence: error ~ dx^2
slope2 = peak_vals[2] * (dx_ref / dx_values[2])**2
axes[1, 1].loglog(dx_ref, slope2, 'k--', alpha=0.5, label='2nd order (dx²)')

axes[1, 1].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[1, 1].set_ylabel('Temperature', fontsize=12)
axes[1, 1].set_title('Convergence Order (Log-Log)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].invert_xaxis()

plt.tight_layout()
plt.savefig('convergence_study_instantaneous.png', dpi=300, bbox_inches='tight')
print("\nConvergence plot saved as 'convergence_study_instantaneous.png'")
plt.show()

print("\n" + "="*70)
print("CONVERGENCE STUDY COMPLETE")
print("="*70)