import numpy as np
import matplotlib.pyplot as plt
from diff_1D_intial_optional_anim import build_problem, run_simulation

# Test different grid spacings
dx_values = [2.0, 1.0, 0.5, 0.25]
results = {}

for dx in dx_values:
    print(f"\n{'='*50}")
    print(f"Running with dx = {dx}")
    print(f"{'='*50}")
    
    # Calculate nx to keep domain size constant (Lx = 400)
    Lx = 400.0
    nx = int(Lx / dx)  # REMOVED the +1
    
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
    
    phi_final, Q_actual = run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        source_value=400.0,
        source_duration=1.0,
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
    print(f"{dx:<8.3f} {r['nx']:<8} {r['actual_Lx']:<12.1f} {r['dt']:<10.6f} {r['Q_actual']:<12.2f} "
          f"{r['peak_at_t100']:<12.4f} {r['center_value']:<12.4f}")

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
    peak_change = abs(peak_fine - peak_coarse) / peak_coarse * 100
    center_change = abs(center_fine - center_coarse) / center_coarse * 100
    
    print(f"\nRefining from dx={dx_coarse} to dx={dx_fine}:")
    print(f"  Q_actual change:     {Q_change:.3f}%")
    print(f"  Peak temp change:    {peak_change:.3f}%")
    print(f"  Center temp change:  {center_change:.3f}%")

# Check if converged (< 1% change in finest refinement)
finest_change = abs(results[dx_values[-1]]['Q_actual'] - results[dx_values[-2]]['Q_actual']) / results[dx_values[-2]]['Q_actual'] * 100

print("\n" + "="*70)
if finest_change < 1.0:
    print(f"✓ CONVERGED: Change in Q_actual = {finest_change:.3f}% < 1%")
else:
    print(f"✗ NOT CONVERGED: Change in Q_actual = {finest_change:.3f}% > 1%")
    print("  (Note: Convergence may still be acceptable if < 5%)")
print("="*70)

# Plot convergence
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Q_actual convergence
Q_vals = [results[dx]['Q_actual'] for dx in dx_values]
axes[0].plot(dx_values, Q_vals, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[0].set_ylabel('Q_actual', fontsize=12)
axes[0].set_title('Total Energy Convergence', fontsize=12)
axes[0].grid(True)
axes[0].set_xscale('log')
axes[0].invert_xaxis()

# Peak temperature convergence
peak_vals = [results[dx]['peak_at_t100'] for dx in dx_values]
axes[1].plot(dx_values, peak_vals, 'o-', linewidth=2, markersize=8)
axes[1].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[1].set_ylabel('Peak Temperature at t=100s', fontsize=12)
axes[1].set_title('Peak Temperature Convergence', fontsize=12)
axes[1].grid(True)
axes[1].set_xscale('log')
axes[1].invert_xaxis()

# Center temperature convergence
center_vals = [results[dx]['center_value'] for dx in dx_values]
axes[2].plot(dx_values, center_vals, 'o-', linewidth=2, markersize=8)
axes[2].set_xlabel('Grid spacing (dx)', fontsize=12)
axes[2].set_ylabel('Center Temperature at t=100s', fontsize=12)
axes[2].set_title('Center Temperature Convergence', fontsize=12)
axes[2].grid(True)
axes[2].set_xscale('log')
axes[2].invert_xaxis()

plt.tight_layout()
plt.savefig('convergence_study_L_constant.png', dpi=300, bbox_inches='tight')
print("\nConvergence plot saved as 'convergence_study_L_constant.png'")
plt.show()