"""
2D CONVERGENCE STUDY WITH ADAPTIVE MESH AND SMOOTH TANH PROFILES

This script imports geometry and smooth profile functions from Functions.py
to eliminate code duplication and ensure consistency.

Key features:
- Imports from Functions.py instead of duplicating code
- Smooth hyperbolic tangent profiles instead of sharp boolean masks
- Adaptive mesh with variable fine_dx parameter
- Multiprocessing for parallel parameter sweep
- Steady-state detection and early stopping
- Non-interactive matplotlib backend (no popup windows)
- All outputs labeled with fine_dx value
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import time as timer
import multiprocessing as mp
import csv
import pandas as pd
import sys

# Import functions from Functions.py
from Functions import (
    initalize_variables,
    intialize_equations,
    smooth_circular_profile,
    create_adaptive_mesh_for_simulation,
    calculate_total_amount
)

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

D_solution = 150.0  # μm²/s
D_gel = 60.0        # μm²/s
k_p = 0.2           # 1/s
k_d_ds = 3e-4       # 1/s
k_d_ss = 3e-4       # 1/s
k_slow = 1e5 * 1e-6 # 1/(μM·s)
k_fast = 1e6 * 1e-6 # 1/(μM·s)

# Initial concentrations (μM)
I1O2_init = 0.1     # 100 nM
I2_init = 0.1       # 100 nM
Th2_init = 5.0      # 5000 nM

# Geometry
node_diameter = 75.0    # μm
node_radius = node_diameter / 2.0
distance_between = 300.0  # μm center-to-center
total_width = 1e4         # 10000 μm = 1 cm
total_height = 1e3        # 1000 μm = 1 mm

# Fixed time step for convergence study
dt_fixed = 30.0  # seconds
total_time = 8 * 3600  # 8 hours in seconds

# Steady-state detection parameters
STEADY_STATE_THRESHOLD = 1e-8  # Relative change threshold
STEADY_STATE_WINDOW = 100      # Number of steps to check
CHECK_INTERVAL = 50            # Check every N steps

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_steady_state(values, threshold=STEADY_STATE_THRESHOLD, window=STEADY_STATE_WINDOW):
    """
    Detect if simulation has reached steady state.
    Returns True if the relative change over the window is below threshold.
    """
    if len(values) < window:
        return False
    
    recent_values = values[-window:]
    mean_val = np.mean(recent_values)
    
    if abs(mean_val) < 1e-12:  # Avoid division by zero
        return True
    
    max_change = np.max(np.abs(np.diff(recent_values)))
    relative_change = max_change / abs(mean_val)
    
    return relative_change < threshold


def save_timeseries_to_csv(time_points, I2_concentration, S2_free_concentration, 
                           S2_total_concentration, fine_dx, distance_between, dt):
    """Save time series data to CSV with fine_dx label in filename."""
    filename = f'timeseries_dx={fine_dx:.1f}_ccd={distance_between:.0f}_dt={dt:.0f}.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (hr)', 'I2 (μM)', 'S2_free (μM)', 'S2_total (μM)'])
        for t, i2, s2_free, s2_total in zip(time_points, I2_concentration, 
                                            S2_free_concentration, S2_total_concentration):
            writer.writerow([t, i2, s2_free, s2_total])
    
    print(f"  Saved: {filename}")
    return filename


def save_spatial_to_csv(x, y, S2_values, I2_values, Th2_values, fine_dx, distance_between):
    """Save final spatial distribution to CSV with fine_dx label in filename."""
    filename = f'spatial_dx={fine_dx:.1f}_ccd={distance_between:.0f}.csv'
    
    df = pd.DataFrame({
        'x (μm)': x,
        'y (μm)': y,
        'S2 (μM)': S2_values,
        'I2 (μM)': I2_values,
        'Th2 (μM)': Th2_values
    })
    df.to_csv(filename, index=False)
    
    print(f"  Saved: {filename}")
    return filename


def plot_timeseries(time_points, I2_concentration, S2_free_concentration, 
                   S2_total_concentration, fine_dx, distance_between, dt, converged):
    """Generate time series plot with fine_dx label."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    conv_status = "CONVERGED" if converged else "MAX TIME"
    fig.suptitle(f'Time Series (dx={fine_dx:.1f}μm, CCD={distance_between:.0f}μm, dt={dt:.0f}s) - {conv_status}', 
                 fontsize=14, weight='bold')
    
    # Plot 1: I2 concentration
    ax1.plot(time_points, np.array(I2_concentration) * 1000, 'b-', linewidth=2, label='I2')
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('I2 Concentration (nM)', fontsize=11)
    ax1.set_title('Receiver Node I2 Depletion', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: S2 concentrations
    ax2.plot(time_points, np.array(S2_free_concentration) * 1000, 'r-', 
             linewidth=2, label='S2 (free)')
    ax2.plot(time_points, np.array(S2_total_concentration) * 1000, 'g--', 
             linewidth=2, label='S2 (total)')
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('S2 Concentration (nM)', fontsize=11)
    ax2.set_title('Signal Strand S2 at Receiver', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    filename = f'timeseries_dx={fine_dx:.1f}_ccd={distance_between:.0f}_dt={dt:.0f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return filename


def plot_spatial(mesh, S2, I2, Th2, fine_dx, distance_between, 
                sender_center_x, receiver_center_x, center_y):
    """Generate spatial distribution plot with fine_dx label."""
    x = mesh.cellCenters[0].value
    y = mesh.cellCenters[1].value
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Final Spatial Distribution (dx={fine_dx:.1f}μm, CCD={distance_between:.0f}μm)', 
                 fontsize=14, weight='bold')
    
    # S2 concentration
    ax = axes[0]
    scatter = ax.tricontourf(x, y, S2.value * 1000, levels=20, cmap='viridis')
    ax.plot(sender_center_x, center_y, 'wo', markersize=10, label='Sender')
    ax.plot(receiver_center_x, center_y, 'ro', markersize=10, label='Receiver')
    ax.set_xlabel('x (μm)', fontsize=11)
    ax.set_ylabel('y (μm)', fontsize=11)
    ax.set_title('S2 Concentration (nM)', fontsize=12)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='S2 (nM)')
    ax.set_aspect('equal')
    
    # I2 concentration
    ax = axes[1]
    scatter = ax.tricontourf(x, y, I2.value * 1000, levels=20, cmap='plasma')
    ax.plot(sender_center_x, center_y, 'wo', markersize=10, label='Sender')
    ax.plot(receiver_center_x, center_y, 'ro', markersize=10, label='Receiver')
    ax.set_xlabel('x (μm)', fontsize=11)
    ax.set_ylabel('y (μm)', fontsize=11)
    ax.set_title('I2 Concentration (nM)', fontsize=12)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='I2 (nM)')
    ax.set_aspect('equal')
    
    # Th2 concentration
    ax = axes[2]
    scatter = ax.tricontourf(x, y, Th2.value * 1000, levels=20, cmap='inferno')
    ax.plot(sender_center_x, center_y, 'wo', markersize=10, label='Sender')
    ax.plot(receiver_center_x, center_y, 'ro', markersize=10, label='Receiver')
    ax.set_xlabel('x (μm)', fontsize=11)
    ax.set_ylabel('y (μm)', fontsize=11)
    ax.set_title('Th2 Concentration (nM)', fontsize=12)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Th2 (nM)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    filename = f'spatial_dx={fine_dx:.1f}_ccd={distance_between:.0f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return filename


# =============================================================================
# SINGLE SIMULATION FUNCTION (for multiprocessing)
# =============================================================================

def run_single_param(fine_dx):
    """
    Run a single simulation with given fine_dx value.
    Returns dictionary with results.
    """
    start_time = timer.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING SIMULATION: fine_dx = {fine_dx} μm")
    print(f"{'='*70}")
    
    # Calculate transition width as multiple of fine_dx
    transition_width = 3.0 * fine_dx
    
    # Create adaptive mesh
    print("Creating adaptive mesh...")
    mesh, sender_center_x, receiver_center_x, sender_center_y = create_adaptive_mesh_for_simulation(
        node_size=node_diameter,
        sender_center=None,
        receiver_center=None,
        fine_dx=fine_dx,
        coarse_dx=40.0,
        box_padding=200.0,
        transition_width=100.0,
        total_width=total_width,
        total_height=total_height
    )
    
    receiver_center_y = sender_center_y
    n_cells = mesh.numberOfCells
    
    print(f"  Mesh created: {n_cells} cells")
    print(f"  Sender at ({sender_center_x:.1f}, {sender_center_y:.1f})")
    print(f"  Receiver at ({receiver_center_x:.1f}, {receiver_center_y:.1f})")
    print(f"  Profile transition width: {transition_width:.1f} μm")
    
    # Get cell centers
    x = mesh.cellCenters[0].value
    y = mesh.cellCenters[1].value
    
    # Initialize variables using Functions.py
    
    S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(
        mesh, x, y,
        sender_center_x, receiver_center_x, receiver_center_y,
        node_radius, I2_init, Th2_init, I1O2_init,
        transition_width, D_gel, D_solution
    )
    
    # Initialize equations using Functions.py
    eq = intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2)
    
    # Find receiver center index for monitoring
    distances = np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2)
    receiver_center_idx = np.argmin(distances)
    
    # Time-stepping loop with steady-state detection
    
    time_points = []
    I2_concentration = []
    S2_free_concentration = []
    S2_total_concentration = []
    
    save_interval_steps = int(60.0 / dt_fixed)  # Save every 60 seconds
    converged = False
    step = 0
    max_steps = int(total_time / dt_fixed)
    
    while step < max_steps:
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        S2_I2.updateOld()
        S2_Th2.updateOld()
        
        res = 1e10
        sweep = 0
        max_sweeps = 10
        
        while res > 1e-6 and sweep < max_sweeps:
            res = eq.sweep(dt=dt_fixed)
            sweep += 1
        
        if step % save_interval_steps == 0:
            current_time = step * dt_fixed
            time_points.append(current_time / 3600)
            
            I2_val = I2.value[receiver_center_idx]
            S2_free_val = S2.value[receiver_center_idx]
            S2_total_val = (S2.value[receiver_center_idx] + 
                           S2_I2.value[receiver_center_idx] + 
                           S2_Th2.value[receiver_center_idx])
            
            I2_concentration.append(I2_val)
            S2_free_concentration.append(S2_free_val)
            S2_total_concentration.append(S2_total_val)
            
            if step % (save_interval_steps * 10) == 0:
                print(f"dx = {fine_dx}"
                    f"  t = {current_time/3600:.2f} hr: "
                      f"I2 = {I2_val*1000:.2f} nM, "
                      f"S2_total = {S2_total_val*1000:.2f} nM")
        
        # Check for steady state
        if step % CHECK_INTERVAL == 0 and step > 0:
            if detect_steady_state(I2_concentration):
                converged = True
                print(f"\n  ✓ STEADY STATE REACHED at t = {step * dt_fixed / 3600:.2f} hr")
                break
        
        step += 1
    
    final_time = step * dt_fixed
    runtime = timer.time() - start_time
    
    if not converged:
        print(f"\n  ⚠ Maximum time reached without convergence")


    print(f"\nSimulation complete for dx={fine_dx}:")
    print(f"  Total steps: {step}")
    print(f"  Final time: {final_time/3600:.2f} hours")
    print(f"  Runtime: {runtime:.2f} seconds")
    print(f"  Converged: {converged}")
    
    # Save results
    print("\nSaving results...")
    save_timeseries_to_csv(time_points, I2_concentration, S2_free_concentration,
                          S2_total_concentration, fine_dx, distance_between, dt_fixed)
    
    save_spatial_to_csv(x, y, S2.value, I2.value, Th2.value, 
                       fine_dx, distance_between)
    
    plot_timeseries(time_points, I2_concentration, S2_free_concentration,
                   S2_total_concentration, fine_dx, distance_between, dt_fixed, converged)
    
    #plot_spatial(mesh, S2, I2, Th2, fine_dx, distance_between,
                #sender_center_x, receiver_center_x, receiver_center_y)
    
    # Return results dictionary
    return {
        'fine_dx': fine_dx,
        'n_cells': n_cells,
        'n_steps': step,
        'final_time': final_time,
        'runtime': runtime,
        'converged': converged,
        'transition_width': transition_width,
        'time': time_points,
        'I2': I2_concentration,
        'S2_free': S2_free_concentration,
        'S2_total': S2_total_concentration
    }


# =============================================================================
# CONVERGENCE ANALYSIS FUNCTIONS
# =============================================================================

def save_results_to_csv(convergence_results, filename='convergence_results.csv'):
    """Save convergence study results to CSV."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fine_dx', 'n_cells', 'n_steps', 'final_time_hr', 
                        'runtime_s', 'converged', 'transition_width',
                        'final_I2', 'final_S2_free', 'final_S2_total'])
        
        for result in convergence_results:
            final_I2 = result['I2'][-1] if len(result['I2']) > 0 else float('nan')
            final_S2_free = result['S2_free'][-1] if len(result['S2_free']) > 0 else float('nan')
            final_S2_total = result['S2_total'][-1] if len(result['S2_total']) > 0 else float('nan')
            
            writer.writerow([
                result['fine_dx'],
                result['n_cells'],
                result['n_steps'],
                result['final_time'] / 3600,
                result['runtime'],
                result['converged'],
                result['transition_width'],
                final_I2,
                final_S2_free,
                final_S2_total
            ])
    
    print(f"\nConvergence results saved to: {filename}")


def plot_convergence_summary(convergence_results, total_parallel_time, n_processes):
    """Generate convergence study summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid
    fig.suptitle('Convergence Study Summary', fontsize=16, weight='bold')
    
    fine_dx_vals = [r['fine_dx'] for r in convergence_results]
    n_cells_vals = [r['n_cells'] for r in convergence_results]
    colors = ['green' if r['converged'] else 'red' for r in convergence_results]
    
    # Plot 1: I2 trajectories vs time for different fine_dx
    ax = axes[0, 0]
    for result in convergence_results:
        if len(result['time']) > 0:
            ax.plot(result['time'], np.array(result['I2']) * 1000,
                   label=f"dx = {result['fine_dx']:.1f} μm", 
                   linewidth=2, alpha=0.7)
    ax.axhline(y=75, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=25, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('[I2] at Receiver (nM)', fontsize=11)
    ax.set_title('I2 vs Time: Effect of Mesh Refinement', fontsize=12, weight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 2: Final I2 value vs fine_dx
    ax = axes[0, 1]
    final_I2_vals = [r['I2'][-1] * 1000 if len(r['I2']) > 0 else float('nan') 
                     for r in convergence_results]
    ax.scatter(fine_dx_vals, final_I2_vals, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.plot(fine_dx_vals, final_I2_vals, 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Fine mesh spacing (μm)', fontsize=11)
    ax.set_ylabel('Final [I2] (nM)', fontsize=11)
    ax.set_title('Spatial Convergence: Final I2 vs dx', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add secondary x-axis with cell count
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(fine_dx_vals)
    ax2.set_xticklabels([f"{n}" for n in n_cells_vals], fontsize=9)
    ax2.set_xlabel('Total cells', fontsize=10, color='gray')
    
    # Plot 3: Final total S2 value vs fine_dx
    ax = axes[0, 2]
    final_S2_vals = [r['S2_total'][-1] * 1000 if len(r['S2_total']) > 0 else float('nan') 
                     for r in convergence_results]
    ax.scatter(fine_dx_vals, final_S2_vals, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.plot(fine_dx_vals, final_S2_vals, 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Fine mesh spacing (μm)', fontsize=11)
    ax.set_ylabel('Final [S2] total (nM)', fontsize=11)
    ax.set_title('Spatial Convergence: Final S2 vs dx', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Computational runtime vs dx
    ax = axes[1, 0]
    runtimes = [r['runtime'] for r in convergence_results]
    ax.bar(range(len(fine_dx_vals)), runtimes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(fine_dx_vals)))
    ax.set_xticklabels([f"{dx:.1f}" for dx in fine_dx_vals], rotation=45)
    ax.set_xlabel('Fine mesh spacing (μm)', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('Computational Cost vs Mesh Refinement', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Number of time steps to convergence
    ax = axes[1, 1]
    n_steps_vals = [r['n_steps'] for r in convergence_results]
    ax.bar(range(len(fine_dx_vals)), n_steps_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(fine_dx_vals)))
    ax.set_xticklabels([f"{dx:.1f}" for dx in fine_dx_vals], rotation=45)
    ax.set_xlabel('Fine mesh spacing (μm)', fontsize=11)
    ax.set_ylabel('Number of time steps', fontsize=11)
    ax.set_title('Steps to Convergence', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    sequential_time = sum(runtimes)
    speedup = sequential_time / total_parallel_time if total_parallel_time > 0 else 0
    efficiency = speedup / n_processes * 100
    
    n_converged = sum(1 for r in convergence_results if r['converged'])
    
    summary_text = f"""
CONVERGENCE STUDY SUMMARY

Domain: {total_width/1e3:.0f} mm × {total_height/1e3:.0f} mm
Smooth tanh profiles

Parallel Performance:
  Processes: {n_processes}
  Parallel time: {total_parallel_time:.2f} s
  Sequential (est): {sequential_time:.2f} s
  Speedup: {speedup:.2f}×
  Efficiency: {efficiency:.1f}%

Parameter Ranges:
  fine_dx: {min(fine_dx_vals):.1f} - {max(fine_dx_vals):.1f} μm
  Cells: {min(n_cells_vals)} - {max(n_cells_vals)}
  
Convergence Status:
  Converged: {n_converged} / {len(convergence_results)}
  Time step: {dt_fixed} s
  
Legend:
  Green = Converged to steady state
  Red = Reached max time
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('convergence_summary.png', dpi=300, bbox_inches='tight')
    print("Convergence summary plot saved: convergence_summary.png")
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("ADAPTIVE MESH CONVERGENCE STUDY (WITH MULTIPROCESSING)")
    print("="*70)
    
    # Test different fine_dx values (spatial convergence)
    fine_dx_values = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]  # μm
    
    print(f"\nTesting {len(fine_dx_values)} fine_dx values: {fine_dx_values}")
    print(f"Fixed parameters:")
    print(f"  dt = {dt_fixed} s")
    print(f"  coarse_dx = 40.0 μm")
    print(f"  box_padding = 200.0 μm")
    print(f"  mesh_transition_width = 100.0 μm")
    print(f"  profile_transition_width = 3.0 × fine_dx")
    
    # Determine number of processes
    n_processes = max(1, mp.cpu_count() - 4)
    print(f"\nUsing {n_processes} parallel processes")
    print("="*70 + "\n")
    
    # Run simulations in parallel
    start_parallel = timer.time()
    
    with mp.Pool(processes=n_processes) as pool:
        convergence_results = pool.map(run_single_param, fine_dx_values)
    
    total_parallel_time = timer.time() - start_parallel
    
    print("\n" + "="*70)
    print(f"All simulations completed in {total_parallel_time:.2f} seconds")
    print("="*70)
    
    # ==========================================================================
    # SAVE RESULTS TO CSV
    # ==========================================================================
    
    save_results_to_csv(convergence_results, 
                       filename='convergence_adaptive_mesh_smooth_profiles.csv')
    
    # ==========================================================================
    # CONVERGENCE STUDY SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("CONVERGENCE STUDY SUMMARY")
    print("="*70)
    print(f"{'fine_dx':<10} {'Cells':<10} {'Steps':<8} {'Time (hr)':<10} "
          f"{'Final I2':<12} {'Runtime (s)':<12} {'Conv'}")
    print("-"*70)
    for result in convergence_results:
        final_I2_val = result['I2'][-1] if len(result['I2']) > 0 else float('nan')
        print(f"{result['fine_dx']:<10.1f} {result['n_cells']:<10} "
              f"{result['n_steps']:<8} {result['final_time']/3600:<10.3f} "
              f"{final_I2_val:<12.6f} {result['runtime']:<12.2f} {result['converged']}")
    print("="*70)
    
    # Recommend optimal parameters
    print("\nRECOMMENDATIONS:")
    print("-"*70)
    
    converged_results = [r for r in convergence_results 
                        if r['converged'] and len(r['I2']) > 0]
    
    if converged_results:
        # Sort by runtime
        converged_results.sort(key=lambda x: x['runtime'])
        
        optimal = converged_results[0]
        print(f"✓ Optimal parameters: fine_dx = {optimal['fine_dx']} μm")
        print(f"  - Mesh cells: {optimal['n_cells']}")
        print(f"  - Profile transition width: {optimal['transition_width']:.1f} μm")
        print(f"  - Converged in {optimal['n_steps']} steps ({optimal['final_time']/3600:.2f} hr)")
        print(f"  - Runtime: {optimal['runtime']:.2f} seconds")
        print(f"  - Final I2: {optimal['I2'][-1]*1000:.2f} nM")
        
        # Check if finer meshes give different results
        finest = min(converged_results, key=lambda x: x['fine_dx'])
        if finest['fine_dx'] != optimal['fine_dx']:
            I2_diff = abs(finest['I2'][-1] - optimal['I2'][-1]) / optimal['I2'][-1] * 100
            print(f"\n  Comparison with finest mesh (dx={finest['fine_dx']:.1f} μm):")
            print(f"  - I2 difference: {I2_diff:.2f}%")
            if I2_diff < 1.0:
                print(f"  → Optimal mesh is accurate (<1% error)")
            else:
                print(f"  → Consider using finer mesh for higher accuracy")
    else:
        print("⚠ No simulations converged to steady state")
        print("  Consider:")
        print("  - Increasing total_time")
        print("  - Adjusting steady-state threshold")
        print("  - Checking initial conditions")
    
    print("="*70)
    
    # ==========================================================================
    # GENERATE CONVERGENCE SUMMARY PLOT
    # ==========================================================================
    
    plot_convergence_summary(convergence_results, total_parallel_time, n_processes)
    
    print("\n" + "="*70)
    print("CONVERGENCE STUDY COMPLETE")
    print("="*70)