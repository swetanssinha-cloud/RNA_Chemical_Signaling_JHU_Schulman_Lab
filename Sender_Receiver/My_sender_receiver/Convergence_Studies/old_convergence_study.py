"""
2D Tethered Genelet Reaction-Diffusion Model using FiPy
WITH MULTIPROCESSING, CSV OUTPUT, AND CIRCULAR NODE GEOMETRY
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import time as timer
import multiprocessing as mp
from functools import partial
import csv
import os

# =============================================================================
# PARAMETERS (from Supplementary Table 1)
# =============================================================================

# Diffusion coefficients (μm²/s)
D_solution = 150.0  # RNA diffusion in solution
D_gel = 60.0        # RNA diffusion in hydrogel

# Reaction rates
k_p = 0.2           # Transcription rate (1/s)
k_d_ds = 3e-4       # Degradation rate of double-stranded RNA (1/s)
k_d_ss = 3e-4       # Degradation rate of single-stranded RNA (1/s)
k_slow = 1e5 * 1e-6 # 5bp toehold (converted to μM⁻¹s⁻¹)
k_fast = 1e6 * 1e-6 # 7bp toehold (converted to μM⁻¹s⁻¹)

# Initial concentrations (μM)
I1O2_init = 0.1     # 100 nM sender switch
I2_init = 0.1       # 100 nM receiver switch
Th2_init = 5.0      # 5 μM threshold (5000 nM)

# Geometry (μm)
node_diameter = 75.0
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 1500  # Center-to-center distance

total_width = 1e4    # 10 mm = 10000 μm
total_height = 1e3   # 1 mm = 1000 μm

# Time parameters
dt_initial = 1.0    # Start with 1 second timestep (we'll test this)
max_time = 8 * 3600 # Maximum 8 hours
check_interval = 100 # Check for steady state every N steps

# Steady-state detection parameters
ss_tolerance = 1e-8  # Relative change threshold for steady state
ss_window = 50       # Number of timesteps to check for steady state

# =============================================================================
# FUNCTION TO BUILD AND RUN MODEL
# =============================================================================

def run_simulation(dt, dx=40, max_time=max_time, check_steady_state=True, verbose=True):
    """
    Run the 2D tethered genelet simulation with circular nodes.
    
    Parameters:
    -----------
    dt : float
        Timestep size (seconds)
    dx : float
        Spatial resolution (μm)
    max_time : float
        Maximum simulation time (seconds)
    check_steady_state : bool
        If True, stop when steady state is reached
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict with results: time_points, I2, S2_free, S2_total, converged, runtime, dx, dt
    """
    
    start_time = timer.time()
    
    # =========================================================================
    # 2D MESH SETUP
    # =========================================================================
    
    dy = dx  # Use same resolution in both directions
    
    nx = int(total_width // dx)
    ny = int(total_height // dy)
    
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    
    # Get cell centers
    x, y = mesh.cellCenters
    
    # Center of domain
    center_x = total_width / 2
    center_y = total_height / 2
    
    # Sender node: circular at left
    sender_center_x = center_x - distance_between / 2
    sender_center_y = center_y
    
    # Receiver node: circular at right  
    receiver_center_x = center_x + distance_between / 2
    receiver_center_y = center_y
    
    # CIRCULAR MASKS using distance formula
    sender_mask = (np.sqrt((x - sender_center_x)**2 + (y - sender_center_y)**2) <= node_radius)
    receiver_mask = (np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2) <= node_radius)
    gel_mask = sender_mask | receiver_mask
    
    if verbose:
        print(f"\n2D Simulation Setup:")
        print(f"  Mesh: {nx} × {ny} = {nx*ny} cells")
        print(f"  Cell size: dx = dy = {dx} μm")
        print(f"  Domain: {total_width} × {total_height} μm²")
        print(f"  Node diameter: {node_diameter} μm (circular)")
        print(f"  Distance: {distance_between} μm (center-to-center)")
        print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
        print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
    
    # =========================================================================
    # DEFINE CELL VARIABLES
    # =========================================================================
    
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
    
    # =========================================================================
    # DEFINE EQUATIONS (2D diffusion!)
    # =========================================================================
    
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
    
    # =========================================================================
    # TIME STEPPING WITH STEADY-STATE DETECTION
    # =========================================================================
    
    # Find receiver center index (2D)
    distances_to_receiver = np.sqrt((x - receiver_center_x)**2 + 
                                    (y - receiver_center_y)**2)
    receiver_center_idx = np.argmin(distances_to_receiver)
    
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
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running simulation with dt = {dt} s, dx = {dx} μm")
        print(f"{'='*70}\n")
    
    while current_time < max_time:
        # Update old values
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        S2_I2.updateOld()
        S2_Th2.updateOld()
        
        # Store pre-step values for change calculation
        S2_old_vals = S2.value.copy()
        I2_old_vals = I2.value.copy()
        Th2_old_vals = Th2.value.copy()
        S2_I2_old_vals = S2_I2.value.copy()
        S2_Th2_old_vals = S2_Th2.value.copy()
        
        # Solve equations
        res = 1e10
        sweep = 0
        max_sweeps = 10
        tolerance = 1e-6
        
        while res > tolerance and sweep < max_sweeps:
            res = eq.sweep(dt=dt)
            sweep += 1
        
        current_time += dt
        step += 1
        
        # Store data periodically
        if step % check_interval == 0:
            time_points.append(current_time / 3600)
            I2_concentration.append(I2.value[receiver_center_idx])
            S2_free_concentration.append(S2.value[receiver_center_idx])
            
            S2_total = (S2.value[receiver_center_idx] + 
                       S2_I2.value[receiver_center_idx] + 
                       S2_Th2.value[receiver_center_idx])
            S2_total_concentration.append(S2_total)
            
            if verbose:
                print(f"t = {current_time/3600:.3f} hr (step {step}): "
                      f"I2 = {I2_concentration[-1]:.6f} μM, "
                      f"S2_free = {S2_free_concentration[-1]:.6f} μM, "
                      f"residual = {res:.2e}")
        
        # =====================================================================
        # STEADY-STATE DETECTION
        # =====================================================================
        
        if check_steady_state and step % check_interval == 0:
            # Calculate maximum relative change across all variables
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
    
    elapsed_time = timer.time() - start_time
    
    if verbose:
        if not converged_to_ss:
            print(f"\nMaximum time reached ({max_time/3600:.1f} hours)")
        print(f"Simulation runtime: {elapsed_time:.2f} seconds")
        print(f"Total steps: {step}")
        print(f"Final time: {current_time/3600:.3f} hours\n")
    
    return {
        'time_points': np.array(time_points),
        'I2': np.array(I2_concentration),
        'S2_free': np.array(S2_free_concentration),
        'S2_total': np.array(S2_total_concentration),
        'converged': converged_to_ss,
        'final_time': current_time,
        'runtime': elapsed_time,
        'n_steps': step,
        'dt': dt,
        'dx': dx,
        'nx': nx,
        'ny': ny
    }


# =============================================================================
# WRAPPER FUNCTION FOR MULTIPROCESSING
# =============================================================================

def run_single_param(params):
    """
    Wrapper function to run a single simulation for multiprocessing.
    
    Parameters:
    -----------
    params : tuple
        (dt, dx) - timestep and spatial resolution
    """
    dt_test, dx_test = params
    print(f"Starting simulation with dt = {dt_test} s, dx = {dx_test} μm...")
    result = run_simulation(dt=dt_test, dx=dx_test, check_steady_state=True, verbose=False)
    print(f"Completed dt = {dt_test} s, dx = {dx_test} μm: "
          f"Converged={result['converged']}, Runtime={result['runtime']:.2f}s, Steps={result['n_steps']}")
    return result


# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

def save_results_to_csv(convergence_results, filename='convergence_study_2D_results.csv'):
    """
    Save convergence study results to a CSV file.
    
    Parameters:
    -----------
    convergence_results : list of dict
        Results from convergence study
    filename : str
        Output CSV filename
    """
    
    # Create summary CSV with key metrics
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['dt_s', 'dx_um', 'nx', 'ny', 'n_cells', 'n_steps', 
                      'final_time_hr', 'final_I2_uM', 'final_S2_free_uM', 
                      'final_S2_total_uM', 'runtime_s', 'converged']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in convergence_results:
            final_I2_val = result['I2'][-1] if len(result['I2']) > 0 else np.nan
            final_S2_free_val = result['S2_free'][-1] if len(result['S2_free']) > 0 else np.nan
            final_S2_total_val = result['S2_total'][-1] if len(result['S2_total']) > 0 else np.nan
            
            writer.writerow({
                'dt_s': result['dt'],
                'dx_um': result['dx'],
                'nx': result['nx'],
                'ny': result['ny'],
                'n_cells': result['nx'] * result['ny'],
                'n_steps': result['n_steps'],
                'final_time_hr': result['final_time']/3600,
                'final_I2_uM': final_I2_val,
                'final_S2_free_uM': final_S2_free_val,
                'final_S2_total_uM': final_S2_total_val,
                'runtime_s': result['runtime'],
                'converged': result['converged']
            })
    
    print(f"\n✓ Summary results saved to: {filename}")
    
    # Save detailed time series data for each parameter combination
    base_name = filename.rsplit('.', 1)[0]
    for result in convergence_results:
        if len(result['time_points']) > 0:
            detail_filename = f"{base_name}_dt_{result['dt']:.1f}s_dx_{result['dx']:.0f}um_timeseries.csv"
            with open(detail_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time_hr', 'I2_uM', 'S2_free_uM', 'S2_total_uM'])
                for i in range(len(result['time_points'])):
                    writer.writerow([
                        result['time_points'][i],
                        result['I2'][i],
                        result['S2_free'][i],
                        result['S2_total'][i]
                    ])
            print(f"  - Time series saved to: {detail_filename}")


# =============================================================================
# CONVERGENCE STUDY WITH MULTIPROCESSING
# =============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("2D CONVERGENCE STUDY (WITH MULTIPROCESSING)")
    print("="*70)
    
    # Test different timesteps and spatial resolutions
    dt_values = [0.25,0.5,1.0, 2.0, 5.0, 10.0, 20.0, 30.0]  # seconds
    dx_values = [40]  # Start with single dx value; can add [40, 50, 60] for spatial convergence

    dt_values=[2]
    dx_values = [0.25,1,2,5,10,20,30]
    
    # Create parameter combinations
    param_combinations = [(dt, dx) for dt in dt_values for dx in dx_values]
    
    # Determine number of processes
    n_processes = max(1, mp.cpu_count() - 3)
    print(f"\nUsing {n_processes} parallel processes")
    print(f"Testing {len(param_combinations)} parameter combinations:")
    print(f"  dt values: {dt_values}")
    print(f"  dx values: {dx_values}")
    print("="*70 + "\n")
    
    # Run simulations in parallel
    start_parallel = timer.time()
    
    with mp.Pool(processes=n_processes) as pool:
        convergence_results = pool.map(run_single_param, param_combinations)
    
    total_parallel_time = timer.time() - start_parallel
    
    print("\n" + "="*70)
    print(f"All simulations completed in {total_parallel_time:.2f} seconds")
    print("="*70)
    
    # ==========================================================================
    # SAVE RESULTS TO CSV
    # ==========================================================================
    
    save_results_to_csv(convergence_results, filename='convergence_study_2D_results.csv')
    
    # ==========================================================================
    # CONVERGENCE STUDY SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("CONVERGENCE STUDY SUMMARY")
    print("="*70)
    print(f"{'dt (s)':<8} {'dx (μm)':<8} {'Mesh':<12} {'Steps':<8} {'Time (hr)':<10} "
          f"{'Final I2':<12} {'Runtime (s)':<12} {'Conv'}")
    print("-"*70)
    for result in convergence_results:
        final_I2_val = result['I2'][-1] if len(result['I2']) > 0 else float('nan')
        mesh_str = f"{result['nx']}×{result['ny']}"
        print(f"{result['dt']:<8.1f} {result['dx']:<8.0f} {mesh_str:<12} "
              f"{result['n_steps']:<8} {result['final_time']/3600:<10.3f} "
              f"{final_I2_val:<12.6f} {result['runtime']:<12.2f} {result['converged']}")
    print("="*70)
    
    # Recommend optimal parameters
    print("\nRECOMMENDATIONS:")
    print("-"*70)
    
    # Find timesteps that converged
    converged_results = [r for r in convergence_results if r['converged'] and len(r['I2']) > 0]
    
    if converged_results:
        # Sort by runtime
        converged_results.sort(key=lambda x: x['runtime'])
        
        optimal = converged_results[0]
        print(f"✓ Optimal parameters: dt = {optimal['dt']} s, dx = {optimal['dx']} μm")
        print(f"  - Mesh: {optimal['nx']} × {optimal['ny']} = {optimal['nx']*optimal['ny']} cells")
        print(f"  - Reaches steady state in {optimal['final_time']/3600:.2f} hours")
        print(f"  - Computation time: {optimal['runtime']:.2f} seconds")
        print(f"  - Final I2 concentration: {optimal['I2'][-1]:.6f} μM")
        
        # Check accuracy
        final_I2 = [r['I2'][-1] if len(r['I2']) > 0 else np.nan for r in convergence_results]
        if len(final_I2) > 1 and not np.isnan(final_I2[0]):
            relative_diff = abs(optimal['I2'][-1] - final_I2[0]) / final_I2[0]
            print(f"  - Relative error vs finest grid: {relative_diff:.2e}")
            
            if relative_diff < 1e-3:
                print(f"  ✓ Excellent accuracy (< 0.1% error)")
            elif relative_diff < 1e-2:
                print(f"  ✓ Good accuracy (< 1% error)")
            else:
                print(f"  ⚠ Consider smaller timestep for better accuracy")
    else:
        print("⚠ No simulations reached steady state. Consider:")
        print("  - Increasing max_time")
        print("  - Relaxing ss_tolerance")
        print("  - Using smaller timestep")
    
    print("="*70 + "\n")
    
    # ==========================================================================
    # CONVERGENCE ANALYSIS PLOTS
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: I2 trajectories for different dt
    ax = axes[0, 0]
    for result in convergence_results:
        if len(result['time_points']) > 0:
            label = f"dt={result['dt']}s, dx={result['dx']}μm"
            ax.plot(result['time_points'], result['I2'], 
                    label=label, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('[I2] (μM)', fontsize=11)
    ax.set_title('I2 Convergence: Effect of Timestep (2D)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final I2 value vs dt
    ax = axes[0, 1]
    final_I2 = [r['I2'][-1] if len(r['I2']) > 0 else np.nan for r in convergence_results]
    dt_vals = [r['dt'] for r in convergence_results]
    ax.plot(dt_vals, final_I2, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Timestep dt (s)', fontsize=11)
    ax.set_ylabel('Final [I2] (μM)', fontsize=11)
    ax.set_title('Steady-State I2 vs Timestep (2D)', fontsize=13)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Computational efficiency
    ax = axes[1, 0]
    runtimes = [r['runtime'] for r in convergence_results]
    ax.plot(dt_vals, runtimes, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Timestep dt (s)', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('Computational Cost vs Timestep (2D)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speedup from parallelization
    ax = axes[1, 1]
    sequential_time = sum(runtimes)
    speedup = sequential_time / total_parallel_time
    efficiency = speedup / n_processes * 100
    
    ax.text(0.5, 0.7, f"2D Parallel Execution Summary", 
            ha='center', va='center', fontsize=14, weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.55, f"Domain: {total_width/1e3:.0f} mm × {total_height/1e3:.0f} mm", 
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.5, 0.45, f"Number of processes: {n_processes}", 
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.5, 0.35, f"Total parallel time: {total_parallel_time:.2f} s", 
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.5, 0.25, f"Sequential time (est.): {sequential_time:.2f} s", 
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.text(0.5, 0.15, f"Speedup: {speedup:.2f}×", 
            ha='center', va='center', fontsize=11, color='green', weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.05, f"Efficiency: {efficiency:.1f}%", 
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('convergence_study_2D_analysis_spatial.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to: convergence_study_2D_analysis.png\n")
    plt.show()