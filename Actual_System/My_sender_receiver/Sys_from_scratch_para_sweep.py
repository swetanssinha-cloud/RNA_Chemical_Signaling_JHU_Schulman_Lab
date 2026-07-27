"""
Tethered Genelet Reaction-Diffusion Model - Parameter Sweep with Multiprocessing
Runs multiple simulations in parallel, sweeping over a specified parameter
This is for Charlie Chen's Paper on DNA circuts written in COMSOL in 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import time as timer
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SWEEP CONFIGURATION - USER FILLS THESE IN
# =============================================================================

SWEEP_PARAMETER = "distance_between"  # Example: "k_p", "Th2_init", "distance_between", etc.
SWEEP_VALUES = [100,300,500,700,1000,1500]  # List of values to test

# Number of repeats per parameter value (for error bars)
N_REPEATS = 1  # Set to 1 if no error bars needed

# Number of parallel processes (None = use all available cores)
N_PROCESSES = cpu_count() -4  # Or set to a specific number like 4, 8, etc.

# =============================================================================
# DEFAULT PARAMETERS (from Supplementary Table 1)
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
Th2_init = 5.0      # 5 μM threshold

# Geometry (μm)
node_size = 50.0
distance_between = 300.0
total_length = 5000.0

# Time parameters
dt_initial = 1.0    # Start with 1 second timestep
max_time = 8 * 3600 # Maximum 8 hours
check_interval = 100 # Check for steady state every N steps

# Steady-state detection parameters
ss_tolerance = 1e-8  # Relative change threshold for steady state
ss_window = 50       # Number of timesteps to check for steady state

# =============================================================================
# SIMULATION FUNCTION
# =============================================================================

def run_single_simulation(params):
    """
    Run a single simulation with given parameters.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all simulation parameters
    
    Returns:
    --------
    dict with results
    """
    
    # Extract parameters
    D_solution = params['D_solution']
    D_gel = params['D_gel']
    k_p = params['k_p']
    k_d_ds = params['k_d_ds']
    k_d_ss = params['k_d_ss']
    k_slow = params['k_slow']
    k_fast = params['k_fast']
    I1O2_init = params['I1O2_init']
    I2_init = params['I2_init']
    Th2_init = params['Th2_init']
    node_size = params['node_size']
    distance_between = params['distance_between']
    total_length = params['total_length']
    dt = params['dt']
    max_time = params['max_time']
    check_interval = params['check_interval']
    ss_tolerance = params['ss_tolerance']
    ss_window = params['ss_window']
    sweep_value = params['sweep_value']
    repeat_idx = params['repeat_idx']
    
    start_time = timer.time()
    
    try:
        # =====================================================================
        # MESH SETUP
        # =====================================================================
        
        nx = 400
        dx = total_length / nx
        mesh = Grid2D(nx=nx, dx=dx)
        x = mesh.cellCenters[0]
        
        # Define node regions
        sender_center = total_length / 2 - distance_between / 2
        receiver_center = total_length / 2 + distance_between / 2
        
        sender_mask = (x >= sender_center - node_size/2) & (x <= sender_center + node_size/2)
        receiver_mask = (x >= receiver_center - node_size/2) & (x <= receiver_center + node_size/2)
        gel_mask = sender_mask | receiver_mask
        
        # =====================================================================
        # DEFINE CELL VARIABLES
        # =====================================================================
        
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
        
        # =====================================================================
        # DEFINE EQUATIONS
        # =====================================================================
        
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
        
        # =====================================================================
        # TIME STEPPING WITH STEADY-STATE DETECTION
        # =====================================================================
        
        receiver_center_idx = np.argmin(np.abs(x.value - receiver_center))
        
        # Storage
        time_points = []
        I2_concentration = []
        S2_free_concentration = []
        S2_total_concentration = []
        
        # For steady-state detection
        recent_changes = []
        
        current_time = 0.0
        step = 0
        converged_to_ss = False
        
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
            
            # =================================================================
            # STEADY-STATE DETECTION
            # =================================================================
            
            if step % check_interval == 0:
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
                        break
        
        elapsed_time = timer.time() - start_time
        
        # =====================================================================
        # CALCULATE OUTPUT METRICS
        # =====================================================================
        
        time_points = np.array(time_points)
        I2_concentration = np.array(I2_concentration)
        S2_free_concentration = np.array(S2_free_concentration)
        S2_total_concentration = np.array(S2_total_concentration)
        
        # Final values (last time point)
        final_I2 = I2_concentration[-1] if len(I2_concentration) > 0 else np.nan
        final_S2 = S2_free_concentration[-1] if len(S2_free_concentration) > 0 else np.nan
        final_S2_total = S2_total_concentration[-1] if len(S2_total_concentration) > 0 else np.nan
        
        # Time to reach halfway point for I2
        # Halfway between initial and final I2
        I2_halfway = (I2_init + final_I2) / 2
        
        # Find time when I2 crosses halfway point
        time_to_half = np.nan
        if len(I2_concentration) > 1:
            # Find first point where I2 drops below halfway
            below_half = I2_concentration < I2_halfway
            if np.any(below_half):
                idx = np.argmax(below_half)
                if idx > 0:
                    # Linear interpolation for more accurate time
                    t1, t2 = time_points[idx-1], time_points[idx]
                    c1, c2 = I2_concentration[idx-1], I2_concentration[idx]
                    time_to_half = t1 + (I2_halfway - c1) * (t2 - t1) / (c2 - c1)
                else:
                    time_to_half = time_points[idx]
        
        print(f"  {SWEEP_PARAMETER}={sweep_value:.4g}, repeat {repeat_idx+1}/{N_REPEATS}: "
              f"final_I2={final_I2:.6f}, t_half={time_to_half:.3f}h, "
              f"runtime={elapsed_time:.1f}s, converged={converged_to_ss}")
        
        return {
            'sweep_value': sweep_value,
            'repeat_idx': repeat_idx,
            'final_I2': final_I2,
            'final_S2': final_S2,
            'final_S2_total': final_S2_total,
            'time_to_half': time_to_half,
            'wall_time': elapsed_time,
            'converged': converged_to_ss,
            'time_points': time_points,
            'I2_concentration': I2_concentration,
            'S2_free_concentration': S2_free_concentration,
            'S2_total_concentration': S2_total_concentration
        }
        
    except Exception as e:
        print(f"  ERROR: {SWEEP_PARAMETER}={sweep_value:.4g}, repeat {repeat_idx}: {str(e)}")
        return {
            'sweep_value': sweep_value,
            'repeat_idx': repeat_idx,
            'final_I2': np.nan,
            'final_S2': np.nan,
            'final_S2_total': np.nan,
            'time_to_half': np.nan,
            'wall_time': np.nan,
            'converged': False
        }

# =============================================================================
# PARAMETER SWEEP SETUP
# =============================================================================

def create_parameter_dict(sweep_value, repeat_idx):
    """Create parameter dictionary with swept parameter updated."""
    
    params = {
        'D_solution': D_solution,
        'D_gel': D_gel,
        'k_p': k_p,
        'k_d_ds': k_d_ds,
        'k_d_ss': k_d_ss,
        'k_slow': k_slow,
        'k_fast': k_fast,
        'I1O2_init': I1O2_init,
        'I2_init': I2_init,
        'Th2_init': Th2_init,
        'node_size': node_size,
        'distance_between': distance_between,
        'total_length': total_length,
        'dt': dt_initial,
        'max_time': max_time,
        'check_interval': check_interval,
        'ss_tolerance': ss_tolerance,
        'ss_window': ss_window,
        'sweep_value': sweep_value,
        'repeat_idx': repeat_idx
    }
    
    # Update the swept parameter
    if SWEEP_PARAMETER in params:
        params[SWEEP_PARAMETER] = sweep_value
    else:
        raise ValueError(f"Unknown parameter: {SWEEP_PARAMETER}")
    
    return params

# =============================================================================
# RUN PARAMETER SWEEP
# =============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*80)
    print("PARAMETER SWEEP WITH MULTIPROCESSING")
    print("="*80)
    print(f"Sweeping parameter: {SWEEP_PARAMETER}")
    print(f"Values: {SWEEP_VALUES}")
    print(f"Repeats per value: {N_REPEATS}")
    print(f"Total simulations: {len(SWEEP_VALUES) * N_REPEATS}")
    
    # Determine number of processes
    if N_PROCESSES is None:
        n_processes = cpu_count()
    else:
        n_processes = min(N_PROCESSES, cpu_count())
    
    print(f"Using {n_processes} parallel processes")
    print("="*80 + "\n")
    
    # Create list of all parameter combinations
    param_list = []
    for sweep_value in SWEEP_VALUES:
        for repeat_idx in range(N_REPEATS):
            param_list.append(create_parameter_dict(sweep_value, repeat_idx))
    
    # Run simulations in parallel
    total_start = timer.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, param_list)
    
    total_time = timer.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"All simulations complete! Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # ==========================================================================
    # PROCESS RESULTS - CALCULATE MEANS AND STANDARD ERRORS
    # ==========================================================================
    
    # Organize results by sweep value
    results_by_value = {val: [] for val in SWEEP_VALUES}
    for result in results:
        results_by_value[result['sweep_value']].append(result)
    
    # Calculate statistics
    sweep_vals = []
    final_I2_mean = []
    final_I2_err = []
    final_S2_mean = []
    final_S2_err = []
    final_S2_total_mean = []
    final_S2_total_err = []
    time_to_half_mean = []
    time_to_half_err = []
    wall_time_mean = []
    wall_time_err = []
    
    for val in SWEEP_VALUES:
        repeats = results_by_value[val]
        
        # Extract data for this parameter value
        I2_vals = [r['final_I2'] for r in repeats if not np.isnan(r['final_I2'])]
        S2_vals = [r['final_S2'] for r in repeats if not np.isnan(r['final_S2'])]
        S2_tot_vals = [r['final_S2_total'] for r in repeats if not np.isnan(r['final_S2_total'])]
        t_half_vals = [r['time_to_half'] for r in repeats if not np.isnan(r['time_to_half'])]
        wt_vals = [r['wall_time'] for r in repeats if not np.isnan(r['wall_time'])]
        
        sweep_vals.append(val)
        
        # Calculate means and standard errors
        final_I2_mean.append(np.mean(I2_vals) if len(I2_vals) > 0 else np.nan)
        final_I2_err.append(np.std(I2_vals) / np.sqrt(len(I2_vals)) if len(I2_vals) > 1 else 0)
        
        final_S2_mean.append(np.mean(S2_vals) if len(S2_vals) > 0 else np.nan)
        final_S2_err.append(np.std(S2_vals) / np.sqrt(len(S2_vals)) if len(S2_vals) > 1 else 0)
        
        final_S2_total_mean.append(np.mean(S2_tot_vals) if len(S2_tot_vals) > 0 else np.nan)
        final_S2_total_err.append(np.std(S2_tot_vals) / np.sqrt(len(S2_tot_vals)) if len(S2_tot_vals) > 1 else 0)
        
        time_to_half_mean.append(np.mean(t_half_vals) if len(t_half_vals) > 0 else np.nan)
        time_to_half_err.append(np.std(t_half_vals) / np.sqrt(len(t_half_vals)) if len(t_half_vals) > 1 else 0)
        
        wall_time_mean.append(np.mean(wt_vals) if len(wt_vals) > 0 else np.nan)
        wall_time_err.append(np.std(wt_vals) / np.sqrt(len(wt_vals)) if len(wt_vals) > 1 else 0)
    
    sweep_vals = np.array(sweep_vals)
    final_I2_mean = np.array(final_I2_mean)
    final_I2_err = np.array(final_I2_err)
    final_S2_mean = np.array(final_S2_mean)
    final_S2_err = np.array(final_S2_err)
    final_S2_total_mean = np.array(final_S2_total_mean)
    final_S2_total_err = np.array(final_S2_total_err)
    time_to_half_mean = np.array(time_to_half_mean)
    time_to_half_err = np.array(time_to_half_err)
    wall_time_mean = np.array(wall_time_mean)
    wall_time_err = np.array(wall_time_err)
    
    # ==========================================================================
    # PLOTTING
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Parameter Sweep: {SWEEP_PARAMETER}', fontsize=16, fontweight='bold')
    
    # Plot 1: Final I2 vs Parameter
    ax = axes[0, 0]
    ax.errorbar(sweep_vals, final_I2_mean, yerr=final_I2_err, 
                marker='o', markersize=8, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel(f'{SWEEP_PARAMETER}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final [I2] (μM)', fontsize=12, fontweight='bold')
    ax.set_title('Steady-State I2 Concentration', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final S2 vs Parameter
    ax = axes[0, 1]
    ax.errorbar(sweep_vals, final_S2_mean, yerr=final_S2_err, 
                marker='s', markersize=8, linewidth=2, capsize=5, capthick=2, color='green')
    ax.set_xlabel(f'{SWEEP_PARAMETER}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final [S2] free (μM)', fontsize=12, fontweight='bold')
    ax.set_title('Steady-State Free S2', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final S2 Total vs Parameter
    ax = axes[0, 2]
    ax.errorbar(sweep_vals, final_S2_total_mean, yerr=final_S2_total_err, 
                marker='^', markersize=8, linewidth=2, capsize=5, capthick=2, color='purple')
    ax.set_xlabel(f'{SWEEP_PARAMETER}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final [S2] total (μM)', fontsize=12, fontweight='bold')
    ax.set_title('Steady-State Total S2', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Time to Half vs Parameter
    ax = axes[1, 0]
    ax.errorbar(sweep_vals, time_to_half_mean, yerr=time_to_half_err, 
                marker='D', markersize=8, linewidth=2, capsize=5, capthick=2, color='red')
    ax.set_xlabel(f'{SWEEP_PARAMETER}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time to I2 Halfway (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Response Time', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Wall Time vs Parameter
    ax = axes[1, 1]
    ax.errorbar(sweep_vals, wall_time_mean, yerr=wall_time_err, 
                marker='*', markersize=10, linewidth=2, capsize=5, capthick=2, color='orange')
    ax.set_xlabel(f'{SWEEP_PARAMETER}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wall Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Cost', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Example time traces (first and last parameter values)
    ax = axes[1, 2]
    # Plot first parameter value
    first_results = results_by_value[SWEEP_VALUES[0]]
    for r in first_results:
        if len(r['time_points']) > 0:
            ax.plot(r['time_points'], r['I2_concentration'], 
                   'b-', alpha=0.5, linewidth=1, label=f'{SWEEP_PARAMETER}={SWEEP_VALUES[0]:.3g}')
            break
    
    # Plot last parameter value
    last_results = results_by_value[SWEEP_VALUES[-1]]
    for r in last_results:
        if len(r['time_points']) > 0:
            ax.plot(r['time_points'], r['I2_concentration'], 
                   'r-', alpha=0.5, linewidth=1, label=f'{SWEEP_PARAMETER}={SWEEP_VALUES[-1]:.3g}')
            break
    
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('[I2] (μM)', fontsize=12, fontweight='bold')
    ax.set_title('Example Time Traces', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'Chen_Sys_parameter_sweep_{SWEEP_PARAMETER}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==========================================================================
    # SAVE RESULTS TO FILE
    # ==========================================================================
    
    output_filename = f'sweep_results_{SWEEP_PARAMETER}.txt'
    with open(output_filename, 'w') as f:
        f.write(f"Parameter Sweep Results: {SWEEP_PARAMETER}\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Value':<15} {'Final_I2':<15} {'I2_err':<15} {'Final_S2':<15} {'S2_err':<15} "
                f"{'S2_total':<15} {'S2tot_err':<15} {'t_half':<15} {'t_err':<15} "
                f"{'Wall_time':<15} {'WT_err':<15}\n")
        f.write("-"*180 + "\n")
        
        for i, val in enumerate(sweep_vals):
            f.write(f"{val:<15.6g} {final_I2_mean[i]:<15.6g} {final_I2_err[i]:<15.6g} "
                   f"{final_S2_mean[i]:<15.6g} {final_S2_err[i]:<15.6g} "
                   f"{final_S2_total_mean[i]:<15.6g} {final_S2_total_err[i]:<15.6g} "
                   f"{time_to_half_mean[i]:<15.6g} {time_to_half_err[i]:<15.6g} "
                   f"{wall_time_mean[i]:<15.6g} {wall_time_err[i]:<15.6g}\n")
    
    print(f"\nResults saved to: {output_filename}")
    print(f"Figure saved to: parameter_sweep_{SWEEP_PARAMETER}.png")
    
    print("\n" + "="*80)
    print("SWEEP COMPLETE!")
    print("="*80)