"""
Parameter Sweep Script for 2D Tethered Genelet Model
Runs multiple simulations with varying parameters using multiprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import sys
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
warnings.filterwarnings('ignore')

from Functions_and_system.Functions import calculate_total_amount, smooth_circular_profile, intialize_equations, initalize_variables
from Mesh.New_simple_mesh import create_gmsh_radial_mesh



# Import functions from your original script
# Assuming the original file is saved as 'Sys_adaptive_mesh_tanh_nodes.py'
# We'll redefine the necessary functions here or you can import them

# =============================================================================
# USER CONFIGURATION - FILL THESE IN
# =============================================================================

SWEEP_PARAMETER = "distance_between"  # Options: "distance_between", "k_p", "k_slow", "k_fast", "D_gel", "Th2_init", "node_diameter"
SWEEP_VALUES = [200,300,500,800,1100,1200,1300,1500]  # List of values to sweep


SWEEP_PARAMETER = "fine_dx"
SWEEP_VALUES = [0.25,0.5,0.75, 1, 1.25]


# Number of parallel processes (use None for auto-detect)
if len(SWEEP_VALUES) < 6:
    N_PROCESSES = len(SWEEP_VALUES) 
else:
    N_PROCESSES = 5
# Number of replicates per parameter value (for error bars)
N_REPLICATES = 1

# =============================================================================
# FIXED PARAMETERS (defaults from original script)
# =============================================================================

DEFAULT_PARAMS = {
    'D_solution': 150.0,
    'D_gel': 60.0,
    'k_p': 0.2,
    'k_d_ds': 3e-4,
    'k_d_ss': 3e-4,
    'k_slow': 1e5 * 1e-6,
    'k_fast': 1e6 * 1e-6,
    'I1O2_init': 0.1,
    'I2_init': 0.1,
    'Th2_init': 5.0,
    'node_size': 50.0,
    'node_diameter': 75.0,
    'bath_margin': 250.0,
    'distance_between': 300.0,
    'total_width': 1e4,
    'total_height': 1e3,
    'dt': 30.0,
    'total_time': 8 * 3600,
    'save_interval_time': 60.0,
    'fine_dx': 0.75, #used to be 5
    'coarse_dx': 50.0, #used to be 40
    'box_padding': 200.0,
    'mesh_transition_width': 100.0,
    'profile_transition_width_factor': 3.0,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_total_amount(concentration, mesh):
    """Calculate total amount of species in the domain."""
    cell_volumes = mesh.cellVolumes
    total = np.sum(concentration * cell_volumes)
    return total



def calculate_half_time(time_array, I2_array, I2_init):
    """Calculate time when I2 drops to halfway between initial and final value."""
    I2_final = I2_array[-1]
    I2_half = (I2_init + I2_final) / 2.0
    
    # Find first time point where I2 drops below half value
    below_half = I2_array < I2_half
    if np.any(below_half):
        idx = np.argmax(below_half)
        return time_array[idx]
    else:
        return np.nan  # Never reached half


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def run_single_simulation(param_value, replicate_id=0):
    """
    Run a single simulation with specified parameter value.
    
    Returns:
    --------
    dict with keys:
        - param_value: the parameter value used
        - replicate_id: replicate number
        - I2_final: final I2 concentration at receiver
        - S2_final: final free S2 concentration at receiver
        - S2_total_final: final total S2 at receiver
        - half_time: time to reach halfway point
        - wall_time: computation time
        - success: True/False
    """
    
    start_wall_time = time.time()
    
    try:
        # Create parameter dictionary
        params = DEFAULT_PARAMS.copy()
        params[SWEEP_PARAMETER] = param_value
        
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
        node_diameter = params['node_diameter']
        node_radius = node_diameter / 2
        distance_between = params['distance_between']
        total_width = params['total_width']
        total_height = params['total_height']
        dt = params['dt']
        total_time = params['total_time']
        save_interval_time = params['save_interval_time']
        fine_dx = params['fine_dx']
        coarse_dx = params['coarse_dx']
        
        n_steps = int(total_time / dt)
        save_interval_steps = int(save_interval_time / dt)
        
        # Create adaptive mesh
        mesh, sender_center_x, receiver_center_x, sender_center_y = create_gmsh_radial_mesh(bath_width = total_width, bath_height=total_height, node_diameter=node_diameter, 
                                                                    distance_between_nodes=distance_between, min_cell_size=fine_dx,
                                                                    max_cell_size=coarse_dx, growth_rate=1.5, mesh_filename='sweep_mesh.msh', 
                                                                    visualize_gmsh=False, verbose=False, )
        
        
        receiver_center_y = sender_center_y
        x, y = mesh.cellCenters
        
        # Create smooth profiles

        S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(mesh, x,y, sender_center_x, receiver_center_x, 
                                                                     receiver_center_y, node_radius, I2_init, Th2_init, 
                                                                     I1O2_init, D_gel, D_solution)
        
        eq = intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2)

        
        # Find receiver center index
        distances_to_receiver = numerix.sqrt((x - receiver_center_x)**2 + 
                                             (y - receiver_center_y)**2)
        receiver_center_idx = numerix.argmin(distances_to_receiver)
        
        # Storage
        time_points = []
        I2_concentration = []
        S2_free_concentration = []
        S2_total_concentration = []
        
# Time stepping with steady-state detection
        STEADY_STATE_THRESHOLD = 1e-8  # Relative change threshold
        STEADY_STATE_WINDOW = 100  # Number of steps to check
        CHECK_INTERVAL = 50  # Check every N steps
        
        steady_state_reached = False
        recent_I2_values = []
        
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
                
                recent_I2_values.append(I2_val)
            
            
            # Check for steady state every CHECK_INTERVAL steps
            if step % (save_interval_steps * CHECK_INTERVAL) == 0 and len(recent_I2_values) > STEADY_STATE_WINDOW:
                recent_window = recent_I2_values[-STEADY_STATE_WINDOW:]
                mean_I2 = np.mean(recent_window)
                
                if mean_I2 > 0:
                    relative_change = np.std(recent_window) / mean_I2
                    
                    if relative_change < STEADY_STATE_THRESHOLD:
                        steady_state_reached = True
                        print(f"  → Steady state reached at t={current_time/3600:.2f} hr "
                              f"(rel. change={relative_change:.2e})")
                        break
            
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
        
        # Calculate outputs
        I2_final = I2_concentration[-1]
        S2_final = S2_free_concentration[-1]
        S2_total_final = S2_total_concentration[-1]
        
        time_array = np.array(time_points)
        I2_array = np.array(I2_concentration)
        half_time = calculate_half_time(time_array, I2_array, I2_init)
        
        wall_time = time.time() - start_wall_time
        
        result = {
            'param_value': param_value,
            'replicate_id': replicate_id,
            'I2_final': I2_final,
            'S2_final': S2_final,
            'S2_total_final': S2_total_final,
            'half_time': half_time,
            'wall_time': wall_time,
            'success': True
        }
        
        print(f"✓ {SWEEP_PARAMETER}={param_value:.2e}, Rep={replicate_id}: "
              f"I2_final={I2_final*1000:.2f} nM, t_half={half_time:.2f} hr, "
              f"time={wall_time:.1f}s")
        
        return result
        
    except Exception as e:
        print(f"✗ {SWEEP_PARAMETER}={param_value:.2e}, Rep={replicate_id}: FAILED - {str(e)}")
        return {
            'param_value': param_value,
            'replicate_id': replicate_id,
            'I2_final': np.nan,
            'S2_final': np.nan,
            'S2_total_final': np.nan,
            'half_time': np.nan,
            'wall_time': time.time() - start_wall_time,
            'success': False
        }


# =============================================================================
# PARALLEL SWEEP EXECUTION
# =============================================================================

def run_parameter_sweep():
    """Run the full parameter sweep with multiprocessing."""
    
    print("="*80)
    print("PARAMETER SWEEP CONFIGURATION")
    print("="*80)
    print(f"Sweep parameter: {SWEEP_PARAMETER}")
    print(f"Sweep values: {SWEEP_VALUES}")
    print(f"Replicates per value: {N_REPLICATES}")
    print(f"Total simulations: {len(SWEEP_VALUES) * N_REPLICATES}")
    
    # Determine number of processes
    n_processes = N_PROCESSES if N_PROCESSES else cpu_count()
    print(f"Using {n_processes} parallel processes")
    print("="*80)
    print()
    
    # Create list of all simulation tasks
    tasks = []
    for param_value in SWEEP_VALUES:
        for rep in range(N_REPLICATES):
            tasks.append((param_value, rep))
    
    # Run simulations in parallel
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(run_single_simulation, tasks)
    
    total_time = time.time() - start_time
    
    print()
    print("="*80)
    print(f"SWEEP COMPLETE - Total time: {total_time/60:.2f} minutes")
    print("="*80)
    
    return results


# =============================================================================
# DATA ANALYSIS AND PLOTTING
# =============================================================================

def analyze_and_plot_results(results):
    """Analyze results and create plots with error bars."""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful runs
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("ERROR: No successful simulations!")
        return
    
    # Calculate statistics for each parameter value
    stats = df_success.groupby('param_value').agg({
        'I2_final': ['mean', 'std'],
        'S2_final': ['mean', 'std'],
        'S2_total_final': ['mean', 'std'],
        'half_time': ['mean', 'std'],
        'wall_time': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f'sweep_results_{SWEEP_PARAMETER}={SWEEP_VALUES}.csv'
    stats.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Parameter Sweep: {SWEEP_PARAMETER}', fontsize=16, fontweight='bold')
    
    param_vals = stats['param_value'].values
    
    # Plot 1: Final I2
    axes[0, 0].errorbar(param_vals, stats['I2_final_mean']*1000, 
                        yerr=stats['I2_final_std']*1000,
                        fmt='o-', capsize=5, linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(SWEEP_PARAMETER, fontsize=12)
    axes[0, 0].set_ylabel('Final [I2] (nM)', fontsize=12)
    axes[0, 0].set_title('Final I2 Concentration', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final free S2
    axes[0, 1].errorbar(param_vals, stats['S2_final_mean']*1000, 
                        yerr=stats['S2_final_std']*1000,
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel(SWEEP_PARAMETER, fontsize=12)
    axes[0, 1].set_ylabel('Final [S2] free (nM)', fontsize=12)
    axes[0, 1].set_title('Final Free S2 Concentration', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final total S2
    axes[0, 2].errorbar(param_vals, stats['S2_total_final_mean']*1000, 
                        yerr=stats['S2_total_final_std']*1000,
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color='red')
    axes[0, 2].set_xlabel(SWEEP_PARAMETER, fontsize=12)
    axes[0, 2].set_ylabel('Final [S2] total (nM)', fontsize=12)
    axes[0, 2].set_title('Final Total S2 Concentration', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Half-time
    axes[1, 0].errorbar(param_vals, stats['half_time_mean'], 
                        yerr=stats['half_time_std'],
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel(SWEEP_PARAMETER, fontsize=12)
    axes[1, 0].set_ylabel('Half-time (hours)', fontsize=12)
    axes[1, 0].set_title('Time to Half I2', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Wall time
    axes[1, 1].errorbar(param_vals, stats['wall_time_mean'], 
                        yerr=stats['wall_time_std'],
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_xlabel(SWEEP_PARAMETER, fontsize=12)
    axes[1, 1].set_ylabel('Wall time (seconds)', fontsize=12)
    axes[1, 1].set_title('Computation Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary table
    axes[1, 2].axis('off')
    table_data = []
    for _, row in stats.iterrows():
        table_data.append([
            f"{row['param_value']:.2e}",
            f"{row['I2_final_mean']*1000:.1f}±{row['I2_final_std']*1000:.1f}",
            f"{row['half_time_mean']:.2f}±{row['half_time_std']:.2f}"
        ])
    
    table = axes[1, 2].table(cellText=table_data,
                             colLabels=[SWEEP_PARAMETER, 'I2 final (nM)', 't_half (hr)'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    plt.tight_layout()
    
    # Save figure
    fig_filename = f'sweep_plots_{SWEEP_PARAMETER}_{timestamp}.png'
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {fig_filename}")
    plt.show()
    
    return stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    2D TETHERED GENELET MODEL - PARAMETER SWEEP SCRIPT        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Run the parameter sweep
    results = run_parameter_sweep()
    
    # Analyze and plot results
    stats = analyze_and_plot_results(results)
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    SWEEP COMPLETE!                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()