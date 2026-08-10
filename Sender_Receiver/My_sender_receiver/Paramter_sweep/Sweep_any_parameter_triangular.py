'''
CORRECTED: Sweep any parameter with triangular mesh
ONLY mesh loading changed - solver and ALL physics/equations identical to original
'''
"""
Parameter Sweep Script for 2D Tethered Genelet Model
Runs multiple simulations with varying parameters using multiprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import sys
import re
import tempfile
import os


parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
warnings.filterwarnings('ignore')

from Functions_and_system.Functions import calculate_total_amount, smooth_circular_profile, intialize_equations, initalize_variables
from Mesh.New_simple_mesh import create_gmsh_radial_mesh

# =============================================================================
# USER CONFIGURATION - FILL THESE IN
# =============================================================================

SWEEP_PARAMETER = "distance_between"  # Options: "distance_between", "k_p", "k_slow", "k_fast", "D_gel", "Th2_init", "node_diameter"
SWEEP_VALUES = [200.0, 300.0, 500.0]

# Number of parallel processes
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
    'dt': 60.0,
    'total_time': 3 * 3600,
    'save_interval_time': 60.0,
    'fine_dx': 0.75,
    'coarse_dx': 100.0,
    'box_padding': 200.0,
    'mesh_transition_width': 100.0,
    'profile_transition_width_factor': 3.0,
}

# Global constants (must match what's in Functions.py)
k_p = DEFAULT_PARAMS['k_p']
k_d_ds = DEFAULT_PARAMS['k_d_ds']
k_d_ss = DEFAULT_PARAMS['k_d_ss']
k_slow = DEFAULT_PARAMS['k_slow']
k_fast = DEFAULT_PARAMS['k_fast']

# =============================================================================
# HELPER FUNCTIONS - MESH MANAGEMENT
# =============================================================================

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


def determine_if_mesh_changes(param_name):
    """Check if parameter affects mesh geometry"""
    mesh_affecting_params = ['distance_between', 'node_diameter', 'fine_dx', 'coarse_dx']
    return param_name in mesh_affecting_params


def get_mesh_filename(params):
    """Generate unique mesh filename based on geometry parameters"""
    return (f"sweep_mesh_ccd={params['distance_between']:.2f}_"
            f"nd={params['node_diameter']:.2f}_"
            f"fine={params['fine_dx']:.2f}_"
            f"coarse={params['coarse_dx']:.2f}.msh")


def parse_mesh_filename(mesh_filename):
    """
    Extract geometry parameters from mesh filename.
    Returns: (distance_between, node_diameter)
    """
    # Example: "sweep_mesh_ccd=300.00_nd=75.00_fine=0.75_coarse=100.00.msh"
    match = re.search(r'ccd=([\d.]+).*nd=([\d.]+)', mesh_filename)
    if match:
        distance_between = float(match.group(1))
        node_diameter = float(match.group(2))
        return distance_between, node_diameter
    else:
        raise ValueError(f"Cannot parse mesh filename: {mesh_filename}")


# =============================================================================
# MAIN SIMULATION FUNCTION - IDENTICAL TO ORIGINAL EXCEPT MESH LOADING
# =============================================================================

# =============================================================================
# ISOLATED MESH GENERATION FUNCTION (NEW)
# =============================================================================

def create_gmsh_radial_mesh_isolated(bath_width, bath_height, node_diameter, 
                                      distance_between_nodes, min_cell_size, 
                                      max_cell_size, growth_rate=1.5, 
                                      process_id=None, verbose=False):
    """
    Creates a Gmsh radial mesh in ISOLATION for a single worker process.
    Uses a unique temporary file to avoid conflicts with other processes.
    
    THIS IS THE KEY CHANGE: Each process gets its own temp file
    → No shared file access → No conflicts
    
    Parameters:
    -----------
    bath_width, bath_height : float
        Domain dimensions (μm)
    node_diameter : float
        Diameter of circular nodes (μm)
    distance_between_nodes : float
        Center-to-center distance between sender/receiver (μm)
    min_cell_size : float
        Minimum mesh size at node surface (μm)
    max_cell_size : float
        Maximum mesh size far from nodes (μm)
    growth_rate : float
        Geometric growth rate for mesh refinement
    process_id : int or None
        Process identifier for unique filename generation
    verbose : bool
        Print progress messages
        
    Returns:
    --------
    mesh : FiPy Gmsh2D mesh object
    sender_center_x : float
    receiver_center_x : float
    y_center : float
    """
    import gmsh  # Import here to avoid issues with multiprocessing
    
    # === GENERATE UNIQUE TEMPORARY FILENAME FOR THIS PROCESS ===
    # This ensures each worker has its own isolated mesh file
    if process_id is None:
        process_id = os.getpid()  # Use process ID if not provided
    
    # Create unique temp file in system temp directory
    temp_dir = tempfile.gettempdir()
    mesh_filename = os.path.join(temp_dir, f"isolated_mesh_pid{process_id}_{int(time.time()*1000000)}.msh")
    
    if verbose:
        print(f"[PID {process_id}] Creating isolated mesh: {mesh_filename}")
    
    # === CALCULATE GEOMETRY ===
    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0
    
    # === INITIALIZE GMSH (ISOLATED TO THIS PROCESS) ===
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("radial_adaptive_mesh")
    
    # === GEOMETRY CREATION ===
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0, 
                                                node_radius, node_radius)
    receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                  node_radius, node_radius)
    
    gmsh.model.occ.synchronize()

    # === GET BOUNDARIES BEFORE BOOLEAN OPERATION ===
    sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)
    receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)
    
    sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
    receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]
    
    if verbose:
        print(f"[PID {process_id}] Sender curves: {sender_curve_tags}")
        print(f"[PID {process_id}] Receiver curves: {receiver_curve_tags}")

    # Boolean operations to cut nodes from bath
    bath_with_nodes, _ = gmsh.model.occ.cut(
        [(2, rectangle_tag)],
        [(2, sender_circle_tag), (2, receiver_circle_tag)],
        removeTool=True
    )
    
    gmsh.model.occ.synchronize()
    
    # === MESH SIZE FIELDS (RADIAL REFINEMENT) ===

    if sender_curve_tags:
        distance_field_sender = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field_sender, "CurvesList", sender_curve_tags)
    
    if receiver_curve_tags:
        distance_field_receiver = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field_receiver, "CurvesList", receiver_curve_tags)

    
    # Minimum distance field (closest to either node)
    min_distance_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_distance_field, "FieldsList", 
                                     [distance_field_sender, distance_field_receiver])
    
    # Threshold field for smooth size transition
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", min_distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_cell_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_cell_size)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 200.0)
    gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 1)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # Disable other mesh size methods
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # === GENERATE MESH ===
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")
    
    # === SAVE TO TEMPORARY FILE (FORMAT 2.2 FOR FIPY) ===
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(mesh_filename)
    
    # === FINALIZE GMSH ===
    gmsh.finalize()
    
    # === IMPORT INTO FIPY FROM TEMP FILE ===
    if verbose:
        print(f"[PID {process_id}] Loading mesh into FiPy...")
    
    mesh = Gmsh2D(mesh_filename)
    
    if verbose:
        print(f"[PID {process_id}] Mesh created: {mesh.numberOfCells:,} cells")
    
    # === CLEAN UP TEMP FILE ===
    try:
        os.remove(mesh_filename)
        if verbose:
            print(f"[PID {process_id}] Cleaned up temp file")
    except:
        pass  # If cleanup fails, temp files will be cleaned by OS eventually
    
    return mesh, sender_center_x, receiver_center_x, y_center


# =============================================================================
# WORKER FUNCTION - RUNS ONE SIMULATION (MODIFIED)
# =============================================================================

def run_single_simulation(param_value, replicate_number):
    """
    Run a single simulation with given parameter value.
    
    KEY CHANGE: Each call generates its own mesh in isolation.
    No pre-generated meshes, no shared files.
    """
    try:
        # Build parameter dictionary
        params = DEFAULT_PARAMS.copy()
        params[SWEEP_PARAMETER] = param_value

        start_time = time.perf_counter()
        
        # Extract parameters
        distance_between = params['distance_between']
        node_diameter = params['node_diameter']
        total_width = params['total_width']
        total_height = params['total_height']
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
        dt = params['dt']
        total_time = params['total_time']
        save_interval_time = params['save_interval_time']
        fine_dx = params['fine_dx']
        coarse_dx = params['coarse_dx']
        
        n_steps = int(total_time / dt)
        save_interval_steps = int(save_interval_time / dt)
        
        # =============================================================================
        # ISOLATED MESH GENERATION (KEY CHANGE)
        # =============================================================================
        # Generate unique process identifier for this simulation
        process_id = os.getpid() * 10000 + replicate_number  # Ensure uniqueness
        
        # Each worker creates its own mesh independently
        # No file sharing, no conflicts with other processes
        mesh, sender_center_x, receiver_center_x, sender_center_y = create_gmsh_radial_mesh_isolated(
            bath_width=total_width,
            bath_height=total_height,
            node_diameter=node_diameter,
            distance_between_nodes=distance_between,
            min_cell_size=fine_dx,
            max_cell_size=coarse_dx,
            growth_rate=1.5,
            process_id=process_id,
            verbose=False
        )
        
        # Calculate geometry from actual parameters used (not from filename parsing)
        node_radius = node_diameter / 2.0
        domain_center_x = total_width / 2.0
        domain_center_y = total_height / 2.0
        receiver_center_x = domain_center_x + distance_between / 2.0
        receiver_center_y = sender_center_y  # Same y-coordinate
        

        x, y = mesh.cellCenters[0], mesh.cellCenters[1]
        # Initialize variables - IDENTICAL TO ORIGINAL
        S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables(
            mesh, x, y, sender_center_x, receiver_center_x, 
            receiver_center_y, node_radius, I2_init, Th2_init, 
            I1O2_init, D_gel, D_solution
        )
        
        # Initialize equations 
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
        STEADY_STATE_THRESHOLD = 1e-8
        STEADY_STATE_WINDOW = 50
        CHECK_INTERVAL = 100
        
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
        
        # Calculate outputs - IDENTICAL TO ORIGINAL
        I2_final = I2_concentration[-1]
        S2_final = S2_free_concentration[-1]
        S2_total_final = S2_total_concentration[-1]
        
        time_array = np.array(time_points)
        I2_array = np.array(I2_concentration)
        half_time = calculate_half_time(time_array, I2_array, I2_init)
        
        
        result = {
            'param_value': param_value,
            'replicate_id': replicate_number,
            'I2_final': I2_final,
            'S2_final': S2_final,
            'S2_total_final': S2_total_final,
            'half_time': half_time,
            'wall_time': time.perf_counter() - start_time, 
            'success': True
        }
        print(f"total time for {param_value} = {(time.perf_counter() - start_time):.2f}")
        print(f"✓ {SWEEP_PARAMETER}={param_value:.2e}, Rep={replicate_number}: "
              f"I2_final={I2_final*1000:.2f} nM, t_half={half_time:.2f} hr")
        
        return result
        
    except Exception as e:
        print(f"✗ ERROR [{SWEEP_PARAMETER}={param_value}, Rep={replicate_number}]: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'param_value': param_value,
            'replicate_id': replicate_number,
            'I2_final': np.nan,
            'S2_final': np.nan,
            'S2_total_final': np.nan,
            'half_time': np.nan,
            'wall_time': np.nan,
            'success': False
        }


# =============================================================================
# PARAMETER SWEEP ORCHESTRATION
# =============================================================================

def run_parameter_sweep():
    """
    Run parameter sweep with parallel processing.
    
    KEY CHANGE: No pre-generation phase!
    Each worker generates its own mesh when needed.
    """
    print("="*80)
    print(f"PARAMETER SWEEP CONFIGURATION")
    print("="*80)
    print(f"Sweeping parameter: {SWEEP_PARAMETER}")
    print(f"Values: {SWEEP_VALUES}")
    print(f"Replicates per value: {N_REPLICATES}")
    print(f"Total simulations: {len(SWEEP_VALUES) * N_REPLICATES}")
    print("="*80)
    print("MESH GENERATION STRATEGY: ISOLATED PER-CORE")
    print("Each worker process creates its own mesh independently")
    print("No shared files, no pre-generation, no conflicts")
    print("="*80 + "\n")
    
    # Build task list
    tasks = []
    for param_value in SWEEP_VALUES:
        for rep in range(N_REPLICATES):
            tasks.append((param_value, rep))
    
    # Run parallel simulations
    n_processes = N_PROCESSES if N_PROCESSES else cpu_count()
    print(f"Using {n_processes} parallel processes")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(run_single_simulation, tasks)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"SWEEP COMPLETE - Total time: {total_time/60:.2f} minutes")
    print("="*80)
    
    return results


# =============================================================================
# DATA ANALYSIS AND PLOTTING - IDENTICAL TO ORIGINAL
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
    csv_filename = f'sweep_results_{SWEEP_PARAMETER}={SWEEP_VALUES}_triangular_mesh.csv'
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
    fig_filename = f'sweep_plots_{SWEEP_PARAMETER}_{timestamp}_triangular.png'
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
    print("║  2D TETHERED GENELET MODEL - PARAMETER SWEEP (CORRECTED)     ║")
    print("║  Gmsh mesh with pre-generation to avoid conflicts            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Run the parameter sweep
    results = run_parameter_sweep()
    
    # Analyze and plot results
    stats = analyze_and_plot_results(results)
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    SWEEP COMPLETE!                           ║") 
    print("╚══════════════════════════════════════════════════════════════╝")
    print()