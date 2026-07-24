# =============================================================================
# PART 2: CONVERGENCE STUDY
# =============================================================================



"""
Tethered Genelet Reaction-Diffusion Model using FiPy
WITH ADAPTIVE STEADY-STATE DETECTION AND CONVERGENCE STUDY
"""

import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix
import time as timer

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
Th2_init = 0.005    # 5 μM threshold

# Geometry (μm)
node_size = 50.0
distance_between = 300.0
total_length = 2000.0

# Time parameters
dt_initial = 1.0    # Start with 1 second timestep (we'll test this)
max_time = 8 * 3600 # Maximum 8 hours
check_interval = 100 # Check for steady state every N steps

# Steady-state detection parameters
ss_tolerance = 1e-8  # Relative change threshold for steady state
ss_window = 50       # Number of timesteps to check for steady state
# Steady state if: max(|dC/dt|) / max(|C|) < ss_tolerance for ss_window steps

# =============================================================================
# FUNCTION TO BUILD AND RUN MODEL
# =============================================================================

def run_simulation(dt, max_time=max_time, check_steady_state=True, verbose=True):
    """
    Run the tethered genelet simulation.
    
    Parameters:
    -----------
    dt : float
        Timestep size (seconds)
    max_time : float
        Maximum simulation time (seconds)
    check_steady_state : bool
        If True, stop when steady state is reached
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict with results: time_points, I2, S2_free, S2_total, converged, runtime
    """
    
    start_time = timer.time()
    
    # =========================================================================
    # MESH SETUP
    # =========================================================================
    
    nx = 400
    dx = total_length / nx
    mesh = Grid1D(nx=nx, dx=dx)
    x = mesh.cellCenters[0]
    
    # Define node regions
    sender_center = total_length / 2 - distance_between / 2
    receiver_center = total_length / 2 + distance_between / 2
    
    sender_mask = (x >= sender_center - node_size/2) & (x <= sender_center + node_size/2)
    receiver_mask = (x >= receiver_center - node_size/2) & (x <= receiver_center + node_size/2)
    gel_mask = sender_mask | receiver_mask
    
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
    # DEFINE EQUATIONS
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
    
    receiver_center_idx = np.argmin(np.abs(x.value - receiver_center))
    
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
        print(f"Running simulation with dt = {dt} s")
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
            # Change = |C(t) - C(t-dt)| / (|C(t)| + epsilon)
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
        'dt': dt
    }




print("\n" + "="*70)
print("PART 2: CONVERGENCE STUDY")
print("="*70)

# Test different timesteps
dt_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # seconds
convergence_results = []

for dt_test in dt_values:
    print(f"\nTesting dt = {dt_test} s...")
    result = run_simulation(dt=dt_test, check_steady_state=True, verbose=False)
    convergence_results.append(result)
    
    print(f"  Converged: {result['converged']}")
    print(f"  Final time: {result['final_time']/3600:.3f} hours")
    print(f"  Steps: {result['n_steps']}")
    print(f"  Runtime: {result['runtime']:.2f} s")
    if len(result['I2']) > 0:
        print(f"  Final I2: {result['I2'][-1]:.6f} μM")

# =============================================================================
# CONVERGENCE ANALYSIS PLOTS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: I2 trajectories for different dt
ax = axes[0, 0]
for result in convergence_results:
    if len(result['time_points']) > 0:
        ax.plot(result['time_points'], result['I2'], 
                label=f"dt = {result['dt']} s", linewidth=2, alpha=0.7)
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('[I2] (μM)', fontsize=11)
ax.set_title('I2 Convergence: Effect of Timestep', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Final I2 value vs dt
ax = axes[0, 1]
final_I2 = [r['I2'][-1] if len(r['I2']) > 0 else np.nan for r in convergence_results]
dt_vals = [r['dt'] for r in convergence_results]
ax.plot(dt_vals, final_I2, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Timestep dt (s)', fontsize=11)
ax.set_ylabel('Final [I2] (μM)', fontsize=11)
ax.set_title('Steady-State I2 vs Timestep', fontsize=13)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Plot 3: Computational efficiency
ax = axes[1, 0]
runtimes = [r['runtime'] for r in convergence_results]
ax.plot(dt_vals, runtimes, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Timestep dt (s)', fontsize=11)
ax.set_ylabel('Runtime (seconds)', fontsize=11)
ax.set_title('Computational Cost vs Timestep', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 4: Relative error vs dt (using smallest dt as reference)
ax = axes[1, 1]
if not np.isnan(final_I2[0]):  # If we have a reference solution
    reference_I2 = final_I2[0]  # Smallest dt is most accurate
    relative_errors = [abs(I2 - reference_I2) / reference_I2 if not np.isnan(I2) else np.nan 
                       for I2 in final_I2]
    ax.loglog(dt_vals, relative_errors, 'go-', linewidth=2, markersize=8, label='Relative error')
    
    # Add reference lines for convergence order
    dt_array = np.array(dt_vals)
    ax.loglog(dt_array, 0.01 * dt_array/dt_array[0], 'k--', alpha=0.5, label='1st order')
    ax.loglog(dt_array, 0.01 * (dt_array/dt_array[0])**2, 'k:', alpha=0.5, label='2nd order')
    
    ax.set_xlabel('Timestep dt (s)', fontsize=11)
    ax.set_ylabel('Relative Error', fontsize=11)
    ax.set_title('Convergence Order Analysis', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_study.png', dpi=300, bbox_inches='tight')
print("\nConvergence study saved as 'convergence_study.png'")
plt.show()

# =============================================================================
# CONVERGENCE SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("CONVERGENCE STUDY SUMMARY")
print("="*70)
print(f"{'dt (s)':<10} {'Steps':<10} {'Time (hr)':<12} {'Final I2':<12} {'Runtime (s)':<12} {'Converged'}")
print("-"*70)
for result in convergence_results:
    final_I2_val = result['I2'][-1] if len(result['I2']) > 0 else float('nan')
    print(f"{result['dt']:<10.1f} {result['n_steps']:<10} "
          f"{result['final_time']/3600:<12.3f} {final_I2_val:<12.6f} "
          f"{result['runtime']:<12.2f} {result['converged']}")
print("="*70)

# Recommend optimal timestep
print("\nRECOMMENDATIONS:")
print("-"*70)

# Find timesteps that converged
converged_results = [r for r in convergence_results if r['converged'] and len(r['I2']) > 0]

if converged_results:
    # Sort by runtime
    converged_results.sort(key=lambda x: x['runtime'])
    
    optimal = converged_results[0]
    print(f"✓ Optimal timestep: dt = {optimal['dt']} s")
    print(f"  - Reaches steady state in {optimal['final_time']/3600:.2f} hours")
    print(f"  - Computation time: {optimal['runtime']:.2f} seconds")
    print(f"  - Final I2 concentration: {optimal['I2'][-1]:.6f} μM")
    
    # Check accuracy
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