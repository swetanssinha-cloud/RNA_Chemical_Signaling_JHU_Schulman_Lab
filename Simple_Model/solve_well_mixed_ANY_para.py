import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from fipy import CellVariable, Grid1D, TransientTerm, ImplicitSourceTerm, numerix
import time
import os
import glob

# ============================================================================
# PARAMETER SWEEP CONFIGURATION - EDIT THIS SECTION
# ============================================================================
# Specify which parameter to sweep and its values
SWEEP_PARAM = "Th2_0"  # Options: "Phi_in", "Th2_0", "I2_0", "k_slow", "k_fast", etc.
SWEEP_VALUES = np.array([1000, 2500, 5000, 7500, 10000]) * 1e-9  # Values in SI units (M for concentrations)

# For Phi_in sweeps, should I2_0 scale with Phi_in? (True for original behavior)
SCALE_I2_WITH_PHI_IN = True

# Base parameters (in SI units: M for concentrations, 1/M/s for rates, s for time)
BASE_PARAMS = {
    'k_slow': 1e5,      # 1/M/s
    'k_fast': 1e6,      # 1/M/s
    'k_d_ds': 3e-4,     # 1/s
    'k_d_ss': 3e-4,     # 1/s
    'Phi_in': 2e-9,     # M/s
    'I2_0': 100e-9,     # M
    'Th2_0': 5000e-9,   # M
    'S2_0': 0.0,        # M
    'C_I2_0': 0.0,      # M
    'C_Th2_0': 0.0,     # M
}

# Simulation settings
SIM_PARAMS = {
    'dt_s': 60,                    # Time step (seconds)
    'max_time_hours': 8,           # Max simulation time
    'abs_threshold': 1e-8,         # Convergence threshold (M) - 10 nM
    'check_interval': 100,         # Steps between convergence checks
    'save_interval': 10,           # Steps between CSV saves
    'max_sweeps': 20,              # Max sweeps per timestep
    'progress_interval_s': 60,     # Progress print interval (wall-clock seconds)
}
# ============================================================================

# Clean up old output files
for pattern in ['run_*.csv', 'summary.csv', 'steady_state_values.csv']:
    for f in glob.glob(pattern):
        os.remove(f)
        print(f"Deleted: {f}")

def run_simulation(args):
    """Run a single FiPy simulation with given parameters"""
    sweep_value, params, sim_params = args
    
    # Extract parameters
    k_slow = params['k_slow']
    k_fast = params['k_fast']
    k_d_ds = params['k_d_ds']
    k_d_ss = params['k_d_ss']
    Phi_in = params['Phi_in']
    
    I2_0 = params['I2_0']
    Th2_0 = params['Th2_0']
    S2_0 = params['S2_0']
    C_I2_0 = params['C_I2_0']
    C_Th2_0 = params['C_Th2_0']
    
    dt_s = sim_params['dt_s']
    max_time_hours = sim_params['max_time_hours']
    abs_threshold = sim_params['abs_threshold']
    check_interval = sim_params['check_interval']
    save_interval = sim_params['save_interval']
    max_sweeps = sim_params['max_sweeps']
    progress_interval_s = sim_params['progress_interval_s']
    
    max_steps = int((max_time_hours * 3600) / dt_s)
    
    # Create mesh (single point for well-mixed)
    mesh = Grid1D(nx=1, dx=1.0)
    
    # Create variables
    S2 = CellVariable(name="S2", mesh=mesh, value=S2_0, hasOld=True)
    I2 = CellVariable(name="I2", mesh=mesh, value=I2_0, hasOld=True)
    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_0, hasOld=True)
    C_I2 = CellVariable(name="C_I2", mesh=mesh, value=C_I2_0, hasOld=True)
    C_Th2 = CellVariable(name="C_Th2", mesh=mesh, value=C_Th2_0, hasOld=True)
    
    # Define equations using implicit source term method
    eq_S2 = (TransientTerm(var=S2) ==
             ImplicitSourceTerm(coeff=-k_slow * I2, var=S2) +
             ImplicitSourceTerm(coeff=-k_fast * Th2, var=S2) +
             k_d_ds * C_I2 +
             k_d_ss * C_Th2 +
             Phi_in)

    eq_I2 = (TransientTerm(var=I2) == 
            k_d_ds * C_I2 +
            ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

    eq_Th2 = (TransientTerm(var=Th2) == 
            k_d_ss * C_Th2 +
            ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

    eq_C_I2 = (TransientTerm(var=C_I2) ==
               k_slow * S2 * I2 +
               ImplicitSourceTerm(coeff=-k_d_ds, var=C_I2))

    eq_C_Th2 = (TransientTerm(var=C_Th2) ==
                k_fast * S2 * Th2 +
                ImplicitSourceTerm(coeff=-k_d_ss, var=C_Th2))
    
    coupled_eq = eq_S2 & eq_I2 & eq_Th2 & eq_C_I2 & eq_C_Th2
    
    # Determine CSV filename based on sweep parameter
    if SWEEP_PARAM == "Phi_in":
        csv_filename = f"run_Phi_in_{sweep_value*1e9:.1f}nMps.csv"
    elif SWEEP_PARAM == "Th2_0":
        csv_filename = f"run_Th2_0_{sweep_value*1e9:.0f}nM.csv"
    elif SWEEP_PARAM == "I2_0":
        csv_filename = f"run_I2_0_{sweep_value*1e9:.0f}nM.csv"
    else:
        csv_filename = f"run_{SWEEP_PARAM}_{sweep_value:.3e}.csv"
    
    time_data = []
    converged = False
    status = "timeout"
    
    prev_values = np.array([I2.value[0], Th2.value[0], C_I2.value[0], C_Th2.value[0]])
    last_max_abs_change = np.inf
    
    start_wall_time = time.time()
    last_progress_time = start_wall_time
    
    for step in range(max_steps):
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        C_I2.updateOld()
        C_Th2.updateOld()
        
        res = 1e10
        for sweep in range(max_sweeps):
            res = coupled_eq.sweep(dt=dt_s)
            if res < 1e-10:
                break
        
        current_time_s = (step + 1) * dt_s
        
        if (step + 1) % save_interval == 0:
            S2_tot = S2.value[0] + C_I2.value[0] + C_Th2.value[0]
            time_data.append({
                'time_s': current_time_s,
                'S2_M': S2.value[0],
                'I2_M': I2.value[0],
                'S2_tot_M': S2_tot
            })
        
        if (step + 1) % check_interval == 0:
            current_values = np.array([I2.value[0], Th2.value[0], C_I2.value[0], C_Th2.value[0]])
            abs_changes = np.abs(current_values - prev_values)
            max_abs_change = np.max(abs_changes)
            last_max_abs_change = max_abs_change
            
            if max_abs_change < abs_threshold:
                converged = True
                status = "converged"
                break
            
            prev_values = current_values.copy()
        
        current_wall_time = time.time()
        if current_wall_time - last_progress_time >= progress_interval_s:
            elapsed_wall = current_wall_time - start_wall_time
            sim_hours = current_time_s / 3600
            print(f"  [{SWEEP_PARAM}={sweep_value:.2e}] t={sim_hours:.2f}h, "
                  f"wall={elapsed_wall:.1f}s, max_Δ={last_max_abs_change:.2e}M")
            last_progress_time = current_wall_time
    
    end_wall_time = time.time()
    wall_time_s = end_wall_time - start_wall_time
    sim_time_to_ss_h = current_time_s / 3600
    
    df = pd.DataFrame(time_data)
    df.to_csv(csv_filename, index=False)
    
    result = {
        SWEEP_PARAM: sweep_value,
        'converged': converged,
        'status': status,
        'sim_time_to_ss_h': sim_time_to_ss_h,
        'wall_time_s': wall_time_s,
        'final_S2_M': S2.value[0],
        'final_I2_M': I2.value[0],
        'final_Th2_M': Th2.value[0],
        'final_C_I2_M': C_I2.value[0],
        'final_C_Th2_M': C_Th2.value[0],
        'final_S2_tot_M': S2.value[0] + C_I2.value[0] + C_Th2.value[0],
        'final_abs_change_M': last_max_abs_change,
        'csv_file': csv_filename
    }
    
    return result

if __name__ == '__main__':
    print(f"Starting parameter sweep: {SWEEP_PARAM}")
    print(f"Values: {SWEEP_VALUES}")
    print(f"Number of simulations: {len(SWEEP_VALUES)}")
    print(f"Convergence threshold: {SIM_PARAMS['abs_threshold']*1e9:.1f} nM")
    print(f"Max simulation time: {SIM_PARAMS['max_time_hours']} hours")
    print()
    
    # Prepare simulation arguments
    sim_args = []
    for sweep_value in SWEEP_VALUES:
        params = BASE_PARAMS.copy()
        params[SWEEP_PARAM] = sweep_value
        
        # Special handling for Phi_in sweep with I2_0 scaling
        if SWEEP_PARAM == "Phi_in" and SCALE_I2_WITH_PHI_IN:
            params['I2_0'] = (sweep_value / 1e-9) * BASE_PARAMS['I2_0']
        
        sim_args.append((sweep_value, params, SIM_PARAMS))
    
    # Run simulations in parallel
    num_processes = max(1, cpu_count() - 1)
    print(f"Running with {num_processes} processes\n")
    
    with Pool(num_processes) as pool:
        results = pool.map(run_simulation, sim_args)
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Create summary CSV with convenient units
    summary_df = results_df.copy()
    
    # Convert sweep parameter to convenient units
    if SWEEP_PARAM in ["Phi_in"]:
        summary_df[f'{SWEEP_PARAM}_nMps'] = summary_df[SWEEP_PARAM] * 1e9
        summary_df = summary_df.drop(columns=[SWEEP_PARAM])
    elif SWEEP_PARAM in ["I2_0", "Th2_0", "S2_0"]:
        summary_df[f'{SWEEP_PARAM}_nM'] = summary_df[SWEEP_PARAM] * 1e9
        summary_df = summary_df.drop(columns=[SWEEP_PARAM])
    
    # Convert final concentrations to nM
    for col in ['final_S2_M', 'final_I2_M', 'final_Th2_M', 'final_C_I2_M', 'final_C_Th2_M', 'final_S2_tot_M']:
        summary_df[col.replace('_M', '_nM')] = summary_df[col] * 1e9
        summary_df = summary_df.drop(columns=[col])
    
    summary_df['final_abs_change_nM'] = summary_df['final_abs_change_M'] * 1e9
    summary_df = summary_df.drop(columns=['final_abs_change_M'])
    
    summary_df.to_csv('summary.csv', index=False)
    
    # Save detailed steady state values (keep SI units)
    results_df.to_csv('steady_state_values.csv', index=False)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - summary.csv (convenient units)")
    print(f"  - steady_state_values.csv (SI units)")
    print(f"  - run_*.csv (time course data)")
    print(f"\nConverged: {summary_df['converged'].sum()} / {len(summary_df)}")