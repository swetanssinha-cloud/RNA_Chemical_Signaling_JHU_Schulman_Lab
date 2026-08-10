import numpy as np
import pandas as pd
import os
from fipy import CellVariable, TransientTerm, ImplicitSourceTerm, Grid1D
import time as simtime
from multiprocessing import Pool, cpu_count

OUTPUT_DIR = "well_mixed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
for f in os.listdir(OUTPUT_DIR):
    if f.startswith("run_Phi_in_") and f.endswith(".csv"):
        os.remove(os.path.join(OUTPUT_DIR, f))
    if f == "summary.csv":
        os.remove(os.path.join(OUTPUT_DIR, f))
    if f == "steady_state_values.csv":
        os.remove(os.path.join(OUTPUT_DIR, f))


def run_simulation_to_steady_state(Phi_in_value, params, 
                                   abs_threshold=1e-12,  # Absolute change threshold (1 pM)
                                   check_interval=100,
                                   max_total_steps=50000,
                                   max_time_hours=24,
                                   save_interval=1,
                                   verbose=True):
    """
    Run FiPy simulation until steady state, saving time course data.
    
    NOTE: Only checks convergence on I2, Th2, C_I2, C_Th2 (NOT S2!)
    S2 keeps increasing due to Phi_in influx, so it never reaches steady state.
    
    Parameters:
    -----------
    abs_threshold : float
        Absolute change threshold for convergence in Molar (default 1e-12 = 1 pM)
        System is at steady state when I2, Th2, C_I2, C_Th2 change by less than this
    check_interval : int
        How often to check for convergence (every N steps)
    max_total_steps : int
        Maximum steps before giving up
    max_time_hours : float
        Maximum SIMULATION time (in hours) before stopping
    save_interval : int
        How often to save data points (every N steps)
    verbose : bool
        Print progress updates
    
    Returns both the full time course AND steady state values.
    """
    
    # Start wall-clock timer
    wall_time_start = simtime.perf_counter()
    last_print_time = wall_time_start
    
    k_slow = params['k_slow']
    k_fast = params['k_fast']
    k_d_ds = params['k_d_ds']
    k_d_ss = params['k_d_ss']
    I2_0 = (Phi_in_value / 1e-9) * params['I2_0']
    Th2_0 = params['Th2_0']
    dt_s = params['dt_s']
    max_sweeps = params['max_sweeps']
    tol = params['tol']

    mesh = Grid1D(nx=1, dx=1.0)

    S2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    I2 = CellVariable(name="I2", mesh=mesh, value=I2_0, hasOld=True)
    Th2 = CellVariable(name="Th2", mesh=mesh, value=Th2_0, hasOld=True)
    C_I2 = CellVariable(name="C_I2", mesh=mesh, value=0.0, hasOld=True)
    C_Th2 = CellVariable(name="C_Th2", mesh=mesh, value=0.0, hasOld=True)

    eq_S2 = (TransientTerm(var=S2) ==
             ImplicitSourceTerm(coeff=-k_slow * I2, var=S2) +
             ImplicitSourceTerm(coeff=-k_fast * Th2, var=S2) +
             ImplicitSourceTerm(coeff=-k_d_ss, var=S2) +
             Phi_in_value)

    eq_I2 = (TransientTerm(var=I2) == 
            k_d_ds * C_I2 +
            ImplicitSourceTerm(coeff=-k_slow * S2, var=I2))

    eq_Th2 = (TransientTerm(var=Th2) == 
            k_d_ds * C_Th2 +
            ImplicitSourceTerm(coeff=-k_fast * S2, var=Th2))

    eq_C_I2 = (TransientTerm(var=C_I2) ==
               k_slow * S2 * I2 +
               ImplicitSourceTerm(coeff=-k_d_ds, var=C_I2))

    eq_C_Th2 = (TransientTerm(var=C_Th2) ==
                k_fast * S2 * Th2 +
                ImplicitSourceTerm(coeff=-k_d_ds, var=C_Th2))

    # Storage for time course
    time_history = []
    S2_history = []
    I2_history = []
    S2_tot_history = []
    tw50 = None
    
    # Storage for convergence check - ONLY for I2, Th2, C_I2, C_Th2 (NOT S2!)
    prev_values = None
    step = 0
    converged = False
    timeout = False
    max_sim_time_s = max_time_hours * 3600
    
    # For progress reporting
    last_max_abs_change = np.inf
    report_interval = check_interval * 10
    
    if verbose:
        print(f"  [Phi={Phi_in_value*1e9:.2f}] Starting simulation (max sim time: {max_time_hours} hours)...")
        print(f"  [Phi={Phi_in_value*1e9:.2f}] Absolute change threshold = {abs_threshold:.2e} M ({abs_threshold*1e9:.6f} nM)")
        print(f"  [Phi={Phi_in_value*1e9:.2f}] NOTE: Only checking I2, Th2, C_I2, C_Th2 for convergence (S2 grows indefinitely)")
    
    while step < max_total_steps and not converged and not timeout:
        S2.updateOld()
        I2.updateOld()
        Th2.updateOld()
        C_I2.updateOld()
        C_Th2.updateOld()

        for sweep in range(max_sweeps):
            eq_S2.sweep(dt=dt_s)
            eq_I2.sweep(dt=dt_s)
            eq_Th2.sweep(dt=dt_s)
            eq_C_I2.sweep(dt=dt_s)
            eq_C_Th2.sweep(dt=dt_s)

        step += 1
        current_time = step * dt_s
        
        # Check for timeout (SIMULATION TIME LIMIT)
        if current_time >= max_sim_time_s:
            timeout = True
            if verbose:
                print(f"  [Phi={Phi_in_value*1e9:.2f}] ⏱ TIMEOUT: Reached max simulation time of {max_time_hours} hours")
        
        # Print progress every 1 minute of WALL TIME (real time)
        current_wall_time = simtime.perf_counter()
        if verbose and (current_wall_time - last_print_time) >= 60:
            elapsed_wall = current_wall_time - wall_time_start
            print(f"  [Phi={Phi_in_value*1e9:.2f}] Still computing... "
                  f"Wall time: {elapsed_wall/60:.1f} min | "
                  f"Sim time: {current_time/3600:.2f} h | "
                  f"Step: {step} | "
                  f"Max abs change: {last_max_abs_change:.2e} M ({last_max_abs_change*1e9:.6f} nM) | "
                  f"Threshold: {abs_threshold:.2e} M")
            last_print_time = current_wall_time
        
        # Save time course data every save_interval steps
        if step % save_interval == 0:
            time_history.append(current_time)
            S2_history.append(S2.value[0])
            I2_history.append(I2.value[0])
            S2_tot_history.append(S2.value[0] + C_I2.value[0] + C_Th2.value[0])
            
            # Check for tw50 crossing
            if tw50 is None and I2.value[0] <= 50e-9 and I2.old[0] > 50e-9:
                tw50 = current_time
        
        # Check for convergence every check_interval steps
        # ONLY CHECK: I2, Th2, C_I2, C_Th2 (NOT S2 - it keeps growing!)
        if step % check_interval == 0:
            current_values = np.array([
                I2.value[0], Th2.value[0], 
                C_I2.value[0], C_Th2.value[0]
            ])
            
            if prev_values is not None:
                # Calculate ABSOLUTE change (not relative)
                abs_change = np.abs(current_values - prev_values)
                
                max_abs_change = np.max(abs_change)
                last_max_abs_change = max_abs_change
                
                # Progress report (every report_interval steps)
                if verbose and step % report_interval == 0:
                    wall_time_elapsed = simtime.perf_counter() - wall_time_start
                    print(f"  [Phi={Phi_in_value*1e9:.2f}] Step {step:6d} | "
                          f"Sim time: {current_time/3600:6.2f} h | "
                          f"Wall time: {wall_time_elapsed:6.1f} s | "
                          f"Max abs change: {max_abs_change:.2e} M ({max_abs_change*1e9:.6f} nM) | "
                          f"Threshold: {abs_threshold:.2e} M")
                
                # Check if all checked species have converged
                if np.all(abs_change < abs_threshold):
                    converged = True
                    if verbose:
                        print(f"  [Phi={Phi_in_value*1e9:.2f}] ✓ Converged at step {step}! "
                              f"(max abs change {max_abs_change:.2e} M < {abs_threshold:.2e} M)")
            
            prev_values = current_values.copy()
    
    # End wall-clock timer
    wall_time_end = simtime.perf_counter()
    wall_time_total = wall_time_end - wall_time_start
    
    if tw50 is None:
        tw50 = np.nan
    
    # Determine status
    if converged:
        status = "converged"
    elif timeout:
        status = "timeout"
    else:
        status = "max_steps"
    
    # Return both time course and steady state info
    return {
        'Phi_in_value': Phi_in_value,
        'time': np.array(time_history),
        'S2': np.array(S2_history),
        'I2': np.array(I2_history),
        'S2_tot': np.array(S2_tot_history),
        'tw50': tw50,
        'S2_ss': S2.value[0],
        'I2_ss': I2.value[0],
        'Th2_ss': Th2.value[0],
        'C_I2_ss': C_I2.value[0],
        'C_Th2_ss': C_Th2.value[0],
        'S2_tot_ss': S2.value[0] + C_I2.value[0] + C_Th2.value[0],
        'converged': converged,
        'status': status,
        'steps': step,
        'sim_time_to_ss_s': step * dt_s,
        'sim_time_to_ss_h': step * dt_s / 3600,
        'wall_time_s': wall_time_total,
        'final_abs_change': last_max_abs_change
    }


def run_single_simulation_wrapper(args):
    """
    Wrapper function for multiprocessing.
    Unpacks arguments and calls run_simulation_to_steady_state.
    """
    Phi_in_value, params, abs_threshold, check_interval, max_total_steps, max_time_hours, save_interval, verbose = args
    
    result = run_simulation_to_steady_state(
        Phi_in_value, params,
        abs_threshold=abs_threshold,
        check_interval=check_interval,
        max_total_steps=max_total_steps,
        max_time_hours=max_time_hours,
        save_interval=save_interval,
        verbose=verbose
    )
    
    return result


def save_run_to_csv(Phi_in_value, result, output_dir=OUTPUT_DIR):
    """Save time course data to CSV."""
    phi_label = f"{Phi_in_value * 1e9:.3f}".replace(".", "p")
    filename = os.path.join(output_dir, f"run_Phi_in_{phi_label}nMps.csv")

    df = pd.DataFrame({
        "time_s": result['time'],
        "S2_M": result['S2'],
        "I2_M": result['I2'],
        "S2_tot_M": result['S2_tot']
    })

    df.to_csv(filename, index=False)
    return filename


if __name__ == "__main__":
    params = {
        'k_slow': 1e5,
        'k_fast': 1e6,
        'k_d_ds': 3e-4,
        'k_d_ss': 3e-4,
        'I2_0': 100e-9,
        'Th2_0': 5000e-9,
        'dt_s': 60.0,
        'max_sweeps': 20,
        'tol': 1e-10
    }

    Phi_in_array = np.array([2,3,4,5]) * 1e-9

    # Simulation settings
    abs_threshold = 1e-8  # 1 pM absolute change threshold
    check_interval = 100
    max_total_steps = 50000
    max_time_hours = 8
    save_interval = 10
    verbose = True

    total_start_time = simtime.perf_counter()

    print("="*80)
    print("RUNNING SIMULATIONS TO STEADY STATE (MULTIPROCESSING)")
    print("="*80)
    print(f"Number of CPUs available: {cpu_count()}")
    print(f"Using {max(1, cpu_count()-1)} processes")
    print(f"Max simulation time per run: {max_time_hours} hours")
    print(f"Absolute change threshold: {abs_threshold:.2e} M ({abs_threshold*1e9:.6f} nM)")
    print(f"NOTE: Only checking I2, Th2, C_I2, C_Th2 for convergence (S2 grows indefinitely)")
    print("="*80)

    # Build list of argument tuples for each simulation
    args_list = [
        (Phi_in_val, params, abs_threshold, check_interval, max_total_steps, 
         max_time_hours, save_interval, verbose)
        for Phi_in_val in Phi_in_array
    ]

    # Run simulations in parallel
    with Pool(max(1, cpu_count()-1)) as pool:
        results = pool.map(run_single_simulation_wrapper, args_list)

    print("\n" + "="*80)
    print("ALL SIMULATIONS COMPLETE - PROCESSING RESULTS")
    print("="*80)

    summary_rows = []
    ss_rows = []

    # Process results
    for result in results:
        Phi_in_val = result['Phi_in_value']
        
        print(f"\nPhi_in = {Phi_in_val*1e9:.2f} nM/s")
        print("-" * 80)
        
        # Save time course to CSV
        csv_file = save_run_to_csv(Phi_in_val, result)
        
        # Store summary data
        summary_rows.append({
            "Phi_in_nMps": Phi_in_val * 1e9,
            "tw50_s": result['tw50'],
            "final_S2_nM": result['S2'][-1] * 1e9,
            "final_S2_tot_nM": result['S2_tot'][-1] * 1e9,
            "final_I2_nM": result['I2'][-1] * 1e9,
            "csv_file": csv_file
        })
        
        # Store steady state data
        ss_rows.append({
            'Phi_in_nMps': Phi_in_val * 1e9,
            'I2_ss_nM': result['I2_ss'] * 1e9,
            'S2_ss_nM': result['S2_ss'] * 1e9,
            'Th2_ss_nM': result['Th2_ss'] * 1e9,
            'S2_tot_ss_nM': result['S2_tot_ss'] * 1e9,
            'sim_time_to_ss_h': result['sim_time_to_ss_h'],
            'wall_time_s': result['wall_time_s'],
            'steps': result['steps'],
            'status': result['status'],
            'converged': result['converged'],
            'final_abs_change_M': result['final_abs_change'],
            'final_abs_change_nM': result['final_abs_change'] * 1e9
        })
        
        # Print result summary
        if result['converged']:
            print(f"  ✓ CONVERGED")
            print(f"    Simulation time to steady state: {result['sim_time_to_ss_h']:.2f} hours ({result['steps']} steps)")
            print(f"    Wall clock time (computation):   {result['wall_time_s']:.2f} seconds")
            print(f"    Final max abs change: {result['final_abs_change']:.2e} M ({result['final_abs_change']*1e9:.6f} nM)")
            print(f"    Threshold: {abs_threshold:.2e} M ({abs_threshold*1e9:.6f} nM)")
            print(f"    I2_ss   = {result['I2_ss']*1e9:.6f} nM")
            print(f"    S2_ss   = {result['S2_ss']*1e9:.6f} nM (still growing)")
            print(f"    Th2_ss  = {result['Th2_ss']*1e9:.6f} nM")
        else:
            print(f"  ✗ DID NOT CONVERGE ({result['status'].upper()})")
            print(f"    Stopped at: {result['sim_time_to_ss_h']:.2f} hours ({result['steps']} steps)")
            print(f"    Wall clock time: {result['wall_time_s']:.2f} seconds")
            print(f"    Final values:")
            print(f"      I2   = {result['I2_ss']*1e9:.6f} nM")
            print(f"      S2   = {result['S2_ss']*1e9:.6f} nM (still growing)")
            print(f"      Th2  = {result['Th2_ss']*1e9:.6f} nM")
            print(f"    Final max absolute change: {result['final_abs_change']:.2e} M ({result['final_abs_change']*1e9:.6f} nM)")
            print(f"    (Target threshold: {abs_threshold:.2e} M)")
        
        print(f"    Saved: {csv_file}")

    # Save summary files
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    
    ss_df = pd.DataFrame(ss_rows)
    ss_df.to_csv(os.path.join(OUTPUT_DIR, "steady_state_values.csv"), index=False)

    total_end_time = simtime.perf_counter()
    total_wall_time = total_end_time - total_start_time
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total wall clock time for all simulations: {total_wall_time:.2f} seconds ({total_wall_time/60:.1f} minutes)")
    print(f"  (Average per simulation: {total_wall_time/len(Phi_in_array):.2f} seconds)")
    print(f"\nSpeedup from multiprocessing:")
    total_sequential_time = sum([row['wall_time_s'] for row in ss_rows])
    print(f"  Sequential time would be: {total_sequential_time:.2f} seconds")
    print(f"  Parallel time was:        {total_wall_time:.2f} seconds")
    print(f"  Speedup factor:           {total_sequential_time/total_wall_time:.2f}x")
    print(f"\nFiles saved:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'summary.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'steady_state_values.csv')}")
    print(f"  - Individual run CSVs in {OUTPUT_DIR}/")
    print("\nConvergence summary:")
    print(ss_df[['Phi_in_nMps', 'status', 'sim_time_to_ss_h', 'wall_time_s', 'converged', 'final_abs_change_nM']])
    print("="*80)