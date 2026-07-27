from __future__ import annotations

from fipy import CellVariable, Grid2D
import argparse
import os
import glob
import time as simtime
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import csv


from sender_receiver_tanh_nodes import (
    smooth_circle_profile, 
    double_peak_diffusion, 
    build_equations, 
    build_geometry, 
    clip_nonnegative, 
    initialize_variables,
    mean_in_mask
)

SWEEP_PARAMETER = 'center_distance_um'
SWEEP_VALUES = np.linspace(100,1500, num =26)
NUM_PROCESSES = cpu_count() - 2

# Available parameters to sweep:
# - 'node_length_um'
# - 'center_distance_um'
# - 'bath_margin_um'
# - 'dx_um'
# - 'total_hours'
# - 'dt_s'
# - 'd_gel_um2_s'
# - 'd_solution_um2_s'
# - 'k_p_s_inv'
# - 'k_d_ds_s_inv'
# - 'k_d_ss_s_inv'
# - 'k_slow_M_inv_s_inv'
# - 'k_fast_M_inv_s_inv'
# - 'sender_switch_nM'
# - 'receiver_switch_nM'
# - 'threshold_uM'
# - 'transition_sharpness'

# Output directory for results
OUTPUT_DIR = Path("Center_center_distance_sweep")
OUTPUT_CSV = OUTPUT_DIR / "sweep_results.csv"

# Simulation control parameters
SIM_CONFIG = {
    'check_interval': 100,      # Steps between convergence checks
    'save_interval': 10,         # Steps between CSV saves
    'abs_threshold': 1e-8,       # Convergence threshold (M) - 10 nM
    'progress_interval_s': 60,   # Progress print interval (wall-clock seconds)
}

# ============================================================================
# Unit conversions
MOLAR = 1.0
NANOMOLAR = 1e-9 * MOLAR
MICROMOLAR = 1e-6 * MOLAR


@dataclass
class SenderReceiverParams:
    node_length_um: float = 50.0
    center_distance_um: float = 300.0
    bath_margin_um: float = 250.0
    dx_um: float = 10.0
    total_hours: float = 10.0 #testing 10 hours to see if this gives me convergence
    dt_s: float = 60.0
    nonlinear_tolerance: float = 1e-9
    max_sweeps_per_step: int = 20

    d_gel_um2_s: float = 60.0
    d_solution_um2_s: float = 150.0
    k_p_s_inv: float = 0.2 # I swore I changed this to 0.2 - in real life it is 0.02
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0
    transition_sharpness: float = 20 # maybe 5 is too sharp, but how could anything be more sharp than mask case scenario? 
    def validate(self) -> None:
        """Validate parameter values."""
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")

def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = False):
    """Run the sender-receiver simulation and return final I2 concentration."""
    params.validate()
    
    # Build geometry and get center coordinates
    (mesh, nx, ny, sender_mask, receiver_mask,
     sender_center_x, sender_center_y,
     receiver_center_x, receiver_center_y) = build_geometry(params)
    
    # Initialize variables with smooth profiles
    vars_by_name = initialize_variables(
        mesh, sender_mask, receiver_mask, params,
        sender_center_x, sender_center_y,
        receiver_center_x, receiver_center_y
    )
    
    # Build equations
    eqs = build_equations(vars_by_name, params)

    # Prepare time-stepping
    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))

    # Variables that evolve in time
    dynamic_vars = (
        vars_by_name["S2"],
        vars_by_name["I2"],
        vars_by_name["S2_I2"],
        vars_by_name["Th2"],
        vars_by_name["S2_Th2"],
    )

    # Time-stepping loop
    for step in range(1, n_steps + 1):
        # Store old values
        for var in dynamic_vars:
            var.updateOld()

        # Nonlinear iteration loop
        residual = np.inf
        sweep_count = 0
        while residual > params.nonlinear_tolerance and sweep_count < params.max_sweeps_per_step:
            residual = 0.0
            residual = max(residual, eqs["S2"].sweep(var=vars_by_name["S2"], dt=params.dt_s))
            residual = max(residual, eqs["I2"].sweep(var=vars_by_name["I2"], dt=params.dt_s))
            residual = max(residual, eqs["S2_I2"].sweep(var=vars_by_name["S2_I2"], dt=params.dt_s))
            residual = max(residual, eqs["Th2"].sweep(var=vars_by_name["Th2"], dt=params.dt_s))
            residual = max(residual, eqs["S2_Th2"].sweep(var=vars_by_name["S2_Th2"], dt=params.dt_s))
            clip_nonnegative(vars_by_name)
            sweep_count += 1

        # Progress output
        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            receiver_i2_nM = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
            print(
                f"  step {step:4d}/{n_steps} | t = {step * params.dt_s / 3600.0:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM:8.3f} nM | "
                f"sweeps = {sweep_count:2d}"
            )

    # Return final I2 concentration
    final_i2_nM = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
    return final_i2_nM


def run_single_simulation(args):
    """
    Wrapper function to run a single simulation (for multiprocessing).
    
    Parameters:
    -----------
    args : tuple
        (index, param_name, param_value, total_count)
    
    Returns:
    --------
    dict : Result dictionary with parameter info and final I2
    """
    index, param_name, param_value, total_count = args
    
    # Create params with default values
    params = SenderReceiverParams()
    
    # Set the parameter to sweep
    setattr(params, param_name, param_value)
    
    # Run simulation
    print(f"[Process {os.getpid()}] Starting simulation {index+1}/{total_count}: {param_name} = {param_value}")
    sim_start = simtime.perf_counter()
    
    try:
        final_i2 = simulate_sender_receiver(params, verbose=False)
        sim_end = simtime.perf_counter()
        
        result = {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'final_i2_nM': final_i2,
            'simulation_time_s': sim_end - sim_start,
            'status': 'success'
        }
        
        print(f"[Process {os.getpid()}] Completed {index+1}/{total_count}: "
              f"{param_name} = {param_value}, I2 = {final_i2:.6f} nM "
              f"(time: {sim_end - sim_start:.2f}s)")
        
        return result
        
    except Exception as e:
        print(f"[Process {os.getpid()}] ERROR in simulation {index+1}/{total_count}: {str(e)}")
        return {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'final_i2_nM': None,
            'simulation_time_s': None,
            'status': f'error: {str(e)}'
        }


def run_parameter_sweep_parallel(param_name: str, param_values: list, output_csv: str, num_processes: int = None):
    """
    Run parameter sweep in parallel and save results to CSV.
    
    Parameters:
    -----------
    param_name : str
        Name of the parameter to sweep (must match a field in SenderReceiverParams)
    param_values : list
        List of values to sweep through
    output_csv : str
        Output CSV filename
    num_processes : int or None
        Number of processes to use (None = use all available cores)
    """
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    num_processes = min(num_processes, len(param_values))  # Don't use more processes than values
    
    print(f"\n{'='*70}")
    print(f"PARALLEL PARAMETER SWEEP: {param_name}")
    print(f"Values: {param_values}")
    print(f"Output: {output_csv}")
    print(f"Number of processes: {num_processes}")
    print(f"Total CPU cores available: {cpu_count()}")
    print(f"{'='*70}\n")
    
    # Prepare arguments for parallel execution
    args_list = [
        (i, param_name, value, len(param_values)) 
        for i, value in enumerate(param_values)
    ]
    
    # Run simulations in parallel
    sweep_start = simtime.perf_counter()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_simulation, args_list)
    
    sweep_end = simtime.perf_counter()
    
    # Sort results by parameter value
    results = sorted(results, key=lambda x: x['parameter_value'])
    
    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['parameter_name', 'parameter_value', 'final_i2_nM', 'simulation_time_s', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n{'='*70}")
    print(f"PARALLEL SWEEP COMPLETE")
    print(f"Total sweep time: {sweep_end - sweep_start:.2f} seconds")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*70}\n")
    
    # Print summary table
    print("\nSummary:")
    print(f"{'Parameter Value':<20} {'Final I2 (nM)':<15} {'Time (s)':<12} {'Status':<10}")
    print("-" * 60)
    for result in results:
        i2_str = f"{result['final_i2_nM']:.6f}" if result['final_i2_nM'] is not None else "N/A"
        time_str = f"{result['simulation_time_s']:.2f}" if result['simulation_time_s'] is not None else "N/A"
        print(f"{result['parameter_value']:<20} {i2_str:<15} {time_str:<12} {result['status']:<10}")
    
    # Calculate speedup estimate
    total_sim_time = sum(r['simulation_time_s'] for r in results if r['simulation_time_s'] is not None)
    if total_sim_time > 0:
        speedup = total_sim_time / (sweep_end - sweep_start)
        print(f"\nEstimated speedup: {speedup:.2f}x")
        print(f"(Total sequential time would be ~{total_sim_time:.2f}s)")


def main():
    """Main execution function."""
    start_time = simtime.perf_counter()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_parameter_sweep_parallel(
        param_name=SWEEP_PARAMETER,
        param_values=SWEEP_VALUES,
        output_csv=OUTPUT_CSV,
        num_processes=NUM_PROCESSES
    )
    
    end_time = simtime.perf_counter()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":

    main()