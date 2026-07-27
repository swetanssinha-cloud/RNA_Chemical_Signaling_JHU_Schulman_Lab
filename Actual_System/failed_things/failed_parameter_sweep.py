"""
Sender-Receiver reaction-diffusion parameter sweep simulation with parallel execution.

This script runs parameter sweeps in parallel with convergence detection and saves 
time series data to CSV files. Compatible with analysis scripts that generate plots 
and metrics.
"""

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
from sender_receiver_tanh_nodes import (
    smooth_circle_profile, 
    double_peak_diffusion, 
    build_equations, 
    build_geometry, 
    clip_nonnegative, 
    mean_in_mask
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================
# Specify which parameter to sweep and the values to test


SWEEP_PARAMETER = 'center_distance_um'
SWEEP_VALUES = np.linspace(100,1500, num =6)

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
OUTPUT_DIR = Path("sweep_results")

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
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")

    def copy_with(self, **kwargs):
        """Create a copy of params with specified fields modified."""
        params_dict = {
            'node_length_um': self.node_length_um,
            'center_distance_um': self.center_distance_um,
            'bath_margin_um': self.bath_margin_um,
            'dx_um': self.dx_um,
            'total_hours': self.total_hours,
            'dt_s': self.dt_s,
            'nonlinear_tolerance': self.nonlinear_tolerance,
            'max_sweeps_per_step': self.max_sweeps_per_step,
            'd_gel_um2_s': self.d_gel_um2_s,
            'd_solution_um2_s': self.d_solution_um2_s,
            'k_p_s_inv': self.k_p_s_inv,
            'k_d_ds_s_inv': self.k_d_ds_s_inv,
            'k_d_ss_s_inv': self.k_d_ss_s_inv,
            'k_slow_M_inv_s_inv': self.k_slow_M_inv_s_inv,
            'k_fast_M_inv_s_inv': self.k_fast_M_inv_s_inv,
            'sender_switch_nM': self.sender_switch_nM,
            'receiver_switch_nM': self.receiver_switch_nM,
            'threshold_uM': self.threshold_uM,
            'transition_sharpness': self.transition_sharpness,
        }
        params_dict.update(kwargs)
        return SenderReceiverParams(**params_dict)


def initialize_variables(
    mesh,
    sender_mask,
    receiver_mask,
    params: SenderReceiverParams,
    sender_center_x: float,
    sender_center_y: float,
    receiver_center_x: float,
    receiver_center_y: float
):
    """
    Initialize all CellVariables with smooth spatial profiles.
    """
    # Get cell coordinates
    x = np.asarray(mesh.cellCenters[0].value)
    y = np.asarray(mesh.cellCenters[1].value)

    # Radius of circular nodes (half of node_length)
    R = (params.node_length_um) / np.sqrt(np.pi)  

    # Sharpness parameter for transitions
    c = params.transition_sharpness

    # 1. DIFFUSION COEFFICIENT (static, spatially-varying)
    D_gel = params.d_gel_um2_s
    D_solution = params.d_solution_um2_s
    H_diffusion = D_gel - D_solution
    U_diffusion = D_solution

    D_values = double_peak_diffusion(
        x, y,
        sender_center_x, sender_center_y,
        receiver_center_x, receiver_center_y,
        R, H_diffusion, U_diffusion, c)

    diffusion = CellVariable(name="D", mesh=mesh, value=D_values)

    # 2. I1O2 (static source term at sender)
    H_i1o2 = params.sender_switch_nM * NANOMOLAR
    U_i1o2 = 0.0

    i1o2_values = smooth_circle_profile(
        x, y,
        sender_center_x, sender_center_y,
        R, H_i1o2, U_i1o2, c)
    i1o2 = CellVariable(name="I1O2", mesh=mesh, value=i1o2_values)

    # 3. I2 (dynamic variable, initial condition at receiver)
    H_i2 = params.receiver_switch_nM * NANOMOLAR
    U_i2 = 0.0

    i2_initial = smooth_circle_profile(
        x, y,
        receiver_center_x, receiver_center_y,
        R, H_i2, U_i2, c
    )
    i2 = CellVariable(name="I2", mesh=mesh, value=i2_initial, hasOld=True)

    # 4. Th2 (dynamic variable, initial condition at receiver)
    H_th2 = params.threshold_uM * MICROMOLAR
    U_th2 = 0.0

    th2_initial = smooth_circle_profile(
        x, y,
        receiver_center_x, receiver_center_y,
        R, H_th2, U_th2, c
    )
    th2 = CellVariable(name="Th2", mesh=mesh, value=th2_initial, hasOld=True)

    # 5. Other variables (start at zero everywhere)
    s2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    s2_i2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
    s2_th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

    return {
        "S2": s2,
        "I2": i2,
        "S2_I2": s2_i2,
        "Th2": th2,
        "S2_Th2": s2_th2,
        "I1O2": i1o2,
        "D": diffusion,
    }


def run_simulation(args):
    """
    Run a single sender-receiver simulation with given parameters.
    This function is designed to be called by multiprocessing.Pool.
    """
    sweep_value, base_params, sweep_param, sim_config = args
    
    # Create parameters with modified value
    params = base_params.copy_with(**{sweep_param: sweep_value})
    params.validate()

    # Extract simulation config
    check_interval = sim_config['check_interval']
    save_interval = sim_config['save_interval']
    abs_threshold = sim_config['abs_threshold']
    progress_interval_s = sim_config['progress_interval_s']

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

    # Prepare time-series storage (sparse)
    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    time_data = []

    # Variables that evolve in time
    dynamic_vars = (
        vars_by_name["S2"],
        vars_by_name["I2"],
        vars_by_name["S2_I2"],
        vars_by_name["Th2"],
        vars_by_name["S2_Th2"],
    )

    # Initial values (save at t=0)
    time_data.append({
        'time_s': 0.0,
        'I2_M': mean_in_mask(vars_by_name["I2"], receiver_mask),
        'Th2_M': mean_in_mask(vars_by_name["Th2"], receiver_mask),
        'S2_M': mean_in_mask(vars_by_name["S2"], receiver_mask),
        'S2_I2_M': mean_in_mask(vars_by_name["S2_I2"], receiver_mask),
        'S2_Th2_M': mean_in_mask(vars_by_name["S2_Th2"], receiver_mask),
    })

    # Convergence tracking - track 5 species including S2
    prev_values = np.array([
        mean_in_mask(vars_by_name["S2"], receiver_mask),
        mean_in_mask(vars_by_name["I2"], receiver_mask),
        mean_in_mask(vars_by_name["S2_I2"], receiver_mask),
        mean_in_mask(vars_by_name["Th2"], receiver_mask),
        mean_in_mask(vars_by_name["S2_Th2"], receiver_mask),
    ])
    
    converged = False
    status = "timeout"
    last_max_abs_change = np.inf

    start_wall_time = simtime.time()
    last_progress_time = start_wall_time

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

        current_time_s = step * params.dt_s

        # Save data at specified intervals
        if step % save_interval == 0:
            time_data.append({
                'time_s': current_time_s,
                'I2_M': mean_in_mask(vars_by_name["I2"], receiver_mask),
                'Th2_M': mean_in_mask(vars_by_name["Th2"], receiver_mask),
                'S2_M': mean_in_mask(vars_by_name["S2"], receiver_mask),
                'S2_I2_M': mean_in_mask(vars_by_name["S2_I2"], receiver_mask),
                'S2_Th2_M': mean_in_mask(vars_by_name["S2_Th2"], receiver_mask),
            })

        # Check convergence at specified intervals
        if step % check_interval == 0:
            current_values = np.array([
                mean_in_mask(vars_by_name["S2"], receiver_mask),
                mean_in_mask(vars_by_name["I2"], receiver_mask),
                mean_in_mask(vars_by_name["S2_I2"], receiver_mask),
                mean_in_mask(vars_by_name["Th2"], receiver_mask),
                mean_in_mask(vars_by_name["S2_Th2"], receiver_mask),
            ])
            abs_changes = np.abs(current_values - prev_values)
            max_abs_change = np.max(abs_changes)
            last_max_abs_change = max_abs_change

            if max_abs_change < abs_threshold:
                converged = True
                status = "converged"
                # Save final point before breaking
                if step % save_interval != 0:
                    time_data.append({
                        'time_s': current_time_s,
                        'I2_M': current_values[1],
                        'Th2_M': current_values[3],
                        'S2_M': current_values[0],
                        'S2_I2_M': current_values[2],
                        'S2_Th2_M': current_values[4],
                    })
                break

            prev_values = current_values.copy()

        # Progress output based on wall-clock time
        current_wall_time = simtime.time()
        if current_wall_time - last_progress_time >= progress_interval_s:
            elapsed_wall = current_wall_time - start_wall_time
            sim_hours = current_time_s / 3600
            print(f"  [{sweep_param}={sweep_value:.3e}] t={sim_hours:.2f}h, "
                  f"wall={elapsed_wall:.1f}s, max_Δ={last_max_abs_change:.2e}M")
            last_progress_time = current_wall_time

    end_wall_time = simtime.time()
    wall_time_s = end_wall_time - start_wall_time
    sim_time_to_ss_h = current_time_s / 3600

    # Get final values
    final_values = {
        'S2_M': mean_in_mask(vars_by_name["S2"], receiver_mask),
        'I2_M': mean_in_mask(vars_by_name["I2"], receiver_mask),
        'Th2_M': mean_in_mask(vars_by_name["Th2"], receiver_mask),
        'S2_I2_M': mean_in_mask(vars_by_name["S2_I2"], receiver_mask),
        'S2_Th2_M': mean_in_mask(vars_by_name["S2_Th2"], receiver_mask),
    }

    # Save time series to CSV
    csv_filename = f"timeseries_{sweep_param}_{sweep_value:.6e}.csv"
    csv_path = OUTPUT_DIR / csv_filename

    df = pd.DataFrame(time_data)
    df.to_csv(csv_path, index=False)

    # Return result summary
    result = {
        sweep_param: sweep_value,
        'converged': converged,
        'status': status,
        'sim_time_to_ss_h': sim_time_to_ss_h,
        'wall_time_s': wall_time_s,
        'final_S2_M': final_values['S2_M'],
        'final_I2_M': final_values['I2_M'],
        'final_Th2_M': final_values['Th2_M'],
        'final_S2_I2_M': final_values['S2_I2_M'],
        'final_S2_Th2_M': final_values['S2_Th2_M'],
        'final_abs_change_M': last_max_abs_change,
        'csv_file': csv_filename
    }

    return result


def format_summary(results_df, sweep_param):
    """
    Convert results dataframe to human-readable units (nM, μM, hours).
    """
    summary_df = results_df.copy()

    # Convert sweep parameter to convenient units
    if sweep_param.endswith('_nM'):
        # Already in nM, keep as is
        pass
    elif sweep_param.endswith('_uM') or sweep_param == 'threshold_uM':
        summary_df[f'{sweep_param}_display'] = summary_df[sweep_param]
    elif sweep_param in ['sender_switch_nM', 'receiver_switch_nM']:
        # Already in nM in the param name
        pass
    elif 'M_inv' in sweep_param or 's_inv' in sweep_param:
        # Rate constants - keep scientific notation
        pass
    else:
        # For concentration parameters, try to detect and convert
        if sweep_param in ['threshold_uM']:
            pass  # Keep as is
        else:
            pass  # Keep as is for other parameters

    # Convert final concentrations to nM
    for col in ['final_S2_M', 'final_I2_M', 'final_Th2_M', 'final_S2_I2_M', 'final_S2_Th2_M']:
        if col in summary_df.columns:
            summary_df[col.replace('_M', '_nM')] = summary_df[col] * 1e9
            summary_df = summary_df.drop(columns=[col])

    if 'final_abs_change_M' in summary_df.columns:
        summary_df['final_abs_change_nM'] = summary_df['final_abs_change_M'] * 1e9
        summary_df = summary_df.drop(columns=['final_abs_change_M'])

    return summary_df


def run_parameter_sweep(sweep_param: str, sweep_values: list, output_dir: Path, 
                        base_params: SenderReceiverParams, sim_config: dict):
    """
    Run parameter sweep in parallel and save results.

    Parameters:
    -----------
    sweep_param : str
        Name of parameter to sweep
    sweep_values : list
        Values to test
    output_dir : Path
        Directory to save results
    base_params : SenderReceiverParams
        Base parameter set
    sim_config : dict
        Simulation control parameters
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old output files
    print("Cleaning up old output files...")
    for pattern in ['timeseries_*.csv', 'summary.csv', 'steady_state_values.csv']:
        for f in glob.glob(str(output_dir / pattern)):
            os.remove(f)
            print(f"  Deleted: {f}")

    print("="*80)
    print(f"PARAMETER SWEEP: {sweep_param}")
    print(f"Values: {sweep_values}")
    print(f"Number of simulations: {len(sweep_values)}")
    print(f"Convergence threshold: {sim_config['abs_threshold']*1e9:.1f} nM")
    print(f"Max simulation time: {base_params.total_hours} hours")
    print("="*80)
    print()

    # Prepare simulation arguments
    sim_args = []
    for sweep_value in sweep_values:
        sim_args.append((sweep_value, base_params, sweep_param, sim_config))

    # Run simulations in parallel
    num_processes = max(1, cpu_count() - 1)
    print(f"Running with {num_processes} processes")
    print()

    start_time = simtime.time()
    
    with Pool(num_processes) as pool:
        results = pool.map(run_simulation, sim_args)

    total_time = simtime.time() - start_time

    # Save results
    results_df = pd.DataFrame(results)

    # Save detailed steady state values (SI units)
    steady_state_path = output_dir / 'steady_state_values.csv'
    results_df.to_csv(steady_state_path, index=False)

    # Create summary CSV with convenient units
    summary_df = format_summary(results_df, sweep_param)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print()
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"\nTotal wall time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nResults saved to:")
    print(f"  - {summary_path} (convenient units)")
    print(f"  - {steady_state_path} (SI units)")
    print(f"  - timeseries_*.csv (time course data)")
    print(f"\nConverged: {summary_df['converged'].sum()} / {len(summary_df)}")
    
    if 'status' in summary_df.columns:
        print(f"\nStatus summary:")
        print(summary_df['status'].value_counts().to_string())


def main():
    """Main execution function."""
    base_params = SenderReceiverParams()
    run_parameter_sweep(
        SWEEP_PARAMETER, 
        SWEEP_VALUES, 
        OUTPUT_DIR, 
        base_params,
        SIM_CONFIG
    )


if __name__ == "__main__":
    start_time = simtime.perf_counter()
    main()
    end_time = simtime.perf_counter()
    print(f"sim time: {end_time - start_time}")