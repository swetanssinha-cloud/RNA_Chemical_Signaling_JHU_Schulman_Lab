"""
Sender-Receiver reaction-diffusion parameter sweep simulation.

This script runs parameter sweeps and saves time series data to CSV files.
Compatible with the analysis script that generates plots and metrics.
"""

from __future__ import annotations

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sender_receiver_tanh_nodes import smooth_circle_profile, double_peak_diffusion, build_equations, build_geometry, clip_nonnegative, mean_in_mask

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================
# Specify which parameter to sweep and the values to test

SWEEP_PARAMETER = 'threshold_uM'  # Parameter name to sweep
SWEEP_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]  # Values to test

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
    total_hours: float = 8.0
    dt_s: float = 60.0
    nonlinear_tolerance: float = 1e-9
    max_sweeps_per_step: int = 20

    d_gel_um2_s: float = 60.0
    d_solution_um2_s: float = 150.0
    k_p_s_inv: float = 0.02
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0
    transition_sharpness: float = 0.1

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
    R = 0.5 * params.node_length_um
    
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



def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = True):
    """
    Run the sender-receiver simulation.
    """
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

    # Prepare time-series storage
    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_s = np.zeros(n_steps + 1)
    receiver_i2_M = np.zeros(n_steps + 1)
    receiver_th2_M = np.zeros(n_steps + 1)
    receiver_s2_M = np.zeros(n_steps + 1)
    receiver_s2_i2_M = np.zeros(n_steps + 1)
    receiver_s2_th2_M = np.zeros(n_steps + 1)

    # Initial values
    receiver_i2_M[0] = mean_in_mask(vars_by_name["I2"], receiver_mask)
    receiver_th2_M[0] = mean_in_mask(vars_by_name["Th2"], receiver_mask)
    receiver_s2_M[0] = mean_in_mask(vars_by_name["S2"], receiver_mask)
    receiver_s2_i2_M[0] = mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
    receiver_s2_th2_M[0] = mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)

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

        # Store results (in Molar units for consistency with analysis script)
        times_s[step] = step * params.dt_s
        receiver_i2_M[step] = mean_in_mask(vars_by_name["I2"], receiver_mask)
        receiver_th2_M[step] = mean_in_mask(vars_by_name["Th2"], receiver_mask)
        receiver_s2_M[step] = mean_in_mask(vars_by_name["S2"], receiver_mask)
        receiver_s2_i2_M[step] = mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
        receiver_s2_th2_M[step] = mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)

        # Progress output
        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:4d}/{n_steps} | t = {times_s[step]/3600:.2f} h | "
                f"receiver I2 = {receiver_i2_M[step]/NANOMOLAR:8.3f} nM | "
                f"receiver Th2 = {receiver_th2_M[step]/NANOMOLAR:8.3f} nM | "
                f"sweeps = {sweep_count:2d} | residual = {residual:.3e}"
            )

    return {
        "times_s": times_s,
        "receiver_i2_M": receiver_i2_M,
        "receiver_th2_M": receiver_th2_M,
        "receiver_s2_M": receiver_s2_M,
        "receiver_s2_i2_M": receiver_s2_i2_M,
        "receiver_s2_th2_M": receiver_s2_th2_M,
    }


def run_parameter_sweep(sweep_param: str, sweep_values: list, output_dir: Path):
    """
    Run parameter sweep and save results.
    
    Parameters:
    -----------
    sweep_param : str
        Name of parameter to sweep
    sweep_values : list
        Values to test
    output_dir : Path
        Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base parameters
    base_params = SenderReceiverParams()
    
    # Storage for summary data
    summary_data = []
    
    print("="*80)
    print(f"PARAMETER SWEEP: {sweep_param}")
    print(f"Values: {sweep_values}")
    print("="*80)
    
    for idx, value in enumerate(sweep_values):
        print(f"\n[{idx+1}/{len(sweep_values)}] Running simulation with {sweep_param} = {value}")
        print("-"*80)
        
        # Create parameters with modified value
        params = base_params.copy_with(**{sweep_param: value})
        
        # Run simulation
        result = simulate_sender_receiver(params, verbose=True)
        
        # Save time series to CSV
        csv_filename = f"timeseries_{sweep_param}_{value:.6e}.csv"
        csv_path = output_dir / csv_filename
        
        df = pd.DataFrame({
            'time_s': result['times_s'],
            'I2_M': result['receiver_i2_M'],
            'Th2_M': result['receiver_th2_M'],
            'S2_M': result['receiver_s2_M'],
            'S2_I2_M': result['receiver_s2_i2_M'],
            'S2_Th2_M': result['receiver_s2_th2_M'],
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved time series to: {csv_path}")
        
        # Store summary information
        summary_data.append({
            'csv_file': str(csv_path),
            sweep_param: value,
            'final_I2_M': result['receiver_i2_M'][-1],
            'final_Th2_M': result['receiver_th2_M'][-1],
            'final_S2_M': result['receiver_s2_M'][-1],
            'converged': True,  # Placeholder - could add convergence check
            'sim_time_to_ss_h': params.total_hours,  # Placeholder
        })
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'steady_state_values.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Saved summary to: {summary_path}")
    print(f"{'='*80}")
    print("\nSweep complete! You can now run the analysis script to generate plots.")


def main():
    """Main execution function."""
    run_parameter_sweep(SWEEP_PARAMETER, SWEEP_VALUES, OUTPUT_DIR)


if __name__ == "__main__":
    main()