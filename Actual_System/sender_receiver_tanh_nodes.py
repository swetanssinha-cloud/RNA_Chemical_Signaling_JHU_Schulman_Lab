"""
Sender-Receiver reaction-diffusion simulation with smooth spatial profiles.

This uses smooth tanh-based functions for initial conditions and spatially-varying
diffusion coefficients, replacing the previous mask-based approach.
"""

from __future__ import annotations

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm

import argparse
import os
from dataclasses import dataclass
import time as simtime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


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
    k_p_s_inv: float = 0.2 #actual value is 0.02 but the paper uses 0.2 to speed things up
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0
    
    # New parameter for smooth transition sharpness
    transition_sharpness: float = 1 #something that should be changed - i think as c goes to inf, sharper and sharper 

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")


def apply_preset(params: SenderReceiverParams, preset: str | None) -> SenderReceiverParams:
    """Apply a named parameter preset."""
    if not preset:
        return params
    if preset == "comsol-2-1":
        params.node_length_um = 75.0
        params.center_distance_um = 175.0
        params.bath_margin_um = 2375.0
        params.d_gel_um2_s = 60.0
        params.d_solution_um2_s = 150.0
        params.k_p_s_inv = 0.2
        params.k_d_ds_s_inv = 3e-4
        params.k_d_ss_s_inv = 3e-4
        params.k_slow_M_inv_s_inv = 1e5
        params.k_fast_M_inv_s_inv = 1e6
        params.sender_switch_nM = 100.0
        params.receiver_switch_nM = 100.0
        params.threshold_uM = 10.0
        return params
    raise ValueError(f"Unknown preset: {preset}")


def smooth_circle_profile(x, y, h, k, R, H, U, c):
    """
    Smooth transition from U (outside) to H+U (inside) a circle.
    
    Uses the formula:
    (H/2) * (tanh(c * (R - r)) + 1) + U
    
    where r = sqrt((x-h)^2 + (y-k)^2)
    
    Parameters:
    -----------
    x, y : np.ndarray
        Coordinate arrays (cell centers)
    h, k : float
        Center coordinates of the circle
    R : float
        Radius of the circle
    H : float
        Height above base value (concentration inside - concentration outside)
    U : float
        Base value (concentration outside the circle)
    c : float
        Sharpness parameter (larger = sharper transition)
    
    Returns:
    --------
    np.ndarray
        Array of values at each (x,y) point
    """
    r = np.sqrt((x - h)**2 + (y - k)**2)
    return (H / 2.0) * (np.tanh(c * (R - r)) + 1.0) + U


def double_peak_diffusion(x, y, h1, k1, h2, k2, R, H, U, c):
    """
    Diffusion coefficient with two peaks (sender and receiver nodes).
    
    Uses the formula:
    (H/2) * (1 - tanh(c * (r1 - R))) + (H/2) * (1 - tanh(c * (r2 - R))) + U
    
    where r1 = distance from first center, r2 = distance from second center
    
    Parameters:
    -----------
    x, y : np.ndarray
        Coordinate arrays (cell centers)
    h1, k1 : float
        Center coordinates of first peak (sender)
    h2, k2 : float
        Center coordinates of second peak (receiver)
    R : float
        Radius of both peaks
    H : float
        Height of peaks above base (D_gel - D_solution)
    U : float
        Base diffusion value (D_solution)
    c : float
        Sharpness parameter
    
    Returns:
    --------
    np.ndarray
        Array of diffusion coefficient values at each (x,y) point
    """
    r1 = np.sqrt((x - h1)**2 + (y - k1)**2)
    r2 = np.sqrt((x - h2)**2 + (y - k2)**2)
    
    peak1 = (H / 2.0) * (1.0 - np.tanh(c * (r1 - R)))
    peak2 = (H / 2.0) * (1.0 - np.tanh(c * (r2 - R)))
    
    return peak1 + peak2 + U


def build_geometry(params: SenderReceiverParams):
    """
    Build the mesh and calculate geometry parameters.
    
    Returns:
    --------
    tuple containing:
        - mesh: FiPy Grid2D mesh
        - nx, ny: number of cells in x and y
        - sender_mask, receiver_mask: boolean masks for averaging/plotting
        - sender_center_x, sender_center_y: sender node center
        - receiver_center_x, receiver_center_y: receiver node center
    """
    width_um = 2.0 * params.bath_margin_um + params.center_distance_um + params.node_length_um
    height_um = 2.0 * params.bath_margin_um + params.node_length_um

    nx = int(np.ceil(width_um / params.dx_um))
    ny = int(np.ceil(height_um / params.dx_um))
    mesh = Grid2D(dx=params.dx_um, dy=params.dx_um, nx=nx, ny=ny)

    x = np.asarray(mesh.cellCenters[0].value)
    y = np.asarray(mesh.cellCenters[1].value)

    # Calculate center positions
    sender_center_x = params.bath_margin_um + 0.5 * params.node_length_um
    sender_center_y = 0.5 * height_um
    receiver_center_x = sender_center_x + params.center_distance_um
    receiver_center_y = sender_center_y

    # Create masks for averaging and plotting (using circular regions)
    R = 0.5 * params.node_length_um
    sender_mask = (np.sqrt((x - sender_center_x)**2 + (y - sender_center_y)**2) <= R)
    receiver_mask = (np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2) <= R)

    return (
        mesh, nx, ny, 
        sender_mask, receiver_mask,
        sender_center_x, sender_center_y,
        receiver_center_x, receiver_center_y
    )


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
    
    Parameters:
    -----------
    mesh : Grid2D
        FiPy mesh
    sender_mask, receiver_mask : np.ndarray
        Boolean masks (kept for compatibility with averaging functions)
    params : SenderReceiverParams
        Simulation parameters
    sender_center_x, sender_center_y : float
        Sender node center coordinates
    receiver_center_x, receiver_center_y : float
        Receiver node center coordinates
    
    Returns:
    --------
    dict
        Dictionary of CellVariables
    """
    # Get cell coordinates
    x = np.asarray(mesh.cellCenters[0].value)
    y = np.asarray(mesh.cellCenters[1].value)
    
    # Radius of circular nodes (half of node_length)
    R = 0.5 * params.node_length_um
    
    # Sharpness parameter for transitions
    c = params.transition_sharpness
    
    # ========================================
    # 1. DIFFUSION COEFFICIENT (static, spatially-varying)
    # ========================================
    # D has high values (d_gel) inside both nodes, low values (d_solution) outside
    D_gel = params.d_gel_um2_s
    D_solution = params.d_solution_um2_s
    H_diffusion = D_gel - D_solution  # Height of peaks above base
    U_diffusion = D_solution  # Base value
    
    D_values = double_peak_diffusion(
        x, y,
        sender_center_x, sender_center_y,
        receiver_center_x, receiver_center_y,
        R, H_diffusion, U_diffusion, c)
    
    diffusion = CellVariable(name="D", mesh=mesh, value=D_values)
    
    # ========================================
    # 2. I1O2 (static source term at sender)
    # ========================================
    # High concentration inside sender, zero outside

    H_i1o2 = params.sender_switch_nM * NANOMOLAR
    U_i1o2 = 0.0
    
    i1o2_values = smooth_circle_profile(
        x, y,
        sender_center_x, sender_center_y,
        R, H_i1o2, U_i1o2, c)
    i1o2 = CellVariable(name="I1O2", mesh=mesh, value=i1o2_values)
    
    # ========================================
    # 3. I2 (dynamic variable, initial condition at receiver)
    # ========================================
    # High concentration inside receiver, zero outside
    H_i2 = params.receiver_switch_nM * NANOMOLAR
    U_i2 = 0.0
    
    i2_initial = smooth_circle_profile(
        x, y,
        receiver_center_x, receiver_center_y,
        R, H_i2, U_i2, c
    )
    i2 = CellVariable(name="I2", mesh=mesh, value=i2_initial, hasOld=True)
    
    # ========================================
    # 4. Th2 (dynamic variable, initial condition at receiver)
    # ========================================
    # High concentration inside receiver, zero outside
    # Same center as I2, but different height
    H_th2 = params.threshold_uM * MICROMOLAR
    U_th2 = 0.0
    
    th2_initial = smooth_circle_profile(
        x, y,
        receiver_center_x, receiver_center_y,
        R, H_th2, U_th2, c
    )
    th2 = CellVariable(name="Th2", mesh=mesh, value=th2_initial, hasOld=True)
    
    # ========================================
    # 5. Other variables (start at zero everywhere)
    # ========================================
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


def build_equations(vars_by_name, params: SenderReceiverParams):
    """
    Build the system of PDEs.
    
    These equations are unchanged from the original code.
    """
    s2 = vars_by_name["S2"]
    i2 = vars_by_name["I2"]
    s2_i2 = vars_by_name["S2_I2"]
    th2 = vars_by_name["Th2"]
    s2_th2 = vars_by_name["S2_Th2"]
    i1o2 = vars_by_name["I1O2"]
    diffusion = vars_by_name["D"]

    eq_s2 = (
        TransientTerm(var=s2)
        == DiffusionTerm(coeff=diffusion, var=s2)
        + params.k_p_s_inv * i1o2
        - ImplicitSourceTerm(
            coeff=(
                params.k_slow_M_inv_s_inv * i2
                + params.k_fast_M_inv_s_inv * th2
                + params.k_d_ss_s_inv
            ),
            var=s2,
        )
    )

    eq_i2 = (
        TransientTerm(var=i2)
        == params.k_d_ds_s_inv * s2_i2
        - ImplicitSourceTerm(coeff=params.k_slow_M_inv_s_inv * s2, var=i2)
    )

    eq_th2 = (
        TransientTerm(var=th2)
        == params.k_d_ds_s_inv * s2_th2
        - ImplicitSourceTerm(coeff=params.k_fast_M_inv_s_inv * s2, var=th2)
    )

    eq_s2_i2 = (
        TransientTerm(var=s2_i2)
        == params.k_slow_M_inv_s_inv * i2 * s2
        - ImplicitSourceTerm(coeff=params.k_d_ds_s_inv, var=s2_i2)
    )

    eq_s2_th2 = (
        TransientTerm(var=s2_th2)
        == params.k_fast_M_inv_s_inv * th2 * s2
        - ImplicitSourceTerm(coeff=params.k_d_ds_s_inv, var=s2_th2)
    )

    return {
        "S2": eq_s2,
        "I2": eq_i2,
        "Th2": eq_th2,
        "S2_I2": eq_s2_i2,
        "S2_Th2": eq_s2_th2,
    }


def clip_nonnegative(vars_by_name):
    """Ensure all concentrations remain non-negative."""
    for name in ("S2", "I2", "S2_I2", "Th2", "S2_Th2"):
        var = vars_by_name[name]
        var.setValue(np.maximum(np.asarray(var.value), 0.0))


def mean_in_mask(var: CellVariable, mask: np.ndarray) -> float:
    """Calculate mean value of a variable within a masked region."""
    values = np.asarray(var.value)
    return float(values[mask].mean())


def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = True):
    """
    Run the sender-receiver simulation.
    
    Parameters:
    -----------
    params : SenderReceiverParams
        Simulation parameters
    verbose : bool
        Print progress updates
    
    Returns:
    --------
    dict
        Results dictionary containing simulation data
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
    times_h = np.zeros(n_steps + 1)
    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_th2_nM = np.zeros(n_steps + 1)
    receiver_s2_nM = np.zeros(n_steps + 1)
    receiver_s2_i2_nM = np.zeros(n_steps + 1)
    receiver_s2_th2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)

    # Initial values
    receiver_i2_nM[0] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
    receiver_th2_nM[0] = mean_in_mask(vars_by_name["Th2"], receiver_mask) / NANOMOLAR
    receiver_s2_nM[0] = mean_in_mask(vars_by_name["S2"], receiver_mask) / NANOMOLAR
    receiver_s2_i2_nM[0] = mean_in_mask(vars_by_name["S2_I2"], receiver_mask) / NANOMOLAR
    receiver_s2_th2_nM[0] = mean_in_mask(vars_by_name["S2_Th2"], receiver_mask) / NANOMOLAR
    receiver_total_rna_nM[0] = receiver_s2_nM[0] + receiver_s2_i2_nM[0] + receiver_s2_th2_nM[0]

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

        # Store results
        times_h[step] = step * params.dt_s / 3600.0
        receiver_i2_nM[step] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
        receiver_th2_nM[step] = mean_in_mask(vars_by_name["Th2"], receiver_mask) / NANOMOLAR
        receiver_s2_nM[step] = mean_in_mask(vars_by_name["S2"], receiver_mask) / NANOMOLAR
        receiver_s2_i2_nM[step] = mean_in_mask(vars_by_name["S2_I2"], receiver_mask) / NANOMOLAR
        receiver_s2_th2_nM[step] = mean_in_mask(vars_by_name["S2_Th2"], receiver_mask) / NANOMOLAR
        receiver_total_rna_nM[step] = receiver_s2_nM[step] + receiver_s2_i2_nM[step] + receiver_s2_th2_nM[step]

        # Progress output
        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:4d}/{n_steps} | t = {times_h[step]:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM[step]:8.3f} nM | "
                f"receiver Th2 = {receiver_th2_nM[step]:8.3f} nM | "
                f"receiver S2 = {receiver_s2_nM[step]:8.3f} nM | "
                f"receiver total RNA = {receiver_total_rna_nM[step]:8.3f} nM | "
                f"sweeps = {sweep_count:2d} | residual = {residual:.3e}"
            )

    return {
        "params": params,
        "mesh": mesh,
        "nx": nx,
        "ny": ny,
        "sender_mask": sender_mask,
        "receiver_mask": receiver_mask,
        "vars": vars_by_name,
        "times_h": times_h,
        "receiver_i2_nM": receiver_i2_nM,
        "receiver_th2_nM": receiver_th2_nM,
        "receiver_s2_nM": receiver_s2_nM,
        "receiver_s2_i2_nM": receiver_s2_i2_nM,
        "receiver_s2_th2_nM": receiver_s2_th2_nM,
        "receiver_total_rna_nM": receiver_total_rna_nM,
    }


def field_to_image(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Convert 1D field values to 2D image array."""
    return np.asarray(values).reshape((nx, ny), order="F").T


def save_kinetics_plot(result, output_path: Path):
    """Save time-series plot of all receiver concentrations."""
    params = result["params"]
    times_h = result["times_h"]
    
    # We need to extract time series for all species in the receiver
    # This requires modifying simulate_sender_receiver to track these
    
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    
    # Plot I2
    axes[0].plot(times_h, result["receiver_i2_nM"], color="#0c5da5", lw=2.5)
    axes[0].set_ylabel("Receiver I2 (nM)")
    axes[0].set_title(
        f"Sender/Receiver kinetics | distance = {params.center_distance_um:.0f} um, "
        f"Th2 = {params.threshold_uM:.2f} uM"
    )
    axes[0].grid(alpha=0.25)
    
    # Plot Th2
    axes[1].plot(times_h, result["receiver_th2_nM"], color="#00b945", lw=2.5)
    axes[1].set_ylabel("Receiver Th2 (nM)")
    axes[1].grid(alpha=0.25)
    
    # Plot S2
    axes[2].plot(times_h, result["receiver_s2_nM"], color="#b54e00", lw=2.5)
    axes[2].set_ylabel("Receiver S2 (nM)")
    axes[2].grid(alpha=0.25)
    
    # Plot S2:I2
    axes[3].plot(times_h, result["receiver_s2_i2_nM"], color="#9400d3", lw=2.5)
    axes[3].set_ylabel("Receiver S2:I2 (nM)")
    axes[3].grid(alpha=0.25)
    
    # Plot S2:Th2
    axes[4].plot(times_h, result["receiver_s2_th2_nM"], color="#ff1493", lw=2.5)
    axes[4].set_xlabel("Time (h)")
    axes[4].set_ylabel("Receiver S2:Th2 (nM)")
    axes[4].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_field_plot(result, output_path: Path):
    """Save spatial distribution plots of concentrations."""
    nx = result["nx"]
    ny = result["ny"]
    vars_by_name = result["vars"]
    receiver_mask = result["receiver_mask"]

    s2 = field_to_image(np.asarray(vars_by_name["S2"].value) / NANOMOLAR, nx, ny)
    i2 = field_to_image(np.asarray(vars_by_name["I2"].value) / NANOMOLAR, nx, ny)
    total_rna = field_to_image(
        (
            np.asarray(vars_by_name["S2"].value)
            + np.asarray(vars_by_name["S2_I2"].value)
            + np.asarray(vars_by_name["S2_Th2"].value)
        )
        / NANOMOLAR,
        nx,
        ny,
    )
    receiver_outline = field_to_image(receiver_mask.astype(float), nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    datasets = (
        (s2, "Free S2 (nM)"),
        (i2, "I2 (nM)"),
        (total_rna, "Total RNA (nM)"),
    )

    for ax, (data, title) in zip(axes, datasets):
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.contour(receiver_outline, levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_distance_sweep_plot(distances_um, receiver_i2_nM, receiver_total_rna_nM, output_path: Path):
    """Save distance-response curve plot."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(distances_um, receiver_i2_nM, marker="o", color="#0c5da5", lw=2)
    axes[0].set_ylabel("Steady-state receiver I2 (nM)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(distances_um, receiver_total_rna_nM, marker="o", color="#b54e00", lw=2)
    axes[1].set_xlabel("Sender/receiver center distance (um)")
    axes[1].set_ylabel("Steady-state receiver total RNA (nM)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["comsol-2-1"],
        default=None,
        help="Apply a named parameter preset.",
    )
    parser.add_argument("--distance-um", type=float, default=300.0, help="Node center distance.")
    parser.add_argument("--node-length-um", type=float, default=50.0, help="Hydrogel side length.")
    parser.add_argument("--bath-margin-um", type=float, default=250.0, help="Bath margin around nodes.")
    parser.add_argument("--dx-um", type=float, default=10.0, help="Square cell size.")
    parser.add_argument("--hours", type=float, default=8.0, help="Total simulated time.")
    parser.add_argument("--dt-s", type=float, default=60.0, help="Time step in seconds.")
    parser.add_argument("--threshold-uM", type=float, default=5.0, help="Initial threshold in receiver.")
    parser.add_argument("--sender-switch-nM", type=float, default=100.0, help="Initial sender I1O2.")
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0, help="Initial receiver I2.")
    parser.add_argument("--transition-sharpness", type=float, default=0.1, 
                       help="Sharpness of smooth transitions (default 0.1)")
    parser.add_argument(
        "--sweep-distances-um",
        type=float,
        nargs="*",
        default=None,
        help="Optional distance sweep. If provided, the script also saves a distance-response curve.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reaction_diffusion_models/sender_receiver"),
        help="Prefix for saved figures.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints.")
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    params = SenderReceiverParams(
        node_length_um=args.node_length_um,
        center_distance_um=args.distance_um,
        bath_margin_um=args.bath_margin_um,
        dx_um=args.dx_um,
        total_hours=args.hours,
        dt_s=args.dt_s,
        sender_switch_nM=args.sender_switch_nM,
        receiver_switch_nM=args.receiver_switch_nM,
        threshold_uM=args.threshold_uM,
        transition_sharpness=args.transition_sharpness,
    )
    params = apply_preset(params, args.preset)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Run main simulation
    result = simulate_sender_receiver(params, verbose=not args.quiet)
    save_kinetics_plot(result, args.output_prefix.with_name(args.output_prefix.name + "_kinetics_tanh.png"))
    save_field_plot(result, args.output_prefix.with_name(args.output_prefix.name + "_fields_tanh.png"))

    print(
        f"Saved kinetics plot to "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_kinetics.png')}"
    )
    print(
        f"Saved field plot to "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_fields.png')}"
    )

    # Optional distance sweep
    if args.sweep_distances_um:
        final_i2 = []
        final_total_rna = []
        for distance_um in args.sweep_distances_um:
            sweep_params = SenderReceiverParams(
                node_length_um=args.node_length_um,
                center_distance_um=distance_um,
                bath_margin_um=args.bath_margin_um,
                dx_um=args.dx_um,
                total_hours=args.hours,
                dt_s=args.dt_s,
                sender_switch_nM=args.sender_switch_nM,
                receiver_switch_nM=args.receiver_switch_nM,
                threshold_uM=args.threshold_uM,
                transition_sharpness=args.transition_sharpness,
            )
            sweep_result = simulate_sender_receiver(sweep_params, verbose=False)
            final_i2.append(sweep_result["receiver_i2_nM"][-1])
            final_total_rna.append(sweep_result["receiver_total_rna_nM"][-1])

        sweep_plot = args.output_prefix.with_name(args.output_prefix.name + "_distance_sweep.png")
        save_distance_sweep_plot(
            np.asarray(args.sweep_distances_um),
            np.asarray(final_i2),
            np.asarray(final_total_rna),
            sweep_plot,
        )
        print(f"Saved distance-response plot to {sweep_plot}")


if __name__ == "__main__":
    start_time = simtime.perf_counter()
    main()
    end_time = simtime.perf_counter()
    print(f"\ntotal sim time: {end_time - start_time:.2f} seconds")
    