from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import time as simtime
from dataclasses import dataclass

# --------------------------------------------------------------------------
# IMPORT YOUR MODEL FUNCTIONS FROM ANOTHER FILE
# --------------------------------------------------------------------------
# Replace `your_model_file` with the actual filename (without .py)
#
# Expected imported functions:
#   - build_geometry(params)
#   - initialize_variables(mesh, sender_mask, receiver_mask, params)
#   - build_equations(vars_by_name, params)
#   - clip_nonnegative(vars_by_name)
#   - mean_in_mask(var, mask)
#
# If your function names differ, just rename them here.
# --------------------------------------------------------------------------


from sender_receiver_tanh_nodes import (
    build_geometry,
    initialize_variables,
    build_equations,
    clip_nonnegative,
    mean_in_mask,
    double_peak_diffusion, 
    smooth_circle_profile
)

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm  # only if needed by imported module

MOLAR = 1.0
NANOMOLAR = 1e-9 * MOLAR
MICROMOLAR = 1e-6 * MOLAR


# ============================================================================
# PARAMETERS
# ============================================================================
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
    k_p_s_inv: float = 0.2
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0

    transition_sharpness: float = 0.5

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")


# ============================================================================
# SIMULATION DRIVER
# ============================================================================
def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = False):
    """
    Run one simulation using geometry / variable / equation helpers imported
    from your model file.
    """
    params.validate()

    mesh, nx, ny, sender_mask, receiver_mask, sender_center_x, sender_center_y, receiver_center_x, receiver_center_y = build_geometry(params)
    vars_by_name = initialize_variables(mesh, sender_mask, receiver_mask, params,sender_center_x, sender_center_y, receiver_center_x, receiver_center_y)
    eqs = build_equations(vars_by_name, params)

    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_h = np.zeros(n_steps + 1)
    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)

    receiver_i2_nM[0] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
    receiver_total_rna_nM[0] = (
        mean_in_mask(vars_by_name["S2"], receiver_mask)
        + mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
        + mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)
    ) / NANOMOLAR

    dynamic_vars = (
        vars_by_name["S2"],
        vars_by_name["I2"],
        vars_by_name["S2_I2"],
        vars_by_name["Th2"],
        vars_by_name["S2_Th2"],
    )

    total_sweeps = 0
    start = simtime.perf_counter()

    for step in range(1, n_steps + 1):
        for var in dynamic_vars:
            var.updateOld()

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
            total_sweeps += 1

        times_h[step] = step * params.dt_s / 3600.0
        receiver_i2_nM[step] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
        receiver_total_rna_nM[step] = (
            mean_in_mask(vars_by_name["S2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)
        ) / NANOMOLAR

        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:4d}/{n_steps} | t = {times_h[step]:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM[step]:8.3f} nM | "
                f"receiver total RNA = {receiver_total_rna_nM[step]:8.3f} nM | "
                f"sweeps = {sweep_count:2d} | residual = {residual:.3e}"
            )

    runtime = simtime.perf_counter() - start

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
        "receiver_total_rna_nM": receiver_total_rna_nM,
        "final_receiver_i2_nM": receiver_i2_nM[-1],
        "final_receiver_total_rna_nM": receiver_total_rna_nM[-1],
        "runtime_s": runtime,
        "total_sweeps": total_sweeps,
        "avg_sweeps_per_step": total_sweeps / n_steps,
    }


# ============================================================================
# CONVERGENCE STUDIES
# ============================================================================
def percent_change(coarse, fine):
    if coarse == 0:
        return np.nan
    return abs(fine - coarse) / abs(coarse) * 100.0


def run_spatial_convergence():
    """
    Fix dt small and vary dx.
    """
    base_params = SenderReceiverParams(
        node_length_um=50.0,
        center_distance_um=300.0,
        bath_margin_um=250.0,
        total_hours=8.0,
        dt_s=10.0,   # small and fixed
    )

    dx_values = [2,1,0.5,0.25] #[20.0, 15.0, 10.0, 7.5, 5.0] # probally should have chosen [2,1,0.5,0.25]
    results = {}

    print("\n" + "=" * 80)
    print("SPATIAL CONVERGENCE STUDY")
    print("=" * 80)
    print(f"Fixed dt = {base_params.dt_s} s")
    print(f"dx values = {dx_values}")

    for dx in dx_values:
        params = SenderReceiverParams(**vars(base_params))
        params.dx_um = dx

        print(f"\nRunning dx = {dx} um, dt = {params.dt_s} s")
        result = simulate_sender_receiver(params, verbose=False)
        results[dx] = result

        print(f"  nx x ny = {result['nx']} x {result['ny']}")
        print(f"  final receiver I2 = {result['final_receiver_i2_nM']:.6f} nM")
        print(f"  final receiver total RNA = {result['final_receiver_total_rna_nM']:.6f} nM")
        print(f"  runtime = {result['runtime_s']:.2f} s")

    # Table
    print("\n" + "=" * 100)
    print("SPATIAL CONVERGENCE TABLE")
    print("=" * 100)
    print(f"{'dx (um)':<10} {'dt (s)':<10} {'nx':<8} {'ny':<8} {'I2_final (nM)':<18} {'Total RNA (nM)':<18}")
    print("-" * 100)
    for dx in dx_values:
        r = results[dx]
        print(f"{dx:<10.3f} {r['params'].dt_s:<10.3f} {r['nx']:<8d} {r['ny']:<8d} "
              f"{r['final_receiver_i2_nM']:<18.6f} {r['final_receiver_total_rna_nM']:<18.6f}")

    # Relative changes
    print("\nRelative change between successive dx refinements:")
    for i in range(len(dx_values) - 1):
        dx_coarse = dx_values[i]
        dx_fine = dx_values[i + 1]
        c = results[dx_coarse]
        f = results[dx_fine]

        print(f"\n  dx {dx_coarse} -> {dx_fine} um")
        print(f"    I2 change:   {percent_change(c['final_receiver_i2_nM'], f['final_receiver_i2_nM']):.4f}%")
        print(f"    RNA change:  {percent_change(c['final_receiver_total_rna_nM'], f['final_receiver_total_rna_nM']):.4f}%")

    # Plot
    dx_arr = np.array(dx_values)
    i2_final = np.array([results[dx]["final_receiver_i2_nM"] for dx in dx_values])
    rna_final = np.array([results[dx]["final_receiver_total_rna_nM"] for dx in dx_values])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(dx_arr, i2_final, "o-", lw=2)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("dx (um)")
    axes[0].set_ylabel("Final receiver I2 (nM)")
    axes[0].set_title("Spatial convergence: I2")
    axes[0].grid(True)

    axes[1].plot(dx_arr, rna_final, "o-", lw=2, color="orange")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("dx (um)")
    axes[1].set_ylabel("Final receiver total RNA (nM)")
    axes[1].set_title("Spatial convergence: total RNA")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("spatial_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


def run_temporal_convergence():
    """
    Fix dx fine and vary dt.
    """
    base_params = SenderReceiverParams(
        node_length_um=50.0,
        center_distance_um=300.0,
        bath_margin_um=250.0,
        total_hours=8.0,
        dx_um=5.0,   # fine and fixed
    )

    dt_values = [240.0, 120.0, 60.0, 30.0, 15.0, 10]  # seconds
    results = {}

    print("\n" + "=" * 80)
    print("TEMPORAL CONVERGENCE STUDY")
    print("=" * 80)
    print(f"Fixed dx = {base_params.dx_um} um")
    print(f"dt values = {dt_values}")

    for dt in dt_values:
        params = SenderReceiverParams(**vars(base_params))
        params.dt_s = dt

        print(f"\nRunning dx = {params.dx_um} um, dt = {dt} s")
        result = simulate_sender_receiver(params, verbose=False)
        results[dt] = result

        print(f"  nx x ny = {result['nx']} x {result['ny']}")
        print(f"  final receiver I2 = {result['final_receiver_i2_nM']:.6f} nM")
        print(f"  final receiver total RNA = {result['final_receiver_total_rna_nM']:.6f} nM")
        print(f"  runtime = {result['runtime_s']:.2f} s")

    # Table
    print("\n" + "=" * 100)
    print("TEMPORAL CONVERGENCE TABLE")
    print("=" * 100)
    print(f"{'dt (s)':<10} {'dx (um)':<10} {'nx':<8} {'ny':<8} {'I2_final (nM)':<18} {'Total RNA (nM)':<18}")
    print("-" * 100)
    for dt in dt_values:
        r = results[dt]
        print(f"{dt:<10.3f} {r['params'].dx_um:<10.3f} {r['nx']:<8d} {r['ny']:<8d} "
              f"{r['final_receiver_i2_nM']:<18.6f} {r['final_receiver_total_rna_nM']:<18.6f}")

    # Relative changes
    print("\nRelative change between successive dt refinements:")
    for i in range(len(dt_values) - 1):
        dt_coarse = dt_values[i]
        dt_fine = dt_values[i + 1]
        c = results[dt_coarse]
        f = results[dt_fine]

        print(f"\n  dt {dt_coarse} -> {dt_fine} s")
        print(f"    I2 change:   {percent_change(c['final_receiver_i2_nM'], f['final_receiver_i2_nM']):.4f}%")
        print(f"    RNA change:  {percent_change(c['final_receiver_total_rna_nM'], f['final_receiver_total_rna_nM']):.4f}%")

    # Plot
    dt_arr = np.array(dt_values)
    i2_final = np.array([results[dt]["final_receiver_i2_nM"] for dt in dt_values])
    rna_final = np.array([results[dt]["final_receiver_total_rna_nM"] for dt in dt_values])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(dt_arr, i2_final, "o-", lw=2)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("dt (s)")
    axes[0].set_ylabel("Final receiver I2 (nM)")
    axes[0].set_title("Temporal convergence: I2")
    axes[0].grid(True)

    axes[1].plot(dt_arr, rna_final, "o-", lw=2, color="orange")
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("dt (s)")
    axes[1].set_ylabel("Final receiver total RNA (nM)")
    axes[1].set_title("Temporal convergence: total RNA")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("temporal_convergence.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    spatial_results = run_spatial_convergence()
    temporal_results = run_temporal_convergence()