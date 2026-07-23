from __future__ import annotations

#!/usr/bin/env python3
"""
FiPy implementation of the 2-node sender/receiver model from SI Section 2.1.

Model assumptions used here:
- S2 is the only diffusing species.
- The sender switch (I1O2), receiver switch (I2), and threshold (Th2) are
  immobilized within their hydrogel nodes.
- Bound complexes (S2:I2 and S2:Th2) are also immobilized.
- The outer bath boundary is reflective (FiPy's default no-flux condition).

Units are kept in micrometers, seconds, and molar concentration.
Concentrations are represented in M (mol/L), so the SI bimolecular rates are
used directly as `1/M/s`.
"""

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import time as simtime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


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
    k_p_s_inv: float = 0.2
    k_d_ds_s_inv: float = 3e-4
    k_d_ss_s_inv: float = 3e-4
    k_slow_M_inv_s_inv: float = 1e5
    k_fast_M_inv_s_inv: float = 1e6

    sender_switch_nM: float = 100.0
    receiver_switch_nM: float = 100.0
    threshold_uM: float = 5.0

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError(
                "center_distance_um must be at least node_length_um to avoid overlapping nodes."
            )
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")


def apply_preset(params: SenderReceiverParams, preset: str | None) -> SenderReceiverParams:
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


def build_geometry(params: SenderReceiverParams):
    width_um = 2.0 * params.bath_margin_um + params.center_distance_um + params.node_length_um
    height_um = 2.0 * params.bath_margin_um + params.node_length_um

    nx = int(np.ceil(width_um / params.dx_um))
    ny = int(np.ceil(height_um / params.dx_um))
    mesh = Grid2D(dx=params.dx_um, dy=params.dx_um, nx=nx, ny=ny)

    x = np.asarray(mesh.cellCenters[0].value)
    y = np.asarray(mesh.cellCenters[1].value)

    sender_center_x = params.bath_margin_um + 0.5 * params.node_length_um
    sender_center_y = 0.5 * height_um
    receiver_center_x = sender_center_x + params.center_distance_um
    receiver_center_y = sender_center_y

    half = 0.5 * params.node_length_um
    sender_mask = (np.abs(x - sender_center_x) <= half) & (np.abs(y - sender_center_y) <= half)
    receiver_mask = (np.abs(x - receiver_center_x) <= half) & (np.abs(y - receiver_center_y) <= half)

    return mesh, nx, ny, sender_mask, receiver_mask


def initialize_variables(mesh, sender_mask, receiver_mask, params: SenderReceiverParams):
    s2 = CellVariable(name="S2", mesh=mesh, value=0.0, hasOld=True)
    i2 = CellVariable(name="I2", mesh=mesh, value=0.0, hasOld=True)
    s2_i2 = CellVariable(name="S2_I2", mesh=mesh, value=0.0, hasOld=True)
    th2 = CellVariable(name="Th2", mesh=mesh, value=0.0, hasOld=True)
    s2_th2 = CellVariable(name="S2_Th2", mesh=mesh, value=0.0, hasOld=True)

    i1o2 = CellVariable(name="I1O2", mesh=mesh, value=0.0)
    diffusion = CellVariable(name="D", mesh=mesh, value=params.d_solution_um2_s)

    diffusion.setValue(params.d_gel_um2_s, where=sender_mask | receiver_mask)
    i1o2.setValue(params.sender_switch_nM * NANOMOLAR, where=sender_mask)
    i2.setValue(params.receiver_switch_nM * NANOMOLAR, where=receiver_mask)
    th2.setValue(params.threshold_uM * MICROMOLAR, where=receiver_mask)

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
    for name in ("S2", "I2", "S2_I2", "Th2", "S2_Th2"):
        var = vars_by_name[name]
        var.setValue(np.maximum(np.asarray(var.value), 0.0))


def mean_in_mask(var: CellVariable, mask: np.ndarray) -> float:
    values = np.asarray(var.value)
    return float(values[mask].mean())


def mean_in_domain(var: CellVariable) -> float:
    values = np.asarray(var.value)
    return float(values.mean())


def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = True):
    params.validate()
    mesh, nx, ny, sender_mask, receiver_mask = build_geometry(params)
    vars_by_name = initialize_variables(mesh, sender_mask, receiver_mask, params)
    eqs = build_equations(vars_by_name, params)

    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_h = np.zeros(n_steps + 1)

    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)
    sender_s2_nM = np.zeros(n_steps + 1)
    receiver_s2_nM = np.zeros(n_steps + 1)
    domain_s2_nM = np.zeros(n_steps + 1)

    receiver_i2_nM[0] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
    receiver_total_rna_nM[0] = (
        mean_in_mask(vars_by_name["S2"], receiver_mask)
        + mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
        + mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)
    ) / NANOMOLAR
    sender_s2_nM[0] = mean_in_mask(vars_by_name["S2"], sender_mask) / NANOMOLAR
    receiver_s2_nM[0] = mean_in_mask(vars_by_name["S2"], receiver_mask) / NANOMOLAR
    domain_s2_nM[0] = mean_in_domain(vars_by_name["S2"]) / NANOMOLAR

    dynamic_vars = (
        vars_by_name["S2"],
        vars_by_name["I2"],
        vars_by_name["S2_I2"],
        vars_by_name["Th2"],
        vars_by_name["S2_Th2"],
    )

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

        times_h[step] = step * params.dt_s / 3600.0
        receiver_i2_nM[step] = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
        receiver_total_rna_nM[step] = (
            mean_in_mask(vars_by_name["S2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)
        ) / NANOMOLAR
        sender_s2_nM[step] = mean_in_mask(vars_by_name["S2"], sender_mask) / NANOMOLAR
        receiver_s2_nM[step] = mean_in_mask(vars_by_name["S2"], receiver_mask) / NANOMOLAR
        domain_s2_nM[step] = mean_in_domain(vars_by_name["S2"]) / NANOMOLAR

        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:4d}/{n_steps} | t = {times_h[step]:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM[step]:8.3f} nM | "
                f"receiver total RNA = {receiver_total_rna_nM[step]:8.3f} nM | "
                f"sender S2 = {sender_s2_nM[step]:8.3f} nM | "
                f"receiver S2 = {receiver_s2_nM[step]:8.3f} nM | "
                f"domain S2 = {domain_s2_nM[step]:8.3f} nM | "
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
        "receiver_total_rna_nM": receiver_total_rna_nM,
        "sender_s2_nM": sender_s2_nM,
        "receiver_s2_nM": receiver_s2_nM,
        "domain_s2_nM": domain_s2_nM,
    }


def field_to_image(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.asarray(values).reshape((nx, ny), order="F").T


def save_kinetics_plot(result, output_path: Path):
    params = result["params"]
    times_h = result["times_h"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(times_h, result["receiver_i2_nM"], color="#0c5da5", lw=2.5)
    axes[0].set_ylabel("Receiver I2 (nM)")
    axes[0].set_title(
        f"Sender/Receiver kinetics | distance = {params.center_distance_um:.0f} um, "
        f"Th2 = {params.threshold_uM:.2f} uM"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(times_h, result["receiver_total_rna_nM"], color="#b54e00", lw=2.5)
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Receiver total RNA (nM)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_diagnostic_plot(result, output_path: Path):
    params = result["params"]
    times_h = result["times_h"]

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(times_h, result["receiver_i2_nM"], lw=2, color="#0c5da5")
    axes[0].set_ylabel("Receiver I2 (nM)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times_h, result["receiver_total_rna_nM"], lw=2, color="#b54e00")
    axes[1].set_ylabel("Receiver total RNA (nM)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times_h, result["sender_s2_nM"], lw=2, color="#6a0dad")
    axes[2].set_ylabel("Sender S2 (nM)")
    axes[2].grid(alpha=0.25)

    axes[3].plot(times_h, result["receiver_s2_nM"], lw=2, label="Receiver S2")
    axes[3].plot(times_h, result["domain_s2_nM"], lw=2, label="Domain-mean S2")
    axes[3].set_ylabel("S2 concentration (nM)")
    axes[3].set_xlabel("Time (h)")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    fig.suptitle(
        f"Diagnostics | distance = {params.center_distance_um:.0f} um, "
        f"Th2 = {params.threshold_uM:.2f} uM",
        y=0.995
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_field_plot(result, output_path: Path):
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["comsol-2-1"], default=None)
    parser.add_argument("--distance-um", type=float, default=300.0)
    parser.add_argument("--node-length-um", type=float, default=50.0)
    parser.add_argument("--bath-margin-um", type=float, default=250.0)
    parser.add_argument("--dx-um", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--dt-s", type=float, default=60.0)
    parser.add_argument("--threshold-uM", type=float, default=5.0)
    parser.add_argument("--sender-switch-nM", type=float, default=100.0)
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0)
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
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
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
    )
    params = apply_preset(params, args.preset)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = simulate_sender_receiver(params, verbose=not args.quiet)
    save_kinetics_plot(result, args.output_prefix.with_name(args.output_prefix.name + "_kinetics.png"))
    save_field_plot(result, args.output_prefix.with_name(args.output_prefix.name + "_fields.png"))
    save_diagnostic_plot(result, args.output_prefix.with_name(args.output_prefix.name + "_diagnostics.png"))

    print(f"Saved kinetics plot to {args.output_prefix.with_name(args.output_prefix.name + '_kinetics.png')}")
    print(f"Saved field plot to {args.output_prefix.with_name(args.output_prefix.name + '_fields.png')}")
    print(f"Saved diagnostic plot to {args.output_prefix.with_name(args.output_prefix.name + '_diagnostics.png')}")

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
            )
            sweep_result = simulate_sender_receiver(sweep_params, verbose=False)
            final_i2.append(sweep_result["receiver_i2_nM"][-1])
            final_total_rna.append(sweep_result["receiver_total_rna_nM"][-1])

        sweep_plot = args.output_prefix.with_name(args.output_prefix.name + "_distance_sweep.png")
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

        axes[0].plot(args.sweep_distances_um, final_i2, marker="o", color="#0c5da5", lw=2)
        axes[0].set_ylabel("Steady-state receiver I2 (nM)")
        axes[0].grid(alpha=0.25)

        axes[1].plot(args.sweep_distances_um, final_total_rna, marker="o", color="#b54e00", lw=2)
        axes[1].set_xlabel("Sender/receiver center distance (um)")
        axes[1].set_ylabel("Steady-state receiver total RNA (nM)")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(sweep_plot, dpi=200)
        plt.close(fig)

        print(f"Saved distance-response plot to {sweep_plot}")


if __name__ == "__main__":
    start_time = simtime.perf_counter()
    main()
    end_time = simtime.perf_counter()
    print(f"\ntotal sim time: {end_time - start_time:.2f} seconds")