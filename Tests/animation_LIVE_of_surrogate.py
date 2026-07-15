#!/usr/bin/env python3
"""
FiPy live animation of the 2-node sender/receiver model from SI Section 2.1.

Model assumptions:
- S2 is the only diffusing species.
- I1O2, I2, Th2, and bound complexes are immobilized in their gel nodes.
- Outer boundary is reflective (no-flux).

Units:
- micrometers, seconds, molar concentration (M)
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fipy import CellVariable, DiffusionTerm, Grid2D, ImplicitSourceTerm, TransientTerm


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
            raise ValueError("center_distance_um must be at least node_length_um.")
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")


def apply_preset(params: SenderReceiverParams, preset: str | None) -> SenderReceiverParams:
    if not preset:
        return params

    if preset == "comsol-2-1":
        params.node_length_um = 75.0
        params.center_distance_um = 175.0
        params.bath_margin_um = 2375.0
        params.d_gel_um2_s = 42.0
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
    sender_mask = (
        (np.abs(x - sender_center_x) <= half) &
        (np.abs(y - sender_center_y) <= half)
    )
    receiver_mask = (
        (np.abs(x - receiver_center_x) <= half) &
        (np.abs(y - receiver_center_y) <= half)
    )

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


def field_to_image(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.asarray(values).reshape((nx, ny), order="F").T


def mean_in_mask(var: CellVariable, mask: np.ndarray) -> float:
    values = np.asarray(var.value)
    return float(values[mask].mean())


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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame-interval", type=int, default=1, help="Simulation steps per animation frame.")
    parser.add_argument("--fps", type=int, default=10, help="Animation playback speed.")
    return parser.parse_args()


def run_live_animation(params: SenderReceiverParams, frame_interval: int = 1, fps: int = 10, verbose: bool = True):
    params.validate()

    mesh, nx, ny, sender_mask, receiver_mask = build_geometry(params)
    vars_by_name = initialize_variables(mesh, sender_mask, receiver_mask, params)
    eqs = build_equations(vars_by_name, params)

    dynamic_vars = (
        vars_by_name["S2"],
        vars_by_name["I2"],
        vars_by_name["S2_I2"],
        vars_by_name["Th2"],
        vars_by_name["S2_Th2"],
    )

    total_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    n_frames = max(1, total_steps // frame_interval)

    step_counter = 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    s2_img = field_to_image(np.asarray(vars_by_name["S2"].value) / NANOMOLAR, nx, ny)
    i2_img = field_to_image(np.asarray(vars_by_name["I2"].value) / NANOMOLAR, nx, ny)
    total_img = field_to_image(
        (
            np.asarray(vars_by_name["S2"].value)
            + np.asarray(vars_by_name["S2_I2"].value)
            + np.asarray(vars_by_name["S2_Th2"].value)
        ) / NANOMOLAR,
        nx, ny
    )

    receiver_outline = field_to_image(receiver_mask.astype(float), nx, ny)

    im0 = axes[0].imshow(s2_img, origin="lower", cmap="viridis")
    axes[0].contour(receiver_outline, levels=[0.5], colors="white", linewidths=0.8)
    axes[0].set_title("Free S2 (nM)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(i2_img, origin="lower", cmap="plasma")
    axes[1].contour(receiver_outline, levels=[0.5], colors="white", linewidths=0.8)
    axes[1].set_title("I2 (nM)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(total_img, origin="lower", cmap="inferno")
    axes[2].contour(receiver_outline, levels=[0.5], colors="white", linewidths=0.8)
    axes[2].set_title("Total RNA (nM)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    def update(frame):
        nonlocal step_counter

        for _ in range(frame_interval):
            if step_counter >= total_steps:
                break

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

            step_counter += 1

        s2_img = field_to_image(np.asarray(vars_by_name["S2"].value) / NANOMOLAR, nx, ny)
        i2_img = field_to_image(np.asarray(vars_by_name["I2"].value) / NANOMOLAR, nx, ny)
        total_img = field_to_image(
            (
                np.asarray(vars_by_name["S2"].value)
                + np.asarray(vars_by_name["S2_I2"].value)
                + np.asarray(vars_by_name["S2_Th2"].value)
            ) / NANOMOLAR,
            nx, ny
        )

        im0.set_array(s2_img)
        im1.set_array(i2_img)
        im2.set_array(total_img)

        current_time_h = step_counter * params.dt_s / 3600.0
        receiver_i2 = mean_in_mask(vars_by_name["I2"], receiver_mask) / NANOMOLAR
        receiver_total_rna = (
            mean_in_mask(vars_by_name["S2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_I2"], receiver_mask)
            + mean_in_mask(vars_by_name["S2_Th2"], receiver_mask)
        ) / NANOMOLAR

        fig.suptitle(
            f"Sender/Receiver Simulation | d = {params.center_distance_um:.0f} μm | t = {current_time_h:.2f} h",
            fontsize=14,
            fontweight="bold"
        )

        if verbose and (frame == 0 or frame % max(1, n_frames // 10) == 0):
            print(
                f"Frame {frame}/{n_frames} | t = {current_time_h:.2f} h | "
                f"Receiver I2 = {receiver_i2:.3f} nM | "
                f"Receiver total RNA = {receiver_total_rna:.3f} nM"
            )

        return [im0, im1, im2]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
        repeat=False
    )

    plt.show()


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

    run_live_animation(
        params,
        frame_interval=args.frame_interval,
        fps=args.fps,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()