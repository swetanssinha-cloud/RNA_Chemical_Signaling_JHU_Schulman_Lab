#!/usr/bin/env python3
"""
FiPy live animation of the 2-node sender/receiver model.

Shows:
- Free S2
- Free I2
- S2:I2
- Free Th2
- S2:Th2
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
    k_p_s_inv: float = 0.02 #this should be smaller because 0.2 is a really big number for experiments
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
    parser.add_argument("--distance-um", type=float, default=300.0)
    parser.add_argument("--node-length-um", type=float, default=50.0)
    parser.add_argument("--bath-margin-um", type=float, default=250.0)
    parser.add_argument("--dx-um", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--dt-s", type=float, default=60.0)
    parser.add_argument("--threshold-uM", type=float, default=5.0)
    parser.add_argument("--sender-switch-nM", type=float, default=100.0)
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()

    sender_outline = field_to_image(sender_mask.astype(float), nx, ny)
    receiver_outline = field_to_image(receiver_mask.astype(float), nx, ny)

    def make_img(var_name):
        return field_to_image(np.asarray(vars_by_name[var_name].value) / NANOMOLAR, nx, ny)

    # Common scale for all panels (nM)
    PLOT_VMIN = 0.0
    PLOT_VMAX = max(
        params.sender_switch_nM,
        params.receiver_switch_nM,
        params.threshold_uM * 1000.0  # convert μM to nM
    )

    panels = [
        ("S2", "Free S2 (nM)", "viridis"),
        ("I2", "Free I2 (nM)", "plasma"),
        ("S2_I2", "S2:I2 (nM)", "magma"),
        ("Th2", "Free Th2 (nM)", "cividis"),
        ("S2_Th2", "S2:Th2 (nM)", "inferno"),
    ]

    ims = []

    PLOT_LIMITS = {
    "S2": 100.0,
    "I2": 100.0,
    "S2_I2": 100.0,
    "Th2": 5000.0,
    "S2_Th2": 5000.0,
}
    for ax, (var_name, title, cmap) in zip(axes, panels):
        im = ax.imshow(
            make_img(var_name),
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=PLOT_LIMITS[var_name])
        ax.contour(sender_outline, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.contour(receiver_outline, levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    # hide the 6th subplot
    axes[-1].axis("off")

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

        for im, var_name in zip(ims, ["S2", "I2", "S2_I2", "Th2", "S2_Th2"]):
            im.set_array(make_img(var_name))

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
                f"Receiver Threshold = {params.threshold_uM:.3f} μM | "
                f"Receiver total RNA = {receiver_total_rna:.3f} nM"
            )

        return ims

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
        repeat=False
    )

    plt.show()
    return anim


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

    run_live_animation(
        params,
        frame_interval=args.frame_interval,
        fps=args.fps,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()