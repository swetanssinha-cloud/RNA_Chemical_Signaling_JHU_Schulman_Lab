#!/usr/bin/env python3
"""
Naive NumPy finite-difference version of the 2-node sender/receiver model.

This is intentionally simpler than the FiPy model:
- regular Cartesian grid only
- explicit Euler time stepping
- pure NumPy update loops
- reflective outer boundaries

It is useful for qualitative checks and student exploration, but it is not as
robust as the FiPy implementation for stiff kinetics or fine meshes.

Concentrations are represented in M (mol/L), so the SI bimolecular rates are
used directly as `1/M/s`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


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
    dt_s: float = 0.05

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
            raise ValueError("Nodes overlap. Increase center_distance_um.")
        if self.dx_um <= 0 or self.dt_s <= 0 or self.total_hours <= 0:
            raise ValueError("dx_um, dt_s, and total_hours must be positive.")
        dt_limit = 0.24 * self.dx_um**2 / max(self.d_solution_um2_s, self.d_gel_um2_s)
        if self.dt_s > dt_limit:
            raise ValueError(
                f"Explicit Euler dt_s={self.dt_s:g} is too large for this mesh. "
                f"Use dt_s <= {dt_limit:.4g} s or increase dx_um."
            )


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

    x = (np.arange(nx) + 0.5) * params.dx_um
    y = (np.arange(ny) + 0.5) * params.dx_um
    xx, yy = np.meshgrid(x, y, indexing="xy")

    sender_center_x = params.bath_margin_um + 0.5 * params.node_length_um
    sender_center_y = 0.5 * height_um
    receiver_center_x = sender_center_x + params.center_distance_um
    receiver_center_y = sender_center_y

    half = 0.5 * params.node_length_um
    sender_mask = (np.abs(xx - sender_center_x) <= half) & (np.abs(yy - sender_center_y) <= half)
    receiver_mask = (
        (np.abs(xx - receiver_center_x) <= half) & (np.abs(yy - receiver_center_y) <= half)
    )

    diffusion = np.full((ny, nx), params.d_solution_um2_s, dtype=float)
    diffusion[sender_mask | receiver_mask] = params.d_gel_um2_s

    return nx, ny, sender_mask, receiver_mask, diffusion


def initialize_fields(params: SenderReceiverParams, sender_mask, receiver_mask, diffusion):
    s2 = np.zeros_like(diffusion)
    i2 = np.zeros_like(diffusion)
    s2_i2 = np.zeros_like(diffusion)
    th2 = np.zeros_like(diffusion)
    s2_th2 = np.zeros_like(diffusion)
    i1o2 = np.zeros_like(diffusion)

    i1o2[sender_mask] = params.sender_switch_nM * NANOMOLAR
    i2[receiver_mask] = params.receiver_switch_nM * NANOMOLAR
    th2[receiver_mask] = params.threshold_uM * MICROMOLAR

    return {
        "S2": s2,
        "I2": i2,
        "S2_I2": s2_i2,
        "Th2": th2,
        "S2_Th2": s2_th2,
        "I1O2": i1o2,
        "D": diffusion,
    }


def divergence_of_diffusive_flux(c: np.ndarray, d: np.ndarray, dx: float) -> np.ndarray:
    c_pad = np.pad(c, ((1, 1), (1, 1)), mode="edge")
    d_pad = np.pad(d, ((1, 1), (1, 1)), mode="edge")

    d_e = 0.5 * (d_pad[1:-1, 1:-1] + d_pad[1:-1, 2:])
    d_w = 0.5 * (d_pad[1:-1, 1:-1] + d_pad[1:-1, :-2])
    d_n = 0.5 * (d_pad[1:-1, 1:-1] + d_pad[:-2, 1:-1])
    d_s = 0.5 * (d_pad[1:-1, 1:-1] + d_pad[2:, 1:-1])

    flux_e = d_e * (c_pad[1:-1, 2:] - c_pad[1:-1, 1:-1]) / dx
    flux_w = d_w * (c_pad[1:-1, 1:-1] - c_pad[1:-1, :-2]) / dx
    flux_n = d_n * (c_pad[1:-1, 1:-1] - c_pad[:-2, 1:-1]) / dx
    flux_s = d_s * (c_pad[2:, 1:-1] - c_pad[1:-1, 1:-1]) / dx

    return (flux_e - flux_w + flux_s - flux_n) / dx


def save_time_series_csv(times_h, receiver_i2_nM, receiver_total_rna_nM, path: Path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_h", "receiver_I2_nM", "receiver_total_RNA_nM"])
        for t, i2_val, total_val in zip(times_h, receiver_i2_nM, receiver_total_rna_nM):
            writer.writerow([t, i2_val, total_val])


def plot_kinetics(times_h, receiver_i2_nM, receiver_total_rna_nM, path: Path):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(times_h, receiver_i2_nM, color="#0c5da5", lw=2.2)
    axes[0].set_ylabel("Receiver I2 (nM)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(times_h, receiver_total_rna_nM, color="#b54e00", lw=2.2)
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Receiver total RNA (nM)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fields(fields, receiver_mask, path: Path):
    if plt is None:
        return
    s2 = fields["S2"] / NANOMOLAR
    i2 = fields["I2"] / NANOMOLAR
    total_rna = (fields["S2"] + fields["S2_I2"] + fields["S2_Th2"]) / NANOMOLAR

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (s2, i2, total_rna),
        ("Free S2 (nM)", "I2 (nM)", "Total RNA (nM)"),
    ):
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.contour(receiver_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def simulate_sender_receiver(params: SenderReceiverParams, verbose: bool = True):
    params.validate()
    nx, ny, sender_mask, receiver_mask, diffusion = build_geometry(params)
    fields = initialize_fields(params, sender_mask, receiver_mask, diffusion)

    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_h = np.zeros(n_steps + 1)
    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)

    receiver_i2_nM[0] = fields["I2"][receiver_mask].mean() / NANOMOLAR
    receiver_total_rna_nM[0] = (
        fields["S2"][receiver_mask].mean()
        + fields["S2_I2"][receiver_mask].mean()
        + fields["S2_Th2"][receiver_mask].mean()
    ) / NANOMOLAR

    for step in range(1, n_steps + 1):
        s2 = fields["S2"]
        i2 = fields["I2"]
        s2_i2 = fields["S2_I2"]
        th2 = fields["Th2"]
        s2_th2 = fields["S2_Th2"]
        i1o2 = fields["I1O2"]
        d = fields["D"]

        diff_s2 = divergence_of_diffusive_flux(s2, d, params.dx_um)

        bind_i2 = params.k_slow_M_inv_s_inv * i2 * s2
        bind_th2 = params.k_fast_M_inv_s_inv * th2 * s2
        unbind_i2 = params.k_d_ds_s_inv * s2_i2
        unbind_th2 = params.k_d_ds_s_inv * s2_th2

        s2 = s2 + params.dt_s * (
            diff_s2
            + params.k_p_s_inv * i1o2
            - bind_i2
            - bind_th2
            - params.k_d_ss_s_inv * s2
        )
        i2 = i2 + params.dt_s * (unbind_i2 - bind_i2)
        th2 = th2 + params.dt_s * (unbind_th2 - bind_th2)
        s2_i2 = s2_i2 + params.dt_s * (bind_i2 - unbind_i2)
        s2_th2 = s2_th2 + params.dt_s * (bind_th2 - unbind_th2)

        np.maximum(s2, 0.0, out=s2)
        np.maximum(i2, 0.0, out=i2)
        np.maximum(th2, 0.0, out=th2)
        np.maximum(s2_i2, 0.0, out=s2_i2)
        np.maximum(s2_th2, 0.0, out=s2_th2)

        fields["S2"] = s2
        fields["I2"] = i2
        fields["Th2"] = th2
        fields["S2_I2"] = s2_i2
        fields["S2_Th2"] = s2_th2

        times_h[step] = step * params.dt_s / 3600.0
        receiver_i2_nM[step] = i2[receiver_mask].mean() / NANOMOLAR
        receiver_total_rna_nM[step] = (
            s2[receiver_mask].mean() + s2_i2[receiver_mask].mean() + s2_th2[receiver_mask].mean()
        ) / NANOMOLAR

        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:6d}/{n_steps} | t = {times_h[step]:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM[step]:8.3f} nM | "
                f"receiver total RNA = {receiver_total_rna_nM[step]:8.3f} nM"
            )

    return {
        "params": params,
        "nx": nx,
        "ny": ny,
        "sender_mask": sender_mask,
        "receiver_mask": receiver_mask,
        "fields": fields,
        "times_h": times_h,
        "receiver_i2_nM": receiver_i2_nM,
        "receiver_total_rna_nM": receiver_total_rna_nM,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["comsol-2-1"], default=None)
    parser.add_argument("--distance-um", type=float, default=300.0)
    parser.add_argument("--node-length-um", type=float, default=50.0)
    parser.add_argument("--bath-margin-um", type=float, default=250.0)
    parser.add_argument("--dx-um", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument(
        "--dt-s",
        type=float,
        default=0.05,
        help="Explicit Euler step. Must satisfy the diffusion CFL limit.",
    )
    parser.add_argument("--threshold-uM", type=float, default=5.0)
    parser.add_argument("--sender-switch-nM", type=float, default=100.0)
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reaction_diffusion_models/sender_receiver_numpy"),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

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
    result = simulate_sender_receiver(params, verbose=not args.quiet)

    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_kinetics.csv")
    npz_path = args.output_prefix.with_name(args.output_prefix.name + "_final_fields.npz")
    kinetics_png = args.output_prefix.with_name(args.output_prefix.name + "_kinetics.png")
    fields_png = args.output_prefix.with_name(args.output_prefix.name + "_fields.png")

    save_time_series_csv(
        result["times_h"],
        result["receiver_i2_nM"],
        result["receiver_total_rna_nM"],
        csv_path,
    )
    np.savez_compressed(
        npz_path,
        S2=result["fields"]["S2"],
        I2=result["fields"]["I2"],
        S2_I2=result["fields"]["S2_I2"],
        Th2=result["fields"]["Th2"],
        S2_Th2=result["fields"]["S2_Th2"],
        sender_mask=result["sender_mask"],
        receiver_mask=result["receiver_mask"],
    )
    plot_kinetics(result["times_h"], result["receiver_i2_nM"], result["receiver_total_rna_nM"], kinetics_png)
    plot_fields(result["fields"], result["receiver_mask"], fields_png)

    print(f"Saved kinetics CSV to {csv_path}")
    print(f"Saved final fields NPZ to {npz_path}")
    if plt is not None:
        print(f"Saved kinetics plot to {kinetics_png}")
        print(f"Saved field plot to {fields_png}")
    else:
        print("Matplotlib not available; skipped PNG plots.")


if __name__ == "__main__":
    main()
