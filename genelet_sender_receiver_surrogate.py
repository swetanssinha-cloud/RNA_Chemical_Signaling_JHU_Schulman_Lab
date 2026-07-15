#!/usr/bin/env python3
"""
Stable reduced-order surrogate for the 2-node sender/receiver model.

This keeps the receiver chemistry from the full model, but replaces the 2D
reaction-diffusion PDE with three well-mixed transport compartments:

- sender gel concentration of S2
- effective path / bath concentration of S2
- receiver gel concentration of S2

Only S2 is transported. The receiver-local chemistry is the same mass-action
system used in the FiPy / NumPy versions:

    S2 + I2  <->  S2:I2
    S2 + Th2 <->  S2:Th2

The time integration uses backward Euler with Newton iterations, so it remains
stable for large time steps even when the reaction rates are stiff.
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
class SurrogateParams:
    node_length_um: float = 50.0
    center_distance_um: float = 300.0
    total_hours: float = 8.0
    dt_s: float = 300.0

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

    min_path_length_factor: float = 0.5
    transport_scale: float = 1.0
    path_loss_scale: float = 2.5
    newton_tolerance: float = 1e-10
    newton_max_iters: int = 20

    def validate(self) -> None:
        if self.center_distance_um < self.node_length_um:
            raise ValueError("center_distance_um must be at least node_length_um.")
        if self.node_length_um <= 0 or self.total_hours <= 0 or self.dt_s <= 0:
            raise ValueError("node_length_um, total_hours, and dt_s must be positive.")
        if self.min_path_length_factor <= 0 or self.transport_scale <= 0 or self.path_loss_scale < 0:
            raise ValueError(
                "min_path_length_factor and transport_scale must be positive, "
                "and path_loss_scale must be nonnegative."
            )


def apply_preset(params: SurrogateParams, preset: str | None) -> SurrogateParams:
    if preset is None:
        return params
    if preset == "comsol-2-1":
        params.node_length_um = 75.0
        params.center_distance_um = 175.0
        params.sender_switch_nM = 100.0
        params.receiver_switch_nM = 100.0
        params.threshold_uM = 10.0
        return params
    raise ValueError(f"Unknown preset: {preset}")


def transport_geometry(params: SurrogateParams) -> dict[str, float]:
    node_area_um2 = params.node_length_um**2
    edge_gap_um = max(params.center_distance_um - params.node_length_um, 0.0)
    min_path_length_um = params.min_path_length_factor * params.node_length_um
    path_length_um = max(edge_gap_um, min_path_length_um)
    path_area_um2 = params.node_length_um * path_length_um

    d_interface_um2_s = 2.0 / (1.0 / params.d_gel_um2_s + 1.0 / params.d_solution_um2_s)
    exchange_distance_um = 0.5 * params.node_length_um + 0.5 * path_length_um
    conductance_um2_s = (
        params.transport_scale
        * d_interface_um2_s
        * params.node_length_um
        / exchange_distance_um
    )
    path_loss_s_inv = params.path_loss_scale * params.d_solution_um2_s / max(
        path_length_um**2,
        1e-12,
    )

    return {
        "node_area_um2": node_area_um2,
        "path_area_um2": path_area_um2,
        "edge_gap_um": edge_gap_um,
        "path_length_um": path_length_um,
        "conductance_um2_s": conductance_um2_s,
        "path_loss_s_inv": path_loss_s_inv,
    }


def rhs(y: np.ndarray, params: SurrogateParams, geom: dict[str, float]) -> np.ndarray:
    s_sender, s_path, s_receiver, i2, s2_i2, th2, s2_th2 = y

    node_area = geom["node_area_um2"]
    path_area = geom["path_area_um2"]
    conductance = geom["conductance_um2_s"]

    sender_exchange = conductance / node_area
    path_from_sender = conductance / path_area
    receiver_exchange = conductance / node_area
    path_to_receiver = conductance / path_area
    path_loss = geom["path_loss_s_inv"]

    source = params.k_p_s_inv * params.sender_switch_nM * NANOMOLAR
    bind_i2 = params.k_slow_M_inv_s_inv * i2 * s_receiver
    bind_th2 = params.k_fast_M_inv_s_inv * th2 * s_receiver
    unbind_i2 = params.k_d_ds_s_inv * s2_i2
    unbind_th2 = params.k_d_ds_s_inv * s2_th2

    out = np.zeros_like(y)
    out[0] = (
        source
        - params.k_d_ss_s_inv * s_sender
        - sender_exchange * (s_sender - s_path)
    )
    out[1] = (
        -params.k_d_ss_s_inv * s_path
        - path_loss * s_path
        + path_from_sender * (s_sender - s_path)
        - path_to_receiver * (s_path - s_receiver)
    )
    out[2] = (
        -params.k_d_ss_s_inv * s_receiver
        + receiver_exchange * (s_path - s_receiver)
        - bind_i2
        - bind_th2
        + unbind_i2
        + unbind_th2
    )
    out[3] = unbind_i2 - bind_i2
    out[4] = bind_i2 - unbind_i2
    out[5] = unbind_th2 - bind_th2
    out[6] = bind_th2 - unbind_th2
    return out


def rhs_jacobian(y: np.ndarray, params: SurrogateParams, geom: dict[str, float]) -> np.ndarray:
    _, _, s_receiver, i2, _, th2, _ = y

    node_area = geom["node_area_um2"]
    path_area = geom["path_area_um2"]
    conductance = geom["conductance_um2_s"]

    sender_exchange = conductance / node_area
    path_from_sender = conductance / path_area
    receiver_exchange = conductance / node_area
    path_to_receiver = conductance / path_area
    path_loss = geom["path_loss_s_inv"]

    jac = np.zeros((7, 7), dtype=float)

    jac[0, 0] = -params.k_d_ss_s_inv - sender_exchange
    jac[0, 1] = sender_exchange

    jac[1, 0] = path_from_sender
    jac[1, 1] = -params.k_d_ss_s_inv - path_loss - path_from_sender - path_to_receiver
    jac[1, 2] = path_to_receiver

    jac[2, 1] = receiver_exchange
    jac[2, 2] = (
        -params.k_d_ss_s_inv
        - receiver_exchange
        - params.k_slow_M_inv_s_inv * i2
        - params.k_fast_M_inv_s_inv * th2
    )
    jac[2, 3] = -params.k_slow_M_inv_s_inv * s_receiver
    jac[2, 4] = params.k_d_ds_s_inv
    jac[2, 5] = -params.k_fast_M_inv_s_inv * s_receiver
    jac[2, 6] = params.k_d_ds_s_inv

    jac[3, 2] = -params.k_slow_M_inv_s_inv * i2
    jac[3, 3] = -params.k_slow_M_inv_s_inv * s_receiver
    jac[3, 4] = params.k_d_ds_s_inv

    jac[4, 2] = params.k_slow_M_inv_s_inv * i2
    jac[4, 3] = params.k_slow_M_inv_s_inv * s_receiver
    jac[4, 4] = -params.k_d_ds_s_inv

    jac[5, 2] = -params.k_fast_M_inv_s_inv * th2
    jac[5, 5] = -params.k_fast_M_inv_s_inv * s_receiver
    jac[5, 6] = params.k_d_ds_s_inv

    jac[6, 2] = params.k_fast_M_inv_s_inv * th2
    jac[6, 5] = params.k_fast_M_inv_s_inv * s_receiver
    jac[6, 6] = -params.k_d_ds_s_inv

    return jac


def backward_euler_step(
    y_prev: np.ndarray,
    dt_s: float,
    params: SurrogateParams,
    geom: dict[str, float],
) -> tuple[np.ndarray, int, float]:
    y = np.maximum(y_prev.copy(), 0.0)
    identity = np.eye(y_prev.size)
    residual_norm = np.inf

    for iteration in range(1, params.newton_max_iters + 1):
        residual = y - y_prev - dt_s * rhs(y, params, geom)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if residual_norm < params.newton_tolerance:
            return np.maximum(y, 0.0), iteration, residual_norm

        jac = identity - dt_s * rhs_jacobian(y, params, geom)
        step = np.linalg.solve(jac, -residual)

        damping = 1.0
        while damping > 1e-6:
            candidate = y + damping * step
            if np.all(candidate >= -1e-15):
                y = np.maximum(candidate, 0.0)
                break
            damping *= 0.5
        else:
            y = np.maximum(y + step, 0.0)

    raise RuntimeError(
        f"Newton solve failed to converge in {params.newton_max_iters} iterations; "
        f"final residual = {residual_norm:.3e}"
    )


def simulate_surrogate(params: SurrogateParams, verbose: bool = True) -> dict[str, object]:
    params.validate()
    geom = transport_geometry(params)

    y = np.array(
        [
            0.0,
            0.0,
            0.0,
            params.receiver_switch_nM * NANOMOLAR,
            0.0,
            params.threshold_uM * MICROMOLAR,
            0.0,
        ],
        dtype=float,
    )

    n_steps = int(np.ceil(params.total_hours * 3600.0 / params.dt_s))
    times_h = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, y.size), dtype=float)
    states[0] = y

    receiver_i2_nM = np.zeros(n_steps + 1)
    receiver_total_rna_nM = np.zeros(n_steps + 1)
    receiver_i2_nM[0] = y[3] / NANOMOLAR
    receiver_total_rna_nM[0] = (y[2] + y[4] + y[6]) / NANOMOLAR

    newton_iterations = np.zeros(n_steps, dtype=int)
    newton_residuals = np.zeros(n_steps, dtype=float)

    for step in range(1, n_steps + 1):
        y, iters, residual = backward_euler_step(y, params.dt_s, params, geom)
        times_h[step] = step * params.dt_s / 3600.0
        states[step] = y
        receiver_i2_nM[step] = y[3] / NANOMOLAR
        receiver_total_rna_nM[step] = (y[2] + y[4] + y[6]) / NANOMOLAR
        newton_iterations[step - 1] = iters
        newton_residuals[step - 1] = residual

        if verbose and (step == 1 or step % max(1, n_steps // 10) == 0 or step == n_steps):
            print(
                f"step {step:4d}/{n_steps} | t = {times_h[step]:5.2f} h | "
                f"receiver I2 = {receiver_i2_nM[step]:8.3f} nM | "
                f"receiver total RNA = {receiver_total_rna_nM[step]:8.3f} nM | "
                f"newton iters = {iters:2d} | residual = {residual:.3e}"
            )

    return {
        "params": params,
        "geometry": geom,
        "times_h": times_h,
        "states_M": states,
        "receiver_i2_nM": receiver_i2_nM,
        "receiver_total_rna_nM": receiver_total_rna_nM,
        "newton_iterations": newton_iterations,
        "newton_residuals": newton_residuals,
    }


def save_time_series_csv(result: dict[str, object], path: Path) -> None:
    states = np.asarray(result["states_M"])
    times_h = np.asarray(result["times_h"])
    receiver_i2_nM = np.asarray(result["receiver_i2_nM"])
    receiver_total_rna_nM = np.asarray(result["receiver_total_rna_nM"])

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_h",
                "S_sender_nM",
                "S_path_nM",
                "S_receiver_nM",
                "I2_nM",
                "S2_I2_nM",
                "Th2_nM",
                "S2_Th2_nM",
                "receiver_I2_nM",
                "receiver_total_RNA_nM",
            ]
        )
        for idx, time_h in enumerate(times_h):
            row_nM = (states[idx] / NANOMOLAR).tolist()
            writer.writerow(
                [
                    time_h,
                    *row_nM,
                    receiver_i2_nM[idx],
                    receiver_total_rna_nM[idx],
                ]
            )


def plot_outputs(result: dict[str, object], path: Path) -> None:
    if plt is None:
        return

    times_h = np.asarray(result["times_h"])
    states = np.asarray(result["states_M"]) / NANOMOLAR
    params = result["params"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(times_h, states[:, 3], color="#0c5da5", lw=2.4, label="I2")
    axes[0].plot(times_h, states[:, 5], color="#7a3b8f", lw=2.0, label="Th2")
    axes[0].set_ylabel("Receiver species (nM)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(times_h, states[:, 2], color="#1f7a1f", lw=2.2, label="free S2")
    axes[1].plot(times_h, states[:, 4], color="#b54e00", lw=2.0, label="S2:I2")
    axes[1].plot(times_h, states[:, 6], color="#d47d00", lw=2.0, label="S2:Th2")
    axes[1].set_ylabel("Receiver RNA pools (nM)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    axes[2].plot(times_h, states[:, 0], color="#255c99", lw=2.0, label="sender")
    axes[2].plot(times_h, states[:, 1], color="#7f8c8d", lw=2.0, label="path")
    axes[2].plot(times_h, states[:, 2], color="#1f7a1f", lw=2.0, label="receiver")
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel("Transport S2 (nM)")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    fig.suptitle(
        "Sender/Receiver surrogate kinetics | "
        f"distance = {params.center_distance_um:.0f} um, "
        f"Th2 = {params.threshold_uM:.2f} uM"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["comsol-2-1"], default=None)
    parser.add_argument("--distance-um", type=float, default=300.0, help="Node center distance.")
    parser.add_argument("--node-length-um", type=float, default=50.0, help="Hydrogel side length.")
    parser.add_argument("--hours", type=float, default=8.0, help="Total simulated time.")
    parser.add_argument("--dt-s", type=float, default=300.0, help="Backward-Euler time step.")
    parser.add_argument("--threshold-uM", type=float, default=5.0, help="Initial threshold in receiver.")
    parser.add_argument("--sender-switch-nM", type=float, default=100.0, help="Initial sender I1O2.")
    parser.add_argument("--receiver-switch-nM", type=float, default=100.0, help="Initial receiver I2.")
    parser.add_argument(
        "--min-path-length-factor",
        type=float,
        default=0.5,
        help="Lower bound on the effective path length as a fraction of node length.",
    )
    parser.add_argument(
        "--transport-scale",
        type=float,
        default=1.0,
        help="Multiplier on the diffusive conductance used by the surrogate.",
    )
    parser.add_argument(
        "--path-loss-scale",
        type=float,
        default=2.5,
        help="Multiplier on the path-to-bath leakage term used to mimic 2D dilution.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("surrogate_sender_receiver/sender_receiver_surrogate"),
        help="Prefix for saved CSV and PNG outputs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress prints.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    params = SurrogateParams(
        node_length_um=args.node_length_um,
        center_distance_um=args.distance_um,
        total_hours=args.hours,
        dt_s=args.dt_s,
        sender_switch_nM=args.sender_switch_nM,
        receiver_switch_nM=args.receiver_switch_nM,
        threshold_uM=args.threshold_uM,
        min_path_length_factor=args.min_path_length_factor,
        transport_scale=args.transport_scale,
        path_loss_scale=args.path_loss_scale,
    )
    params = apply_preset(params, args.preset)

    result = simulate_surrogate(params, verbose=not args.quiet)

    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_kinetics.csv")
    png_path = args.output_prefix.with_name(args.output_prefix.name + "_kinetics.png")
    save_time_series_csv(result, csv_path)
    plot_outputs(result, png_path)

    print(f"Saved kinetics CSV to {csv_path}")
    geom = result["geometry"]
    print(
        "effective_path_length_um "
        f"{geom['path_length_um']:.6f}"
    )
    print(
        "conductance_um2_s "
        f"{geom['conductance_um2_s']:.6f}"
    )
    print(f"path_loss_s_inv {geom['path_loss_s_inv']:.6f}")
    print(f"final_receiver_I2_nM {result['receiver_i2_nM'][-1]:.6f}")
    print(f"final_receiver_total_RNA_nM {result['receiver_total_rna_nM'][-1]:.6f}")
    if plt is not None:
        print(f"Saved kinetics plot to {png_path}")
    else:
        print("Matplotlib not available; skipped PNG plot.")


if __name__ == "__main__":
    main()
