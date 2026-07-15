#!/usr/bin/env python3
"""
Animated heatmap visualization for 2-node sender/receiver hydrogel model.
Shows mRNA (S2) propagating from sender to receiver in real-time.

Requirements:
    pip install numpy matplotlib

Usage:
    python3 animate_hydrogel_nodes.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from dataclasses import dataclass

# Constants
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

    newton_max_iters: int = 50
    newton_tolerance: float = 1e-10

    def validate(self):
        assert self.node_length_um > 0
        assert self.center_distance_um > self.node_length_um


def transport_geometry(params: SurrogateParams) -> dict[str, float]:
    """Calculate effective transport geometry for 3-compartment model."""
    node_area_um2 = params.node_length_um**2
    path_length_um = max(
        params.center_distance_um - params.node_length_um,
        params.min_path_length_factor * params.node_length_um,
    )
    path_area_um2 = params.node_length_um * path_length_um
    
    harmonic_d_um2_s = (
        2.0 * params.d_gel_um2_s * params.d_solution_um2_s
        / (params.d_gel_um2_s + params.d_solution_um2_s)
    )
    conductance_um2_s = (
        params.transport_scale * harmonic_d_um2_s * params.node_length_um
    )
    path_loss_s_inv = (
        params.path_loss_scale
        * params.d_solution_um2_s
        / (path_length_um * params.node_length_um)
    )

    return {
        "node_area_um2": node_area_um2,
        "path_length_um": path_length_um,
        "path_area_um2": path_area_um2,
        "conductance_um2_s": conductance_um2_s,
        "path_loss_s_inv": path_loss_s_inv,
    }


def rhs(y: np.ndarray, params: SurrogateParams, geom: dict[str, float]) -> np.ndarray:
    """Right-hand side of ODE system."""
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
    """Jacobian matrix for Newton solver."""
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
    """Single backward Euler step with Newton solver."""
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


def simulate_surrogate(params: SurrogateParams) -> dict:
    """Run the full simulation."""
    params.validate()
    geom = transport_geometry(params)

    # Initial state: [s_sender, s_path, s_receiver, i2, s2_i2, th2, s2_th2]
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

    for step in range(1, n_steps + 1):
        y, iters, residual = backward_euler_step(y, params.dt_s, params, geom)
        times_h[step] = step * params.dt_s / 3600.0
        states[step] = y

    return {
        "params": params,
        "geometry": geom,
        "times_h": times_h,
        "states_M": states,
    }


def create_spatial_grid(params: SurrogateParams, geom: dict, nx: int = 200):
    """Create a spatial grid representing sender -> path -> receiver."""
    node_len = params.node_length_um
    path_len = geom["path_length_um"]
    total_len = node_len + path_len + node_len
    
    x = np.linspace(0, total_len, nx)
    
    # Define regions
    sender_mask = x < node_len
    path_mask = (x >= node_len) & (x < node_len + path_len)
    receiver_mask = x >= node_len + path_len
    
    return x, sender_mask, path_mask, receiver_mask, total_len


def interpolate_spatial_concentration(states, x, sender_mask, path_mask, receiver_mask):
    """Map 3-compartment model to spatial profile for visualization."""
    s_sender = states[0] / NANOMOLAR  # Convert to nM
    s_path = states[1] / NANOMOLAR
    s_receiver = states[2] / NANOMOLAR
    
    concentration = np.zeros_like(x)
    concentration[sender_mask] = s_sender
    concentration[path_mask] = s_path
    concentration[receiver_mask] = s_receiver
    
    return concentration


def animate_hydrogel_interaction(result: dict, save_path: str = None):
    """Create animated heatmap of mRNA propagation."""
    params = result["params"]
    geom = result["geometry"]
    times_h = result["times_h"]
    states = result["states_M"]
    
    # Create spatial grid
    nx = 200
    ny = 50
    x, sender_mask, path_mask, receiver_mask, total_len = create_spatial_grid(params, geom, nx)
    
    # Set up figure
    fig, (ax_heat, ax_conc, ax_chem) = plt.subplots(3, 1, figsize=(14, 10), 
                                                      gridspec_kw={'height_ratios': [2, 1.5, 1.5]})
    
    # Initialize heatmap
    conc_profile = interpolate_spatial_concentration(states[0], x, sender_mask, path_mask, receiver_mask)
    max_conc = np.max(states[:, 0:3] / NANOMOLAR) * 1.1  # Max concentration for colorscale
    
    # Create 2D heatmap data
    heatmap_data = np.tile(conc_profile, (ny, 1))
    
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                        extent=[0, total_len, 0, params.node_length_um],
                        vmin=0, vmax=max_conc, origin='lower')
    
    # Add node boundaries
    node_len = params.node_length_um
    path_len = geom["path_length_um"]
    ax_heat.axvline(node_len, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax_heat.axvline(node_len + path_len, color='white', linestyle='--', linewidth=2, alpha=0.7)
    
    # Labels for regions
    ax_heat.text(node_len/2, params.node_length_um*0.95, 'SENDER', 
                 ha='center', va='top', color='white', fontsize=12, weight='bold')
    ax_heat.text(node_len + path_len/2, params.node_length_um*0.95, 'PATH', 
                 ha='center', va='top', color='white', fontsize=11, weight='bold')
    ax_heat.text(node_len + path_len + node_len/2, params.node_length_um*0.95, 'RECEIVER', 
                 ha='center', va='top', color='white', fontsize=12, weight='bold')
    
    ax_heat.set_xlabel('Position (μm)', fontsize=11)
    ax_heat.set_ylabel('Width (μm)', fontsize=11)
    ax_heat.set_title('mRNA (S2) Spatial Distribution', fontsize=13, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax_heat, label='S2 Concentration (nM)')
    
    # Concentration profile plot
    line_conc, = ax_conc.plot(x, conc_profile, 'b-', linewidth=2.5)
    ax_conc.axvline(node_len, color='gray', linestyle='--', alpha=0.5)
    ax_conc.axvline(node_len + path_len, color='gray', linestyle='--', alpha=0.5)
    ax_conc.set_ylabel('S2 (nM)', fontsize=11)
    ax_conc.set_ylim(0, max_conc)
    ax_conc.set_xlim(0, total_len)
    ax_conc.grid(alpha=0.3)
    ax_conc.set_title('Concentration Profile', fontsize=12)
    
    # Receiver chemistry plot
    line_s2, = ax_chem.plot([], [], 'g-', linewidth=2, label='Free S2')
    line_s2i2, = ax_chem.plot([], [], 'r-', linewidth=2, label='S2:I2')
    line_s2th2, = ax_chem.plot([], [], 'orange', linewidth=2, label='S2:Th2')
    ax_chem.set_xlabel('Time (h)', fontsize=11)
    ax_chem.set_ylabel('Receiver RNA (nM)', fontsize=11)
    ax_chem.set_xlim(0, times_h[-1])
    ax_chem.set_ylim(0, np.max([states[:, 2].max(), states[:, 4].max(), states[:, 6].max()]) / NANOMOLAR * 1.1)
    ax_chem.grid(alpha=0.3)
    ax_chem.legend(loc='upper right', frameon=False)
    ax_chem.set_title('Receiver Chemistry', fontsize=12)
    
    time_text = ax_heat.text(0.02, 0.02, '', transform=ax_heat.transAxes, 
                             fontsize=12, color='white', weight='bold',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    def init():
        """Initialize animation."""
        line_s2.set_data([], [])
        line_s2i2.set_data([], [])
        line_s2th2.set_data([], [])
        return im, line_conc, line_s2, line_s2i2, line_s2th2, time_text
    
    def update(frame):
        """Update animation frame."""
        # Update heatmap
        conc_profile = interpolate_spatial_concentration(states[frame], x, sender_mask, 
                                                         path_mask, receiver_mask)
        heatmap_data = np.tile(conc_profile, (ny, 1))
        im.set_array(heatmap_data)
        
        # Update concentration profile
        line_conc.set_data(x, conc_profile)
        
        # Update receiver chemistry
        line_s2.set_data(times_h[:frame+1], states[:frame+1, 2] / NANOMOLAR)
        line_s2i2.set_data(times_h[:frame+1], states[:frame+1, 4] / NANOMOLAR)
        line_s2th2.set_data(times_h[:frame+1], states[:frame+1, 6] / NANOMOLAR)
        
        # Update time text
        time_text.set_text(f'Time: {times_h[frame]:.2f} h')
        
        return im, line_conc, line_s2, line_s2i2, line_s2th2, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init, 
                                   frames=len(times_h), interval=50, 
                                   blit=True, repeat=True)
    
    fig.suptitle(f'Hydrogel Node Communication | Distance = {params.center_distance_um:.0f} μm', 
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        Writer = animation.writers['pillow']
        writer = Writer(fps=20, metadata=dict(artist='Hydrogel Simulator'), bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved!")
    
    plt.show()
    
    return anim


def main():
    """Main execution function."""
    print("=" * 60)
    print("Hydrogel Node Communication - Real-Time Animation")
    print("=" * 60)
    
    # Configure simulation parameters
    params = SurrogateParams(
        node_length_um=50.0,
        center_distance_um=300.0,
        total_hours=8.0,
        dt_s=300.0,  # 5-minute time steps
        sender_switch_nM=100.0,
        receiver_switch_nM=100.0,
        threshold_uM=5.0,
    )
    
    print(f"\nSimulation Parameters:")
    print(f"  Node size: {params.node_length_um} μm")
    print(f"  Distance between centers: {params.center_distance_um} μm")
    print(f"  Total time: {params.total_hours} hours")
    print(f"  Time step: {params.dt_s} seconds")
    
    print("\nRunning simulation...")
    result = simulate_surrogate(params)
    
    print(f"Simulation complete! ({len(result['times_h'])} time points)")
    print(f"Creating animation...")
    
    # Create animation (saves as GIF)
    anim = animate_hydrogel_interaction(result, save_path="hydrogel_animation.gif")
    
    print("\n" + "=" * 60)
    print("Animation complete! Close the window to exit.")
    print("=" * 60)


if __name__ == "__main__":
    main()