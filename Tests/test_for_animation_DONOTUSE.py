#!/usr/bin/env python3
"""
2D Spatial Animation of Sender/Receiver System

Creates an animated heatmap showing how S2 signal propagates through
the 2D domain from sender node to receiver node over time.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# If you want to use the FiPy version, import from there
# Otherwise, we'll create a simplified 2D finite difference solver

NANOMOLAR = 1e-9
MICROMOLAR = 1e-6


@dataclass
class AnimationParams:
    """Parameters for the 2D spatial simulation."""
    node_length_um: float = 50.0
    center_distance_um: float = 300.0
    bath_margin_um: float = 250.0
    dx_um: float = 10.0
    total_hours: float = 8.0
    dt_s: float = 60.0
    
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


class SpatialSimulator2D:
    """2D finite difference solver for reaction-diffusion system."""
    
    def __init__(self, params: AnimationParams):
        self.params = params
        self.setup_grid()
        self.initialize_fields()
        self.current_time = 0.0
        self.step_count = 0
        
    def setup_grid(self):
        """Create 2D spatial grid."""
        p = self.params
        
        # Domain size
        self.width_um = 2.0 * p.bath_margin_um + p.center_distance_um + p.node_length_um
        self.height_um = 2.0 * p.bath_margin_um + p.node_length_um
        
        # Grid resolution
        self.nx = int(np.ceil(self.width_um / p.dx_um))
        self.ny = int(np.ceil(self.height_um / p.dx_um))
        
        # Actual grid spacing (may be slightly adjusted)
        self.dx = self.width_um / self.nx
        self.dy = self.height_um / self.ny
        
        # Create coordinate arrays
        self.x = np.linspace(0, self.width_um, self.nx)
        self.y = np.linspace(0, self.height_um, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Define node locations
        sender_center_x = p.bath_margin_um + 0.5 * p.node_length_um
        sender_center_y = 0.5 * self.height_um
        receiver_center_x = sender_center_x + p.center_distance_um
        receiver_center_y = sender_center_y
        
        half = 0.5 * p.node_length_um
        
        # Create masks for sender and receiver nodes
        self.sender_mask = (
            (np.abs(self.X - sender_center_x) <= half) & 
            (np.abs(self.Y - sender_center_y) <= half)
        )
        
        self.receiver_mask = (
            (np.abs(self.X - receiver_center_x) <= half) & 
            (np.abs(self.Y - receiver_center_y) <= half)
        )
        
        # Store node positions for visualization
        self.sender_rect = (
            sender_center_x - half,
            sender_center_y - half,
            p.node_length_um,
            p.node_length_um
        )
        
        self.receiver_rect = (
            receiver_center_x - half,
            receiver_center_y - half,
            p.node_length_um,
            p.node_length_um
        )
        
        print(f"Grid: {self.nx} x {self.ny} = {self.nx * self.ny} cells")
        print(f"Domain: {self.width_um:.1f} x {self.height_um:.1f} μm")
        
    def initialize_fields(self):
        """Initialize concentration fields."""
        p = self.params
        
        # S2 and its complexes (2D fields)
        self.S2 = np.zeros((self.ny, self.nx))
        
        # Receiver chemistry (only in receiver node)
        self.I2 = np.zeros((self.ny, self.nx))
        self.S2_I2 = np.zeros((self.ny, self.nx))
        self.Th2 = np.zeros((self.ny, self.nx))
        self.S2_Th2 = np.zeros((self.ny, self.nx))
        
        # Initialize receiver node with I2 and Th2
        self.I2[self.receiver_mask] = p.receiver_switch_nM * NANOMOLAR
        self.Th2[self.receiver_mask] = p.threshold_uM * MICROMOLAR
        
        # Sender concentration (constant in sender node)
        self.I1O2 = np.zeros((self.ny, self.nx))
        self.I1O2[self.sender_mask] = p.sender_switch_nM * NANOMOLAR
        
        # Diffusion coefficient field
        self.D = np.ones((self.ny, self.nx)) * p.d_solution_um2_s
        self.D[self.sender_mask] = p.d_gel_um2_s
        self.D[self.receiver_mask] = p.d_gel_um2_s
        
        # Storage for time series
        self.times_h = [0.0]
        self.receiver_i2_mean = [np.mean(self.I2[self.receiver_mask]) / NANOMOLAR]
        self.receiver_s2_mean = [np.mean(self.S2[self.receiver_mask]) / NANOMOLAR]
        self.receiver_total_rna = [
            np.mean((self.S2 + self.S2_I2 + self.S2_Th2)[self.receiver_mask]) / NANOMOLAR
        ]
        
    def diffusion_step(self, C, D, dt):
        """
        Compute diffusion using explicit finite difference.
        ∂C/∂t = ∇·(D∇C)
        
        For uniform grid with variable D, using central differences.
        """
        # Preallocate output
        dC_dt = np.zeros_like(C)
        
        # Interior points (avoid boundaries)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                # D at cell faces (harmonic mean for better accuracy)
                D_right = 2 * D[i, j] * D[i, j+1] / (D[i, j] + D[i, j+1] + 1e-20)
                D_left = 2 * D[i, j] * D[i, j-1] / (D[i, j] + D[i, j-1] + 1e-20)
                D_up = 2 * D[i, j] * D[i+1, j] / (D[i, j] + D[i+1, j] + 1e-20)
                D_down = 2 * D[i, j] * D[i-1, j] / (D[i, j] + D[i-1, j] + 1e-20)
                
                # Flux divergence
                flux_x = (D_right * (C[i, j+1] - C[i, j]) - 
                         D_left * (C[i, j] - C[i, j-1])) / self.dx**2
                
                flux_y = (D_up * (C[i+1, j] - C[i, j]) - 
                         D_down * (C[i, j] - C[i-1, j])) / self.dy**2
                
                dC_dt[i, j] = flux_x + flux_y
        
        # No-flux boundary conditions (automatically satisfied by not updating boundaries)
        
        return dC_dt
    
    def reaction_terms(self):
        """Compute reaction terms for all species."""
        p = self.params
        
        # Production in sender
        production = p.k_p_s_inv * self.I1O2
        
        # Degradation everywhere
        degradation_S2 = p.k_d_ss_s_inv * self.S2
        
        # Binding reactions (only in receiver node)
        bind_I2 = np.zeros_like(self.S2)
        bind_Th2 = np.zeros_like(self.S2)
        unbind_I2 = np.zeros_like(self.S2)
        unbind_Th2 = np.zeros_like(self.S2)
        
        # Only compute where receiver is present
        bind_I2[self.receiver_mask] = (
            p.k_slow_M_inv_s_inv * 
            self.S2[self.receiver_mask] * 
            self.I2[self.receiver_mask]
        )
        
        bind_Th2[self.receiver_mask] = (
            p.k_fast_M_inv_s_inv * 
            self.S2[self.receiver_mask] * 
            self.Th2[self.receiver_mask]
        )
        
        unbind_I2[self.receiver_mask] = (
            p.k_d_ds_s_inv * self.S2_I2[self.receiver_mask]
        )
        
        unbind_Th2[self.receiver_mask] = (
            p.k_d_ds_s_inv * self.S2_Th2[self.receiver_mask]
        )
        
        return {
            'production': production,
            'degradation_S2': degradation_S2,
            'bind_I2': bind_I2,
            'bind_Th2': bind_Th2,
            'unbind_I2': unbind_I2,
            'unbind_Th2': unbind_Th2,
        }
    
    def step(self):
        """Advance simulation by one time step using operator splitting."""
        dt = self.params.dt_s
        
        # Step 1: Diffusion (explicit)
        diffusion = self.diffusion_step(self.S2, self.D, dt)
        
        # Step 2: Reactions
        reactions = self.reaction_terms()
        
        # Update S2
        self.S2 += dt * (
            diffusion + 
            reactions['production'] - 
            reactions['degradation_S2'] -
            reactions['bind_I2'] -
            reactions['bind_Th2'] +
            reactions['unbind_I2'] +
            reactions['unbind_Th2']
        )
        
        # Update receiver species
        self.I2 += dt * (reactions['unbind_I2'] - reactions['bind_I2'])
        self.S2_I2 += dt * (reactions['bind_I2'] - reactions['unbind_I2'])
        self.Th2 += dt * (reactions['unbind_Th2'] - reactions['bind_Th2'])
        self.S2_Th2 += dt * (reactions['bind_Th2'] - reactions['unbind_Th2'])
        
        # Enforce non-negativity
        self.S2 = np.maximum(self.S2, 0.0)
        self.I2 = np.maximum(self.I2, 0.0)
        self.S2_I2 = np.maximum(self.S2_I2, 0.0)
        self.Th2 = np.maximum(self.Th2, 0.0)
        self.S2_Th2 = np.maximum(self.S2_Th2, 0.0)
        
        # Update time
        self.current_time += dt
        self.step_count += 1
        
        # Store statistics
        self.times_h.append(self.current_time / 3600.0)
        self.receiver_i2_mean.append(np.mean(self.I2[self.receiver_mask]) / NANOMOLAR)
        self.receiver_s2_mean.append(np.mean(self.S2[self.receiver_mask]) / NANOMOLAR)
        self.receiver_total_rna.append(
            np.mean((self.S2 + self.S2_I2 + self.S2_Th2)[self.receiver_mask]) / NANOMOLAR
        )
        
        # Check stability (CFL condition for explicit diffusion)
        max_D = max(self.params.d_gel_um2_s, self.params.d_solution_um2_s)
        cfl = max_D * dt / min(self.dx**2, self.dy**2)
        if cfl > 0.5 and self.step_count == 1:
            print(f"WARNING: CFL number = {cfl:.3f} (should be < 0.5 for stability)")
            print(f"Consider reducing dt_s or increasing dx_um")


class AnimatedHeatmap:
    """Create animated visualization of 2D spatial dynamics."""
    
    def __init__(self, simulator: SpatialSimulator2D):
        self.sim = simulator
        self.setup_figure()
        
    def setup_figure(self):
        """Create figure with heatmap and time series plots."""
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[2, 1], 
                     hspace=0.3, wspace=0.3)
        
        # Main heatmap
        self.ax_heat = self.fig.add_subplot(gs[0, :])
        
        # Time series plots
        self.ax_i2 = self.fig.add_subplot(gs[1, 0])
        self.ax_rna = self.fig.add_subplot(gs[1, 1])
        
        # Initialize heatmap
        self.im = self.ax_heat.imshow(
            self.sim.S2 / NANOMOLAR,
            origin='lower',
            cmap='hot',
            interpolation='bilinear',
            extent=[0, self.sim.width_um, 0, self.sim.height_um],
            vmin=0,
            vmax=10,  # Will auto-adjust
        )
        
        # Draw node outlines
        sender_rect = Rectangle(
            (self.sim.sender_rect[0], self.sim.sender_rect[1]),
            self.sim.sender_rect[2], self.sim.sender_rect[3],
            linewidth=2, edgecolor='cyan', facecolor='none',
            linestyle='--', label='Sender'
        )
        self.ax_heat.add_patch(sender_rect)
        
        receiver_rect = Rectangle(
            (self.sim.receiver_rect[0], self.sim.receiver_rect[1]),
            self.sim.receiver_rect[2], self.sim.receiver_rect[3],
            linewidth=2, edgecolor='lime', facecolor='none',
            linestyle='--', label='Receiver'
        )
        self.ax_heat.add_patch(receiver_rect)
        
        self.ax_heat.set_xlabel('X Position (μm)', fontsize=12)
        self.ax_heat.set_ylabel('Y Position (μm)', fontsize=12)
        self.ax_heat.set_title('S2 Signal Concentration (nM)', fontsize=14, weight='bold')
        self.ax_heat.legend(loc='upper right', fontsize=10)
        
        # Colorbar
        self.cbar = plt.colorbar(self.im, ax=self.ax_heat, fraction=0.046, pad=0.04)
        self.cbar.set_label('Concentration (nM)', fontsize=11)
        
        # Time text
        self.time_text = self.ax_heat.text(
            0.02, 0.98, '', transform=self.ax_heat.transAxes,
            fontsize=14, weight='bold', color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
        )
        
        # Initialize time series plots
        self.line_i2, = self.ax_i2.plot([], [], 'b-', lw=2.5, label='I2')
        self.ax_i2.set_xlabel('Time (h)', fontsize=11)
        self.ax_i2.set_ylabel('Receiver I2 (nM)', fontsize=11)
        self.ax_i2.set_xlim(0, self.sim.params.total_hours)
        self.ax_i2.set_ylim(0, self.sim.params.receiver_switch_nM * 1.1)
        self.ax_i2.grid(alpha=0.3)
        self.ax_i2.legend()
        
        self.line_rna, = self.ax_rna.plot([], [], 'r-', lw=2.5, label='Total RNA')
        self.line_s2, = self.ax_rna.plot([], [], 'g--', lw=2, label='Free S2', alpha=0.7)
        self.ax_rna.set_xlabel('Time (h)', fontsize=11)
        self.ax_rna.set_ylabel('Receiver Concentration (nM)', fontsize=11)
        self.ax_rna.set_xlim(0, self.sim.params.total_hours)
        self.ax_rna.set_ylim(0, 50)
        self.ax_rna.grid(alpha=0.3)
        self.ax_rna.legend()
        
        p = self.sim.params
        self.fig.suptitle(
            f'2D Sender-Receiver Dynamics | Distance: {p.center_distance_um:.0f} μm | '
            f'Threshold: {p.threshold_uM:.1f} μM',
            fontsize=15, weight='bold'
        )
        
    def update(self, frame):
        """Update function for animation."""
        # Run simulation step
        self.sim.step()
        
        # Update heatmap
        S2_nM = self.sim.S2 / NANOMOLAR
        self.im.set_array(S2_nM)
        
        # Auto-scale colormap
        vmax = max(np.percentile(S2_nM, 99), 1.0)  # Use 99th percentile to avoid outliers
        self.im.set_clim(0, vmax)
        
        # Update time text
        self.time_text.set_text(f't = {self.sim.current_time/3600:.2f} h')
        
        # Update time series
        times = np.array(self.sim.times_h)
        self.line_i2.set_data(times, self.sim.receiver_i2_mean)
        self.line_rna.set_data(times, self.sim.receiver_total_rna)
        self.line_s2.set_data(times, self.sim.receiver_s2_mean)
        
        # Auto-scale RNA plot
        max_rna = max(max(self.sim.receiver_total_rna), 1.0)
        self.ax_rna.set_ylim(0, max_rna * 1.1)
        
        # Print progress
        total_steps = int(np.ceil(self.sim.params.total_hours * 3600.0 / self.sim.params.dt_s))
        if frame % max(1, total_steps // 20) == 0:
            print(f"Step {frame}/{total_steps} | t = {self.sim.current_time/3600:.2f} h | "
                  f"Max S2 = {np.max(S2_nM):.2f} nM | "
                  f"Receiver I2 = {self.sim.receiver_i2_mean[-1]:.2f} nM")
        
        return self.im, self.time_text, self.line_i2, self.line_rna
    
    def animate(self, output_path: Path, fps: int = 10, dpi: int = 100):
        """Create and save animation."""
        n_frames = int(np.ceil(self.sim.params.total_hours * 3600.0 / self.sim.params.dt_s))
        
        print(f"Creating animation: {n_frames} frames at {fps} fps")
        print(f"This will take approximately {n_frames/fps:.1f} seconds of video")
        
        anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=n_frames,
            interval=1000/fps,
            blit=False,
            repeat=False
        )
        
        # Save
        print("Saving animation (this may take a while)...")
        writer = animation.FFMpegWriter(
            fps=fps, 
            bitrate=3000, 
            codec='h264',
            extra_args=['-pix_fmt', 'yuv420p']  # Better compatibility
        )
        anim.save(output_path, writer=writer, dpi=dpi)
        
        print(f"Animation saved to {output_path}")
        plt.close(self.fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance-um', type=float, default=300.0,
                       help='Center-to-center distance between nodes')
    parser.add_argument('--node-length-um', type=float, default=50.0,
                       help='Side length of square nodes')
    parser.add_argument('--bath-margin-um', type=float, default=250.0,
                       help='Bath margin around nodes')
    parser.add_argument('--dx-um', type=float, default=10.0,
                       help='Spatial grid resolution')
    parser.add_argument('--hours', type=float, default=8.0,
                       help='Simulation duration')
    parser.add_argument('--dt-s', type=float, default=60.0,
                       help='Time step (must satisfy CFL condition)')
    parser.add_argument('--threshold-uM', type=float, default=5.0,
                       help='Receiver threshold concentration')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation frames per second')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Animation resolution (lower = faster)')
    parser.add_argument('--output-dir', type=Path, default=Path('animations_2d'),
                       help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup parameters
    params = AnimationParams(
        node_length_um=args.node_length_um,
        center_distance_um=args.distance_um,
        bath_margin_um=args.bath_margin_um,
        dx_um=args.dx_um,
        total_hours=args.hours,
        dt_s=args.dt_s,
        threshold_uM=args.threshold_uM,
    )
    
    # Create simulator
    print("Initializing 2D spatial simulator...")
    simulator = SpatialSimulator2D(params)
    
    # Create animator
    animator = AnimatedHeatmap(simulator)
    
    # Generate animation
    output_path = args.output_dir / f'spatial_2d_d{args.distance_um:.0f}um.mp4'
    animator.animate(output_path, fps=args.fps, dpi=args.dpi)
    
    print("\n=== Final Results ===")
    print(f"Final receiver I2: {simulator.receiver_i2_mean[-1]:.2f} nM")
    print(f"Final receiver total RNA: {simulator.receiver_total_rna[-1]:.2f} nM")
    print(f"Switch depletion: {(1 - simulator.receiver_i2_mean[-1]/params.receiver_switch_nM)*100:.1f}%")


if __name__ == '__main__':
    main()