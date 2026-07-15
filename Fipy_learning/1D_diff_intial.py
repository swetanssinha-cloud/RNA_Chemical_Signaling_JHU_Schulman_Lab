from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
from fipy.tools import numerix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

'''
what is included in this file:
1. Create a heatmap animation of of a 1D diffusion problem with an intial source at the center of the domain
2. Plot the value of the variable at different time points
3. Plot the analytical solution for a point source in an infinite domain for comparison
'''

def build_problem(nx, dx, D):
    """Create mesh, variable, equation, and center source mask."""

    mesh = Grid1D(dx=dx, nx=nx)
    phi = CellVariable(name="temperature", mesh=mesh, value=0.0)

    # Domain size
    Lx = nx * dx

    # Center 1-cell source region
    x = mesh.cellCenters[0]
    center_x = Lx / 2.0
    half_size = dx / 2.0

    center_source = (
        (x >= center_x - half_size) & (x < center_x + half_size)
    )

    # Diffusion equation
    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, center_source


def run_simulation(mesh, phi, eq, center_source, D=1.0,
                   source_value=1.0, source_duration=1, total_time=10.0,
                   snapshot_interval_time=20.0):
    """Run transient diffusion with animated line plot and heatmap."""

    dt = 0.9 * (mesh.dx ** 2) / (2.0 * D)
    nsteps = int(total_time / dt)

    # Calculate snapshot interval in steps
    snapshot_interval = int(snapshot_interval_time / dt)

    # Store the exact snapshot times we want
    snapshot_times = np.arange(0, total_time + snapshot_interval_time, snapshot_interval_time)
    snapshot_index = 0
    next_snapshot_time = snapshot_times[snapshot_index]

    # Clean up old CSV files
    for f in glob.glob('phi_values_time_*.csv'):
        os.remove(f)

    # Matplotlib setup with two subplots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    x_vals = mesh.cellCenters[0]
    phi_vals = phi.value
    
    # Line plot (top)
    line, = ax1.plot(x_vals, phi_vals, marker='o')
    ax1.set_xlabel("x")
    ax1.set_ylabel("phi")
    ax1.set_title("phi vs x (1D Diffusion)")
    ax1.set_ylim(0.0, source_value * 1.05)
    ax1.grid(True)
    
    # Heatmap (bottom)
    phi_2d = phi_vals.reshape(1, -1)
    im = ax2.imshow(phi_2d, aspect='auto', cmap='hot', 
                    vmin=0.0, vmax=source_value,
                    extent=[x_vals[0]-mesh.dx/2, x_vals[-1]+mesh.dx/2, -0.5, 0.5])
    
    ax2.set_xlabel("x")
    ax2.set_yticks([])
    ax2.set_title("1D Heat Diffusion (Heatmap)")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("phi (temperature)")
    
    plt.tight_layout()

    time = 0.0
    Q_actual = None  # ← FIX: Initialize Q_actual before the loop

    for step in range(nsteps):
        # Apply source only during the first source_duration seconds
        if time < source_duration:
            phi.setValue(source_value, where=center_source)

        # Diffusion step
        eq.solve(var=phi, dt=dt)

        # Calculate Q right after source turns off
        if Q_actual is None and time >= source_duration:
            Q_actual = np.trapezoid(phi.value, x_vals)
            print(f"Total energy at t={time:.2f}s: Q = {Q_actual}")

        # Update line plot
        phi_vals = phi.value
        line.set_ydata(phi_vals)
        ax1.set_title(f"phi vs x (t = {time:.2f}s)")

        # Save snapshot when we cross the next snapshot time
        if time >= next_snapshot_time and snapshot_index < len(snapshot_times):
            df = pd.DataFrame({
                'x': x_vals,
                'phi': phi_vals
            })
            df.to_csv(f'phi_values_time_{next_snapshot_time:.2f}s.csv', index=False)
            print(f"Saved snapshot at t = {time:.2f}s (target: {next_snapshot_time:.2f}s)")
            snapshot_index += 1
            if snapshot_index < len(snapshot_times):
                next_snapshot_time = snapshot_times[snapshot_index]
        
        # Update heatmap
        phi_2d = phi_vals.reshape(1, -1)
        im.set_data(phi_2d)
        ax2.set_title(f"1D Heat Diffusion Heatmap (t = {time:.2f}s)")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        time += dt

    plt.ioff()
    return phi, Q_actual


def analytical_solution_gaussian(x, t, x0, D, Q):
    """Analytical solution for point source in infinite domain"""
    if t <= 0:
        return np.zeros_like(x)
    return (Q / np.sqrt(4 * np.pi * D * t)) * np.exp(-(x - x0)**2 / (4 * D * t))


if __name__ == "__main__":
    mesh, phi, eq, center_source = build_problem(nx=401, dx=1.0, D=1.0) #try 399 if it goes too far in the other direction. 

    
    # Animate both line plot and heatmap
    phi, Q_actual = run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        source_value=400.0,
        source_duration=1.0,  # 1 second source duration
        total_time=100.0,
        snapshot_interval_time=20.0
    )
    
    print(f"\nUsing Q_actual = {Q_actual:.2f} for analytical solution\n")

    # Read and plot all snapshots
    csv_files = sorted(glob.glob('phi_values_time_*.csv'))
    
    if csv_files:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot numerical simulation results
        for file in csv_files:
            df = pd.read_csv(file)
            # Extract time from filename for label
            time_str = file.replace('phi_values_time_', '').replace('.csv', '')
            ax.plot(df["x"], df["phi"], label=f'Simulation t = {time_str}', 
                   marker='o', markersize=3, alpha=0.7)

        # Add analytical solutions
        x_analytical = np.linspace(0, 400, 1000)
        Q = Q_actual  # Use the calculated total energy
        
        for time_val in [20, 40, 60, 80, 100]:  # times in seconds
            phi_analytical = analytical_solution_gaussian(x_analytical, time_val, 200, 1.0, Q)
            ax.plot(x_analytical, phi_analytical, '--', 
                   label=f'Analytical t = {time_val}.00s', alpha=0.7, linewidth=2)

        ax.set_xlabel("x (position)")
        ax.set_ylabel("phi (temperature)")
        ax.set_title("Numerical Simulation vs Analytical Solution")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No snapshot files found!")
    
    plt.close('all')