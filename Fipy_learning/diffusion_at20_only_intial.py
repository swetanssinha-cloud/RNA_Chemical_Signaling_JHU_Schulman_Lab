from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix
import matplotlib.pyplot as plt
import numpy as np


def build_problem(nx, ny=None, dx=1.0, D=1.0):
    """Create mesh, variable, equation, and center source mask."""

    if ny is None:
        ny = nx

    mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)
    phi = CellVariable(name="temperature", mesh=mesh, value=0.0)

    # Domain size
    Lx = nx * dx
    Ly = ny * dx

    # Center 5x5 source region
    x, y = mesh.cellCenters
    center_x = Lx / 2.0
    center_y = Ly / 2.0
    half_size = 2.5 * dx

    center_source = (
        (x >= center_x - half_size) & (x < center_x + half_size) &
        (y >= center_y - half_size) & (y < center_y + half_size)
    )

    # leaving these in gives zero temperature on outer boundary
    # commenting these out gives non-zero temperature on outer boundary
    # phi.constrain(0.0, mesh.facesLeft)
    # phi.constrain(0.0, mesh.facesRight)
    # phi.constrain(0.0, mesh.facesTop)
    # phi.constrain(0.0, mesh.facesBottom)

    # Diffusion equation
    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, center_source


def get_slice_at_y(mesh, phi, y_target):
    """Return x and phi values for cells closest to y_target."""
    x, y = mesh.cellCenters
    mask = numerix.abs(y - y_target) < (mesh.dy / 2.0 + 1e-12)

    x_vals = x[mask]
    phi_vals = phi.value[mask]

    order = numerix.argsort(x_vals)
    return x_vals[order], phi_vals[order]


def run_simulation(mesh, phi, eq, center_source, D=1.0,
                   source_value=1.0, source_duration=2.0, total_time=10.0,
                   y_target=None):
    """Run transient diffusion with animated 2D heatmap and 1D slice plot."""

    dt = 0.9 * (mesh.dx ** 2) / (2.0 * D)
    nsteps = int(total_time / dt)

    if y_target is None:
        # middle row
        y_target = mesh.dy * mesh.ny / 2.0

    # Matplotlib setup with two subplots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D Heatmap (left)
    x, y = mesh.cellCenters
    # Reshape phi values to 2D grid
    phi_2d = phi.value.reshape(mesh.ny, mesh.nx)
    
    im = ax1.imshow(phi_2d, origin='lower', cmap='hot', 
                    vmin=0.0, vmax=source_value,
                    extent=[0, mesh.nx * mesh.dx, 0, mesh.ny * mesh.dy],
                    aspect='equal')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("2D Heat Diffusion (Heatmap)")
    cbar1 = plt.colorbar(im, ax=ax1)
    cbar1.set_label("variable ")
    
    # Add horizontal line to show slice position
    slice_line = ax1.axhline(y=y_target, color='cyan', linestyle='--', linewidth=2, label=f'Slice at y={y_target}')
    ax1.legend()
    
    # 1D Slice plot (right)
    x_vals, phi_vals = get_slice_at_y(mesh, phi, y_target)
    line, = ax2.plot(x_vals, phi_vals, marker='o', color='blue')
    ax2.set_xlabel("x")
    ax2.set_ylabel("phi")
    ax2.set_title(f"phi vs x at y = {y_target}")
    ax2.set_ylim(0.0, source_value * 1.05)
    ax2.grid(True)
    
    plt.tight_layout()

    time = 0.0

    for step in range(nsteps):
        # Apply source only during the first source_duration seconds
        if time < source_duration:
            phi.setValue(source_value, where=center_source)

        # Diffusion step
        eq.solve(var=phi, dt=dt)

        # Update 2D heatmap
        phi_2d = phi.value.reshape(mesh.ny, mesh.nx)
        im.set_data(phi_2d)
        ax1.set_title(f"2D Heat Diffusion (t = {time:.2f}s)")
        
        # Update 1D slice plot
        x_vals, phi_vals = get_slice_at_y(mesh, phi, y_target)
        line.set_data(x_vals, phi_vals)
        ax2.set_title(f"phi vs x at y = {y_target} (t = {time:.2f}s)")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        time += dt

    plt.ioff()
    return phi


if __name__ == "__main__":
    size = 100
    mesh, phi, eq, center_source = build_problem(nx=size, dx=1.0, D=1.0)

    # Animate both the 2D heatmap and the 1D slice plot
    run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        source_value=400.0,
        source_duration=2.0,
        total_time=100.0,
        y_target=None      # middle row for a 40x40 grid
    )

    plt.show()
    plt.close('all')