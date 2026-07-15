from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm
from fipy.tools import numerix
import matplotlib.pyplot as plt


def build_problem(nx=40, ny=None, dx=1.0, D=1.0):
    """Create mesh, variable, equation, and masks."""

    if ny is None:
        ny = nx

    mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)
    phi = CellVariable(name="solution variable", mesh=mesh, value=0.0)

    # Domain size
    Lx = nx * dx
    Ly = ny * dx

    # Build a 5x5 center square mask using cell centers
    x, y = mesh.cellCenters
    center_x = Lx / 2.0
    center_y = Ly / 2.0
    half_size = 2.5 * dx   # 5 cells total

    center_square = (
        (x >= center_x - half_size) & (x < center_x + half_size) &
        (y >= center_y - half_size) & (y < center_y + half_size)
    )

    # Set center square to 1 initially
    phi.setValue(1.0, where=center_square)

    # Outer boundary conditions: keep boundary at 0
    phi.constrain(0.0, mesh.facesLeft)
    phi.constrain(0.0, mesh.facesRight)
    phi.constrain(0.0, mesh.facesTop)
    phi.constrain(0.0, mesh.facesBottom)

    # Diffusion equation
    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, center_square


def get_slice_at_y(mesh, phi, y_target):
    """Return x and phi values for cells closest to y_target."""
    x, y = mesh.cellCenters
    mask = numerix.abs(y - y_target) < (mesh.dy / 2.0 + 1e-12)

    x_vals = x[mask]
    phi_vals = phi.value[mask]

    order = numerix.argsort(x_vals)
    return x_vals[order], phi_vals[order]


def run_transient_with_plots(mesh, phi, eq, center_square, y_target=20.0,
                             time_step_factor=0.9, steps=500, D=1.0,
                             field_viewer=True):
    """Run transient solve and animate both field and phi-vs-x slice."""

    dt = time_step_factor * (mesh.dx ** 2) / (2.0 * D)

    # FiPy field viewer
    v = None
    if field_viewer:
        try:
            v = Viewer(vars=phi, datamin=0.0, datamax=1.0)
        except Exception as exc:
            print("FiPy Viewer not available:", exc)
            v = None

    # Matplotlib slice plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals, phi_vals = get_slice_at_y(mesh, phi, y_target)
    line, = ax.plot(x_vals, phi_vals, marker='o')
    ax.set_xlabel("x")
    ax.set_ylabel("phi")
    ax.set_title(f"phi vs x at y = {y_target}")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)

    # Time stepping
    for step in range(steps):
        eq.solve(var=phi, dt=dt)

        # Keep center square fixed
        phi.setValue(1.0, where=center_square)

        # Update FiPy field viewer
        if v is not None:
            v.plot()

        # Update slice plot
        x_vals, phi_vals = get_slice_at_y(mesh, phi, y_target)
        line.set_data(x_vals, phi_vals)
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)
        ax.set_ylim(0.0, 1.05)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    return phi


if __name__ == "__main__":
    mesh, phi, eq, center_square = build_problem(nx=40, dx=1.0, D=1.0)

    # Animate both the 2D field and the 1D slice
    run_transient_with_plots(
        mesh, phi, eq, center_square,
        y_target=20.0,
        steps=500,
        D=1.0,
        field_viewer=True
    )

    # Keep figures open at the end
    plt.show()