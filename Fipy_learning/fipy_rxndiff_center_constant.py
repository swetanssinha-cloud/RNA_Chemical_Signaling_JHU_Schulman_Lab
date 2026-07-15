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

    # ----- Build a 5x5 center square mask using cell centers -----
    x, y = mesh.cellCenters

    center_x = Lx / 2.0
    center_y = Ly / 2.0

    half_size = 2.5 * dx   # 5 cells total: from center - 2.5dx to center + 2.5dx

    center_square = (
        (x >= center_x - half_size) & (x < center_x + half_size) &
        (y >= center_y - half_size) & (y < center_y + half_size)
    )

    # Set the center square to 1 initially
    phi.setValue(1.0, where=center_square)

    # Outer boundary conditions: keep boundary at 0
    phi.constrain(0.0, mesh.facesLeft)
    phi.constrain(0.0, mesh.facesRight)
    phi.constrain(0.0, mesh.facesTop)
    phi.constrain(0.0, mesh.facesBottom)

    # Diffusion equation
    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, center_square


def run_transient(mesh, phi, eq, center_square, time_step_factor=0.9, steps=100, D=1.0, viewer=False): #orginally 10 steps
    """Run transient solve and keep center square fixed at 1."""
    
    dt = time_step_factor * (mesh.dx ** 2) / (2.0 * D)

    v = None
    if viewer:
        try:
            v = Viewer(vars=phi, datamin=0.0, datamax=1.0)
        except Exception as exc:
            print("Viewer not available:", exc)

    for step in range(steps):
        eq.solve(var=phi, dt=dt)

        # Re-impose the center square value so it stays fixed
        phi.setValue(1.0, where=center_square)

        if v is not None:
            v.plot()

    return phi


def solve_steady(phi, center_square):
    """Solve steady-state diffusion with center square fixed at 1."""
    
    eq_steady = DiffusionTerm()

    # Re-impose center square before solving
    phi.setValue(1.0, where=center_square)

    # Solve steady-state
    eq_steady.solve(var=phi)

    # Re-impose again in case solver changes it
    phi.setValue(1.0, where=center_square)




if __name__ == "__main__":
    mesh, phi, eq, center_square = build_problem(nx=40, dx=1.0, D=1.0)

    # Transient solve
    run_transient(mesh, phi, eq, center_square, steps=500, D=1.0, viewer=True)

    # Steady-state solve
    solve_steady(phi, center_square)

    # Plot final result
    try:
        v = Viewer(vars=phi, datamin=0.0, datamax=1.0)
        v.plot()
    except Exception as exc:
        print("Viewer not available for steady-state plot:", exc)
