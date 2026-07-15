from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm
from fipy.tools import numerix
import matplotlib.pyplot as plt

def build_problem(nx=20, ny=None, dx=1.0, D=1.0):
    """Create mesh, variable, equation, and boundary condition masks."""
    
    if ny is None:
        ny = nx
    dy = dx
    L = dx * nx
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    phi = CellVariable(name="solution variable", mesh=mesh, value=0.0)

    # Set up Dirichlet (fixed boundary conditions) faces for top-left and bottom-right corner regions
    X, Y = mesh.faceCenters
    facesTopLeft = ((mesh.facesLeft & (Y > L / 2)) | (mesh.facesTop & (X < L / 2)))
    facesBottomRight = ((mesh.facesRight & (Y < L / 2)) | (mesh.facesBottom & (X > L / 2)))

    valueTopLeft = 0.0
    valueBottomRight = 1.0
    phi.constrain(valueTopLeft, facesTopLeft)
    phi.constrain(valueBottomRight, facesBottomRight)

    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, L, valueBottomRight

def run_transient(mesh, phi, eq, L, valueBottomRight, \
time_step_factor=10 * 0.9, steps=10, D=1.0, viewer=False):
    """Run transient solve and optionally plot at each step."""

    timeStepDuration = time_step_factor * (mesh.dx**2) / (2 * D)
    if viewer:
        try:
            v = Viewer(vars=phi, datamin=0., datamax=1.)
        except Exception as exc:
            print("Viewer not available:", exc)
            v = None
        else:
            v = None


    for step in range(steps):
        eq.solve(var=phi, dt=timeStepDuration)
        if v is not None:
            v.plot()

    # Test value of bottom-right corner cell (as in original)
    is_close = numerix.allclose(phi(((L,), (0,))), valueBottomRight, atol=1e-2)
    return is_close

def solve_steady(phi):
    """Solve steady-state diffusion and return result (phi updated in-place)."""
    
    DiffusionTerm().solve(var=phi)

if __name__ == "__main__":
    mesh, phi, eq, L, valueBottomRight = build_problem(nx=40, dx=1.0, D=1.0)
    ok_transient = run_transient(mesh, phi, eq, L, valueBottomRight, time_step_factor=10 * 0.9, steps=10, D=1.0, viewer=True)
    print("Transient bottom-right close to expected:", bool(ok_transient))

# Steady-state solve (re-using phi)
solve_steady(phi)
try:
    v = Viewer(vars=phi, datamin=0., datamax=1.)
    v.plot()
except Exception as exc:
    print("Viewer not available for steady-state plot:", exc)

ok_steady = numerix.allclose(phi(((L,), (0,))), valueBottomRight, atol=1e-2)
print("Steady-state bottom-right close to expected:", bool(ok_steady))

plt.show(block=True)