from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def build_problem(nx=20, ny=None, dx=1.0, D=1.0):
    """Create mesh, variable, equation, and boundary condition masks."""

    if ny is None:
        ny = nx

    mesh = Grid2D(dx=dx, dy=dx, nx=nx, ny=ny)
    phi = CellVariable(name="solution variable", mesh=mesh, value=0.0)

    X, Y = mesh.faceCenters
    Lx = nx * dx
    Ly = ny * dx

    facesTopLeft = ((mesh.facesLeft & (Y > Ly / 2)) |
                    (mesh.facesTop & (X < Lx / 2)))
    facesBottomRight = ((mesh.facesRight & (Y < Ly / 2)) |
                        (mesh.facesBottom & (X > Lx / 2)))

    phi.constrain(0.0, facesTopLeft)
    phi.constrain(1.0, facesBottomRight)

    eq = TransientTerm() == DiffusionTerm(coeff=D)
    

    return mesh, phi, eq

def animate_diffusion(mesh, phi, eq, D=1.0, steps=100, interval=50, save_as=None):
    """Create a live animation of diffusion."""

    time_step = 0.9 * (mesh.dx ** 2) / (2 * D)

    fig, ax = plt.subplots(figsize=(8, 6))

    nx = mesh.nx
    ny = mesh.ny

    def get_grid():
        # FiPy values are flattened; reshape into 2D for imshow
        return phi.value.reshape((ny, nx))

    im = ax.imshow(
        get_grid(),
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[0, nx * mesh.dx, 0, ny * mesh.dy],
        interpolation="nearest"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("Diffusion: Step 0")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("phi")

    def update(frame):
        eq.solve(var=phi, dt=time_step)

        im.set_array(get_grid())
        title.set_text(f"Diffusion: Step {frame + 1}")

        return [im, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval,
        blit=False,   # safer than True for many environments
        repeat=False
    )

    if save_as:
        anim.save(save_as, writer="pillow", fps=max(1, 1000 // interval))
        print(f"Animation saved as {save_as}")

    return anim

if __name__ == "__main__":
    mesh, phi, eq = build_problem(nx=40, dx=1.0, D=1.0)

    anim = animate_diffusion(
        mesh, phi, eq,
        D=1.0,
        steps=200,
        interval=50
    )

    plt.show()