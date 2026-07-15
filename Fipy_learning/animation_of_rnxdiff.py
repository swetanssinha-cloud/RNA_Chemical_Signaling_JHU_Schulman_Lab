from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def build_problem(nx=20, ny=None, dx=1.0, D=1.0):
    """Create mesh, variable, equation, and boundary condition masks."""
    
    if ny is None:
        ny = nx
    dy = dx
    L = dx * nx
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

    phi = CellVariable(name="solution variable", mesh=mesh, value=0.0)

    X, Y = mesh.faceCenters
    facesTopLeft = ((mesh.facesLeft & (Y > L / 2)) | (mesh.facesTop & (X < L / 2)))
    facesBottomRight = ((mesh.facesRight & (Y < L / 2)) | (mesh.facesBottom & (X > L / 2)))

    valueTopLeft = 0.0
    valueBottomRight = 1.0
    phi.constrain(valueTopLeft, facesTopLeft)
    phi.constrain(valueBottomRight, facesBottomRight)

    eq = TransientTerm() == DiffusionTerm(coeff=D)

    return mesh, phi, eq, L, valueBottomRight

def animate_diffusion(mesh, phi, eq, time_step_factor=10 * 0.9, 
                      steps=100, D=1.0, interval=50, save_as=None):
    """Create animation of diffusion process."""
    
    timeStepDuration = time_step_factor * (mesh.dx**2) / (2 * D)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Reshape phi values to 2D grid for imshow
    nx = mesh.nx
    ny = mesh.ny
    phi_grid = phi.value.reshape((ny, nx))
    
    # Create initial plot
    im = ax.imshow(phi_grid, origin='lower', cmap='viridis', 
                   vmin=0, vmax=1, extent=[0, mesh.nx*mesh.dx, 0, mesh.ny*mesh.dy])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Diffusion: Step 0')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('phi')
    
    def update(frame):
        """Update function for animation."""
        eq.solve(var=phi, dt=timeStepDuration)
        phi_grid = phi.value.reshape((ny, nx))
        im.set_array(phi_grid)
        ax.set_title(f'Diffusion: Step {frame+1}')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=steps, 
                                  interval=interval, blit=True, repeat=False)
    
    # Save if requested
    if save_as:
        anim.save(save_as, writer='pillow', fps=20)
        print(f"Animation saved as {save_as}")
    
    return anim

if __name__ == "__main__":
    mesh, phi, eq, L, valueBottomRight = build_problem(nx=40, dx=1.0, D=1.0)
    
    # Create and show animation
    anim = animate_diffusion(mesh, phi, eq, time_step_factor=10 * 0.9, 
                            steps=100, D=1.0, interval=50, 
                            save_as='diffusion.gif')  # Remove to skip saving
    
    plt.show()
    