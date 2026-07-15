from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

'''
Run 1D diffusion simulation with INSTANTANEOUS source at t=0.
Energy is injected at the beginning and then conserved throughout.
'''

def build_problem(nx, dx, D):
    """Create mesh, variable, equation, and center source mask."""
    mesh = Grid1D(dx=dx, nx=nx)
    phi = CellVariable(name="temperature", mesh=mesh, value=0.0)
    

    
    Lx = nx * dx
    x = mesh.cellCenters[0]
    center_x = Lx / 2.0
    half_size = dx / 2.0
    
    center_source = (
        (x >= center_x - half_size) & (x < center_x + half_size)
    )
    
    eq = TransientTerm() == DiffusionTerm(coeff=D)
    return mesh, phi, eq, center_source


def run_simulation(mesh, phi, eq, center_source, D=1.0,
                   total_energy=400.0, total_time=100.0,
                   snapshot_interval_time=20.0, show_animation=False, 
                   save_animation=False, animation_filename='diffusion_animation.mp4'):
    """
    Run transient diffusion with instantaneous energy injection at t=0.
    
    Parameters:
    -----------
    total_energy : float
        Total energy to inject at t=0 (distributed over source region)
    """
    
    dt = 0.9 * (mesh.dx ** 2) / (2.0 * D)
    nsteps = int(total_time / dt)
    
    snapshot_times = np.arange(0, total_time + snapshot_interval_time, snapshot_interval_time)
    snapshot_index = 0
    next_snapshot_time = snapshot_times[snapshot_index]
    
    # Clean up old CSV files
    for f in glob.glob('phi_values_time_*.csv'):
        os.remove(f)
    
    x_vals = np.array(mesh.cellCenters[0])
    
    # INJECT ENERGY AT t=0
    # Calculate source volume and initial concentration
    source_volume = np.sum(center_source.value) * mesh.dx
    initial_concentration = total_energy / source_volume
    
    print(f"\n=== INITIAL ENERGY INJECTION ===")
    print(f"Source region: {np.sum(center_source.value)} cells")
    print(f"Source volume: {source_volume:.6f}")
    print(f"Initial concentration: {initial_concentration:.2f}")
    
    phi.setValue(initial_concentration, where=center_source)
    
    # Measure initial energy
    Q_initial = np.trapezoid(phi.value, x_vals)
    print(f"Initial total energy (Q): {Q_initial:.6f}")
    print(f"Target energy: {total_energy:.6f}")
    print(f"Injection error: {abs(Q_initial - total_energy)/total_energy * 100:.4f}%")
    print("="*35 + "\n")
    
    # Animation setup
    if show_animation or save_animation:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        line, = ax1.plot(x_vals, phi.value, marker='o')
        ax1.set_xlabel("x")
        ax1.set_ylabel("phi")
        ax1.set_title("phi vs x (1D Diffusion)")
        ax1.set_ylim(0.0, initial_concentration * 1.1)
        ax1.grid(True)
        
        phi_2d = phi.value.reshape(1, -1)
        im = ax2.imshow(phi_2d, aspect='auto', cmap='hot', 
                        vmin=0.0, vmax=initial_concentration,
                        extent=[x_vals[0]-mesh.dx/2, x_vals[-1]+mesh.dx/2, -0.5, 0.5])
        ax2.set_xlabel("x")
        ax2.set_yticks([])
        ax2.set_title("1D Heat Diffusion (Heatmap)")
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("phi (temperature)")
        plt.tight_layout()
        
        if show_animation:
            plt.ion()
        
        frames = []
    
    time = 0.0
    
    # Track energy conservation
    energy_checks = []
    
    for step in range(nsteps):
        # NO SOURCE - just diffuse!
        eq.solve(var=phi, dt=dt)
        
        time += dt
        
        # Check energy conservation periodically
        if step % 100 == 0:
            Q_current = np.trapezoid(phi.value, x_vals)
            energy_checks.append((time, Q_current))
        
        # Save snapshot
        if time >= next_snapshot_time and snapshot_index < len(snapshot_times):
            Q_current = np.trapezoid(phi.value, x_vals)
            df = pd.DataFrame({
                'x': x_vals,
                'phi': phi.value
            })
            df.to_csv(f'phi_values_time_{next_snapshot_time:.2f}s.csv', index=False)
            print(f"Snapshot at t={time:.2f}s: Q={Q_current:.6f}, "
                  f"conservation error={(Q_current-Q_initial)/Q_initial*100:.4f}%")
            
            snapshot_index += 1
            if snapshot_index < len(snapshot_times):
                next_snapshot_time = snapshot_times[snapshot_index]
        
        # Update animation
        if show_animation or save_animation:
            line.set_ydata(phi.value)
            ax1.set_title(f"phi vs x (t = {time:.2f}s)")
            phi_2d = phi.value.reshape(1, -1)
            im.set_data(phi_2d)
            ax2.set_title(f"1D Heat Diffusion Heatmap (t = {time:.2f}s)")
            
            if save_animation:
                frames.append((time, phi.value.copy()))
            
            if show_animation:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
    
    # Final energy check
    Q_final = np.trapezoid(phi.value, x_vals)
    conservation_error = abs(Q_final - Q_initial) / Q_initial * 100
    
    print(f"\n=== ENERGY CONSERVATION CHECK ===")
    print(f"Initial energy: {Q_initial:.6f}")
    print(f"Final energy:   {Q_final:.6f}")
    print(f"Conservation error: {conservation_error:.4f}%")
    print("="*35 + "\n")
    
    # Save animation
    if save_animation:
        print(f"\nCreating animation with {len(frames)} frames...")
        
        def update_frame(frame_num):
            t, phi_vals = frames[frame_num]
            line.set_ydata(phi_vals)
            ax1.set_title(f"phi vs x (t = {t:.2f}s)")
            phi_2d = phi_vals.reshape(1, -1)
            im.set_data(phi_2d)
            ax2.set_title(f"1D Heat Diffusion Heatmap (t = {t:.2f}s)")
            return line, im
        
        frame_skip = max(1, len(frames) // 500)
        frame_indices = range(0, len(frames), frame_skip)
        
        anim = FuncAnimation(fig, update_frame, frames=frame_indices, 
                           interval=50, blit=False)
        
        try:
            writer = FFMpegWriter(fps=20, bitrate=1800)
            anim.save(animation_filename, writer=writer)
            print(f"Animation saved as {animation_filename}")
        except:
            print("FFmpeg not found, saving as GIF instead...")
            gif_filename = animation_filename.replace('.mp4', '.gif')
            writer = PillowWriter(fps=20)
            anim.save(gif_filename, writer=writer)
            print(f"Animation saved as {gif_filename}")
    
    if show_animation:
        plt.ioff()
    
    if show_animation or save_animation:
        plt.close(fig)
    
    # Save Q_initial (the conserved quantity) to file
    with open('Q_actual.txt', 'w') as f:
        f.write(str(Q_initial))
    
    print(f"Simulation complete!")
    print(f"Saved {len(snapshot_times)} snapshots and Q_actual.txt\n")
    
    return phi, Q_initial


if __name__ == "__main__":
    print("Starting 1D diffusion simulation with instantaneous source...\n")
    
    mesh, phi, eq, center_source = build_problem(nx=400, dx=1.0, D=1.0)
    
    x_vals = np.array(mesh.cellCenters[0])
    print(f"Domain: x = {x_vals[0]:.1f} to {x_vals[-1]:.1f}")
    print(f"Domain length: {len(x_vals) * 1.0:.1f}")
    print(f"Center position: {x_vals[len(x_vals)//2]:.1f}\n")
    
    phi, Q_actual = run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        total_energy=400.0,
        total_time=100.0,
        snapshot_interval_time=20.0,
        show_animation=True,
        save_animation=False,
        animation_filename='diffusion_1D_instantaneous.mp4'
    )
    
    print("Simulation finished! Run analyze_results.py to view results.")