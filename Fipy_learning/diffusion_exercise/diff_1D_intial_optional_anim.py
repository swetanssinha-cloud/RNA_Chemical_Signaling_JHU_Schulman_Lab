from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

'''
Run 1D diffusion simulation and save results to CSV files.
This can run in the background without displaying plots.
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
                   source_value=1.0, source_duration=1.0, total_time=10.0,
                   snapshot_interval_time=20.0, show_animation=False, save_animation=False, animation_filename = 'diffusion_animation.mp4'):
    """Run transient diffusion and save snapshots."""
    
    dt = 0.9 * (mesh.dx ** 2) / (2.0 * D)
    nsteps = int(total_time / dt)
    
    snapshot_times = np.arange(0, total_time + snapshot_interval_time, snapshot_interval_time)
    snapshot_index = 0
    next_snapshot_time = snapshot_times[snapshot_index]
    
    # Clean up old CSV files
    for f in glob.glob('phi_values_time_*.csv'):
        os.remove(f)
    
    x_vals = mesh.cellCenters[0]

    if show_animation or save_animation:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        line, = ax1.plot(x_vals, phi.value, marker='o')
        ax1.set_xlabel("x")
        ax1.set_ylabel("phi")
        ax1.set_title("phi vs x (1D Diffusion)")
        ax1.set_ylim(0.0, source_value * 1.05)
        ax1.grid(True)
        
        phi_2d = phi.value.reshape(1, -1)
        im = ax2.imshow(phi_2d, aspect='auto', cmap='hot', 
                        vmin=0.0, vmax=source_value,
                        extent=[x_vals[0]-mesh.dx/2, x_vals[-1]+mesh.dx/2, -0.5, 0.5])
        ax2.set_xlabel("x")
        ax2.set_yticks([])
        ax2.set_title("1D Heat Diffusion (Heatmap)")
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("phi (temperature)")
        plt.tight_layout()
        
        if show_animation:
            plt.ion()
        
        # For saving: store frames
        frames = []
    
    time = 0.0
    Q_actual = None
    
    for step in range(nsteps):
        if time < source_duration:
            phi.setValue(source_value, where=center_source)
        
        eq.solve(var=phi, dt=dt)
        
        if Q_actual is None and time >= source_duration:
            Q_actual = np.trapezoid(phi.value, x_vals)
            print(f"Total energy at t={time:.2f}s: Q = {Q_actual}")
        
        # Save snapshot
        if time >= next_snapshot_time and snapshot_index < len(snapshot_times):
            df = pd.DataFrame({
                'x': x_vals,
                'phi': phi.value
            })
            df.to_csv(f'phi_values_time_{next_snapshot_time:.2f}s.csv', index=False)
            print(f"Saved snapshot at t = {time:.2f}s (target: {next_snapshot_time:.2f}s)")
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
                # Store frame data
                frames.append((time, phi.value.copy()))
            
            if show_animation:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
        
        time += dt
    
    # Save animation to file
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
        
        # Choose every Nth frame to speed up
        frame_skip = max(1, len(frames) // 500)  # Max 500 frames in video
        frame_indices = range(0, len(frames), frame_skip)
        
        anim = FuncAnimation(fig, update_frame, frames=frame_indices, 
                           interval=50, blit=False)
        
        # Try MP4 first (requires ffmpeg), fall back to GIF
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
    
    # # Optional animation setup
    # if show_animation or save_animation:
    #     plt.ion()
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
    #     line, = ax1.plot(x_vals, phi.value, marker='o')
    #     ax1.set_xlabel("x")
    #     ax1.set_ylabel("phi")
    #     ax1.set_title("phi vs x (1D Diffusion)")
    #     ax1.set_ylim(0.0, source_value * 1.05)
    #     ax1.grid(True)
        
    #     phi_2d = phi.value.reshape(1, -1)
    #     im = ax2.imshow(phi_2d, aspect='auto', cmap='hot', 
    #                     vmin=0.0, vmax=source_value,
    #                     extent=[x_vals[0]-mesh.dx/2, x_vals[-1]+mesh.dx/2, -0.5, 0.5])
    #     ax2.set_xlabel("x")
    #     ax2.set_yticks([])
    #     ax2.set_title("1D Heat Diffusion (Heatmap)")
    #     cbar = plt.colorbar(im, ax=ax2)
    #     cbar.set_label("phi (temperature)")
    #     plt.tight_layout()
    
    # time = 0.0
    # Q_actual = None
    
    # for step in range(nsteps):
    #     if time < source_duration:
    #         phi.setValue(source_value, where=center_source)
        
    #     eq.solve(var=phi, dt=dt)
        
    #     # Calculate Q right after source turns off
    #     if Q_actual is None and time >= source_duration:
    #         Q_actual = np.trapezoid(phi.value, x_vals)
    #         print(f"Total energy at t={time:.2f}s: Q = {Q_actual}")
        
    #     # Save snapshot
    #     if time >= next_snapshot_time and snapshot_index < len(snapshot_times):
    #         df = pd.DataFrame({
    #             'x': x_vals,
    #             'phi': phi.value
    #         })
    #         df.to_csv(f'phi_values_time_{next_snapshot_time:.2f}s.csv', index=False)
    #         print(f"Saved snapshot at t = {time:.2f}s (target: {next_snapshot_time:.2f}s)")
    #         snapshot_index += 1
    #         if snapshot_index < len(snapshot_times):
    #             next_snapshot_time = snapshot_times[snapshot_index]
        
    #     # Update animation if showing
    #     if show_animation:
    #         line.set_ydata(phi.value)
    #         ax1.set_title(f"phi vs x (t = {time:.2f}s)")
    #         phi_2d = phi.value.reshape(1, -1)
    #         im.set_data(phi_2d)
    #         ax2.set_title(f"1D Heat Diffusion Heatmap (t = {time:.2f}s)")
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()
    #         plt.pause(0.001)
        
    #     time += dt
    
    # if show_animation:
    #     plt.ioff()
    
    # Save Q_actual to a file


    with open('Q_actual.txt', 'w') as f:
        f.write(str(Q_actual))
    
    print(f"\nSimulation complete! Q_actual = {Q_actual:.2f}")
    print(f"Saved {len(snapshot_times)} snapshots and Q_actual.txt\n")
    
    return phi, Q_actual


if __name__ == "__main__":
    print("Starting simulation...")
    
    mesh, phi, eq, center_source = build_problem(nx=401, dx=1.0, D=1.0)
    
    x_vals = mesh.cellCenters[0]
    print(f"Domain spans: {x_vals[0]} to {x_vals[-1]}")
    print(f"Middle cell index: {len(x_vals)//2}")
    print(f"Middle cell position: {x_vals[len(x_vals)//2]}")
    print(f"Source applied at cells where x is between {200-0.5} and {200+0.5}")

    
    phi, Q_actual = run_simulation(
        mesh, phi, eq, center_source,
        D=1.0,
        source_value=400.0,
        source_duration=1.0,
        total_time=100.0,
        snapshot_interval_time=20.0,
        show_animation=True, save_animation=True, animation_filename='diffusion_1D.mp4' # Set to True to watch simulation
    )
    
    print("Simulation finished! Run analyze_results.py to view results.")