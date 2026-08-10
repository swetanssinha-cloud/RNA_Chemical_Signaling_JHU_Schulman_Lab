import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

'''
Analyze saved simulation results and compare with analytical solution.
Run this after run_simulation.py completes.
'''

def analytical_solution_gaussian(x, t, x0, D, Q):
    """Analytical solution for point source in infinite domain"""
    if t <= 0:
        return np.zeros_like(x)
    return (Q / np.sqrt(4 * np.pi * D * t)) * np.exp(-(x - x0)**2 / (4 * D * t))


def load_Q_actual():
    """Load Q_actual from file"""
    if not os.path.exists('Q_actual.txt'):
        raise FileNotFoundError("Q_actual.txt not found! Run run_simulation.py first.")
    with open('Q_actual.txt', 'r') as f:
        return float(f.read().strip())


def analyze_and_plot():
    """Load CSV files, calculate differences, and plot results"""
    
    # Load Q_actual
    Q_actual = load_Q_actual()
    print(f"Loaded Q_actual = {Q_actual:.2f}")
    
    # Find all CSV files
    csv_files = sorted(glob.glob('phi_values_time_*.csv'))
    
    if not csv_files:
        print("No CSV files found! Run run_simulation.py first.")
        return
    
    print(f"Found {len(csv_files)} snapshot files\n")
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # ==================== Calculate differences ====================
    times_list = []
    mean_differences = []
    mean_abs_differences = []
    max_abs_differences = []
    max_differences = []

    for file in csv_files:
        df = pd.read_csv(file)
        x_sim = df["x"].values
        phi_sim = df["phi"].values
        
        # Extract time from filename
        time_str = file.replace('phi_values_time_', '').replace('.csv', '')
        time_val = float(time_str.replace('s', ''))
        
        # Skip t=0
        if time_val <= 0:
            ax1.plot(x_sim, phi_sim, label=f'Simulation t = {time_str}', 
                   marker='o', markersize=3, alpha=0.7)
            continue
        
        # Calculate analytical solution at simulation x-points
        phi_analytical_at_sim_points = analytical_solution_gaussian(
            x_sim, time_val, 200.5, 1.0, Q_actual
        )
        
        # Calculate differences
        differences = phi_sim - phi_analytical_at_sim_points
        mean_diff = np.mean(differences)
        mean_abs_diff = np.mean(np.abs(differences))
        max_abs_diff = np.max(np.abs(differences))
        max_diff = np.max(differences)

        times_list.append(time_val)
        mean_differences.append(mean_diff)
        mean_abs_differences.append(mean_abs_diff)
        max_abs_differences.append(max_abs_diff)
        max_differences.append(max_diff)
        
        print(f"Time {time_val:5.2f}s: Mean diff = {mean_diff:8.4f}, "
              f"Mean |diff| = {mean_abs_diff:8.4f}, Max |diff| = {max_abs_diff:8.4f}")
        
        # Plot simulation
        ax1.plot(x_sim, phi_sim, label=f'Simulation t = {time_str}', 
               marker='o', markersize=3, alpha=0.7)
        
        # Plot difference profile at this time
        ax3.plot(x_sim, differences, label=f't = {time_val:.0f}s', alpha=0.7)
    
    # ==================== Plot 1: Comparison ====================
    x_analytical = np.linspace(0, 400, 1000)
    
    for time_val in times_list:
        if time_val > 0:
            phi_analytical = analytical_solution_gaussian(x_analytical, time_val, 200.5, 1.0, Q_actual)
            ax1.plot(x_analytical, phi_analytical, '--', 
                   label=f'Analytical t = {time_val:.0f}s', alpha=0.7, linewidth=2)
    
    ax1.set_xlabel("x (position)")
    ax1.set_ylabel("phi (temperature)")
    ax1.set_title("Numerical Simulation vs Analytical Solution")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True)
    
    # ==================== Plot 2: Mean differences vs time ====================
    ax2.plot(times_list, mean_differences, 'o-', linewidth=2, markersize=8, 
            label='Mean Difference (signed)', color='blue')
    ax2.plot(times_list, mean_abs_differences, 's-', linewidth=2, markersize=8,
            label='Mean Absolute Difference', color='red')
    ax2.plot(times_list, max_abs_differences, '^-', linewidth=2, markersize=8,
            label='Max Absolute Difference', color='orange')
    ax2.plot(times_list, max_differences, 'x-', linewidth=2, markersize=8,
            label='Max Difference', color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Difference in phi")
    ax2.set_title("Error Metrics: Simulation - Analytical Solution")
    ax2.legend()
    ax2.grid(True)
    
    # ==================== Plot 3: Difference profiles ====================
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel("x (position)")
    ax3.set_ylabel("Difference (Simulation - Analytical)")
    ax3.set_title("Spatial Distribution of Differences at Each Time")
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    print("Analyzing simulation results...\n")
    analyze_and_plot()