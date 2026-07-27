"""
plot_sweep_results.py

Reads CSV output from parameter sweep and creates plots:
1. Final I2 concentration vs center-center distance
2. Final S2 (free) vs center-center distance
3. Final S2 (total) vs center-center distance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
INPUT_CSV = Path("sweep_results/steady_state_values.csv")
OUTPUT_DIR = Path("sweep_results/plots")
PLOT_FORMAT = "png"  # or "pdf", "svg"
DPI = 300

def load_sweep_data(csv_path):
    """
    Load sweep results from CSV file.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file
    
    Returns
    -------
    df : pd.DataFrame
        Loaded data
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def extract_final_values(df):
    """
    Extract final concentration values with proper unit conversion.
    
    Parameters
    ----------
    df : pd.DataFrame
        Sweep results
    
    Returns
    -------
    dict with arrays:
        'parameter_values': swept parameter values (assuming center_distance_um)
        'i2': final I2 concentrations (nM)
        's2_free': final free S2 concentrations (nM)
        's2_total': final total S2 concentrations (nM)
    """
    # Get parameter values (center_distance_um column)
    param_values = df['center_distance_um'].values
    
    # Convert from M to nM
    MOLAR = 1.0
    NANOMOLAR = 1e-9 * MOLAR
    
    i2 = df['final_I2_M'].values / NANOMOLAR
    s2_free = df['final_S2_M'].values / NANOMOLAR
    s2_i2 = df['final_S2_I2_M'].values / NANOMOLAR
    s2_th2 = df['final_S2_Th2_M'].values / NANOMOLAR
    s2_total = s2_free + s2_i2 + s2_th2
    
    return {
        'parameter_values': param_values,
        'i2': i2,
        's2_free': s2_free,
        's2_total': s2_total
    }

def create_plots(data, parameter_name, output_dir):
    """
    Create three plots: I2, S2 (free), and S2 (total) vs swept parameter.
    
    Parameters
    ----------
    data : dict
        Dictionary with 'parameter_values', 'i2', 's2_free', 's2_total' arrays
    parameter_name : str
        Name of swept parameter for axis label
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    param_vals = data['parameter_values']
    
    # Determine x-axis label
    if 'distance' in parameter_name.lower():
        xlabel = 'Center-Center Distance (μm)'
    else:
        xlabel = parameter_name.replace('_', ' ').title()
    
    # Plot 1: Final I2 concentration
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_vals, data['i2'], 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Final I2 Concentration (nM)', fontsize=12)
    ax.set_title('Final I2 vs ' + xlabel, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f'final_i2_vs_{parameter_name}.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Plot 2: Final S2 (free)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_vals, data['s2_free'], 'o-', linewidth=2, markersize=8, color='tab:orange')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Final S2 (Free) Concentration (nM)', fontsize=12)
    ax.set_title('Final Free S2 vs ' + xlabel, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f'final_s2_free_vs_{parameter_name}.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Plot 3: Final S2 (total)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_vals, data['s2_total'], 'o-', linewidth=2, markersize=8, color='tab:green')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Final S2 (Total) Concentration (nM)', fontsize=12)
    ax.set_title('Final Total S2 vs ' + xlabel, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f'final_s2_total_vs_{parameter_name}.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Combined plot (all three on same figure)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    axes[0].plot(param_vals, data['i2'], 'o-', linewidth=2, markersize=8, color='tab:blue')
    axes[0].set_ylabel('I2 (nM)', fontsize=11)
    axes[0].set_title('Final Concentrations vs ' + xlabel, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(param_vals, data['s2_free'], 'o-', linewidth=2, markersize=8, color='tab:orange')
    axes[1].set_ylabel('S2 Free (nM)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(param_vals, data['s2_total'], 'o-', linewidth=2, markersize=8, color='tab:green')
    axes[2].set_xlabel(xlabel, fontsize=12)
    axes[2].set_ylabel('S2 Total (nM)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'final_concentrations_vs_{parameter_name}.{PLOT_FORMAT}'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Main plotting routine."""
    print("=" * 60)
    print("Parameter Sweep Plotting Script")
    print("=" * 60)
    
    # Load data
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    
    df = load_sweep_data(INPUT_CSV)
    
    # Check convergence status
    if 'status' in df.columns:
        failed = df[df['status'] != 'converged']
        if len(failed) > 0:
            print(f"\nWarning: {len(failed)} simulations did not converge:")
            print(failed[['center_distance_um', 'status']])
            print("\nProceeding with converged simulations only...")
            df = df[df['status'] == 'converged']
    
    if len(df) == 0:
        raise ValueError("No converged simulations to plot!")
    
    # Extract parameter name
    parameter_name = 'center_distance_um'
    print(f"\nSwept parameter: {parameter_name}")
    print(f"Number of values: {len(df)}")
    print(f"Range: {df[parameter_name].min():.2f} to {df[parameter_name].max():.2f}")
    
    # Extract final values
    data = extract_final_values(df)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"I2 concentration:")
    print(f"  Min:  {data['i2'].min():.3f} nM")
    print(f"  Max:  {data['i2'].max():.3f} nM")
    print(f"  Mean: {data['i2'].mean():.3f} nM")
    print(f"\nS2 (free) concentration:")
    print(f"  Min:  {data['s2_free'].min():.3f} nM")
    print(f"  Max:  {data['s2_free'].max():.3f} nM")
    print(f"  Mean: {data['s2_free'].mean():.3f} nM")
    print(f"\nS2 (total) concentration:")
    print(f"  Min:  {data['s2_total'].min():.3f} nM")
    print(f"  Max:  {data['s2_total'].max():.3f} nM")
    print(f"  Mean: {data['s2_total'].mean():.3f} nM")
    
    # Create plots
    print("\n" + "=" * 60)
    print("Creating Plots")
    print("=" * 60)
    create_plots(data, parameter_name, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Plotting Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()