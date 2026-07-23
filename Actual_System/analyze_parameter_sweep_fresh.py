import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_parameter_sweep_results(csv_path):
    """
    Load sweep results from CSV and create plots.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file with sweep results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out any failed simulations
    df_success = df[df['status'] == 'success'].copy()
    
    if len(df_success) == 0:
        print("No successful simulations found in the CSV file!")
        return
    
    print(f"Loaded {len(df_success)} successful simulations out of {len(df)} total")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Final I2 vs center distance
    ax.plot(df_success['parameter_value'], df_success['final_i2_nM'], 
            marker='o', markersize=8, linewidth=2, color='#0c5da5', 
            label='Final I2 concentration')
    
    # Formatting
    ax.set_xlabel('Center-to-Center Distance (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Final I2 Concentration (nM)', fontsize=14, fontweight='bold')
    ax.set_title('Receiver I2 Response vs Sender-Receiver Distance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Make the plot look nicer
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(csv_path).parent / "final_i2_vs_distance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Parameter swept: {df_success['parameter_name'].iloc[0]}")
    print(f"Number of successful simulations: {len(df_success)}")
    print(f"\nDistance range: {df_success['parameter_value'].min():.1f} - {df_success['parameter_value'].max():.1f} μm")
    print(f"I2 concentration range: {df_success['final_i2_nM'].min():.3f} - {df_success['final_i2_nM'].max():.3f} nM")
    print(f"\nMax I2 at distance: {df_success.loc[df_success['final_i2_nM'].idxmax(), 'parameter_value']:.1f} μm")
    print(f"Min I2 at distance: {df_success.loc[df_success['final_i2_nM'].idxmin(), 'parameter_value']:.1f} μm")
    print(f"\nAverage simulation time: {df_success['simulation_time_s'].mean():.2f} seconds")
    print(f"Total simulation time: {df_success['simulation_time_s'].sum():.2f} seconds")
    print("="*70)
    
    return df_success


# Usage:
if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "Center_center_distance_sweep/sweep_results.csv"
    
    # Create the plot
    df = plot_parameter_sweep_results(csv_file)