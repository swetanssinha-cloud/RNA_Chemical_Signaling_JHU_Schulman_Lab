"""
Script to compare COMSOL and Python simulation results for final concentrations
(I2, S2 free, and S2 total). Overlays all three datasets and shows the
difference as a function of center-center distance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# =============================================================================
# LOAD COMSOL DATA
# =============================================================================

print("="*70)
print("LOADING COMSOL DATA")
print("="*70)

# Load the COMSOL data from the saved CSV file
# (This assumes you've already run the COMSOL processing script)
try:
    df_comsol = pd.read_csv('COMSOL_final_concentrations.csv')
    comsol_ccd = df_comsol['CCD_um'].values
    comsol_I2_final = df_comsol['I2_final_nM'].values
    comsol_S2_free_final = df_comsol['S2_free_final_nM'].values
    comsol_S2_total_final = df_comsol['S2_total_final_nM'].values
    print(f"✓ Loaded COMSOL data: {len(comsol_ccd)} data points")
    print(f"  CCD range: {comsol_ccd.min()} - {comsol_ccd.max()} μm")
except FileNotFoundError:
    print("ERROR: 'COMSOL_final_concentrations.csv' not found!")
    print("Please run the COMSOL processing script first.")
    exit()

# =============================================================================
# LOAD PYTHON SIMULATION DATA
# =============================================================================

print("\n" + "="*70)
print("LOADING PYTHON SIMULATION DATA")
print("="*70)
file_name_python = "Parameter_sweep_Improved_meshV4.csv"
# Load the Python simulation data from CSV
try:
    df_python = pd.read_csv(file_name_python, skipinitialspace=True)

    # Extract the relevant columns
    python_ccd = df_python['param_value'].values
    python_I2_final = df_python['I2_center_final_nM'].values
    python_S2_free_final = df_python['S2_free_center_final_nM'].values
    python_S2_total_final = df_python['S2_total_center_final_nM'].values

    print(f"✓ Loaded Python data: {len(python_ccd)} data points")
    print(f"  CCD range: {python_ccd.min()} - {python_ccd.max()} μm")
except FileNotFoundError:
    print("ERROR: 'TOPLOT_center_center_distance_intial_mesh.csv' not found!")
    exit()
except KeyError as e:
    print(f"ERROR: Column not found in CSV: {e}")
    print("Available columns:", df_python.columns.tolist())
    exit()


# =============================================================================
# LOAD OLD PYTHON SIMULATION DATA
# ============================================================================
print("\n" + "="*70)
print("LOADING PYTHON SIMULATION DATA")
print("="*70)
file_name_python_old = "TOPLOT_center_center_distance_intial_mesh.csv"
# Load the Python simulation data from CSV

df_python_old = pd.read_csv(file_name_python_old, skipinitialspace=True)

# Extract the relevant columns
python_ccd_old = df_python_old['param_value'].values
python_I2_final_old = df_python_old['I2_final_mean'].values * 1000
python_S2_free_final_old = df_python_old['S2_final_mean'].values * 1000
python_S2_total_final_old = df_python_old['S2_total_final_mean'].values * 1000

print(f"✓ Loaded Python data: {len(python_ccd)} data points")
print(f"  CCD range: {python_ccd.min()} - {python_ccd.max()} μm")


# =============================================================================
# COMPARISON HELPER
# =============================================================================
# Runs the full common-point comparison, table, plots, and CSV export for a
# single quantity (I2, S2 free, or S2 total), mirroring the original I2-only
# workflow above.

def compare_quantity(label, units, file_tag,
                      comsol_values, python_values, python_values_old):
    print("\n" + "="*70)
    print(f"FINDING COMMON DATA POINTS ({label})")
    print("="*70)

    common_ccd = np.intersect1d(comsol_ccd, python_ccd)
    print(f"Common CCD values: {common_ccd}")

    comsol_common = []
    python_common = []

    for ccd in common_ccd:
        comsol_idx = np.where(comsol_ccd == ccd)[0][0]
        python_idx = np.where(python_ccd == ccd)[0][0]

        comsol_common.append(comsol_values[comsol_idx])
        python_common.append(python_values[python_idx])

    comsol_common = np.array(comsol_common)
    python_common = np.array(python_common)

    # Calculate the difference (COMSOL - Python)
    difference = comsol_common - python_common

    # Calculate relative difference (as percentage)
    average_value = (comsol_common + python_common) / 2
    relative_difference_percent = (difference / average_value) * 100

    # -------------------------------------------------------------------
    # PRINT COMPARISON TABLE
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"COMPARISON TABLE ({label})")
    print("="*70)
    print(f"{'CCD (μm)':<12} {'COMSOL':<15} {'Python':<15} {'Difference':<15} {'Rel. Diff %':<15}")
    print("-"*70)
    for i, ccd in enumerate(common_ccd):
        print(f"{ccd:<12.1f} {comsol_common[i]:<15.6f} {python_common[i]:<15.6f} "
              f"{difference[i]:<15.6f} {relative_difference_percent[i]:<15.2f}")
    print("="*70)

    # -------------------------------------------------------------------
    # CREATE OVERLAY PLOT
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"CREATING OVERLAY PLOT ({label})")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(comsol_ccd, comsol_values, 'o-', linewidth=2, markersize=8,
             color='blue', label='COMSOL', alpha=0.7)
    ax1.plot(python_ccd, python_values, 's-', linewidth=2, markersize=8,
             color='red', label='Python Simulation', alpha=0.7)
    ax1.plot(python_ccd_old, python_values_old, 'd-', linewidth=2, markersize=8,
             color='orange', label='Python Simulation (Old)', alpha=0.7)

    ax1.set_xlabel('Center-Center Distance (μm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'Final {label} Concentration ({units})', fontsize=14, fontweight='bold')
    ax1.set_title(f'Comparison: COMSOL vs Python Simulation\nFinal {label} Concentration',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    ax2.plot(common_ccd, difference, 'o-', linewidth=2, markersize=8,
             color='purple', label='COMSOL - Python')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Center-Center Distance (μm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'Difference in {label} ({units})\n(COMSOL - Python)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Difference Between COMSOL and Python Simulations ({label})',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(f'COMSOL_vs_Python_{file_tag}_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: COMSOL_vs_Python_{file_tag}_comparison.png")
    plt.show()

    # -------------------------------------------------------------------
    # CREATE RELATIVE DIFFERENCE PLOT
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"CREATING RELATIVE DIFFERENCE PLOT ({label})")
    print("="*70)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(common_ccd, relative_difference_percent, 'o-', linewidth=2, markersize=8,
            color='green', label='Relative Difference')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Center-Center Distance (μm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Difference (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Relative Difference: (COMSOL - Python) / Average × 100% ({label})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(f'COMSOL_vs_Python_{file_tag}_relative_difference.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: COMSOL_vs_Python_{file_tag}_relative_difference.png")
    plt.show()

    # -------------------------------------------------------------------
    # SAVE COMPARISON DATA TO CSV
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"SAVING COMPARISON DATA ({label})")
    print("="*70)

    df_comparison = pd.DataFrame({
        'CCD_um': common_ccd,
        f'COMSOL_{file_tag}_final_nM': comsol_common,
        f'Python_{file_tag}_final_nM': python_common,
        'Difference_nM': difference,
        'Relative_Difference_percent': relative_difference_percent
    })

    df_comparison.to_csv(f'COMSOL_vs_Python_comparison_{file_tag}_Improved_mesh_V4.csv', index=False)
    print(f"✓ Saved: COMSOL_vs_Python_comparison_{file_tag}_Improved_mesh_V4.csv")

    # -------------------------------------------------------------------
    # STATISTICS SUMMARY
    # -------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"STATISTICAL SUMMARY ({label})")
    print("="*70)
    print(f"Mean absolute difference: {np.mean(np.abs(difference)):.6f} nM")
    print(f"Max absolute difference: {np.max(np.abs(difference)):.6f} nM (at CCD = {common_ccd[np.argmax(np.abs(difference))]} μm)")
    print(f"Min absolute difference: {np.min(np.abs(difference)):.6f} nM (at CCD = {common_ccd[np.argmin(np.abs(difference))]} μm)")
    print(f"\nMean relative difference: {np.mean(relative_difference_percent):.2f}%")
    print(f"Mean absolute relative difference: {np.mean(np.abs(relative_difference_percent)):.2f}%")
    print(f"Max relative difference: {np.max(np.abs(relative_difference_percent)):.2f}%")
    print("="*70)


# =============================================================================
# RUN COMPARISON FOR I2, S2 FREE, AND S2 TOTAL
# =============================================================================

compare_quantity('I2', 'nM', 'I2',
                  comsol_I2_final, python_I2_final, python_I2_final_old)

compare_quantity('S2 Free', 'nM', 'S2_free',
                  comsol_S2_free_final, python_S2_free_final, python_S2_free_final_old)

compare_quantity('S2 Total', 'nM', 'S2_total',
                  comsol_S2_total_final, python_S2_total_final, python_S2_total_final_old)

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files (for each of I2, S2_free, S2_total):")
print("  - COMSOL_vs_Python_<tag>_comparison.png (overlay + difference)")
print("  - COMSOL_vs_Python_<tag>_relative_difference.png")
print("  - COMSOL_vs_Python_comparison_<tag>_Improved_mesh_V4.csv")
print("="*70)
