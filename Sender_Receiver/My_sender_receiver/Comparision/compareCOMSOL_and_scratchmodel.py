"""
Compare COMSOL and FiPy simulation results
Overlays plots and calculates differences between the two models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy import interpolate

Value = 300
fine_dx = 1

COMSOL_file = f'Single_sender_receiver-{Value}_um.txt'

python_file = f'timeseries_for_comparision_ccd={Value}_triangular_mesh_dx={fine_dx}.csv'

overlay_save_file_name = f'comsol_vs_python_ccd={Value}um_overlay_Triangular_mesh_dx={fine_dx}.png'

difference_file_name = f'comsol_vs_python_ccd={Value}um_differences_triangular_mesh_dx={fine_dx}.png'

csv_file_name = f'comsol_vs_python_differences_ccd={Value}um_data.csv'
# =============================================================================
# LOAD COMSOL DATA (from TXT file)
# =============================================================================

def load_comsol_data(fileName):
    """
    Load COMSOL data from text file
    Returns time (hours), I2 (nM), S2_free (nM), S2_total (nM)
    """
    t = []
    p = [[], [], []]  # [rS6, rTh, rGRep6]
    kk = 0
    header = 5
    
    with open(fileName) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            kk += 1
            jj = 0
            if kk > header:
                for ii in row:
                    if str(ii) != "":
                        jj += 1
                        if jj == 1:
                            t.append(float(ii))
                        elif jj > 1 and jj < 5:  # numRows + 2
                            p[jj-2].append(float(ii))
    
    # Convert time from minutes to hours
    t = np.array(t) / 60
    p = np.array(p)
    
    # Convert from mol/m³ to nM (multiply by 1e6)
    p = p * 1e6
    
    # Calculate derived quantities
    S2_free = p[0]  # rS6
    Th2_bound = p[1]  # rTh (bound threshold)
    reporter = p[2]  # rGRep6
    
    # Calculate I2 = Initial reporter - Current reporter
    initial_reporter = 100
    I2 = initial_reporter - reporter
    
    # Calculate total S2 = rS6 + rTh + rGRep6
    S2_total = p[0] + p[1] + p[2]
    
    return t, I2, S2_free, S2_total

# =============================================================================
# LOAD FIPY DATA (from CSV file)
# =============================================================================

def load_fipy_data(fileName):
    """
    Load FiPy simulation data from CSV file
    Returns time (hours), I2 (nM), S2_free (nM), S2_total (nM)
    """
    df = pd.read_csv(fileName)
    
    # Updated column names to match the actual CSV format
    time = df['Time (hours)'].values
    I2 = df['I2 (nM)'].values
    S2_free = df['S2_free (nM)'].values
    S2_total = df['S2_total (nM)'].values
    
    return time, I2, S2_free, S2_total

# =============================================================================
# INTERPOLATION FOR DIFFERENCE CALCULATION
# =============================================================================

def interpolate_to_common_time(t1, data1, t2, data2):
    """
    Interpolate both datasets to a common time grid
    Returns common_time, data1_interp, data2_interp
    """
    # Use the time grid with more points
    if len(t1) > len(t2):
        common_time = t1
        data2_interp = np.interp(common_time, t2, data2)
        data1_interp = data1
    else:
        common_time = t2
        data1_interp = np.interp(common_time, t1, data1)
        data2_interp = data2
    
    return common_time, data1_interp, data2_interp

# =============================================================================
# MAIN COMPARISON
# =============================================================================

# Load data
print("Loading COMSOL data...")
comsol_time, comsol_I2, comsol_S2_free, comsol_S2_total = load_comsol_data(COMSOL_file)

print("Loading FiPy data...")
fipy_time, fipy_I2, fipy_S2_free, fipy_S2_total = load_fipy_data(python_file)

print(f"COMSOL: {len(comsol_time)} points, time range: {comsol_time[0]:.2f} - {comsol_time[-1]:.2f} hr")
print(f"FiPy: {len(fipy_time)} points, time range: {fipy_time[0]:.2f} - {fipy_time[-1]:.2f} hr")

# =============================================================================
# OVERLAY PLOTS
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: I2 concentration
axes[0].plot(comsol_time, comsol_I2, 'b-', linewidth=2, label='COMSOL', alpha=0.7)
axes[0].plot(fipy_time, fipy_I2, 'r--', linewidth=2, label='FiPy', alpha=0.7)
axes[0].axhline(y=75, color='g', linestyle=':', alpha=0.5, linewidth=1, label='75% ON')
axes[0].axhline(y=25, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='25% OFF')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title(f'I2 Concentration Comparison (distance = {Value} μm)', fontsize=14, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

# Plot 2: Free S2 concentration
axes[1].plot(comsol_time, comsol_S2_free, 'b-', linewidth=2, label='COMSOL', alpha=0.7)
axes[1].plot(fipy_time, fipy_S2_free, 'r--', linewidth=2, label='FiPy', alpha=0.7)
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('Free S2 Concentration Comparison', fontsize=14, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

# Plot 3: Total S2 concentration
axes[2].plot(comsol_time, comsol_S2_total, 'b-', linewidth=2, label='COMSOL', alpha=0.7)
axes[2].plot(fipy_time, fipy_S2_total, 'r--', linewidth=2, label='FiPy', alpha=0.7)
axes[2].set_xlabel('Time (hours)', fontsize=12)
axes[2].set_ylabel('[S2] total (nM)', fontsize=12)
axes[2].set_title('Total S2 Concentration Comparison', fontsize=14, fontweight='bold')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(overlay_save_file_name, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# DIFFERENCE PLOTS
# =============================================================================

# Interpolate to common time grid
print("\nCalculating differences...")
time_I2, comsol_I2_interp, fipy_I2_interp = interpolate_to_common_time(
    comsol_time, comsol_I2, fipy_time, fipy_I2)

time_S2_free, comsol_S2_free_interp, fipy_S2_free_interp = interpolate_to_common_time(
    comsol_time, comsol_S2_free, fipy_time, fipy_S2_free)

time_S2_total, comsol_S2_total_interp, fipy_S2_total_interp = interpolate_to_common_time(
    comsol_time, comsol_S2_total, fipy_time, fipy_S2_total)

# Calculate differences
diff_I2 = fipy_I2_interp - comsol_I2_interp
diff_S2_free = fipy_S2_free_interp - comsol_S2_free_interp
diff_S2_total = fipy_S2_total_interp - comsol_S2_total_interp

# Calculate percent differences
percent_diff_I2 = 100 * diff_I2 / 100  # Avoid division by zero
percent_diff_S2_free = 100 * diff_S2_free / 100 #not sure if I should divide by anything or not for free S2
percent_diff_S2_total = 100 * diff_S2_total / 5100 #threshold + receiver molecule total

# Plot differences
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Absolute differences (left column)
axes[0, 0].plot(time_I2, diff_I2, 'purple', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].set_xlabel('Time (hours)', fontsize=11)
axes[0, 0].set_ylabel('Δ[I2] (nM)', fontsize=11)
axes[0, 0].set_title('Absolute Difference: I2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(time_S2_free, diff_S2_free, 'purple', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Time (hours)', fontsize=11)
axes[1, 0].set_ylabel('Δ[S2] free (nM)', fontsize=11)
axes[1, 0].set_title('Absolute Difference: Free S2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(time_S2_total, diff_S2_total, 'purple', linewidth=2)
axes[2, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[2, 0].set_xlabel('Time (hours)', fontsize=11)
axes[2, 0].set_ylabel('Δ[S2] total (nM)', fontsize=11)
axes[2, 0].set_title('Absolute Difference: Total S2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

# Percent differences (right column)
axes[0, 1].plot(time_I2, percent_diff_I2, 'darkorange', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].set_xlabel('Time (hours)', fontsize=11)
axes[0, 1].set_ylabel('% Difference', fontsize=11)
axes[0, 1].set_title('Percent Difference: I2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(time_S2_free, percent_diff_S2_free, 'darkorange', linewidth=2)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Time (hours)', fontsize=11)
axes[1, 1].set_ylabel('% Difference', fontsize=11)
axes[1, 1].set_title('Percent Difference: Free S2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(time_S2_total, percent_diff_S2_total, 'darkorange', linewidth=2)
axes[2, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[2, 1].set_xlabel('Time (hours)', fontsize=11)
axes[2, 1].set_ylabel('% Difference', fontsize=11)
axes[2, 1].set_title('Percent Difference: Total S2\n(FiPy - COMSOL)', fontsize=12, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(difference_file_name, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STATISTICS SUMMARY
# =============================================================================

print("\n" + "="*70)
print("COMPARISON STATISTICS")
print("="*70)

print("\n[I2] Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_I2)):.4f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_I2)):.4f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_I2**2)):.4f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(percent_diff_I2)):.2f}%")
print(f"  Max percent difference:   {np.max(np.abs(percent_diff_I2)):.2f}%")

print("\n[S2] Free Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_S2_free)):.4f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_S2_free)):.4f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_S2_free**2)):.4f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(percent_diff_S2_free)):.2f}%")
print(f"  Max percent difference:   {np.max(np.abs(percent_diff_S2_free)):.2f}%")

print("\n[S2] Total Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_S2_total)):.4f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_S2_total)):.4f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_S2_total**2)):.4f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(percent_diff_S2_total)):.2f}%")
print(f"  Max percent difference:   {np.max(np.abs(percent_diff_S2_total)):.2f}%")

print("\n" + "="*70)

# =============================================================================
# SAVE DIFFERENCE DATA TO CSV
# =============================================================================

# Save differences to CSV
df_diff = pd.DataFrame({
    'Time_hours': time_I2,
    'I2_diff_nM': diff_I2,
    'I2_percent_diff': percent_diff_I2,
    'S2_free_diff_nM': diff_S2_free[:len(time_I2)],
    'S2_free_percent_diff': percent_diff_S2_free[:len(time_I2)],
    'S2_total_diff_nM': diff_S2_total[:len(time_I2)],
    'S2_total_percent_diff': percent_diff_S2_total[:len(time_I2)],
    'COMSOL_I2_nM': comsol_I2_interp,
    'FiPy_I2_nM': fipy_I2_interp
})

df_diff.to_csv(csv_file_name, index=False)
print(f"\nDifference data saved to: {csv_file_name}")

print("\nAnalysis complete!")

