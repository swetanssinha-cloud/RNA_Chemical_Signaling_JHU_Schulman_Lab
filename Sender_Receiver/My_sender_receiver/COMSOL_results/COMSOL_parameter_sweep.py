import numpy as np
import csv
import matplotlib.pyplot as plt

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
# MAIN SCRIPT
# =============================================================================

# Define your center-center distance values that correspond to each file
ccd_values = [200,300,500,800,1000,1100,1200,1300,1500]  # Adjust these to match your actual files

# Initialize lists to store the final values
ccd_list = []
I2_final = []
S2_free_final = []
S2_total_final = []

# Loop through each center-center distance value
for ccd in ccd_values:
    # Construct the filename
    filename = f'Single_sender_receiver-{ccd}_um.txt'
    
    try:
        # Load the COMSOL data
        time, I2, S2_free, S2_total = load_comsol_data(filename)
        
        # Get the last time step (final values)
        ccd_list.append(ccd)
        I2_final.append(I2[-1])
        S2_free_final.append(S2_free[-1])
        S2_total_final.append(S2_total[-1])
        
        print(f"Processed {filename}: Final time = {time[-1]:.2f} hr, I2 = {I2[-1]:.4f} nM")
        
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping...")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# =============================================================================
# CREATE PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot I2 vs center-center distance
axes[0].plot(ccd_list, I2_final, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('Center-Center Distance (μm)', fontsize=12)
axes[0].set_ylabel('I2 (nM)', fontsize=12)
axes[0].set_title('Final I2 Concentration vs Center-Center Distance', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot S2_free vs center-center distance
axes[1].plot(ccd_list, S2_free_final, 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Center-Center Distance (μm)', fontsize=12)
axes[1].set_ylabel('S2_free (nM)', fontsize=12)
axes[1].set_title('Final S2_free Concentration vs Center-Center Distance', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Plot S2_total vs center-center distance
axes[2].plot(ccd_list, S2_total_final, 'o-', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('Center-Center Distance (μm)', fontsize=12)
axes[2].set_ylabel('S2_total (nM)', fontsize=12)
axes[2].set_title('Final S2_total Concentration vs Center-Center Distance', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('COMSOL_concentration_vs_ccd.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for i in range(len(ccd_list)):
    print(f"CCD = {ccd_list[i]} μm: I2 = {I2_final[i]:.4f} nM, S2_free = {S2_free_final[i]:.4f} nM, S2_total = {S2_total_final[i]:.4f} nM")
print("="*70)