import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define your dx values that correspond to each file
dx_values = [0.8,1.0,3.0,5.0]  # Adjust these to match your actual dx values

# Initialize lists to store the final values
dx_list = []
I2_final = []
S2_free_final = []
S2_total_final = []

# Loop through each dx value
for dx in dx_values:
    # Construct the filename
    filename = f'timeseries_dx={dx}_ccd=300_dt=30.csv'
    
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Get the last row (final time step)
        last_row = df.iloc[-1]
        
        # Extract the values
        dx_list.append(dx)
        I2_final.append(last_row['I2 (μM)'])
        S2_free_final.append(last_row['S2_free (μM)'])
        S2_total_final.append(last_row['S2_total (μM)'])
        
        print(f"Processed {filename}: Time = {last_row['Time (hr)']} hr")
        
    except FileNotFoundError:
        print(f"File {filename} not found. Skipping...")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Create the plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot I2 vs dx
axes[0].plot(dx_list, I2_final, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('dx', fontsize=12)
axes[0].set_ylabel('I2 (μM)', fontsize=12)
axes[0].set_title('Final I2 Concentration vs dx', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot S2_free vs dx
axes[1].plot(dx_list, S2_free_final, 'o-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('dx', fontsize=12)
axes[1].set_ylabel('S2_free (μM)', fontsize=12)
axes[1].set_title('Final S2_free Concentration vs dx', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Plot S2_total vs dx
axes[2].plot(dx_list, S2_total_final, 'o-', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('dx', fontsize=12)
axes[2].set_ylabel('S2_total (μM)', fontsize=12)
axes[2].set_title('Final S2_total Concentration vs dx', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('concentration_vs_dx_zoomed_in.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
print("\n--- Summary ---")
for i in range(len(dx_list)):
    print(f"dx = {dx_list[i]}: I2 = {I2_final[i]:.6f}, S2_free = {S2_free_final[i]:.6f}, S2_total = {S2_total_final[i]:.6f}")