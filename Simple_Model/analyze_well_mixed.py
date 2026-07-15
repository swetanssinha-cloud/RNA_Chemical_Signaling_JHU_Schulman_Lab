import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_DIR = "well_mixed_results"

summary_file = os.path.join(INPUT_DIR, "summary.csv")
summary = pd.read_csv(summary_file)

# Plot time courses
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))

for _, row in summary.iterrows():
    Phi_in_nMps = row["Phi_in_nMps"]
    csv_file = row["csv_file"]

    df = pd.read_csv(csv_file)

    time_h = df["time_s"] / 3600
    S2_nM = df["S2_M"] * 1e9
    S2_tot_nM = df["S2_tot_M"] * 1e9
    I2_nM = df["I2_M"] * 1e9

    label = f"Φ_in = {Phi_in_nMps:.2f} nM/s"

    ax1.plot(time_h, S2_nM, label=label, linewidth=2)
    ax2.plot(time_h, S2_tot_nM, label=label, linewidth=2)
    ax3.plot(time_h, I2_nM, label=label, linewidth=2)

ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("[S2 free] (nM)")
ax1.set_title("Free Signal S2 vs Time")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, ncol=2)

ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("[S2 total] (nM)")
ax2.set_title("Total Signal (S2 + C_I2 + C_Th2) vs Time")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, ncol=2)

ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("[I2] (nM)")
ax3.set_title("Receiver I2 vs Time")
ax3.grid(True, alpha=0.3)
ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50 nM threshold')
ax3.legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.show()

# Plot tw50 vs Phi_in
plt.figure(figsize=(10, 6))
valid = ~np.isnan(summary["tw50_s"])

plt.plot(
    summary.loc[valid, "Phi_in_nMps"],
    summary.loc[valid, "tw50_s"] / 3600,
    'o-',
    linewidth=2,
    markersize=8
)

plt.xlabel("Input Flux Φ_in (nM/s)")
plt.ylabel("Time to [I2] = 50 nM (hours)")
plt.title("Time for I2 to Drop to 50 nM vs Input Flux")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print final values
print("\nFinal values:")
for _, row in summary.iterrows():
    tw50_str = f"{row['tw50_s']/3600:.2f} hrs" if not np.isnan(row["tw50_s"]) else "Never"
    print(
        f"Φ_in = {row['Phi_in_nMps']:.2f} nM/s → "
        f"[S2 free] = {row['final_S2_nM']:.1f} nM, "
        f"[S2 total] = {row['final_S2_tot_nM']:.1f} nM, "
        f"[I2] = {row['final_I2_nM']:.1f} nM, "
        f"tw50 = {tw50_str}"
    )