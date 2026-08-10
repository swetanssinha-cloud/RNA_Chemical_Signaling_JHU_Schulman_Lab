import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_DIR = "well_mixed_results"


def find_crossing_time(df, target_value, time_col="time_s", value_col="I2_M"):
    """
    Find the first time where value_col drops to target_value.
    Uses linear interpolation between adjacent points.
    """
    t = df[time_col].to_numpy()
    y = df[value_col].to_numpy()

    below = np.where(y <= target_value)[0]
    if len(below) == 0:
        return np.nan

    idx = below[0]

    if idx == 0:
        return t[0]

    t1, t2 = t[idx - 1], t[idx]
    y1, y2 = y[idx - 1], y[idx]

    if y2 == y1:
        return t2

    return t1 + (target_value - y1) * (t2 - t1) / (y2 - y1)


# Load summary file
summary_file = os.path.join(INPUT_DIR, "summary.csv")
summary = pd.read_csv(summary_file)

# Storage for computed times
tw50_initial_list = []
tw50_offfraction_list = []

# Loop over each run
for _, row in summary.iterrows():
    Phi_in_nMps = row["Phi_in_nMps"]
    csv_file = row["csv_file"]

    df = pd.read_csv(csv_file)

    # Initial and final I2 from the CSV
    I2_initial = df["I2_M"].iloc[0]
    I2_final = df["I2_M"].iloc[-1]

    # 1) 50% of initial I2
    target_initial_50 = 0.5 * I2_initial

    # 2) 50% of the off-fraction range
    #    = final + 0.5*(initial - final)
    target_off_50 = I2_final + 0.5 * (I2_initial - I2_final)

    # Find crossing times
    tw50_initial = find_crossing_time(df, target_initial_50)
    tw50_off = find_crossing_time(df, target_off_50)

    tw50_initial_list.append(tw50_initial)
    tw50_offfraction_list.append(tw50_off)

    print(
        f"Φ_in = {Phi_in_nMps:.2f} nM/s | "
        f"I2_0 = {I2_initial*1e9:.2f} nM | "
        f"I2_final = {I2_final*1e9:.2f} nM | "
        f"t_50% initial = {tw50_initial/3600:.3f} h | "
        f"t_50% off-range = {tw50_off/3600:.3f} h"
    )

# Add results to summary dataframe
summary["t50_initial_s"] = tw50_initial_list
summary["t50_offrange_s"] = tw50_offfraction_list

# Save updated summary if you want
summary.to_csv(os.path.join(INPUT_DIR, "summary_with_thresholds.csv"), index=False)


'''
Plot of phi vs time where 50% of I2 is used up
'''


# plt.figure(figsize=(10, 6))
# valid = ~np.isnan(summary["t50_initial_s"])

# plt.plot(
#     summary.loc[valid, "Phi_in_nMps"],
#     summary.loc[valid, "t50_initial_s"] / 3600,
#     "o-",
#     linewidth=2,
#     markersize=8
# )

# plt.xlabel("Input Flux Φ_in (nM/s)")
# plt.ylabel("Time to 50% of initial [I2] (hours)")
# plt.title("Time for [I2] to drop to 50% of its initial value")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# '''
# Plot of phi vs time it takes for [I2] to 
# '''

# plt.figure(figsize=(10, 6))
# valid = ~np.isnan(summary["t50_offrange_s"])

# plt.plot(
#     summary.loc[valid, "Phi_in_nMps"],
#     summary.loc[valid, "t50_offrange_s"] / 3600,
#     "o-",
#     linewidth=2,
#     markersize=8,
#     color="green"
# )

# plt.xlabel("Input Flux Φ_in (nM/s)")
# plt.ylabel("Time to 50% of off-fraction [I2] (hours)")
# plt.title("Time for [I2] to drop to midpoint between initial and final values")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # '''plot of t50 * phi in vs phi in'''
# # plt.figure(figsize=(10, 6))
# # valid = ~np.isnan(summary["t50_initial_s"])

# # plt.plot(summary["Phi_in_nMps"], summary["Phi_in_nMps"] * summary["t50_initial_s"], "o-", linewidth = 2, markersize = 8)

# # plt.xlabel("Input Flux Φ_in (nM/s)")
# # plt.ylabel("Time to 50% of initial [I2] (hours) * Input Flux")
# # plt.title("Time for [I2] to drop to 50% of its initial value * input flux - to determine proprotionality")
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.show()


# '''
# Plot of I2
# '''
# fig, ax = plt.subplots(figsize=(10, 6))

# for _, row in summary.iterrows():
#     Phi_in_nMps = row["Phi_in_nMps"]
#     df = pd.read_csv(row["csv_file"])

#     t_h = df["time_s"] / 3600
#     I2_nM = df["I2_M"] * 1e9

#     ax.plot(t_h, I2_nM, label=f"Φ_in = {Phi_in_nMps:.2f} nM/s")

# ax.set_xlabel("Time (hours)")
# ax.set_ylabel("[I2] (nM)")
# ax.set_title("I2 vs time for all input fluxes")
# ax.grid(True, alpha=0.3)
# ax.legend(fontsize=8, ncol=2)
# plt.tight_layout()
# plt.show()

# '''
# Plot of final [I2] vs Phi_in
# '''
# plt.figure(figsize=(10, 6))

# plt.plot(
#     summary["Phi_in_nMps"],
#     summary["final_I2_nM"],
#     "o-",
#     linewidth=2,
#     markersize=8,
#     color="purple"
# )

# plt.xlabel("Input Flux Φ_in (nM/s)")
# plt.ylabel("Final [I2] (nM)")
# plt.title("Steady-state [I2] vs Input Flux")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



'''
Plot of simulation time to convergence vs Phi_in
'''



# Add this to your analysis code after the tw50 calculation

# Time for I2 to drop to 1% of initial (99% depleted)
tw99_list = []

for _, row in summary.iterrows():
    csv_file = row["csv_file"]
    df = pd.read_csv(csv_file)
    
    I2_initial = df["I2_M"].iloc[0]
    target_1percent = 0.01 * I2_initial
    
    tw99 = find_crossing_time(df, target_1percent)
    tw99_list.append(tw99)

summary["t99_s"] = tw99_list

# Then plot both
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(summary["Phi_in_nMps"], summary["t50_initial_s"]/3600, 
        "o-", label="Time to 50%", linewidth=2, markersize=8)
ax.plot(summary["Phi_in_nMps"], summary["t99_s"]/3600, 
        "s-", label="Time to 99% depletion", linewidth=2, markersize=8)

# Add convergence time from steady state file
ss_data = pd.read_csv(os.path.join(INPUT_DIR, "steady_state_values.csv"))
converged_mask = ss_data["converged"] == True
ax.plot(ss_data.loc[converged_mask, "Phi_in_nMps"], 
        ss_data.loc[converged_mask, "sim_time_to_ss_h"],
        "^-", label="Time to convergence", linewidth=2, markersize=8)

ax.set_xlabel("Input Flux Φ_in (nM/s)")
ax.set_ylabel("Time (hours)")
ax.set_title("Comparison of different time metrics")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()