import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('center_center_distance_sweep_Rmesh.csv')
file_name = '1cmx1mm_parameter_sweep_ccd.png'

df = pd.read_csv('Chen25_sweep_results_distance_between_final.csv')
file_name = 'Chen25_parameter_sweep_ccd.png'

center_center_distance = df["param_value"]
I2_final = df["I2_final_mean"]
S2_final = df["S2_final_mean"]
S2_total_final = df["S2_total_final_mean"]
half_mean_time = df["half_time_mean"]


fig, axes = plt.subplots(4, 1, figsize=(12, 10))


axes[0].plot(center_center_distance, I2_final, marker='o', label='Concetration ON')
axes[0].axhline(y=0.025, color="red", linestyle = '--')
axes[0].set_xlabel('Center-to-Center Distance (um)')
axes[0].set_ylabel('Concentration ON')
axes[0].set_title('I2_final_mean vs Center-to-Center Distance')
axes[0].grid(True, alpha=0.3)

axes[1].plot(center_center_distance, S2_final, marker='o', label='Free S2', color='orange')   
axes[1].set_xlabel('Center-to-Center Distance (um)')
axes[1].set_ylabel('S2_free_final')
axes[1].set_title('S2_free_final vs Center-to-Center Distance')
axes[1].grid(True, alpha=0.3)

axes[2].plot(center_center_distance, S2_total_final, marker = 'o', label='Total S2')
axes[2].set_xlabel('Center-to-Center Distance (um)')
axes[2].set_ylabel('S2_total')
axes[2].set_title('S2_total vs Center-to-Center Distance')
axes[2].grid(True, alpha=0.3)

axes[3].plot(center_center_distance, half_mean_time, marker='o', label='half_time_mean', color='green')
axes[3].set_xlabel('Center-to-Center Distance (um)')
axes[3].set_ylabel('half_time_mean')
axes[3].set_title('half_time vs Center-to-Center Distance')
axes[3].grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig(file_name, dpi=300)

