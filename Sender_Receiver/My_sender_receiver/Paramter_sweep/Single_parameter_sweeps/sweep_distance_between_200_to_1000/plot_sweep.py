import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv('results_ccd_sweep.csv')
file_name = 'for_presentation_ccd_sweep.png'
I2_final = df['I2_center_final_nM']
S2_free_final = df['S2_free_center_final_nM']
S2_tot_final = df['S2_total_center_final_nM']
distance = df['param_value']
half_time = df['half_time_center_hr']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))


axes[0,0].plot(distance, I2_final, marker='o', label='Concetration ON')
axes[0,0].set_xlabel('Center-to-Center Distance (um)')
axes[0,0].set_ylabel('Concentration ON')
axes[0,0].set_title('I2_final_mean vs Center-to-Center Distance')
axes[0,0].grid(True, alpha=0.3)

axes[1,0].plot(distance, S2_free_final, marker='o', label='Free S2', color='orange')   
axes[1,0].set_xlabel('Center-to-Center Distance (um)')
axes[1,0].set_ylabel('S2_free_final')
axes[1,0].set_title('S2_free_final vs Center-to-Center Distance')
axes[1,0].grid(True, alpha=0.3)

axes[0,1].plot(distance, S2_tot_final, marker = 'o', label='Total S2')
axes[0,1].set_xlabel('Center-to-Center Distance (um)')
axes[0,1].set_ylabel('S2_total')
axes[0,1].set_title('S2_total vs Center-to-Center Distance')
axes[0,1].grid(True, alpha=0.3)

axes[1,1].plot(distance, half_time, marker='o', label='half_time_mean', color='green')
axes[1,1].set_xlabel('Center-to-Center Distance (um)')
axes[1,1].set_ylabel('half_time_mean')
axes[1,1].set_title('half_time vs Center-to-Center Distance')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig(file_name, dpi=300)

plt.show()