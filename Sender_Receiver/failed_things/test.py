import pandas as pd
import numpy as np

# Load steady state values
ss = pd.read_csv('sweep_results/steady_state_values.csv')

# Find 300 µm entry
idx_300 = np.argmin(np.abs(ss['center_distance_um'] - 300))
row_300 = ss.iloc[idx_300]

print("="*80)
print("DIAGNOSTIC FOR 300 µm DISTANCE")
print("="*80)

print(f"\nActual center_distance_um: {row_300['center_distance_um']:.1f} µm")
print(f"CSV filename: {row_300['csv_file']}")

print(f"\nFrom steady_state_values.csv:")
print(f"  final_I2_M: {row_300['final_I2_M']:.10e} M")
print(f"  final_I2_nM: {row_300['final_I2_M']*1e9:.6f} nM")

# Load the corresponding timeseries
ts_file = f"sweep_results/{row_300['csv_file']}"
ts = pd.read_csv(ts_file)

print(f"\nFrom timeseries CSV ({row_300['csv_file']}):")
print(f"  Initial I2_M: {ts['I2_M'].iloc[0]:.10e} M")
print(f"  Initial I2_nM: {ts['I2_M'].iloc[0]*1e9:.6f} nM")
print(f"  Final I2_M: {ts['I2_M'].iloc[-1]:.10e} M")
print(f"  Final I2_nM: {ts['I2_M'].iloc[-1]*1e9:.6f} nM")

print(f"\nDo they match?")
print(f"  steady_state final_I2_M == timeseries final I2_M: {np.isclose(row_300['final_I2_M'], ts['I2_M'].iloc[-1])}")

print(f"\nTimeseries info:")
print(f"  Number of time points: {len(ts)}")
print(f"  Time range: {ts['time_s'].iloc[0]:.1f} to {ts['time_s'].iloc[-1]:.1f} seconds")
print(f"  Time range: {ts['time_s'].iloc[0]/3600:.3f} to {ts['time_s'].iloc[-1]/3600:.3f} hours")

print(f"\nLast 5 I2 values (nM):")
print(ts['I2_M'].iloc[-5:].values * 1e9)

print("\n" + "="*80)
print("COMPARING TO SINGLE RUN AT 300 µm")
print("="*80)
print("\nDid you save the single-run results?")
print("If yes, what was the final I2 concentration you got?")