"""
Compare TG_Rmesh_tanh.py ("norm", the original 5-variable coupled solve)
against TG_Rmesh_fast.py ("fast", the split S2-only solve) on the same
parameters. Overlays both time series and plots the difference, same logic
as Comparision/compareCOMSOL_and_python_one_simulation.py, but both sides are
FiPy CSVs so no COMSOL text-file parsing is needed -- one loader for both.

Run this AFTER both TG_Rmesh_tanh.py and TG_Rmesh_fast.py have finished for
the same distance_between / fine_dx, from this directory (Functions_and_system),
since that's where both scripts write their CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Must match whatever distance_between / fine_dx the two scripts were run with.
distance_between = 200
fine_dx = 5

# NOTE: these must match the csv_filename lines at the bottom of
# TG_Rmesh_tanh.py and TG_Rmesh_fast.py respectively -- update here if either
# script's output naming changes.
norm_file = f'SLOW_simulation_ccd={distance_between}.csv'
fast_file = f'Fast_for_Comparision_ccd={distance_between}.csv'

overlay_save_file_name = f'compare_fast_vs_norm.png'
difference_file_name = f'compare_fast_vs_norm.png'
csv_file_name = f'compare_fast_vs_norm.csv'

# The 2% ceiling from the conversation: anything above this and the split
# solver's result should be treated as suspect, not just "a bit different".
ACCEPTABLE_PERCENT_DIFF = 2.0

# =============================================================================
# LOAD DATA (both files are the same FiPy CSV format)
# =============================================================================

def load_fipy_data(fileName):
    """
    Load a FiPy simulation time series.
    Returns time (hours), I2 (nM), S2_free (nM), S2_total (nM)
    """
    df = pd.read_csv(fileName)
    time = df['Time (hours)'].values
    I2 = df['I2 (nM)'].values
    S2_free = df['S2_free (nM)'].values
    S2_total = df['S2_total (nM)'].values
    return time, I2, S2_free, S2_total


def interpolate_to_common_time(t1, data1, t2, data2):
    """Interpolate both datasets onto whichever time grid has more points."""
    if len(t1) > len(t2):
        common_time = t1
        data2_interp = np.interp(common_time, t2, data2)
        data1_interp = data1
    else:
        common_time = t2
        data1_interp = np.interp(common_time, t1, data1)
        data2_interp = data2
    return common_time, data1_interp, data2_interp


def percent_diff(reference, other, floor_nM=1e-3):
    """
    Percent difference relative to the reference (norm) value. floor_nM
    guards against blow-ups when the reference is near zero (e.g. S2 at
    t=0) -- below that floor the absolute difference is reported instead of
    a meaningless percentage.
    """
    denom = np.maximum(np.abs(reference), floor_nM)
    return 100.0 * (other - reference) / denom


# =============================================================================
# MAIN COMPARISON
# =============================================================================

print(f"Loading norm (TG_Rmesh_tanh.py):  {norm_file}")
norm_time, norm_I2, norm_S2_free, norm_S2_total = load_fipy_data(norm_file)

print(f"Loading fast (TG_Rmesh_fast.py):  {fast_file}")
fast_time, fast_I2, fast_S2_free, fast_S2_total = load_fipy_data(fast_file)

print(f"norm: {len(norm_time)} points, time range: {norm_time[0]:.2f} - {norm_time[-1]:.2f} hr")
print(f"fast: {len(fast_time)} points, time range: {fast_time[0]:.2f} - {fast_time[-1]:.2f} hr")

# =============================================================================
# OVERLAY PLOTS
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(norm_time, norm_I2, 'b-', linewidth=2, label='norm (TG_Rmesh_tanh)', alpha=0.7)
axes[0].plot(fast_time, fast_I2, 'r--', linewidth=2, label='fast (TG_Rmesh_fast)', alpha=0.7)
axes[0].axhline(y=75, color='g', linestyle=':', alpha=0.5, linewidth=1, label='75% ON')
axes[0].axhline(y=25, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='25% OFF')
axes[0].set_xlabel('Time (hours)', fontsize=12)
axes[0].set_ylabel('[I2] (nM)', fontsize=12)
axes[0].set_title(f'I2 Concentration Comparison (distance = {distance_between} μm)', fontsize=14, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(norm_time, norm_S2_free, 'b-', linewidth=2, label='norm (TG_Rmesh_tanh)', alpha=0.7)
axes[1].plot(fast_time, fast_S2_free, 'r--', linewidth=2, label='fast (TG_Rmesh_fast)', alpha=0.7)
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('[S2] free (nM)', fontsize=12)
axes[1].set_title('Free S2 Concentration Comparison', fontsize=14, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

axes[2].plot(norm_time, norm_S2_total, 'b-', linewidth=2, label='norm (TG_Rmesh_tanh)', alpha=0.7)
axes[2].plot(fast_time, fast_S2_total, 'r--', linewidth=2, label='fast (TG_Rmesh_fast)', alpha=0.7)
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

print("\nCalculating differences...")
time_I2, norm_I2_interp, fast_I2_interp = interpolate_to_common_time(
    norm_time, norm_I2, fast_time, fast_I2)

time_S2_free, norm_S2_free_interp, fast_S2_free_interp = interpolate_to_common_time(
    norm_time, norm_S2_free, fast_time, fast_S2_free)

time_S2_total, norm_S2_total_interp, fast_S2_total_interp = interpolate_to_common_time(
    norm_time, norm_S2_total, fast_time, fast_S2_total)

diff_I2 = fast_I2_interp - norm_I2_interp
diff_S2_free = fast_S2_free_interp - norm_S2_free_interp
diff_S2_total = fast_S2_total_interp - norm_S2_total_interp

pct_diff_I2 = percent_diff(norm_I2_interp, fast_I2_interp)
pct_diff_S2_free = percent_diff(norm_S2_free_interp, fast_S2_free_interp)
pct_diff_S2_total = percent_diff(norm_S2_total_interp, fast_S2_total_interp)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

axes[0, 0].plot(time_I2, diff_I2, 'purple', linewidth=2)
axes[0, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].set_xlabel('Time (hours)', fontsize=11)
axes[0, 0].set_ylabel('Δ[I2] (nM)', fontsize=11)
axes[0, 0].set_title('Absolute Difference: I2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(time_S2_free, diff_S2_free, 'purple', linewidth=2)
axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Time (hours)', fontsize=11)
axes[1, 0].set_ylabel('Δ[S2] free (nM)', fontsize=11)
axes[1, 0].set_title('Absolute Difference: Free S2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(time_S2_total, diff_S2_total, 'purple', linewidth=2)
axes[2, 0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[2, 0].set_xlabel('Time (hours)', fontsize=11)
axes[2, 0].set_ylabel('Δ[S2] total (nM)', fontsize=11)
axes[2, 0].set_title('Absolute Difference: Total S2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time_I2, pct_diff_I2, 'darkorange', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].axhline(y=ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[0, 1].axhline(y=-ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[0, 1].set_xlabel('Time (hours)', fontsize=11)
axes[0, 1].set_ylabel('% Difference', fontsize=11)
axes[0, 1].set_title('Percent Difference: I2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(time_S2_free, pct_diff_S2_free, 'darkorange', linewidth=2)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].axhline(y=ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[1, 1].axhline(y=-ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[1, 1].set_xlabel('Time (hours)', fontsize=11)
axes[1, 1].set_ylabel('% Difference', fontsize=11)
axes[1, 1].set_title('Percent Difference: Free S2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(time_S2_total, pct_diff_S2_total, 'darkorange', linewidth=2)
axes[2, 1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
axes[2, 1].axhline(y=ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[2, 1].axhline(y=-ACCEPTABLE_PERCENT_DIFF, color='r', linestyle=':', linewidth=1, alpha=0.6)
axes[2, 1].set_xlabel('Time (hours)', fontsize=11)
axes[2, 1].set_ylabel('% Difference', fontsize=11)
axes[2, 1].set_title('Percent Difference: Total S2\n(fast - norm)', fontsize=12, fontweight='bold')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(difference_file_name, dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STATISTICS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPARISON STATISTICS  (fast vs norm)")
print("=" * 70)

print("\n[I2] Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_I2)):.5f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_I2)):.5f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_I2**2)):.5f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(pct_diff_I2)):.4f}%")
print(f"  Max percent difference:   {np.max(np.abs(pct_diff_I2)):.4f}%")

print("\n[S2] Free Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_S2_free)):.5f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_S2_free)):.5f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_S2_free**2)):.5f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(pct_diff_S2_free)):.4f}%")
print(f"  Max percent difference:   {np.max(np.abs(pct_diff_S2_free)):.4f}%")

print("\n[S2] Total Concentration:")
print(f"  Mean absolute difference: {np.mean(np.abs(diff_S2_total)):.5f} nM")
print(f"  Max absolute difference:  {np.max(np.abs(diff_S2_total)):.5f} nM")
print(f"  RMSE:                     {np.sqrt(np.mean(diff_S2_total**2)):.5f} nM")
print(f"  Mean percent difference:  {np.mean(np.abs(pct_diff_S2_total)):.4f}%")
print(f"  Max percent difference:   {np.max(np.abs(pct_diff_S2_total)):.4f}%")

worst_pct = max(np.max(np.abs(pct_diff_I2)),
                np.max(np.abs(pct_diff_S2_free)),
                np.max(np.abs(pct_diff_S2_total)))

print("\n" + "=" * 70)
if worst_pct <= ACCEPTABLE_PERCENT_DIFF:
    print(f"PASS: worst % difference ({worst_pct:.4f}%) is within the "
          f"{ACCEPTABLE_PERCENT_DIFF}% ceiling.")
else:
    print(f"FAIL: worst % difference ({worst_pct:.4f}%) EXCEEDS the "
          f"{ACCEPTABLE_PERCENT_DIFF}% ceiling.")
print("=" * 70)

# =============================================================================
# SAVE DIFFERENCE DATA TO CSV
# =============================================================================

df_diff = pd.DataFrame({
    'Time_hours': time_I2,
    'I2_norm_nM': norm_I2_interp,
    'I2_fast_nM': fast_I2_interp,
    'I2_diff_nM': diff_I2,
    'I2_percent_diff': pct_diff_I2,
    'S2_free_norm_nM': norm_S2_free_interp,
    'S2_free_fast_nM': fast_S2_free_interp,
    'S2_free_diff_nM': diff_S2_free,
    'S2_free_percent_diff': pct_diff_S2_free,
    'S2_total_norm_nM': norm_S2_total_interp,
    'S2_total_fast_nM': fast_S2_total_interp,
    'S2_total_diff_nM': diff_S2_total,
    'S2_total_percent_diff': pct_diff_S2_total,
})

df_diff.to_csv(csv_file_name, index=False)
print(f"\nDifference data saved to: {csv_file_name}")

print("\nAnalysis complete!")
