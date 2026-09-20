"""
Reads the full spatial-field CSV produced by TG_Rmesh_visulization_fast.py
(one row per mesh cell: x_um, y_um, S2_free_nM, ...) and plots the S2
profile on BOTH sides of the sender node, overlaid on a single "distance
from sender" x-axis so they can be compared point-by-point:

  - "Toward receiver":  [S2] at (sender_center_x + d, y_center)  for d >= 0
  - "Away from receiver / mirrored": [S2] at (sender_center_x - d, y_center)
    for d >= 0, reflected onto the same positive-d axis.

Since both curves share the same x-axis, the receiver (at d = distance_between
on the "toward" curve) and the mirrored/pseudo-receiver reference point (at
d = distance_between on the "away" curve, see Compare_Receiver_vs_MirrorPoint.py)
land at the exact same x position -- one dashed vertical line marks both.

Does NOT re-run the simulation -- reads the newest matching
Fast_Visualization_spatial_field_ccd=*_t=*hr.csv file in this directory,
same as Compare_Receiver_vs_MirrorPoint.py.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

# ---- geometry parameters (must match the run that produced the CSV) --------
# identical to TG_Rmesh_visulization_fast.py / TG_Rmesh_fast.py
node_diameter = 75
node_radius = node_diameter / 2
distance_between = 200
total_width = 5000
total_height = 5000

domain_center_x = total_width / 2.0
y_center = total_height / 2.0
sender_center_x = domain_center_x - distance_between / 2.0
receiver_center_x = domain_center_x + distance_between / 2.0

HERE = Path(__file__).parent

# ---- find the spatial-field CSV ---------------------------------------------
candidates = sorted(HERE.glob('Fast_Visualization_spatial_field_ccd=*_t=*hr.csv'),
                    key=lambda p: p.stat().st_mtime)
if not candidates:
    raise FileNotFoundError(
        "No 'Fast_Visualization_spatial_field_ccd=*_t=*hr.csv' file found in "
        f"{HERE}. Run TG_Rmesh_visulization_fast.py first.")
csv_path = candidates[-1]  # newest
print(f"Reading: {csv_path.name}")

m = re.search(r'ccd=(\d+(?:\.\d+)?)_t=([\d.]+)hr', csv_path.name)
snapshot_hr = None
if m:
    csv_ccd = float(m.group(1))
    snapshot_hr = float(m.group(2))
    if not np.isclose(csv_ccd, distance_between):
        print(f"WARNING: filename says ccd={csv_ccd} um but this script's "
              f"distance_between={distance_between} um -- update the "
              f"parameter block above to match, or the geometry below will "
              f"be wrong.")

df = pd.read_csv(csv_path)
x = df['x_um'].values
y = df['y_um'].values
S2_free_nM = df['S2_free_nM'].values

# ---- triangulated interpolation (same method as TG_Rmesh_visulization_fast.py)
triangulation = mtri.Triangulation(x, y)
S2_interp = mtri.LinearTriInterpolator(triangulation, S2_free_nM)

# ---- sample both sides on a shared distance-from-sender axis ----------------
# Capped so neither side samples past a domain edge.
d_max = min(sender_center_x, total_width - sender_center_x)
d = np.linspace(0, d_max, 2000)

S2_toward = np.ma.filled(S2_interp(sender_center_x + d, np.full_like(d, y_center)), np.nan)
S2_away = np.ma.filled(S2_interp(sender_center_x - d, np.full_like(d, y_center)), np.nan)

# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.plot(d, S2_toward, '-', color='blue', linewidth=2,
        label='Toward receiver (sender + d)')
ax.plot(d, S2_away, '-', color='gray', linewidth=2,
        label='Away from receiver / mirrored (sender - d)')

ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Sender (d = 0)')
ax.axvline(x=distance_between, color='black', linestyle='--', linewidth=2, alpha=0.8,
           label=f'Receiver / mirrored point (d = {distance_between:.0f} um)')

ax.set_xlabel('Distance from sender, d (um)', fontsize=12)
ax.set_ylabel('[S2] (nM)', fontsize=12)
title = 'S2 vs distance from sender: toward receiver vs. mirrored away side'
if snapshot_hr is not None:
    title += f'\n(t = {snapshot_hr:.1f} hr, distance_between = {distance_between:.0f} um)'
ax.set_title(title, fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(bottom=0)
ax.set_xlim(0, d_max)

plt.tight_layout()
out_path = HERE / f'Sender_LeftRight_Overlay_ccd={distance_between:.0f}.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path.name}")
plt.show()

# ---- also print the numbers at d = distance_between, for a quick readout ----
idx = np.argmin(np.abs(d - distance_between))
print(f"\nAt d = {d[idx]:.1f} um:")
print(f"  Toward receiver (actual receiver): {S2_toward[idx]:.4f} nM")
print(f"  Away / mirrored point:             {S2_away[idx]:.4f} nM")
print(f"  Difference (away - toward):        {S2_away[idx] - S2_toward[idx]:+.4f} nM")
