"""
Reads the full spatial-field CSV produced by TG_Rmesh_visulization_fast.py
(one row per mesh cell: x_um, y_um, S2_free_nM, ..., S2_total_nM) and reports
the numerical [S2] difference between the receiver node and the "mirrored"
point on the far side of the sender (sender_center_x - distance_between,
y_center) -- the same two locations compared visually in
Fast_Visualization_box_comparison_ccd=*.png.

Does NOT re-run the simulation -- it just reads the newest matching
Fast_Visualization_spatial_field_ccd=*_t=*hr.csv file in this directory.
The geometry parameters below must match the run that produced that CSV
(they mirror TG_Rmesh_visulization_fast.py's parameter block); the ccd
embedded in the CSV filename is checked against `distance_between` below
as a sanity check.
"""

import re
import numpy as np
import pandas as pd
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
mirror_center_x = sender_center_x - distance_between
mirror_center_y = y_center

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
              f"parameter block above to match, or the point locations "
              f"below will be wrong.")

df = pd.read_csv(csv_path)


# ---- nearest-cell lookup (same idea the simulation itself uses for
#      "receiver_center_idx") -------------------------------------------------
def nearest_cell(cx, cy):
    d2 = (df['x_um'] - cx) ** 2 + (df['y_um'] - cy) ** 2
    return df.loc[d2.idxmin()]


receiver_cell = nearest_cell(receiver_center_x, y_center)
mirror_cell = nearest_cell(mirror_center_x, mirror_center_y)


# ---- node-radius-averaged values too -- more robust than one nearest cell,
#      and gives an "apples to apples" comparison at the mirrored point,
#      which has no real node to anchor a single reading to. -----------------
def disk_average(cx, cy, radius):
    r = np.sqrt((df['x_um'] - cx) ** 2 + (df['y_um'] - cy) ** 2)
    inside = r <= radius
    if not inside.any():
        return None
    return df.loc[inside, ['S2_free_nM', 'S2_total_nM']].mean()


receiver_disk = disk_average(receiver_center_x, y_center, node_radius)
mirror_disk = disk_average(mirror_center_x, mirror_center_y, node_radius)

# ---- report ------------------------------------------------------------------
print()
if snapshot_hr is not None:
    print(f"Snapshot time: t = {snapshot_hr:.1f} hr")
print(f"Receiver center:       ({receiver_center_x:.1f}, {y_center:.1f}) um")
print(f"Mirrored point center: ({mirror_center_x:.1f}, {mirror_center_y:.1f}) um "
      f"(= sender_center_x - distance_between)")
print(f"Distance between (ccd): {distance_between:.1f} um")
print()

print("Nearest-cell values:")
print(f"  Receiver        [S2]_free  = {receiver_cell['S2_free_nM']:.6f} nM   "
      f"[S2]_total = {receiver_cell['S2_total_nM']:.6f} nM")
print(f"  Mirrored point  [S2]_free  = {mirror_cell['S2_free_nM']:.6f} nM   "
      f"[S2]_total = {mirror_cell['S2_total_nM']:.6f} nM")
diff_free = mirror_cell['S2_free_nM'] - receiver_cell['S2_free_nM']
diff_total = mirror_cell['S2_total_nM'] - receiver_cell['S2_total_nM']
print(f"  Difference (mirrored - receiver): "
      f"[S2]_free = {diff_free:.6f} nM,  [S2]_total = {diff_total:.6f} nM")

if receiver_disk is not None and mirror_disk is not None:
    print()
    print(f"Node-radius-averaged values (mean over cells within {node_radius:.1f} um "
          f"of each center):")
    print(f"  Receiver        [S2]_free  = {receiver_disk['S2_free_nM']:.6f} nM   "
          f"[S2]_total = {receiver_disk['S2_total_nM']:.6f} nM")
    print(f"  Mirrored point  [S2]_free  = {mirror_disk['S2_free_nM']:.6f} nM   "
          f"[S2]_total = {mirror_disk['S2_total_nM']:.6f} nM")
    diff_free_avg = mirror_disk['S2_free_nM'] - receiver_disk['S2_free_nM']
    diff_total_avg = mirror_disk['S2_total_nM'] - receiver_disk['S2_total_nM']
    print(f"  Difference (mirrored - receiver): "
          f"[S2]_free = {diff_free_avg:.6f} nM,  [S2]_total = {diff_total_avg:.6f} nM")
else:
    print("\n(Node-radius disk average unavailable -- no cells found within "
          "node_radius of one of the points; the nearest-cell values above "
          "still stand.)")
