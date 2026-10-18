"""
Final-concentration, odd/even-node plot for a completed Transmission_Line.py run.

Standalone: reads the CSV Transmission_Line.py already writes (plus its
.meta.json sidecar, if present) and re-derives everything from that -- it
does NOT re-run any simulation, so this can be re-plotted/re-styled instantly
without paying for another multi-hour FiPy run.

What it plots
--------------
For every node k = 1..N, the LAST recorded [I_k] (the template/receiver
concentration -- see Transmission_Line.py's overlay-plot section for why
those are the same quantity here) against that node's distance from node 1.
Nodes are then split by parity of their 1-indexed node number (odd: 1, 3,
5, ...; even: 2, 4, ...) and each group is drawn as its own connected
line+marker series in its own color, rather than one zigzag line through
every node in order -- this is what turns the raw alternating pattern into
two readable trends.

Expected pattern (see the paper's relay mechanism): node 1 is a fixed,
never-depleted source, so it drives node 2 down hard (low). Node 2, having
been depleted, is a weak/decaying source for node 3, so node 3 stays high
-- and the pattern repeats down the line. Odd nodes should trend high, even
nodes should trend low. Whether the two trends stay as parallel plateaus or
converge toward each other as the chain gets longer is an open question this
plot is meant to help answer (see the header of Transmission_Line.py and the
eventual parameter sweep over chain length / rate constants).
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --- which run to plot -------------------------------------------------------
# Usage:
#   python plot_final_receiver_concentrations.py path/to/TransmissionLine_N5_ccd=300.csv
#   python plot_final_receiver_concentrations.py path/to/some_folder      (newest matching CSV in it)
#   python plot_final_receiver_concentrations.py                          (newest matching CSV in the cwd)
# The PNG is written next to the CSV it was made from, so each dataset's
# figure stays with its own data.
target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

if target.is_dir():
    candidates = sorted(target.glob('TransmissionLine_N*_ccd=*.csv'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No TransmissionLine_N*_ccd=*.csv found in {target.resolve()}. "
            f"Pass the CSV path (or a folder containing one) as an argument.")
    csv_path = candidates[0]
else:
    csv_path = target

CSV_FILENAME = csv_path.name
print(f"Reading {csv_path}")
df = pd.read_csv(csv_path)

# --- geometry: distance_between is encoded in the filename (Transmission_Line.py
# writes f'TransmissionLine_N{N_NODES}_ccd={distance_between:.0f}.csv'), not
# stored as a CSV column -- parse it back out rather than hardcode it here.
ccd_match = re.search(r'ccd=(\d+(?:\.\d+)?)', CSV_FILENAME)
if ccd_match is None:
    raise ValueError(
        f"Could not find 'ccd=<value>' in filename {CSV_FILENAME!r} -- "
        f"can't recover node spacing to build the x-axis.")
distance_between = float(ccd_match.group(1))

# --- N_NODES and per-node final [I] values: read every "I_k (nM)" column
i_col_pattern = re.compile(r'^I_(\d+) \(nM\)$')
node_numbers = sorted(
    int(m.group(1)) for c in df.columns if (m := i_col_pattern.match(c))
)
if not node_numbers:
    raise ValueError(f"No 'I_k (nM)' columns found in {CSV_FILENAME}.")
N_NODES = max(node_numbers)

final_row = df.iloc[-1]
x_um = [(k - 1) * distance_between for k in node_numbers]
y_nM = [final_row[f'I_{k} (nM)'] for k in node_numbers]

# --- steady-state status, if the .meta.json sidecar exists ------------------
meta_path = csv_path.with_name(csv_path.name.rsplit('.csv', 1)[0] + '.meta.json')
meta = None
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)

# Independent convergence check straight from the data (same statistic the
# simulation uses): std/mean of the last node's [I] over the trailing 60
# samples. Reported whether or not a .meta.json exists, so older runs that
# predate the sidecar still get an honest caption.
tail = df[f'I_{N_NODES} (nM)'].iloc[-60:]
drift = tail.std() / tail.mean() if tail.mean() > 0 else float('nan')

if meta is not None:
    status = ("steady state reached" if meta['converged_to_ss']
              else "NOT fully converged -- ceiling reached")
    caption = f"t = {meta['final_time_hr']:.2f} hr ({status})"
else:
    caption = (f"t = {final_row['Time (hours)']:.2f} hr "
               f"(no .meta.json; last-node drift {drift:.1e} std/mean)")

print("\nFinal [I_k] (nM):")
for k, y in zip(node_numbers, y_nM):
    print(f"  node {k} ({'odd' if k % 2 else 'even'}): {y:.3f}")
print(f"Last-node [I_{N_NODES}] drift over trailing 60 samples: {drift:.2e} (std/mean)")

# --- split by parity of the 1-indexed node number ---------------------------
odd_x = [x for x, k in zip(x_um, node_numbers) if k % 2 == 1]
odd_y = [y for y, k in zip(y_nM, node_numbers) if k % 2 == 1]
even_x = [x for x, k in zip(x_um, node_numbers) if k % 2 == 0]
even_y = [y for y, k in zip(y_nM, node_numbers) if k % 2 == 0]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(odd_x, odd_y, '-o', color='red', linewidth=2, markersize=8,
        label='Odd nodes (1, 3, 5, ...)', zorder=3)
ax.plot(even_x, even_y, '-o', color='blue', linewidth=2, markersize=8,
        label='Even nodes (2, 4, ...)', zorder=3)

# Node 1 is a fixed constant source (never solved -- see Transmission_Line.py's
# header), not a converged result of the chain dynamics like the rest of the
# odd-node line, so it's drawn hollow to distinguish it at a glance.
ax.plot(x_um[0], y_nM[0], 'o', markerfacecolor='none', markeredgecolor='red',
        markeredgewidth=2, markersize=12, zorder=4)

ax.set_xlabel('Distance from Node 1 (um)', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Template / Receiver Concentration [I] (nM)',
              fontsize=12, fontweight='bold')
ax.set_title(f'{N_NODES}-Node Transmission Line: Final [I] by Node\n{caption}',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
out_path = csv_path.with_name(f'FinalConcentration_N{N_NODES}_OddEven.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Wrote {out_path}")
plt.show()
