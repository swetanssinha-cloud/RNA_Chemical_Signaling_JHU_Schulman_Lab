"""
Verification: does Transmission_Line.py with N_NODES = 2 reproduce TG_Rmesh_fast.py?

With two nodes, the transmission line is meant to be the same model as the
original sender -> receiver file: node 1 is the fixed source (I1O2) and node 2
is the receiver (I2 / Th2 / S2_I2 / S2_Th2). Given identical inputs the two
should agree. This script compares the receiver's time series from both runs
and reports how large the differences are.

Usage (run from anywhere; relative names are looked up in this folder):
    python Transmission_Line_verification.py
    python Transmission_Line_verification.py <line_csv> <fast_csv>

Inputs
------
  LINE_CSV : written by Transmission_Line.py (N_NODES = 2), e.g.
             TransmissionLine_N2_ccd=300_k_p=0.2.csv
  REF_CSV  : written by TG_Rmesh_fast.py, e.g.
             For_Comparision_with_Transmission_Line_ccd=300.csv

Compared quantities (receiver node, nM):  [I], free [S], total [S].

Outputs (written next to LINE_CSV)
-------
  <line_csv stem>_verification.png          overlay + difference plots
  <line_csv stem>_verification_metrics.csv  the error table printed below

For a meaningful comparison BOTH runs must use the same inputs. The CSVs do
not record k_p for the reference run, so the script also checks the first
saved row: free+bound [S] right after the first step is roughly proportional
to k_p, so a ratio far from 1 almost certainly means the two runs used
different k_p (or different initial concentrations).

Differences that are EXPECTED even with matching rate constants:
  * Domain width. TG_Rmesh_fast.py uses a 5000 um wide bath; Transmission_Line.py
    sizes the bath as (N-1)*ccd + node_diameter + 2*bath_margin, which is
    5075 um for N=2, ccd=300, bath_margin=2350. Setting bath_margin = 2312.5
    makes the N=2 geometry identical (5000 um) to the reference.
  * Any resulting difference in the triangulation (the probe cell is the one
    nearest the node centre, so it depends on the mesh).
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --- what to compare ---------------------------------------------------------
LINE_CSV = 'TransmissionLine_N2_ccd=300_k_p=0.2.csv'
REF_CSV = 'For_Comparision_with_Transmission_Line_ccd=300.csv'
LINE_NODE = 2   # which node of the line run plays the role of the receiver

# Adjustable, NOT an established standard: a quantity is reported as "within
# tolerance" if its largest difference is below this % of that quantity's
# full-scale (max |reference|) value.
PASS_TOL_PCT_OF_RANGE = 0.1

# (label, column in the line CSV, column in the reference CSV)
QUANTITIES = [
    ('[I] receiver switch', f'I_{LINE_NODE} (nM)', 'I2 (nM)'),
    ('free [S]',            f'S_{LINE_NODE}_free (nM)', 'S2_free (nM)'),
    ('total [S]',           f'S_{LINE_NODE}_total (nM)', 'S2_total (nM)'),
]

if len(sys.argv) > 1:
    LINE_CSV = sys.argv[1]
if len(sys.argv) > 2:
    REF_CSV = sys.argv[2]


def resolve(name):
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return HERE / p


line_path = resolve(LINE_CSV)
ref_path = resolve(REF_CSV)
for p in (line_path, ref_path):
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")

line = pd.read_csv(line_path)
ref = pd.read_csv(ref_path)

for label, line_col, ref_col in QUANTITIES:
    if line_col not in line.columns:
        raise KeyError(f"'{line_col}' not in {line_path.name}. Columns: {list(line.columns)}")
    if ref_col not in ref.columns:
        raise KeyError(f"'{ref_col}' not in {ref_path.name}. Columns: {list(ref.columns)}")

kp_match = re.search(r'k_p=([0-9.eE+-]+?)\.csv$', line_path.name)
line_kp = kp_match.group(1) if kp_match else 'unknown'

print("=" * 78)
print("TRANSMISSION LINE (N=2) vs TG_Rmesh_fast.py")
print("=" * 78)
print(f"line CSV : {line_path.name}   (k_p in filename: {line_kp})")
print(f"ref  CSV : {ref_path.name}   (k_p not recorded -- confirm it matches)")

# --- align on the saved time stamps -----------------------------------------
line['_t'] = line['Time (hours)'].round(9)
ref['_t'] = ref['Time (hours)'].round(9)
merged = line.merge(ref, on='_t', suffixes=('_line', '_ref'))

n_line, n_ref, n_both = len(line), len(ref), len(merged)
print(f"rows     : line={n_line}, ref={n_ref}, matched on time={n_both}")
if n_both == 0:
    raise RuntimeError("No common time stamps between the two CSVs.")
if n_both < min(n_line, n_ref):
    print(f"WARNING: only {n_both} of {min(n_line, n_ref)} rows share a time stamp. "
          f"One run probably stopped early (steady-state exit) or used a different "
          f"save interval; comparing the overlapping range only.")

t = merged['_t'].to_numpy()

# --- input sanity check from the very first saved row -----------------------
label0, line_col0, ref_col0 = QUANTITIES[2]
first_line = float(merged[line_col0].iloc[0])
first_ref = float(merged[ref_col0].iloc[0])
if first_ref > 0:
    ratio = first_line / first_ref
    print(f"first-row total [S] ratio (line / ref) = {ratio:.4f}")
    if abs(ratio - 1.0) > 0.02:
        print("!" * 78)
        print("WARNING: the first saved row already differs by more than 2%.")
        print("Early [S] scales with k_p, so the two runs very likely used different")
        print("inputs (k_p and/or initial concentrations). Fix that before reading")
        print("anything into the errors below.")
        print("!" * 78)


def error_metrics(a, b):
    """a = line run, b = reference run (both numpy arrays, same length)."""
    d = a - b
    scale = np.nanmax(np.abs(b))
    floor = max(1e-3 * scale, 1e-12)      # keeps relative error finite near 0
    i = int(np.nanargmax(np.abs(d)))
    return {
        'max_abs_err_nM': float(abs(d[i])),
        'time_of_max_err_hr': float(t[i]),
        'rms_err_nM': float(np.sqrt(np.nanmean(d ** 2))),
        'mean_abs_err_nM': float(np.nanmean(np.abs(d))),
        'max_rel_err_pct': float(100 * np.nanmax(np.abs(d) / np.maximum(np.abs(b), floor))),
        'max_err_pct_of_range': float(100 * abs(d[i]) / scale) if scale > 0 else float('nan'),
        'final_line_nM': float(a[-1]),
        'final_ref_nM': float(b[-1]),
        'final_diff_nM': float(d[-1]),
    }


rows = []
series = {}
for label, line_col, ref_col in QUANTITIES:
    a = merged[line_col].to_numpy(dtype=float)
    b = merged[ref_col].to_numpy(dtype=float)
    series[label] = (a, b)
    m = error_metrics(a, b)
    m['quantity'] = label
    rows.append(m)

metrics = pd.DataFrame(rows).set_index('quantity')
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:.4g}')
print("\nError metrics (line minus reference):")
print(metrics.T.to_string())

print(f"\nVerdict at the adjustable tolerance of {PASS_TOL_PCT_OF_RANGE}% of full scale:")
for label, row in metrics.iterrows():
    ok = row['max_err_pct_of_range'] <= PASS_TOL_PCT_OF_RANGE
    print(f"  {label:<22} max error = {row['max_err_pct_of_range']:.4g}% of range "
          f"-> {'within tolerance' if ok else 'EXCEEDS tolerance'}")

metrics_path = line_path.with_name(line_path.stem + '_verification_metrics.csv')
metrics.to_csv(metrics_path)

# --- plots: overlay (left) and difference (right) per quantity ---------------
fig, axes = plt.subplots(len(QUANTITIES), 2, figsize=(14, 3.8 * len(QUANTITIES)),
                         squeeze=False)
fig.suptitle(f'Transmission_Line.py (N=2, k_p={line_kp}) vs TG_Rmesh_fast.py\n'
             f'{line_path.name}  |  {ref_path.name}',
             fontsize=13, fontweight='bold')

for r, (label, _, _) in enumerate(QUANTITIES):
    a, b = series[label]
    ax_o, ax_d = axes[r]

    ax_o.plot(t, a, color='tab:blue', linewidth=2.4, label='Transmission_Line.py (N=2)')
    ax_o.plot(t, b, color='tab:red', linestyle='--', linewidth=1.6, label='TG_Rmesh_fast.py')
    ax_o.set_ylabel(f'{label} (nM)')
    ax_o.set_title(f'{label}: overlay', fontweight='bold')
    ax_o.grid(True, alpha=0.3)
    ax_o.legend(fontsize=9)

    ax_d.plot(t, a - b, color='black', linewidth=1.5)
    ax_d.axhline(0, color='gray', linewidth=0.8)
    ax_d.set_ylabel('line - reference (nM)')
    ax_d.set_title(f'{label}: difference '
                   f'(max |diff| = {metrics.loc[label, "max_abs_err_nM"]:.3g} nM, '
                   f'{metrics.loc[label, "max_err_pct_of_range"]:.3g}% of range)',
                   fontweight='bold', fontsize=10)
    ax_d.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel('Time (hours)')

plt.tight_layout()
fig_path = line_path.with_name(line_path.stem + '_verification.png')
plt.savefig(fig_path, dpi=250, bbox_inches='tight')
print(f"\nWrote {fig_path}")
print(f"Wrote {metrics_path}")
plt.show()
