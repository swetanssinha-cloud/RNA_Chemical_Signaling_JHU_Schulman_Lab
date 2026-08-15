"""
2D Tethered Genelet Model - RADIAL / SPATIAL CHARACTERIZATION OF THE NODES

Same physics, mesh and solver as TG_Rmesh_tanh.py. The difference is the
OUTPUT: instead of reporting only the single cell nearest the receiver
centre, this script characterises each hydrogel node as a full 2D disk at
steady state.

Produced for each node (sender and receiver):

  1. Radial profiles  <C>(r)  for all five species (+ S2_total), obtained by
     binning cells by distance from the node centre in constant-width rings
     of dr = 1 um and taking a CELL-VOLUME-WEIGHTED mean inside each ring.
     Three variants per species:
        - full ring      (indiscriminate, the true area-average of the annulus)
        - facing half    (the half of the ring pointing at the OTHER node)
        - away half      (the opposite half)
  2. Heatmaps of the steady-state field.
  3. A CSV of the radial profiles, and a CSV of the raw per-cell data around
     each node so the binning can be redone without re-running the solve.

Physics notes that motivate the design:
  * eq_I2 carries NO DiffusionTerm - I2 is tethered and immobile. Its radial
    profile is therefore a readout of how far the diffusing S2 penetrated,
    not of I2 transport.
  * I2 is initialised with a hard mask (I2_init * receiver_mask), so it is
    identically zero for r > node_radius and there is a step at the boundary.
    That is why the I2 heatmap is clipped to the node disk.
  * The field is symmetric about the centre-to-centre (x) axis but NOT about
    the y axis, since the sender sits on one side. The facing/away split
    exposes that asymmetry; the full-ring average deliberately averages over
    it. Upper-half vs lower-half agreement is printed as a mesh symmetry check.

Everything is written to ./Radial_Visuals_ofNode/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
from matplotlib.patches import Circle
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Gmsh2D
from fipy.tools import numerix
import pandas as pd
import time as timer
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from Functions import (calculate_total_amount, smooth_circular_profile, intialize_equations,
                       initalize_variables, initalize_variables_speedup)
from Mesh.New_simple_mesh import create_gmsh_radial_mesh, create_conformal_radial_mesh

# =============================================================================
# PARAMETERS  (identical to TG_Rmesh_tanh.py except fine_dx)
# =============================================================================
wall_start_time = timer.time()

D_solution = 150.0
D_gel = 60.0
k_p = 0.01          # 1/s
k_d_ds = 3e-4       # 1/s
k_d_ss = 3e-4       # 1/s
k_slow = 5e4 * 1e-6  # 1/(Ms) * microMolar
k_fast = 1e6 * 1e-6  # 1/(Ms) * microMolar

I1O2_init = 0.1     # uM - 100 nM
I2_init = 0.1       # uM - 100 nM
Th2_init = I2_init * 4

node_diameter = 75
node_radius = node_diameter / 2
bath_margin = 250
distance_between = 300
total_width = 5000
total_height = 5000

fine_dx = 2          # <-- refined from 5 so that dr = 1 um rings are supported
cells_per_level = 3
coarse_dx = 100

dt = 60.0
total_time = 8 * 3600
n_steps = int(total_time / dt)

save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)
check_steady_state = True
ss_tolerance = 1e-8
ss_window = 50
verbose = True
check_interval = 100

# ---- radial binning ---------------------------------------------------------
dr = 1.0             # um, constant ring width

# ---- output -----------------------------------------------------------------
OUT_DIR = Path(__file__).parent / 'Radial_Visuals_ofNode'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = f'ccd={distance_between:.0f}_dx={fine_dx}_dr={dr:g}'
print(f"Output directory: {OUT_DIR}")

# =============================================================================
# 2D ADAPTIVE MESH SETUP
# =============================================================================

print("Creating adaptive mesh...")

# Distinct filename so this run does not clobber the .msh used by
# TG_Rmesh_tanh.py, which is built at a different fine_dx.
mesh_filename, sender_center_x, receiver_center_x, sender_center_y = create_conformal_radial_mesh(
    bath_width=total_width,
    bath_height=total_height,
    node_diameter=75.0,
    distance_between_nodes=distance_between,
    min_cell_size=fine_dx,
    max_cell_size=coarse_dx,
    growth_rate=1.5,
    cells_per_level=cells_per_level,
    mesh_filename=f'radial_mesh_visuals_dx={fine_dx}.msh',
    visualize_gmsh=False,
    verbose=True)

mesh = Gmsh2D(mesh_filename)
receiver_center_y = sender_center_y

x, y = mesh.cellCenters
x = np.asarray(x)
y = np.asarray(y)
cell_vols = np.asarray(mesh.cellVolumes)

print(f"\n2D Simulation Setup TRIANGULAR MESH:")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width} * {total_height} um^2")
print(f"  Node diameter: {node_diameter} um")
print(f"  Distance: {distance_between} um (center-to-center)")
print(f"  Sender at: ({sender_center_x:.0f}, {sender_center_y:.0f})")
print(f"  Receiver at: ({receiver_center_x:.0f}, {receiver_center_y:.0f})")
print()

# =============================================================================
# CELL VARIABLES
# =============================================================================

S2, I2, Th2, S2_I2, S2_Th2, I1O2, D_S2 = initalize_variables_speedup(
    mesh, x, y, sender_center_x, receiver_center_x, receiver_center_y,
    node_radius, I2_init, Th2_init, I1O2_init, D_gel, D_solution)

eq = intialize_equations(S2, D_S2, I1O2, I2, Th2, S2_I2, S2_Th2,
                         k_p, k_slow, k_fast, k_d_ss, k_d_ds)

# =============================================================================
# FIND RECEIVER CENTER INDEX (for monitoring)
# =============================================================================

distances_to_receiver = np.sqrt((x - receiver_center_x)**2 + (y - receiver_center_y)**2)
receiver_center_idx = int(np.argmin(distances_to_receiver))

time_points = []
I2_concentration = []
S2_free_concentration = []
S2_total_concentration = []

recent_changes = []

current_time = 0.0
step = 0
converged_to_ss = False

# =============================================================================
# TIME STEPPING
# =============================================================================

print("Starting 2D simulation with adaptive mesh...")
print(f"Total steps: {n_steps}")

for step in range(n_steps):
    S2.updateOld()
    I2.updateOld()
    Th2.updateOld()
    S2_I2.updateOld()
    S2_Th2.updateOld()

    S2_old_vals = S2.value.copy()
    I2_old_vals = I2.value.copy()
    Th2_old_vals = Th2.value.copy()
    S2_I2_old_vals = S2_I2.value.copy()
    S2_Th2_old_vals = S2_Th2.value.copy()

    res = 1e10
    sweep = 0
    max_sweeps = 10

    while res > 1e-6 and sweep < max_sweeps:
        res = eq.sweep(dt=dt)
        sweep += 1

    if step % save_interval_steps == 0:
        current_time = step * dt
        time_points.append(current_time / 3600)

        I2_val = I2.value[receiver_center_idx]
        S2_free_val = S2.value[receiver_center_idx]
        S2_total_val = (S2.value[receiver_center_idx] +
                        S2_I2.value[receiver_center_idx] +
                        S2_Th2.value[receiver_center_idx])

        I2_concentration.append(I2_val)
        S2_free_concentration.append(S2_free_val)
        S2_total_concentration.append(S2_total_val)

        if check_steady_state and step % check_interval == 0:
            epsilon = 1e-10

            changes = [
                np.max(np.abs(S2.value - S2_old_vals) / (np.abs(S2.value) + epsilon)),
                np.max(np.abs(I2.value - I2_old_vals) / (np.abs(I2.value) + epsilon)),
                np.max(np.abs(Th2.value - Th2_old_vals) / (np.abs(Th2.value) + epsilon)),
                np.max(np.abs(S2_I2.value - S2_I2_old_vals) / (np.abs(S2_I2.value) + epsilon)),
                np.max(np.abs(S2_Th2.value - S2_Th2_old_vals) / (np.abs(S2_Th2.value) + epsilon))
            ]

            max_change = np.max(changes)
            recent_changes.append(max_change)

            if len(recent_changes) > ss_window:
                recent_changes.pop(0)

            if len(recent_changes) >= ss_window:
                if all(c < ss_tolerance for c in recent_changes):
                    converged_to_ss = True
                    if verbose:
                        print(f"\n{'='*70}")
                        print(f"STEADY STATE REACHED at t = {current_time/3600:.3f} hours")
                        print(f"Maximum relative change: {max_change:.2e} < {ss_tolerance:.2e}")
                        print(f"{'='*70}\n")
                    break

        if step % (save_interval_steps * 10) == 0:
            print(f"t = {current_time/3600:.2f} hr: "
                  f"I2 = {I2_val*1000:.2f} nM, "
                  f"S2_total = {S2_total_val*1000:.2f} nM")

print("\nSimulation complete!")

snapshot_time_hr = current_time / 3600
wall_time = timer.time() - wall_start_time
print(f'total seconds of time for simulation: {wall_time:.3f}')
print(f'Snapshot taken at t = {snapshot_time_hr:.3f} hours '
      f'(steady-state detector triggered: {converged_to_ss})')
if recent_changes:
    print(f'Final max relative change per step: {recent_changes[-1]:.3e} '
          f'(tolerance {ss_tolerance:.1e})')

# =============================================================================
# STEADY-STATE FIELDS  (all in nM)
# =============================================================================

FIELDS = {
    'S2_free':  np.asarray(S2.value) * 1000.0,
    'I2':       np.asarray(I2.value) * 1000.0,
    'Th2':      np.asarray(Th2.value) * 1000.0,
    'S2_I2':    np.asarray(S2_I2.value) * 1000.0,
    'S2_Th2':   np.asarray(S2_Th2.value) * 1000.0,
}
FIELDS['S2_total'] = FIELDS['S2_free'] + FIELDS['S2_I2'] + FIELDS['S2_Th2']

SPECIES = ['I2', 'S2_free', 'Th2', 'S2_I2', 'S2_Th2', 'S2_total']

# =============================================================================
# RADIAL BINNING
# =============================================================================

# Constant-width rings. The last edge is clipped to the node radius so the
# outermost bin does not reach past the gel boundary.
bin_edges = np.arange(0.0, node_radius, dr)
if bin_edges[-1] < node_radius:
    bin_edges = np.append(bin_edges, node_radius)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
n_bins = len(bin_centers)


def ring_average(field, r, subset, edges):
    """Cell-volume-weighted mean of `field` in each radial ring.

    Averaging must be volume-weighted: the triangular mesh has unequal cell
    areas, so a plain mean over cells would over-count wherever cells happen
    to be small. Returns (mean_per_bin, cell_count_per_bin); bins containing
    no cells come back as NaN rather than 0 so that undersampled radii are
    visible as gaps instead of masquerading as real zeros.
    """
    nb = len(edges) - 1
    idx = np.digitize(r, edges) - 1
    valid = (idx >= 0) & (idx < nb) & subset

    b = idx[valid]
    w = cell_vols[valid]

    num = np.bincount(b, weights=w * field[valid], minlength=nb)
    den = np.bincount(b, weights=w, minlength=nb)
    cnt = np.bincount(b, minlength=nb)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return mean, cnt


class NodeAnalysis:
    """Radial characterisation of one hydrogel node."""

    def __init__(self, name, cx, cy, other_cx):
        self.name = name
        self.cx = cx
        self.cy = cy

        self.r = np.sqrt((x - cx)**2 + (y - cy)**2)
        self.in_node = self.r <= node_radius

        # "facing" = the half of the node pointing at the other node.
        facing_sign = np.sign(other_cx - cx)
        self.facing = self.in_node & (facing_sign * (x - cx) > 0)
        self.away = self.in_node & (facing_sign * (x - cx) < 0)

        # Upper / lower half, used only as a mesh symmetry check: the true
        # solution is symmetric about the centre-to-centre axis, so these two
        # should agree to round-off if the mesh resolves both halves equally.
        self.upper = self.in_node & (y > cy)
        self.lower = self.in_node & (y < cy)

        self.profiles = {}
        for sp in SPECIES:
            f = FIELDS[sp]
            full, cnt_full = ring_average(f, self.r, self.in_node, bin_edges)
            face, cnt_face = ring_average(f, self.r, self.facing, bin_edges)
            awy,  cnt_away = ring_average(f, self.r, self.away, bin_edges)
            up,   _ = ring_average(f, self.r, self.upper, bin_edges)
            lo,   _ = ring_average(f, self.r, self.lower, bin_edges)
            self.profiles[sp] = dict(full=full, facing=face, away=awy,
                                     upper=up, lower=lo)

        self.counts_full = cnt_full
        self.counts_facing = cnt_face
        self.counts_away = cnt_away
        self.n_cells = int(self.in_node.sum())

    def symmetry_check(self, species='I2'):
        """Max relative upper/lower disagreement over bins where both exist."""
        p = self.profiles[species]
        up, lo = p['upper'], p['lower']
        both = np.isfinite(up) & np.isfinite(lo) & (np.abs(up) + np.abs(lo) > 0)
        if not both.any():
            return np.nan
        return np.max(np.abs(up[both] - lo[both]) /
                      (0.5 * (np.abs(up[both]) + np.abs(lo[both]))))

    def to_dataframe(self):
        d = {'r_um': bin_centers,
             'r_inner_um': bin_edges[:-1],
             'r_outer_um': bin_edges[1:],
             'n_cells_full': self.counts_full,
             'n_cells_facing': self.counts_facing,
             'n_cells_away': self.counts_away}
        for sp in SPECIES:
            p = self.profiles[sp]
            d[f'{sp}_full_nM'] = p['full']
            d[f'{sp}_facing_nM'] = p['facing']
            d[f'{sp}_away_nM'] = p['away']
        return pd.DataFrame(d)


print("\nBinning radial profiles...")
receiver = NodeAnalysis('Receiver', receiver_center_x, receiver_center_y, sender_center_x)
sender = NodeAnalysis('Sender', sender_center_x, sender_center_y, receiver_center_x)

for node in (receiver, sender):
    empty = int(np.sum(node.counts_full == 0))
    print(f"  {node.name}: {node.n_cells} cells in node, {n_bins} rings of dr={dr:g} um, "
          f"{empty} empty ring(s); min cells/ring = {node.counts_full.min()}")
    # Report on both species: the sender carries no I2 at all (it is seeded only
    # in the receiver and cannot diffuse), so an I2 check there is vacuous NaN.
    for sp in ('I2', 'S2_free'):
        val = node.symmetry_check(sp)
        note = '  (species is identically zero here - check is vacuous)' \
            if not np.isfinite(val) else ''
        shown = 'n/a' if not np.isfinite(val) else f'{val:.3e}'
        print(f"    upper/lower symmetry check ({sp}): "
              f"{shown} max relative difference{note}")

# =============================================================================
# SAVE CSVs
# =============================================================================

print("\nSaving CSV files...")

for node in (receiver, sender):
    f = OUT_DIR / f'radial_profile_{node.name.lower()}_{TAG}.csv'
    node.to_dataframe().to_csv(f, index=False)
    print(f"  {f.name}")

# Raw per-cell dump within 2*node_radius of each centre, so the binning choice
# can be revisited without paying for the solve again.
for node in (receiver, sender):
    sel = node.r <= 2 * node_radius
    raw = pd.DataFrame({
        'x_um': x[sel],
        'y_um': y[sel],
        'dx_from_center_um': x[sel] - node.cx,
        'dy_from_center_um': y[sel] - node.cy,
        'r_um': node.r[sel],
        'theta_rad': np.arctan2(y[sel] - node.cy, x[sel] - node.cx),
        'cell_area_um2': cell_vols[sel],
        'inside_node': node.in_node[sel],
    })
    for sp in SPECIES:
        raw[f'{sp}_nM'] = FIELDS[sp][sel]
    f = OUT_DIR / f'raw_cells_{node.name.lower()}_{TAG}.csv'
    raw.to_csv(f, index=False)
    print(f"  {f.name}  ({sel.sum()} cells)")

# Centre-point time series, kept as a sanity check against TG_Rmesh_tanh.py
df_ts = pd.DataFrame({
    'Time (hours)': time_points,
    'I2 (nM)': np.array(I2_concentration) * 1000,
    'S2_free (nM)': np.array(S2_free_concentration) * 1000,
    'S2_total (nM)': np.array(S2_total_concentration) * 1000})
f = OUT_DIR / f'timeseries_center_{TAG}.csv'
df_ts.to_csv(f, index=False)
print(f"  {f.name}")

# =============================================================================
# PLOT PANELS
#
# Each panel is a function taking an Axes. They are called once to build the
# combined overview figure and once each to build a standalone figure, so the
# two never drift apart.
# =============================================================================

C_FULL, C_FACE, C_AWAY = '#1f77b4', '#d62728', '#2ca02c'


def panel_split(ax, node, species, title, ylabel):
    """Full-ring average plus the facing / away halves."""
    p = node.profiles[species]
    ax.plot(bin_centers, p['full'], '-', color=C_FULL, lw=2.5,
            label='Full ring average', zorder=3)
    ax.plot(bin_centers, p['facing'], '--', color=C_FACE, lw=1.8,
            label='Facing other node')
    ax.plot(bin_centers, p['away'], '--', color=C_AWAY, lw=1.8,
            label='Facing away')
    ax.set_xlabel('Radius from node center, r (um)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(0, node_radius)


def panel_full_only(ax, node, species, title, ylabel):
    """Only the indiscriminate full-ring average."""
    ax.plot(bin_centers, node.profiles[species]['full'], '-',
            color=C_FULL, lw=2.5)
    ax.set_xlabel('Radius from node center, r (um)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, node_radius)


def panel_all_species(ax, node, title):
    """All five species (+ total) as full-ring averages, log axis."""
    for sp in SPECIES:
        prof = node.profiles[sp]['full']
        ls = '--' if sp == 'S2_total' else '-'
        ax.plot(bin_centers, prof, ls, lw=2, label=sp)
    ax.set_yscale('log')
    ax.set_xlabel('Radius from node center, r (um)', fontsize=11)
    ax.set_ylabel('Concentration (nM)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, node_radius)


def panel_heatmap(ax, node, species, title, cmap, clip_to_disk, extent_factor=2.0):
    """Steady-state field near a node.

    clip_to_disk=True keeps only cells inside the gel, which is what I2 wants:
    the hard initial mask makes I2 identically zero outside, so including the
    bath would waste the whole colour range on a discontinuity.
    """
    if clip_to_disk:
        sel = node.in_node
        half = node_radius
    else:
        half = extent_factor * node_radius
        sel = (np.abs(x - node.cx) <= half) & (np.abs(y - node.cy) <= half)

    xs = x[sel] - node.cx
    ys = y[sel] - node.cy
    vals = FIELDS[species][sel]

    tri = mtri.Triangulation(xs, ys)
    tpc = ax.tripcolor(tri, vals, shading='gouraud', cmap=cmap)
    cb = plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f'[{species}] (nM)', fontsize=9)

    ax.add_patch(Circle((0, 0), node_radius, fill=False, ec='white',
                        lw=1.5, ls='--'))
    ax.set_aspect('equal')
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel('x from node center (um)', fontsize=11)
    ax.set_ylabel('y from node center (um)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Arrow pointing toward the other node, so the asymmetry has a reference.
    # Stroked in black so it stays legible over the light end of the colormap.
    stroke = [pe.withStroke(linewidth=2.5, foreground='black')]
    other_dir = np.sign((sender_center_x - node.cx) if node.name == 'Receiver'
                        else (receiver_center_x - node.cx))
    arrow = ax.annotate('', xy=(other_dir * half * 0.95, half * 0.87),
                        xytext=(other_dir * half * 0.60, half * 0.87),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2))
    arrow.arrow_patch.set_path_effects(stroke)
    ax.text(0, half * 0.87, 'toward other node', color='white', fontsize=8,
            ha='center', va='center', fontweight='bold', path_effects=stroke)


def panel_timeseries(ax):
    ax.plot(df_ts['Time (hours)'], df_ts['I2 (nM)'], '-',
            color=C_FULL, lw=2, label='I2')
    ax.plot(df_ts['Time (hours)'], df_ts['S2_free (nM)'], '-',
            color=C_AWAY, lw=2, label='S2 free')
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Concentration (nM)', fontsize=11)
    ax.set_title('Receiver CENTER cell vs time\n(sanity check)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def panel_counts(ax):
    """How many cells actually support each ring - the honesty panel."""
    ax.bar(bin_centers, receiver.counts_full, width=dr * 0.9,
           color=C_FULL, alpha=0.7, label='Receiver')
    ax.bar(bin_centers, sender.counts_full, width=dr * 0.5,
           color=C_FACE, alpha=0.7, label='Sender')
    ax.axhline(1, color='k', ls=':', lw=1)
    ax.set_xlabel('Radius from node center, r (um)', fontsize=11)
    ax.set_ylabel('Cells in ring', fontsize=11)
    ax.set_title(f'Cells per ring (dr={dr:g} um, mesh dx={fine_dx} um)\n'
                 'small r is undersampled', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    ax.set_xlim(0, node_radius)


PANELS = [
    ('receiver_I2_radial',
     lambda ax: panel_split(ax, receiver, 'I2',
                            'Receiver: [I2] vs radius', '[I2] (nM)')),
    ('receiver_I2_radial_ringaverage',
     lambda ax: panel_full_only(ax, receiver, 'I2',
                                'Receiver: [I2] vs radius\n(full-ring average only)',
                                '[I2] (nM)')),
    ('receiver_S2free_radial',
     lambda ax: panel_split(ax, receiver, 'S2_free',
                            'Receiver: free [S2] vs radius', '[S2] free (nM)')),
    ('sender_S2_radial',
     lambda ax: panel_split(ax, sender, 'S2_free',
                            'Sender: free [S2] vs radius', '[S2] free (nM)')),
    ('receiver_all_species_radial',
     lambda ax: panel_all_species(ax, receiver, 'Receiver: all species vs radius')),
    ('sender_all_species_radial',
     lambda ax: panel_all_species(ax, sender, 'Sender: all species vs radius')),
    ('receiver_I2_heatmap',
     lambda ax: panel_heatmap(ax, receiver, 'I2', 'Receiver: [I2] steady state\n(node only)',
                              'viridis', clip_to_disk=True)),
    ('receiver_S2free_heatmap',
     lambda ax: panel_heatmap(ax, receiver, 'S2_free',
                              'Receiver: free [S2] steady state\n(node + surroundings)',
                              'magma', clip_to_disk=False)),
    ('sender_S2free_heatmap',
     lambda ax: panel_heatmap(ax, sender, 'S2_free',
                              'Sender: free [S2] steady state\n(node + surroundings)',
                              'magma', clip_to_disk=False)),
    ('center_timeseries', panel_timeseries),
    ('cells_per_ring', panel_counts),
]

# ---------------------------------------------------------------- combined
print("\nPlotting...")

n_panels = len(PANELS)
ncols = 3
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.2 * nrows))
axes = np.atleast_1d(axes).ravel()

for ax, (name, drawer) in zip(axes, PANELS):
    drawer(ax)
for ax in axes[n_panels:]:
    ax.axis('off')

fig.suptitle(
    f'Radial characterisation of the hydrogel nodes at steady state '
    f'(t = {snapshot_time_hr:.2f} hr)\n'
    f'Domain {total_width/1e3:.0f}mm x {total_height/1e3:.0f}mm, '
    f'center-to-center = {distance_between:.0f} um, mesh dx = {fine_dx} um, '
    f'ring width dr = {dr:g} um',
    fontsize=17, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.965])
combined_path = OUT_DIR / f'ALL_radial_visuals_{TAG}.png'
fig.savefig(combined_path, dpi=200, bbox_inches='tight')
print(f"  {combined_path.name}")

# ---------------------------------------------------------------- standalone
for name, drawer in PANELS:
    f1, a1 = plt.subplots(figsize=(7.5, 6))
    drawer(a1)
    f1.tight_layout()
    p = OUT_DIR / f'{name}_{TAG}.png'
    f1.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(f1)
    print(f"  {p.name}")

plt.show()

print(f"\nDone. Everything written to {OUT_DIR}")
