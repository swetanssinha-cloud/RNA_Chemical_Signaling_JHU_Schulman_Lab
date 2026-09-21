"""
2D Tethered Genelet Model -- N-Node Transmission Line (Chen '25, Supp. 2.3.3-2.3.5)
=====================================================================================

What this generalizes
----------------------
TG_Rmesh_fast.py solves ONE sender -> ONE receiver: a fixed, non-reacting
sender switch (I1O2) sitting in the bath and continuously feeding a signal
(S2) into a single downstream receiver node (I2/Th2/S2_I2/S2_Th2).

The paper's transmission line chains that same relay N times: node k's own
switch, once triggered, becomes the thing that drives node k+1's signal.
Model ODEs (12)-(36) in the supplement give the pattern explicitly -- every
node from 2 onward is a verbatim copy of the 2-node model's receiver, just
re-indexed, with node k's production term k_p*[I_(k-1)] pointing at the
*previous* node's switch instead of a fixed sender concentration:

    S_k   (diffuses):  dS_k/dt  = k_p*I_(k-1) - k_slow*I_k*S_k - k_fast*Th_k*S_k - k_d_ss*S_k
    I_k   (local):     dI_k/dt  = k_d_ds*(S_k:I_k) - k_slow*I_k*S_k
    Th_k  (local):     dTh_k/dt = k_d_ds*(S_k:Th_k) - k_fast*Th_k*S_k
    S_k:I_k  (local):  d(..)/dt = -k_d_ds*(S_k:I_k) + k_slow*I_k*S_k
    S_k:Th_k (local):  d(..)/dt = -k_d_ds*(S_k:Th_k) + k_fast*Th_k*S_k

Node 1 is the head of the chain and has no upstream term at all (eq. 12 has
no k_p*[...] production for S1). With S1(0)=0 and nothing to produce it, S1
never leaves zero, so I1 never has anything to react with either -- node 1
is therefore numerically identical to the original model's I1O2: a fixed,
non-evolving source. This file implements it exactly that way (see
"DECISIONS" below) rather than paying for four inert ODEs per run.

DECISIONS
--------------------------------------------------------------
2. Node 1 is a fixed constant source (I1 held at its initial value forever,
   no S1/Th1/complexes solved) -- matches how the sender is already handled
   in the existing 2-node model, and is mathematically what the paper's
   own equations reduce to.
3. Every node is tracked and plotted (I_k, free S_k, total S_k), not just
   the last one, so you can see the signal propagate/decay down the line.
4. Defaults: N_NODES = 5, total_time = 8 hours (paper's worked example).

How the loop generalization works
----------------------------------
CellVariables are kept in Python lists indexed 0..N_NODES-1 (index i is the
paper's "node i+1"). Index 0 only ever holds a fixed I[0]. For i = 1 ..
N_NODES-1, one loop body builds S[i]/I[i]/Th[i]/SI[i]/STh[i] and their
FiPy equation, sourced by I[i-1] -- which is *either* the fixed node-0
source *or* a previously-built solved node, transparently, since both are
just CellVariables. Changing N_NODES is a one-line edit; nothing else in
the file references node count directly.

Solver strategy: identical split-equation idea as TG_Rmesh_fast.py, just
repeated per node. Only S_k diffuses, so each node gets its own small
(Ncells-unknown) FiPy diffusion solve; I_k/Th_k/S_k:I_k/S_k:Th_k update via
the same closed-form backward-Euler step (reaction_pair_step, unchanged).
Within one Picard sweep, nodes are updated in order 1 -> N so that node k
already sees node (k-1)'s freshest guess this sweep (a chain-aware
Gauss-Seidel ordering) -- convergence is judged by the worst (max) residual
across all N-1 diffusion solves.

Mesh: New geometry helper create_conformal_line_mesh() below generalizes
Mesh/New_simple_mesh.py's create_conformal_radial_mesh() from 2 disks to N
disks arranged in a single row, spaced `distance_between` apart
center-to-center (paper: 300 um), fragmented into one conformal bath so
every node shares mesh nodes with its neighbors bath. Domain width
auto-scales with N so adding nodes doesn't crowd the bath margins.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm, Gmsh2D, LinearLUSolver
from fipy.tools import numerix
import pandas as pd
import time as timer
import gmsh
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from Mesh.New_simple_mesh import check_mesh_is_conformal

# =============================================================================
# PARAMETERS (reaction rates identical to TG_Rmesh_fast.py)
# =============================================================================
wall_start_time = timer.time()

D_solution = 150.0
D_gel = 60.0
k_p = 0.01
k_d_ds = 3e-4
k_d_ss = 3e-4
k_slow = 5e4 * 1e-6
k_fast = 1e6 * 1e-6

# --- transmission-line topology --------------------------------------------
N_NODES = 5                 # change this to add/remove senders/receivers
distance_between = 300.0    # center-to-center spacing (paper: "300 um away
                             # from each neighboring node")
node_diameter = 75.0
node_radius = node_diameter / 2.0

# Every node's switch I_k starts at 100 nM (paper IC, all designs).
I_init = 0.1                # uM
I_init_list = [I_init] * N_NODES

# Localized Threshold per node (TG Design 3). The paper only specifies these
# five values for its 5-node worked example (Th1..Th5 = 5, 5, 10, 5, 5 uM).
# Th_init_list[i] is unused for i == 0 since node 1 has no Threshold species
# (see DECISIONS #2 above) -- it's kept in the list only so the indexing
# lines up 1:1 with the paper's node numbering.
    
Th_init_list = [I_init_list[k] * 4.0 for k in range(N_NODES)]

# --- domain sizing (auto-scales with N_NODES so the bath margin around the
#     whole chain stays generous no matter how many nodes are in the line) --
bath_margin = 2350.0
chain_span = (N_NODES - 1) * distance_between
total_width = chain_span + node_diameter + 2 * bath_margin
total_height = 5000.0

fine_dx = 5
cells_per_level = 3
coarse_dx = 100
growth_rate = 1.5

dt = 60.0
total_time = 8 * 3600
n_steps = int(total_time / dt)

save_interval_time = 60.0
save_interval_steps = int(save_interval_time / dt)
verbose = True

# Steady-state detection, ported from Paramter_sweep/Single_parameter_sweeps/
# sweep_core.py: std/mean of a trailing window of saved samples on the LAST
# node's readout (I_N), rather than a step-to-step relative-change threshold
# aggregated over every species at every node. Checking only the last node
# is sufficient for the whole chain: node k's equation only ever depends on
# node (k-1), never anything downstream, so the tail of the chain cannot
# look steady unless every upstream node feeding into it has already been
# steady for the same trailing window.
check_steady_state = True
STEADY_STATE_WINDOW = 60        # trailing saved samples used for mean/std
STEADY_STATE_THRESHOLD = 1e-10  # std/mean of the trailing window
CHECK_INTERVAL = 1              # check every CHECK_INTERVAL saved samples

max_sweeps = 15
sweep_residual_target = 1e-8
sweep_plateau_tol = 1e-9


# =============================================================================
# N-NODE MESH GENERATION
# (generalizes Mesh/New_simple_mesh.py's create_conformal_radial_mesh from
#  2 disks to N disks in a row -- everything else about the ring-based size
#  field and the fragment()-for-conformality trick is unchanged)
# =============================================================================

def create_conformal_line_mesh(
    n_nodes,
    bath_width,
    bath_height,
    node_diameter=75.0,
    distance_between_nodes=300.0,
    min_cell_size=5.0,
    max_cell_size=100.0,
    growth_rate=1.5,
    cells_per_level=3,
    mesh_filename=None,
    verbose=True,
):
    """
    Build a radially-refined triangular mesh containing `n_nodes` circular
    hydrogel nodes arranged in a single horizontal row, evenly spaced by
    `distance_between_nodes` (center-to-center), all fragmented into one
    conformal bath (see Mesh/New_simple_mesh.py's docstring for why
    fragment() -- without it the disks mesh as disconnected islands and
    nothing can diffuse between a node and the bath).

    Returns
    -------
    mesh_filename : str, path to the written .msh (Gmsh format 2.2)
    node_centers_x : list[float], x-coordinate of each node's center
    y_center : float
    """
    if mesh_filename is None:
        mesh_filename = "conformal_line_mesh.msh"
    mesh_filename = str(mesh_filename)

    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1 (got {n_nodes})")
    if min_cell_size < max_cell_size and growth_rate <= 1.0:
        raise ValueError(
            f"growth_rate must be > 1.0 to coarsen from {min_cell_size} um to "
            f"{max_cell_size} um (got {growth_rate}); the ring schedule would "
            f"never terminate.")
    if cells_per_level <= 0:
        raise ValueError(f"cells_per_level must be positive (got {cells_per_level})")

    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    domain_center_x = bath_width / 2.0
    span = (n_nodes - 1) * distance_between_nodes
    node_centers_x = [domain_center_x - span / 2.0 + i * distance_between_nodes
                       for i in range(n_nodes)]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("conformal_line_mesh")

    # ---------------------------------------------------------------- geometry
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    node_disk_tags = [gmsh.model.occ.addDisk(cx, y_center, 0, node_radius, node_radius)
                       for cx in node_centers_x]

    fragments, _ = gmsh.model.occ.fragment(
        [(2, rectangle_tag)],
        [(2, tag) for tag in node_disk_tags],
    )
    gmsh.model.occ.synchronize()

    # fragment() renumbers everything, so re-identify each face by geometry.
    surface_tags = [tag for dim, tag in fragments if dim == 2]
    node_surfaces = [None] * n_nodes
    bath_surfaces = []

    for tag in surface_tags:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        matched = False
        for i, ncx in enumerate(node_centers_x):
            if abs(cx - ncx) < 1.0 and abs(cy - y_center) < 1.0:
                node_surfaces[i] = tag
                matched = True
                break
        if not matched:
            bath_surfaces.append(tag)

    if any(s is None for s in node_surfaces):
        gmsh.finalize()
        raise RuntimeError(
            "Could not identify all node surfaces after fragment(). "
            f"Found surfaces {surface_tags} for nodes at x={node_centers_x}."
        )

    # ------------------------------------------------------------ size fields
    # Distance to every node's surface AND every node's center; taking the
    # minimum over all of them keeps each node's interior finely resolved
    # too (same trick as the 2-node mesh).
    distance_fields = []
    for surf in node_surfaces:
        curves = [abs(t) for _, t in gmsh.model.getBoundary(
            [(2, surf)], oriented=False, combined=False, recursive=False)]
        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "CurvesList", curves)
        gmsh.model.mesh.field.setNumber(f, "Sampling", 200)
        distance_fields.append(f)

    center_points = [gmsh.model.occ.addPoint(cx, y_center, 0) for cx in node_centers_x]
    gmsh.model.occ.synchronize()
    for point in center_points:
        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "PointsList", [point])
        gmsh.model.mesh.field.setNumber(f, "Sampling", 200)
        distance_fields.append(f)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", distance_fields)

    # ------------------------------------------------ geometric ring schedule
    # (identical staircase-of-Thresholds idea as create_conformal_radial_mesh;
    # it only depends on distance-to-nearest-node, so it needs no change to
    # generalize from 2 nodes to N)
    levels = [min_cell_size]
    while levels[-1] < max_cell_size:
        levels.append(min(levels[-1] * growth_rate, max_cell_size))

    boundaries = [0.0]
    for size in levels[:-1]:
        boundaries.append(boundaries[-1] + cells_per_level * size)

    if verbose:
        print(f"\nStepped size field: {len(levels)} rings, "
              f"growth_rate={growth_rate}, {cells_per_level} cells per ring")
        for i, size in enumerate(levels):
            lo = boundaries[i]
            hi = boundaries[i + 1] if i + 1 < len(boundaries) else None
            span_str = f"[{lo:7.1f}, {hi:7.1f})" if hi is not None else f"[{lo:7.1f},     inf)"
            print(f"  ring {i}: {size:7.2f} um  for distance {span_str} um")

    threshold_ids = []
    for i, size in enumerate(levels):
        tid = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tid, "InField", min_field)
        if i < len(levels) - 1:
            transition = boundaries[i + 1]
            eps = max(0.02, 0.02 * transition)
            gmsh.model.mesh.field.setNumber(tid, "SizeMin", size)
            gmsh.model.mesh.field.setNumber(tid, "SizeMax", 1e6)
            gmsh.model.mesh.field.setNumber(tid, "DistMin", max(transition - eps, 0.0))
            gmsh.model.mesh.field.setNumber(tid, "DistMax", transition + eps)
        else:
            gmsh.model.mesh.field.setNumber(tid, "SizeMin", size)
            gmsh.model.mesh.field.setNumber(tid, "SizeMax", size)
            gmsh.model.mesh.field.setNumber(tid, "DistMin", boundaries[i])
            gmsh.model.mesh.field.setNumber(tid, "DistMax", boundaries[i])
        gmsh.model.mesh.field.setNumber(tid, "Sigmoid", 0)
        threshold_ids.append(tid)

    combined_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined_field, "FieldsList", threshold_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined_field)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

    # ------------------------------------------------- physical groups + write
    gmsh.model.addPhysicalGroup(2, bath_surfaces, name="bath")
    for i, surf in enumerate(node_surfaces):
        gmsh.model.addPhysicalGroup(2, [surf], name=f"node_{i + 1}")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(mesh_filename)
    gmsh.finalize()

    n_components = check_mesh_is_conformal(mesh_filename, verbose=verbose)
    if n_components != 1:
        raise RuntimeError(
            f"Mesh {mesh_filename} has {n_components} disconnected pieces "
            f"(expected 1). The nodes are not joined to the bath."
        )

    return mesh_filename, node_centers_x, y_center


# =============================================================================
# BUILD MESH
# =============================================================================

print(f"Creating {N_NODES}-node transmission-line mesh...")

mesh_filename, node_centers_x, y_center = create_conformal_line_mesh(
    n_nodes=N_NODES,
    bath_width=total_width,
    bath_height=total_height,
    node_diameter=node_diameter,
    distance_between_nodes=distance_between,
    min_cell_size=fine_dx,
    max_cell_size=coarse_dx,
    growth_rate=growth_rate,
    cells_per_level=cells_per_level,
    mesh_filename=f'line_mesh_N{N_NODES}.msh',
    verbose=True,
)

mesh = Gmsh2D(mesh_filename)
x, y = mesh.cellCenters

print(f"\n2D Simulation Setup -- {N_NODES}-node transmission line:")
print(f"  Mesh: {mesh.numberOfCells} cells (adaptive)")
print(f"  Domain: {total_width:.0f} x {total_height:.0f} um^2")
print(f"  Node diameter: {node_diameter} um, spacing: {distance_between} um")
for i, cx in enumerate(node_centers_x):
    print(f"  Node {i + 1} at: ({cx:.0f}, {y_center:.0f})")
print()

# =============================================================================
# PER-NODE MASKS AND SHARED DIFFUSION FIELD
# =============================================================================

node_masks = [np.sqrt((x - cx) ** 2 + (y - y_center) ** 2) <= node_radius
              for cx in node_centers_x]

any_node_mask = node_masks[0]
for m in node_masks[1:]:
    any_node_mask = any_node_mask | m

D_S = CellVariable(name="D_S", mesh=mesh, value=D_solution)
D_S.setValue(D_gel * any_node_mask + D_solution * (~any_node_mask))

# =============================================================================
# CELL VARIABLES AND EQUATIONS -- ONE LOOP BUILDS THE WHOLE CHAIN
# =============================================================================
# Node 0 (paper's Node 1): fixed constant source, exactly like I1O2 in the
# 2-node model (see DECISIONS #2 in the header). No TransientTerm, no
# updateOld -- it never changes after this point.
I = [None] * N_NODES
I[0] = CellVariable(name="I_1", mesh=mesh, value=0.0)
I[0].setValue(I_init_list[0] * node_masks[0])

S = [None] * N_NODES
Th = [None] * N_NODES
SI = [None] * N_NODES     # S_k : I_k complex
STh = [None] * N_NODES    # S_k : Th_k complex
eq_S = [None] * N_NODES

for k in range(1, N_NODES):
    S[k] = CellVariable(name=f"S_{k + 1}", mesh=mesh, value=0.0, hasOld=True)

    I[k] = CellVariable(name=f"I_{k + 1}", mesh=mesh, value=0.0, hasOld=True)
    I[k].setValue(I_init_list[k] * node_masks[k])

    Th[k] = CellVariable(name=f"Th_{k + 1}", mesh=mesh, value=0.0, hasOld=True)
    Th[k].setValue(Th_init_list[k] * node_masks[k])

    SI[k] = CellVariable(name=f"S{k + 1}_I{k + 1}", mesh=mesh, value=0.0, hasOld=True)
    STh[k] = CellVariable(name=f"S{k + 1}_Th{k + 1}", mesh=mesh, value=0.0, hasOld=True)

    # k_p * I[k - 1]: node k's signal is produced by the PREVIOUS node's
    # switch, whether that's the fixed node 0 or an already-built solved
    # node -- both are just CellVariables, so the same line works for all k.
    eq_S[k] = (TransientTerm(var=S[k]) ==
               DiffusionTerm(coeff=D_S, var=S[k]) +
               k_p * I[k - 1] +
               ImplicitSourceTerm(coeff=-(k_slow * I[k] + k_fast * Th[k] + k_d_ss), var=S[k]))

line_solver = LinearLUSolver(tolerance=1e-10)


def reaction_pair_step(S_now, X_old, C_old, k_on, k_off, dt):
    """Closed-form backward-Euler step for one exchange pair (X <-> C):
        dX/dt = -k_on*S*X + k_off*C
        dC/dt = +k_on*S*X - k_off*C
    S is held fixed at the current Picard-sweep guess. Exact solution of the
    resulting per-cell 2x2 linear system -- fully vectorized, no sparse
    solve needed. See TG_Rmesh_fast.py for the derivation/validation notes.
    """
    a = k_on * S_now
    d = k_off
    det = 1.0 + dt * (a + d)
    X_new = ((1.0 + dt * d) * X_old + dt * d * C_old) / det
    C_new = (dt * a * X_old + (1.0 + dt * a) * C_old) / det
    return X_new, C_new


# =============================================================================
# FIND EACH NODE'S CENTER CELL (for monitoring)
# =============================================================================

node_center_idx = []
for cx in node_centers_x:
    distances = numerix.sqrt((x - cx) ** 2 + (y - y_center) ** 2)
    node_center_idx.append(numerix.argmin(distances))

time_points = []
# node_data[i] holds this node's time series; node 0 has no diffusing S, so
# its S_free/S_total entries stay NaN throughout (see DECISIONS #2).
node_data = {i: {'I_nM': [], 'S_free_nM': [], 'S_total_nM': []} for i in range(N_NODES)}

recent_I_last_values = []
current_time = 0.0
step = 0
converged_to_ss = False

# =============================================================================
# TIME STEPPING
# =============================================================================

print(f"Starting {N_NODES}-node transmission-line simulation...")
print(f"Total steps: {n_steps}")

for step in range(n_steps):
    for k in range(1, N_NODES):
        S[k].updateOld()
        I[k].updateOld()
        Th[k].updateOld()
        SI[k].updateOld()
        STh[k].updateOld()

    old_vals = {k: {
        'S': S[k].value.copy(),
        'I': I[k].value.copy(),
        'Th': Th[k].value.copy(),
        'SI': SI[k].value.copy(),
        'STh': STh[k].value.copy(),
    } for k in range(1, N_NODES)}

    sweep = 0
    prev_res = None

    while sweep < max_sweeps:
        max_res = 0.0

        # March upstream -> downstream so node k reacts to node (k-1)'s
        # freshest guess within this same sweep (chain-aware Gauss-Seidel).
        for k in range(1, N_NODES):
            S_guess = S[k].value

            I_new, SI_new = reaction_pair_step(
                S_guess, old_vals[k]['I'], old_vals[k]['SI'], k_slow, k_d_ds, dt)
            Th_new, STh_new = reaction_pair_step(
                S_guess, old_vals[k]['Th'], old_vals[k]['STh'], k_fast, k_d_ds, dt)

            I[k].setValue(I_new)
            SI[k].setValue(SI_new)
            Th[k].setValue(Th_new)
            STh[k].setValue(STh_new)

            res = eq_S[k].sweep(dt=dt, solver=line_solver)
            max_res = max(max_res, res)

        sweep += 1

        if max_res < sweep_residual_target:
            break
        if prev_res is not None and abs(max_res - prev_res) < sweep_plateau_tol:
            break
        prev_res = max_res

    if step % save_interval_steps == 0:
        current_time = step * dt
        time_points.append(current_time / 3600)

        for i in range(N_NODES):
            idx = node_center_idx[i]
            if i == 0:
                node_data[i]['I_nM'].append(I[0].value[idx] * 1000)
                node_data[i]['S_free_nM'].append(np.nan)
                node_data[i]['S_total_nM'].append(np.nan)
            else:
                I_val = I[i].value[idx]
                S_free_val = S[i].value[idx]
                S_total_val = S[i].value[idx] + SI[i].value[idx] + STh[i].value[idx]
                node_data[i]['I_nM'].append(I_val * 1000)
                node_data[i]['S_free_nM'].append(S_free_val * 1000)
                node_data[i]['S_total_nM'].append(S_total_val * 1000)

        if check_steady_state:
            recent_I_last_values.append(node_data[N_NODES - 1]['I_nM'][-1])

            if (step % (save_interval_steps * CHECK_INTERVAL) == 0
                    and len(recent_I_last_values) > STEADY_STATE_WINDOW):
                recent_window = recent_I_last_values[-STEADY_STATE_WINDOW:]
                mean_I_last = np.mean(recent_window)

                if mean_I_last > 0:
                    relative_change = np.std(recent_window) / mean_I_last

                    if relative_change < STEADY_STATE_THRESHOLD:
                        converged_to_ss = True
                        if verbose:
                            print(f"\n{'=' * 70}")
                            print(f"STEADY STATE REACHED at t = {current_time / 3600:.3f} hours")
                            print(f"[I_{N_NODES}] std/mean over trailing "
                                  f"{STEADY_STATE_WINDOW} samples: "
                                  f"{relative_change:.2e} < {STEADY_STATE_THRESHOLD:.2e}")
                            print(f"{'=' * 70}\n")
                        break

        if step % (save_interval_steps * 10) == 0:
            last = N_NODES - 1
            print(f"t = {current_time / 3600:.2f} hr: "
                  f"I_1 = {node_data[0]['I_nM'][-1]:.2f} nM, "
                  f"I_{N_NODES} = {node_data[last]['I_nM'][-1]:.2f} nM, "
                  f"S_{N_NODES}_total = {node_data[last]['S_total_nM'][-1]:.2f} nM, "
                  f"sweeps = {sweep}")

print("\nSimulation complete!")

wall_time_end = timer.time()
wall_time = wall_time_end - wall_start_time
print(f'total seconds of time for simulation: {wall_time:.3f}')

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

print("Saving results to CSV files...")

csv_data = {'Time (hours)': time_points}
for i in range(N_NODES):
    csv_data[f'I_{i + 1} (nM)'] = node_data[i]['I_nM']
    csv_data[f'S_{i + 1}_free (nM)'] = node_data[i]['S_free_nM']
    csv_data[f'S_{i + 1}_total (nM)'] = node_data[i]['S_total_nM']

df = pd.DataFrame(csv_data)
csv_filename = f'TransmissionLine_N{N_NODES}_ccd={distance_between:.0f}.csv'
df.to_csv(csv_filename, index=False)

# Metadata that isn't recoverable from the CSV alone -- same sidecar pattern
# as Paramter_sweep/Single_parameter_sweeps/sweep_core.py's .meta.json.
# Lets a standalone script reading this CSV later (e.g. a final-concentration
# plot) know whether "final" means "converged" or just "ran out of time".
meta_filename = csv_filename.rsplit('.csv', 1)[0] + '.meta.json'
with open(meta_filename, 'w') as f:
    json.dump({
        'N_NODES': N_NODES,
        'converged_to_ss': bool(converged_to_ss),
        'final_time_hr': current_time / 3600,
        'total_time_hr': total_time / 3600,
    }, f, indent=2)

# =============================================================================
# PLOTTING -- ONE ROW PER NODE
# =============================================================================

fig, axes = plt.subplots(N_NODES, 3, figsize=(18, 4.5 * N_NODES), squeeze=False)
fig.suptitle(f'{N_NODES}-Node Transmission Line: Domain {total_width / 1e3:.1f}mm x '
             f'{total_height / 1e3:.1f}mm, spacing={distance_between:.0f}um',
             fontsize=16, fontweight='bold')

for i in range(N_NODES):
    ax_I, ax_free, ax_tot = axes[i]

    ax_I.plot(time_points, node_data[i]['I_nM'], 'b-', linewidth=2)
    ax_I.set_xlabel('Time (hours)')
    ax_I.set_ylabel('[I] (nM)')
    ax_I.set_title(f'Node {i + 1}: [I_{i + 1}]', fontweight='bold')
    ax_I.grid(True, alpha=0.3)
    ax_I.set_ylim(bottom=0)

    if i == 0:
        for ax in (ax_free, ax_tot):
            ax.axis('off')
            ax.text(0.5, 0.5, "Node 1 is a fixed source\n(no diffusing S -- see file header)",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, style='italic')
    else:
        ax_free.plot(time_points, node_data[i]['S_free_nM'], 'g-', linewidth=2)
        ax_free.set_xlabel('Time (hours)')
        ax_free.set_ylabel('[S] free (nM)')
        ax_free.set_title(f'Node {i + 1}: free [S_{i + 1}]', fontweight='bold')
        ax_free.grid(True, alpha=0.3)
        ax_free.set_ylim(bottom=0)

        ax_tot.plot(time_points, node_data[i]['S_total_nM'], 'r-', linewidth=2)
        ax_tot.set_xlabel('Time (hours)')
        ax_tot.set_ylabel('[S] total (nM)')
        ax_tot.set_title(f'Node {i + 1}: total [S_{i + 1}]', fontweight='bold')
        ax_tot.grid(True, alpha=0.3)
        ax_tot.set_ylim(bottom=0)

plt.tight_layout()
plot_filename = f'TransmissionLine_N{N_NODES}_ccd={distance_between:.0f}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

# =============================================================================
# TEMPLATE / RECEIVER CONCENTRATION -- ALL NODES OVERLAID ON ONE AXIS
# =============================================================================
# I[k] already plays both roles at once: it's node k's own receiver readout
# (consumed by k_slow*I[k]*S[k] inside eq_S[k]) AND, completely unchanged,
# the exact same CellVariable referenced again as the production source
# k_p*I[k-1] when building node (k+1)'s equation a few sections up. There
# was never a separate "template" variable to plot -- this just overlays
# that one shared quantity for every node so the whole chain's cascade is
# visible on a single time axis, instead of split across separate rows.

fig_overlay, ax_overlay = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, N_NODES))

for i in range(N_NODES):
    ax_overlay.plot(time_points, node_data[i]['I_nM'], color=colors[i],
                     linewidth=2, label=f'Node {i + 1}: [I_{i + 1}]')

ax_overlay.set_xlabel('Time (hours)')
ax_overlay.set_ylabel('Template / Receiver Concentration [I] (nM)')
ax_overlay.set_title(f'{N_NODES}-Node Transmission Line: [I] at Every Node',
                      fontweight='bold')
ax_overlay.legend()
ax_overlay.grid(True, alpha=0.3)
ax_overlay.set_ylim(bottom=0)

plt.tight_layout()
overlay_filename = f'TransmissionLine_N{N_NODES}_overlay_I.png'
plt.savefig(overlay_filename, dpi=300, bbox_inches='tight')

# =============================================================================
# FINAL-TIMESTEP SPATIAL HEAT MAP -- ALL NODES AT ONCE
# =============================================================================
# Two fields, both evaluated at the last time step:
#   left  = receiver concentration [I_k] -- localized to each node (I_k is
#           only nonzero within node k's own circular mask), so this panel
#           reads as N colored blobs laid out along the line, one per node.
#   right = total diffusing signal (S_k + S_k:I_k + S_k:Th_k, summed over
#           every node k=1..N-1) -- this one DOES spread through the whole
#           bath, so it shows how far each relay's signal has propagated.
# The bath is deliberately much larger than the node cluster (so the
# reflective outer boundary doesn't affect near-node behavior), which makes
# a full-domain view show the nodes as a tiny speck -- especially for [I],
# which never leaves its own node. So each field gets both a full-domain row
# (top) and a row zoomed on just the node cluster (bottom), where the actual
# per-node detail is visible.
# Built with a dense scatter of cell centers (same technique already used in
# Mesh/New_simple_mesh.py's mesh visualizations) rather than a triangulated
# tripcolor, since it needs no assumption about FiPy's internal cell-to-
# triangle ordering matching the raw .msh file.

receiver_field_nM = I[0].value.copy()
for k in range(1, N_NODES):
    receiver_field_nM = receiver_field_nM + I[k].value
receiver_field_nM = receiver_field_nM * 1000

signal_field_nM = np.zeros(len(x))
for k in range(1, N_NODES):
    signal_field_nM = signal_field_nM + (S[k].value + SI[k].value + STh[k].value)
signal_field_nM = signal_field_nM * 1000

zoom_pad = 3.0 * node_radius
zoom_xlim = (min(node_centers_x) - zoom_pad, max(node_centers_x) + zoom_pad)
zoom_ylim = (y_center - zoom_pad, y_center + zoom_pad)

fig_hm, axes_hm = plt.subplots(2, 2, figsize=(20, 12))
fig_hm.suptitle(f'{N_NODES}-Node Transmission Line: Final-Timestep Spatial Heat Map '
                f'(t = {current_time / 3600:.2f} hr)', fontsize=16, fontweight='bold')

subsample = max(1, len(x) // 60000)
fields = [
    (receiver_field_nM, 'inferno', 'cyan', '[I] receiver concentration (nM)',
     'Receiver ([I]) concentration by node'),
    (signal_field_nM, 'viridis', 'red', 'Total diffusing signal [S] (nM)',
     'Diffusing signal spread across the bath'),
]

for col, (field_nM, cmap, circle_color, cbar_label, title) in enumerate(fields):
    for row, (xlim, ylim, zoomed) in enumerate([
            (None, None, False),
            (zoom_xlim, zoom_ylim, True)]):
        ax = axes_hm[row, col]
        marker_size = 30 if zoomed else 4
        stride = 1 if zoomed else subsample
        sc = ax.scatter(x[::stride], y[::stride], c=field_nM[::stride],
                         s=marker_size, cmap=cmap)
        for i, cx in enumerate(node_centers_x):
            circle = plt.Circle((cx, y_center), node_radius, fill=False,
                                 edgecolor=circle_color, linewidth=1.5)
            ax.add_patch(circle)
            ax.text(cx, y_center + node_radius + (30 if zoomed else 100), f'Node {i + 1}',
                    ha='center', fontsize=9, fontweight='bold')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label, fontweight='bold')
        ax.set_xlabel('X Position (um)')
        ax.set_ylabel('Y Position (um)')
        ax.set_title(f'{title}{" (zoomed on nodes)" if zoomed else " (full domain)"}',
                     fontweight='bold')
        ax.set_aspect('equal')
        if zoomed:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

plt.tight_layout()
heatmap_filename = f'TransmissionLine_N{N_NODES}_heatmap_final.png'
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')

plt.show()
