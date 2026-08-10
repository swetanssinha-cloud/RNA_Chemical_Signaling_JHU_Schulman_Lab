"""
diagnose_mesh.py  --  READ-ONLY diagnostic. Changes nothing in your project.

WHAT THIS ANSWERS
-----------------
Three separate questions that the parameter-sweep failures could come from:

  Q1. Is the mesh actually ONE connected region?
      New_simple_mesh.py builds a rectangle and two disks but never does a
      boolean operation, so the disks may sit ON TOP of the rectangle as
      separate, disconnected islands instead of being part of it.

  Q2. Which cell does the probe land on?
      Every script measures the receiver with
          receiver_center_idx = argmin(distance to receiver center)
      If that cell belongs to a disconnected island, it can never receive S2,
      and the run reports exactly 0.0 forever.

  Q3. Does generation ORDER change the mesh?  (Swetan's hypothesis)
      create_gmsh_radial_mesh() calls gmsh.initialize() / gmsh.finalize() and
      reuses the model name "radial_adaptive_mesh" every time. If gmsh state
      leaks between calls, meshes built later in a batch differ from meshes
      built first. We test this by building the same set of meshes three ways
      and comparing.

HOW TO READ THE OUTPUT
----------------------
For each mesh you get one line per check. The two that matter most:

    components = 1                  -> GOOD, mesh is one connected region
    components = 3                  -> BAD, the nodes are separate islands

    probe connected to sender = YES -> GOOD, probe can actually see signal
    probe connected to sender = NO  -> BAD, probe is in a sealed pocket

Then a comparison table at the end. If the three build methods produce
different cell counts for the same ccd, order/state leakage is real.

Written meshes go to a scratch folder and are deleted at the end.
"""

import collections
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make "Mesh.New_simple_mesh" importable no matter where this is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SETTINGS  -- edit these if you want
# =============================================================================

# Distances to test. Includes the ones that worked (200/300/500) and the ones
# that failed (800/1200) so we can compare them side by side.
CCD_VALUES = [200.0, 300.0, 500.0, 800.0, 1200.0]

# Gold-standard mesh resolution from TG_Rmesh_tanh.py.
FINE_DX = 5.0
COARSE_DX = 100.0

NODE_DIAMETER = 75.0
BATH_WIDTH = 1e4
BATH_HEIGHT = 1e3

SCRATCH = Path(tempfile.mkdtemp(prefix="mesh_diag_"))


# =============================================================================
# LOW-LEVEL: read a .msh file directly, without FiPy
# =============================================================================

def read_msh(path):
    """Parse a Gmsh 2.2 ASCII file. Returns (coords dict, triangles, surf tags)."""
    text = Path(path).read_text().split("\n")

    i = text.index("$Nodes")
    n_nodes = int(text[i + 1])
    coords = {}
    for k in range(n_nodes):
        parts = text[i + 2 + k].split()
        coords[int(parts[0])] = (float(parts[1]), float(parts[2]))

    j = text.index("$Elements")
    n_elem = int(text[j + 1])
    triangles = []
    surf_tags = []
    for k in range(n_elem):
        parts = text[j + 2 + k].split()
        elem_type = int(parts[1])
        n_tags = int(parts[2])
        if elem_type == 2:  # 2 == 3-node triangle
            triangles.append([int(v) for v in parts[3 + n_tags:]])
            surf_tags.append(int(parts[3 + n_tags - 1]))

    return coords, np.array(triangles), np.array(surf_tags)


def connected_components(triangles):
    """Group triangles that share at least one vertex. Returns label per triangle."""
    parent = list(range(len(triangles)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    node_to_tri = collections.defaultdict(list)
    for ti, tri in enumerate(triangles):
        for v in tri:
            node_to_tri[v].append(ti)

    for tri_list in node_to_tri.values():
        for t in tri_list[1:]:
            union(tri_list[0], t)

    return np.array([find(t) for t in range(len(triangles))])


# =============================================================================
# ANALYSIS OF ONE MESH FILE
# =============================================================================

def analyse(mesh_path, ccd, label):
    coords, triangles, surf_tags = read_msh(mesh_path)

    node_radius = NODE_DIAMETER / 2.0
    y_center = BATH_HEIGHT / 2.0
    sender_x = BATH_WIDTH / 2.0 - ccd / 2.0
    receiver_x = BATH_WIDTH / 2.0 + ccd / 2.0

    centroids = np.array([
        [np.mean([coords[v][0] for v in tri]), np.mean([coords[v][1] for v in tri])]
        for tri in triangles
    ])

    comp = connected_components(triangles)
    n_comp = len(set(comp))

    d_sender = np.hypot(centroids[:, 0] - sender_x, centroids[:, 1] - y_center)
    d_receiver = np.hypot(centroids[:, 0] - receiver_x, centroids[:, 1] - y_center)

    in_sender = d_sender <= node_radius
    in_receiver = d_receiver <= node_radius

    # This is the exact line every one of your scripts uses to pick the probe.
    probe = int(np.argmin(d_receiver))

    # Can the probe cell actually reach the sender through the mesh?
    sender_comps = set(comp[in_sender]) if in_sender.any() else set()
    probe_connected = comp[probe] in sender_comps

    print(f"  [{label}] ccd={ccd:.0f}")
    print(f"      triangles                  : {len(triangles)}")
    print(f"      surface tag groups         : {dict(collections.Counter(surf_tags))}")
    print(f"      connected components       : {n_comp}"
          f"   {'<-- BAD (should be 1)' if n_comp != 1 else '(good)'}")
    print(f"      cells inside sender node   : {int(in_sender.sum())}")
    print(f"      cells inside receiver node : {int(in_receiver.sum())}")
    print(f"      probe cell distance to ctr : {d_receiver[probe]:.4f} um")
    print(f"      probe connected to sender  : "
          f"{'YES (good)' if probe_connected else 'NO  <-- BAD, sealed pocket'}")

    return {
        "ccd": ccd,
        "triangles": len(triangles),
        "components": n_comp,
        "in_sender": int(in_sender.sum()),
        "in_receiver": int(in_receiver.sum()),
        "probe_connected": bool(probe_connected),
    }


# =============================================================================
# BUILDING MESHES
# =============================================================================

def build_one(ccd, out_path):
    """Build a single mesh in THIS process."""
    from Mesh.New_simple_mesh import create_gmsh_radial_mesh

    create_gmsh_radial_mesh(
        bath_width=BATH_WIDTH,
        bath_height=BATH_HEIGHT,
        node_diameter=NODE_DIAMETER,
        distance_between_nodes=ccd,
        min_cell_size=FINE_DX,
        max_cell_size=COARSE_DX,
        growth_rate=1.5,
        mesh_filename=str(out_path),
        visualize_gmsh=False,
        verbose=False,
    )


def build_in_subprocess(ccd, out_path):
    """Build a mesh in a FRESH python process, so no gmsh state can leak in."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--build-one",
         str(ccd), str(out_path)],
        check=True,
        capture_output=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("MESH DIAGNOSTIC")
    print(f"fine_dx={FINE_DX}  coarse_dx={COARSE_DX}  node_diameter={NODE_DIAMETER}")
    print(f"scratch folder: {SCRATCH}")
    print("=" * 78)

    results = {}

    # --- Method A: each mesh built in its own fresh process (cleanest possible)
    print("\nMETHOD A: each mesh in a FRESH subprocess (no state can leak)\n")
    results["A_subprocess"] = []
    for ccd in CCD_VALUES:
        path = SCRATCH / f"A_{ccd:.0f}.msh"
        build_in_subprocess(ccd, path)
        results["A_subprocess"].append(analyse(path, ccd, "A"))

    # --- Method B: all meshes in ONE process, forward order (what your sweeps do)
    print("\nMETHOD B: all meshes in ONE process, FORWARD order"
          "  (this is what generate_all_meshes_sequentially does)\n")
    results["B_forward"] = []
    for ccd in CCD_VALUES:
        path = SCRATCH / f"B_{ccd:.0f}.msh"
        build_one(ccd, path)
        results["B_forward"].append(analyse(path, ccd, "B"))

    # --- Method C: all meshes in ONE process, reversed order
    print("\nMETHOD C: all meshes in ONE process, REVERSED order\n")
    results["C_reversed"] = []
    for ccd in reversed(CCD_VALUES):
        path = SCRATCH / f"C_{ccd:.0f}.msh"
        build_one(ccd, path)
        results["C_reversed"].append(analyse(path, ccd, "C"))
    results["C_reversed"].reverse()

    # --- Comparison
    print("\n" + "=" * 78)
    print("COMPARISON  --  does build order change the mesh?")
    print("=" * 78)
    print(f"{'ccd':>8} | {'A subproc':>12} | {'B forward':>12} | {'C reversed':>12} | same?")
    print("-" * 78)

    order_matters = False
    for idx, ccd in enumerate(CCD_VALUES):
        a = results["A_subprocess"][idx]["triangles"]
        b = results["B_forward"][idx]["triangles"]
        c = results["C_reversed"][idx]["triangles"]
        same = (a == b == c)
        if not same:
            order_matters = True
        print(f"{ccd:>8.0f} | {a:>12} | {b:>12} | {c:>12} | "
              f"{'yes' if same else 'NO  <-- order matters!'}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    any_disconnected = any(
        r["components"] != 1 for group in results.values() for r in group
    )
    any_sealed = any(
        not r["probe_connected"] for group in results.values() for r in group
    )

    print(f"  Mesh is non-conformal (islands)      : "
          f"{'YES - needs occ.fragment()' if any_disconnected else 'no'}")
    print(f"  Probe ever lands in a sealed pocket  : "
          f"{'YES - this is the 0.0 bug' if any_sealed else 'no'}")
    print(f"  Build order changes the mesh         : "
          f"{'YES - gmsh state leaks between calls' if order_matters else 'no'}")
    print("=" * 78)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"\nScratch folder deleted. Nothing in your project was modified.\n")


if __name__ == "__main__":
    # Internal hook used by build_in_subprocess(); not for manual use.
    if len(sys.argv) == 4 and sys.argv[1] == "--build-one":
        build_one(float(sys.argv[2]), sys.argv[3])
    else:
        main()
