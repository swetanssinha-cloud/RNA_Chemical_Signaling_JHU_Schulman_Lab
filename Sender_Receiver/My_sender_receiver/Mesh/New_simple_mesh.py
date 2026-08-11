"""
Gmsh Radial Adaptive Mesh for FiPy - TRUE Radial Refinement
FIXED: Uses Gmsh format 2.2 for FiPy compatibility
"""

import numpy as np
import gmsh
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import sys
import os
import tempfile
import time
from fipy import Gmsh2D


def create_gmsh_radial_mesh(
    bath_width=10000.0,           # μm (1 cm)
    bath_height=1000.0,           # μm (1 mm)
    node_diameter=75.0,           # μm
    distance_between_nodes=300.0, # μm (center-to-center)
    min_cell_size=0.75,          # μm (finest mesh at node surface)
    max_cell_size=50.0,          # μm (coarsest mesh far from nodes)
    growth_rate=1.5,             # How fast mesh grows with distance
    mesh_filename=None,
    visualize_gmsh=False,        # Show Gmsh GUI
    verbose=True
):
    """
    Create TRUE radial adaptive mesh using Gmsh.
    Fine mesh ONLY near circular nodes (radially), coarse everywhere else.
    
    FIXED: Saves mesh in Gmsh format 2.2 for FiPy compatibility.
    
    Parameters:
    -----------
    bath_width, bath_height : float
        Domain dimensions in μm
    node_diameter : float
        Diameter of circular hydrogel nodes in μm
    distance_between_nodes : float
        Center-to-center distance between sender and receiver nodes in μm
    min_cell_size : float
        Finest mesh size at node surfaces in μm (must be 0.75 for tanh compatibility)
    max_cell_size : float
        Coarsest mesh size far from nodes in μm
    growth_rate : float
        How quickly mesh transitions from fine to coarse (1.3 = slow, 2.0 = fast)
    mesh_filename : str
        Output filename for mesh file
    visualize_gmsh : bool
        If True, opens Gmsh GUI to visualize mesh
    verbose : bool
        Print mesh statistics
    
    Returns:
    --------
    mesh_filename : path to saved Gmsh mesh file
    sender_center_x : float
    receiver_center_x : float
    y_center : float
    node_radius : float
    """
    if mesh_filename is None:
        mesh_filename = "default_mesh.msh"

    node_radius = node_diameter / 2.0
    y_center = bath_height / 2.0
    
    # Position nodes symmetrically around domain center
    domain_center_x = bath_width / 2.0
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    
    # Create new model
    gmsh.model.add("radial_adaptive_mesh")
    
    # =============================================================================
    # GEOMETRY CREATION
    # =============================================================================
    
    # Create the rectangular bath domain
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    
    # Create circular sender node
    sender_circle_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0, 
                                                node_radius, node_radius)
    
    # Create circular receiver node
    receiver_circle_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                                  node_radius, node_radius)
    
    # Synchronize CAD entities before mesh operations
    gmsh.model.occ.synchronize()
    
# =============================================================================
# MESH SIZE FIELD - TRUE RADIAL REFINEMENT + FINE INSIDE NODES
# =============================================================================

    # Get boundaries (curves) of the circular nodes
    sender_boundary = gmsh.model.getBoundary([(2, sender_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)
    receiver_boundary = gmsh.model.getBoundary([(2, receiver_circle_tag)], 
                                            oriented=False, combined=False, recursive=False)

    # Extract curve tags (boundaries are 1D entities)
    sender_curve_tags = [abs(tag) for dim, tag in sender_boundary]
    receiver_curve_tags = [abs(tag) for dim, tag in receiver_boundary]

    if verbose:
        print(f"\nSender boundary curves: {sender_curve_tags}")
        print(f"Receiver boundary curves: {receiver_curve_tags}")

    # CREATE EXPLICIT CENTER POINTS for distance fields
    sender_center_point = gmsh.model.occ.addPoint(sender_center_x, y_center, 0)
    receiver_center_point = gmsh.model.occ.addPoint(receiver_center_x, y_center, 0)

    # Synchronize to register the new points
    gmsh.model.occ.synchronize()

    if verbose:
        print(f"Created sender center point: {sender_center_point}")
        print(f"Created receiver center point: {receiver_center_point}")

    # Distance field from SENDER node boundary (edge)
    distance_field_sender_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_boundary, "CurvesList", sender_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_sender_boundary, "Sampling", 200)

    # Distance field from RECEIVER node boundary (edge)
    distance_field_receiver_boundary = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_boundary, "CurvesList", receiver_curve_tags)
    gmsh.model.mesh.field.setNumber(distance_field_receiver_boundary, "Sampling", 200)

    # Distance field from SENDER CENTER POINT
    distance_field_sender_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_sender_center, "PointsList", [sender_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_sender_center, "Sampling", 200)

    # Distance field from RECEIVER CENTER POINT
    distance_field_receiver_center = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field_receiver_center, "PointsList", [receiver_center_point])
    gmsh.model.mesh.field.setNumber(distance_field_receiver_center, "Sampling", 200)

# Take MINIMUM distance to any boundary OR center
    min_distance_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_distance_field, "FieldsList", 
                                    [distance_field_sender_boundary, 
                                    distance_field_receiver_boundary,
                                    distance_field_sender_center,
                                    distance_field_receiver_center])

    # =============================================================================
    # Create Area where refinement is applied
    # =============================================================================
    # Define refinement zone: fine mesh extends this far from nodes
    refinement_radius = 200.0  # μm 

    # Create Threshold field for smooth transition
    # Keeps mesh fine out to refinement_radius, then gradually coarsens
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", min_distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", min_cell_size)      # 0.75 μm at nodes
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_cell_size)      # 50 μm far away
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)                # Fine mesh starts at node
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_radius)  # Fine mesh ends here
    gmsh.model.mesh.field.setNumber(threshold_field, "Sigmoid", 1)                  # Smooth transition
        
    # Set this as the background mesh field
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # Disable other mesh size determination methods
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Algorithm options
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # =============================================================================
    # GENERATE MESH
    # =============================================================================
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Optional: optimize mesh quality

    gmsh.model.mesh.optimize("Netgen")
    
    # =============================================================================
    # STATISTICS
    # =============================================================================
    
    if verbose:
        # Get mesh statistics
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags_list, element_node_tags = gmsh.model.mesh.getElements(dim=2)
        
        n_nodes = len(node_tags)
        n_triangles = sum(len(tags) for tags in element_tags_list)
    
    # =============================================================================
    # SAVE MESH IN FORMAT 2.2 (FIPY COMPATIBLE)
    # =============================================================================
    
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    gmsh.write(mesh_filename)

    # Finalize Gmsh
    gmsh.finalize()
    
    return mesh_filename, sender_center_x, receiver_center_x, y_center


def visualize_gmsh_mesh_comprehensive(
    mesh, sender_x, receiver_x, y_center, node_radius, distance_between,
    save_filename='gmsh_radial_mesh_visualization.png'
):
    """
    Comprehensive 4-panel visualization of the Gmsh radial mesh.
    Shows that mesh is fine ONLY near nodes (radially), not in bands.
    """
    
    x_coords, y_coords = mesh.cellCenters
    x_coords = x_coords
    y_coords = y_coords
    
    # Calculate distance to nearest node surface
    def distance_to_nearest_node_surface(x, y):
        nodes = np.array([[sender_x, y_center], [receiver_x, y_center]])
        distances = []
        for node_x, node_y in nodes:
            dist_to_center = np.sqrt((x - node_x)**2 + (y - node_y)**2)
            dist_to_surface = max(0.0, dist_to_center - node_radius)
            distances.append(dist_to_surface)
        return min(distances)
    
    distances = np.array([distance_to_nearest_node_surface(x, y) 
                         for x, y in zip(x_coords, y_coords)])
    
    # Create masks for nodes
    sender_mask = np.sqrt((x_coords - sender_x)**2 + 
                          (y_coords - y_center)**2) <= node_radius
    receiver_mask = np.sqrt((x_coords - receiver_x)**2 + 
                            (y_coords - y_center)**2) <= node_radius
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    # =============================================================================
    # Panel 1: Full domain with distance coloring
    # =============================================================================
    ax1 = plt.subplot(2, 2, 1)
    
    subsample = max(1, len(x_coords) // 50000)
    scatter1 = ax1.scatter(x_coords[::subsample], y_coords[::subsample], 
                          c=distances[::subsample], s=2, cmap='RdYlGn_r',
                          alpha=0.7, vmin=0, vmax=min(300, distances.max()))
    
    # Draw node circles
    sender_circle = plt.Circle((sender_x, y_center), node_radius,
                               fill=False, edgecolor='blue', linewidth=2.5, 
                               linestyle='-', label='Sender node')
    receiver_circle = plt.Circle((receiver_x, y_center), node_radius,
                                 fill=False, edgecolor='red', linewidth=2.5,
                                 linestyle='-', label='Receiver node')
    ax1.add_patch(sender_circle)
    ax1.add_patch(receiver_circle)
    
    # Distance annotation
    ax1.plot([sender_x, receiver_x], [y_center, y_center],
            'k--', linewidth=1.5, alpha=0.6)
    mid_x = (sender_x + receiver_x) / 2
    ax1.text(mid_x, y_center + 80, f'{distance_between:.0f} μm', 
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Distance to Nearest Node Surface (μm)', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax1.set_title('Full Domain: True Radial Refinement (NO BANDS!)', 
                 fontsize=15, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # =============================================================================
    # Panel 2: Zoomed view of sender node
    # =============================================================================
    ax2 = plt.subplot(2, 2, 2)
    
    zoom_size = node_radius * 6
    x_min, x_max = sender_x - zoom_size, sender_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    zoom_mask = ((x_coords >= x_min) & (x_coords <= x_max) & 
                 (y_coords >= y_min) & (y_coords <= y_max))
    
    scatter2 = ax2.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
                          c=distances[zoom_mask], s=20, cmap='RdYlGn_r',
                          alpha=0.8, vmin=0, vmax=100)
    
    sender_circle_zoom = plt.Circle((sender_x, y_center), node_radius,
                                    fill=False, edgecolor='blue', linewidth=3)
    ax2.add_patch(sender_circle_zoom)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Distance (μm)', fontsize=11)
    
    ax2.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Sender Node (R={node_radius:.1f} μm) - Radial Refinement', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    
    # =============================================================================
    # Panel 3: Zoomed view of receiver node
    # =============================================================================
    ax3 = plt.subplot(2, 2, 3)
    
    x_min, x_max = receiver_x - zoom_size, receiver_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    zoom_mask = ((x_coords >= x_min) & (x_coords <= x_max) & 
                 (y_coords >= y_min) & (y_coords <= y_max))
    
    scatter3 = ax3.scatter(x_coords[zoom_mask], y_coords[zoom_mask], 
                          c=distances[zoom_mask], s=20, cmap='RdYlGn_r',
                          alpha=0.8, vmin=0, vmax=100)
    
    receiver_circle_zoom = plt.Circle((receiver_x, y_center), node_radius,
                                      fill=False, edgecolor='red', linewidth=3)
    ax3.add_patch(receiver_circle_zoom)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Distance (μm)', fontsize=11)
    
    ax3.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax3.set_title(f'Receiver Node (R={node_radius:.1f} μm) - Radial Refinement', 
                 fontsize=14, fontweight='bold', pad=10)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([y_min, y_max])
    
    # =============================================================================
    # Panel 4: Cross-sections showing NO BANDS
    # =============================================================================
    ax4 = plt.subplot(2, 2, 4)
    
    # Horizontal cross-section at Y = y_center
    y_tol = 20  # μm
    h_mask = np.abs(y_coords - y_center) < y_tol
    x_h = x_coords[h_mask]
    d_h = distances[h_mask]
    sort_idx = np.argsort(x_h)
    x_h, d_h = x_h[sort_idx], d_h[sort_idx]
    
    # Vertical cross-section at X = midpoint between nodes
    mid_x = (sender_x + receiver_x) / 2
    x_tol = 50  # μm
    v_mask = np.abs(x_coords - mid_x) < x_tol
    y_v = y_coords[v_mask]
    d_v = distances[v_mask]
    sort_idx = np.argsort(y_v)
    y_v, d_v = y_v[sort_idx], d_v[sort_idx]
    
    ax4.plot(x_h, d_h, 'b-', linewidth=2.5, label='Horizontal (Y=center)', alpha=0.8)
    ax4.plot(y_v, d_v, 'r-', linewidth=2.5, label='Vertical (X=midpoint)', alpha=0.8)
    
    # Mark node positions on horizontal
    ax4.axvline(sender_x, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax4.axvline(receiver_x, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax4.text(sender_x, ax4.get_ylim()[1]*0.9, 'Sender', ha='center', 
             fontsize=10, color='blue', fontweight='bold')
    ax4.text(receiver_x, ax4.get_ylim()[1]*0.9, 'Receiver', ha='center',
             fontsize=10, color='red', fontweight='bold')
    
    ax4.set_xlabel('Position (μm)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Distance to Node Surface (μm)', fontsize=13, fontweight='bold')
    ax4.set_title('Cross-Sections: Proving NO BANDS (distance increases radially)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comprehensive visualization saved: {save_filename}")
    plt.show()



def visualize_triangle_mesh(mesh_filename, sender_x, receiver_x, y_center, node_radius,
                           zoom_sender=True, zoom_size=150.0,
                           save_filename='gmsh_triangle_mesh.png'):
    """
    Visualize the actual triangular mesh structure (edges and vertices).
    Shows individual triangles near the nodes.
    
    Parameters:
    -----------
    mesh : FiPy Gmsh2D mesh
    sender_x, receiver_x : float
        Node center X coordinates
    y_center : float
        Node center Y coordinate
    node_radius : float
        Node radius
    zoom_sender : bool
        If True, zoom on sender; if False, zoom on receiver
    zoom_size : float
        Size of zoom window (μm from node center)
    """
    mesh = Gmsh2D(mesh_filename)
    # Get mesh vertices and face connectivity
    x_verts = mesh.vertexCoords[0]
    y_verts = mesh.vertexCoords[1]
    
    # Get face vertex IDs (triangles)
    face_vertex_ids = mesh.faceVertexIDs
    
    # Choose zoom center
    zoom_center_x = sender_x if zoom_sender else receiver_x
    node_name = "Sender" if zoom_sender else "Receiver"
    node_color = 'blue' if zoom_sender else 'red'
    
    # Create zoom mask
    x_min, x_max = zoom_center_x - zoom_size, zoom_center_x + zoom_size
    y_min, y_max = y_center - zoom_size, y_center + zoom_size
    
    # Filter vertices in zoom region
    vert_mask = ((x_verts >= x_min) & (x_verts <= x_max) & 
                 (y_verts >= y_min) & (y_verts <= y_max))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot all mesh edges (triangle edges)
    print(f"Plotting triangular mesh edges...")
    for i in range(face_vertex_ids.shape[1]):
        v0, v1 = face_vertex_ids[:, i]
        
        # Check if either vertex is in zoom region
        if vert_mask[v0] or vert_mask[v1]:
            x_line = [x_verts[v0], x_verts[v1]]
            y_line = [y_verts[v0], y_verts[v1]]
            
            # Only plot if line segment is in zoom region
            if (min(x_line) < x_max and max(x_line) > x_min and
                min(y_line) < y_max and max(y_line) > y_min):
                ax.plot(x_line, y_line, 'k-', linewidth=0.5, alpha=0.6)
    
    # Plot vertices in zoom region
    ax.scatter(x_verts[vert_mask], y_verts[vert_mask], 
              c='darkblue', s=8, alpha=0.8, zorder=5, label='Mesh vertices')
    
    # Draw node circle
    node_circle = plt.Circle((zoom_center_x, y_center), node_radius,
                            fill=False, edgecolor=node_color, linewidth=3,
                            label=f'{node_name} node')
    ax.add_patch(node_circle)
    
    ax.set_xlabel('X Position (μm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (μm)', fontsize=13, fontweight='bold')
    ax.set_title(f'Triangular Mesh Structure: {node_name} Node (zoom {zoom_size:.0f} μm)', 
                fontsize=15, fontweight='bold', pad=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Triangle mesh visualization saved: {save_filename}")
    plt.show()
    
    return fig

# =============================================================================
# CONFORMAL MESH (use this one)
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# create_gmsh_radial_mesh() above builds a rectangle and two disks but never
# performs a boolean operation on them. Gmsh therefore meshes all three
# surfaces INDEPENDENTLY, so the disks end up as separate, disconnected
# islands sitting on top of the rectangle mesh. They share no vertices with
# it, so no molecule can ever diffuse between a node and the bath.
#
# Diagnostic output on every distance tested (200-1200 um):
#     connected components : 3      <- should be 1
#     ~369 orphan cells per node, frozen at their initial values forever
#
# The fix is occ.fragment(), which imprints the disk boundaries into the
# rectangle. The result is ONE conformal region: three faces that share their
# boundary curves and their mesh nodes. Unlike occ.cut(), fragment KEEPS the
# node interiors as part of the domain, which is what we need since I1O2, I2
# and Th2 all live inside the gel.
# =============================================================================

def check_mesh_is_conformal(mesh_filename, verbose=True):
    """
    Parse a Gmsh 2.2 file and count how many disconnected pieces it contains.

    Returns the number of connected components. Should always be 1.
    Cheap enough to call after every mesh build.
    """
    import collections

    with open(mesh_filename) as handle:
        text = handle.read().split("\n")

    j = text.index("$Elements")
    n_elem = int(text[j + 1])

    triangles = []
    for k in range(n_elem):
        parts = text[j + 2 + k].split()
        if int(parts[1]) == 2:  # element type 2 == 3-node triangle
            n_tags = int(parts[2])
            triangles.append([int(v) for v in parts[3 + n_tags:]])

    parent = list(range(len(triangles)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    node_to_tri = collections.defaultdict(list)
    for ti, tri in enumerate(triangles):
        for v in tri:
            node_to_tri[v].append(ti)

    for tri_list in node_to_tri.values():
        root = find(tri_list[0])
        for t in tri_list[1:]:
            rt = find(t)
            if rt != root:
                parent[rt] = root

    n_components = len({find(t) for t in range(len(triangles))})

    if verbose:
        status = "OK" if n_components == 1 else "BAD - mesh is not conformal!"
        print(f"  conformality check: {len(triangles)} triangles, "
              f"{n_components} connected component(s)  [{status}]")

    return n_components


def create_conformal_radial_mesh(
    bath_width=10000.0,            # um
    bath_height=1000.0,            # um
    node_diameter=75.0,            # um
    distance_between_nodes=300.0,  # um, centre-to-centre
    min_cell_size=5.0,             # um, finest cells at the node surfaces
    max_cell_size=100.0,           # um, coarsest cells far away
    growth_rate=1.5,               # cell size multiplier between rings
    cells_per_level=3,             # how many cells wide each ring is
    mesh_filename=None,
    visualize_gmsh=False,          # open the Gmsh GUI after meshing
    verbose=True,
):
    """
    Build a radially-refined triangular mesh in which the two hydrogel nodes
    are genuinely part of the domain.

    Two differences from create_gmsh_radial_mesh():

    1. occ.fragment() imprints the node disks into the bath, so the mesh is
       ONE connected region instead of three overlapping pieces.

    2. Cell size grows in discrete geometric RINGS rather than along a single
       smooth sigmoid ramp. Ring i holds a constant size

           size_i = min_cell_size * growth_rate**i

       for roughly `cells_per_level` cells' worth of radial distance, then
       steps up. The last ring is clamped to max_cell_size and reaches the
       domain edge.

       Why rings instead of a sigmoid: the sigmoid squeezed the entire
       5 -> 100 um transition into a narrow band and produced neighbouring
       cells differing in size by up to 1.99x. Rings bound that ratio by
       construction (measured worst case 1.55 at growth_rate=1.5) and spread
       the coarsening over a much wider band, at a cost of ~30% more cells.
       Ring boundaries are derived from the cell sizes themselves, so they
       rescale automatically when min_cell_size or max_cell_size change --
       unlike the sigmoid, whose transition was pinned to a hard-coded
       absolute distance.

    NOTE: unlike the older create_gmsh_radial_mesh(), which accepted a
    `growth_rate` argument and then silently ignored it, every parameter here
    is used.

    Returns
    -------
    mesh_filename    : str, path to the written .msh (Gmsh format 2.2)
    sender_center_x  : float
    receiver_center_x: float
    y_center         : float
    """
    if mesh_filename is None:
        mesh_filename = "conformal_mesh.msh"
    mesh_filename = str(mesh_filename)

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
    sender_center_x = domain_center_x - distance_between_nodes / 2.0
    receiver_center_x = domain_center_x + distance_between_nodes / 2.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("conformal_radial_mesh")

    # ---------------------------------------------------------------- geometry
    rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, bath_width, bath_height)
    sender_tag = gmsh.model.occ.addDisk(sender_center_x, y_center, 0,
                                        node_radius, node_radius)
    receiver_tag = gmsh.model.occ.addDisk(receiver_center_x, y_center, 0,
                                          node_radius, node_radius)

    # THE FIX. fragment() imprints the disks into the rectangle so that all
    # three faces share boundary curves and, after meshing, share nodes.
    fragments, _ = gmsh.model.occ.fragment(
        [(2, rectangle_tag)],
        [(2, sender_tag), (2, receiver_tag)],
    )
    gmsh.model.occ.synchronize()

    # fragment() renumbers everything, so re-identify the faces by geometry.
    surface_tags = [tag for dim, tag in fragments if dim == 2]

    sender_surface = receiver_surface = None
    bath_surfaces = []

    for tag in surface_tags:
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if abs(cx - sender_center_x) < 1.0 and abs(cy - y_center) < 1.0:
            sender_surface = tag
        elif abs(cx - receiver_center_x) < 1.0 and abs(cy - y_center) < 1.0:
            receiver_surface = tag
        else:
            bath_surfaces.append(tag)

    if sender_surface is None or receiver_surface is None:
        gmsh.finalize()
        raise RuntimeError(
            "Could not identify the node surfaces after fragment(). "
            f"Found surfaces {surface_tags} for nodes at x={sender_center_x} "
            f"and x={receiver_center_x}."
        )

    # ------------------------------------------------------------ size fields
    sender_curves = [abs(t) for _, t in gmsh.model.getBoundary(
        [(2, sender_surface)], oriented=False, combined=False, recursive=False)]
    receiver_curves = [abs(t) for _, t in gmsh.model.getBoundary(
        [(2, receiver_surface)], oriented=False, combined=False, recursive=False)]

    # Distance to each node's surface, and to each node's centre. Taking the
    # minimum of all four keeps the interior of the nodes finely resolved too.
    center_points = [
        gmsh.model.occ.addPoint(sender_center_x, y_center, 0),
        gmsh.model.occ.addPoint(receiver_center_x, y_center, 0),
    ]
    gmsh.model.occ.synchronize()

    distance_fields = []
    for curves in (sender_curves, receiver_curves):
        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "CurvesList", curves)
        gmsh.model.mesh.field.setNumber(f, "Sampling", 200)
        distance_fields.append(f)

    for point in center_points:
        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "PointsList", [point])
        gmsh.model.mesh.field.setNumber(f, "Sampling", 200)
        distance_fields.append(f)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", distance_fields)

    # ------------------------------------------------ geometric ring schedule
    # levels[i]     = the cell size held throughout ring i
    # boundaries[i] = distance from the node surface at which ring i begins
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
            span = f"[{lo:7.1f}, {hi:7.1f})" if hi is not None else f"[{lo:7.1f},     inf)"
            print(f"  ring {i}: {size:7.2f} um  for distance {span} um")

    # Each ring is a Threshold that returns its own plateau size once past the
    # ring's start, and a huge value before it. Taking the Min over all rings
    # therefore yields, at any point, the size of the innermost ring that has
    # already begun -- i.e. a staircase rather than a ramp.
    threshold_ids = []
    for i, size in enumerate(levels):
        tid = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tid, "InField", min_field)
        if i < len(levels) - 1:
            transition = boundaries[i + 1]
            eps = max(0.02, 0.02 * transition)   # narrow but non-zero riser
            gmsh.model.mesh.field.setNumber(tid, "SizeMin", size)
            gmsh.model.mesh.field.setNumber(tid, "SizeMax", 1e6)
            gmsh.model.mesh.field.setNumber(tid, "DistMin", max(transition - eps, 0.0))
            gmsh.model.mesh.field.setNumber(tid, "DistMax", transition + eps)
        else:
            # Coarsest ring: constant everywhere, so the Min combination can
            # never exceed max_cell_size no matter how far out we go.
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
    # Tagging every surface means Gmsh still writes all elements. SaveAll is
    # belt-and-braces in case a surface is missed.
    gmsh.model.addPhysicalGroup(2, bath_surfaces, name="bath")
    gmsh.model.addPhysicalGroup(2, [sender_surface], name="sender_node")
    gmsh.model.addPhysicalGroup(2, [receiver_surface], name="receiver_node")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(mesh_filename)

    if visualize_gmsh:
        gmsh.fltk.run()

    gmsh.finalize()

    n_components = check_mesh_is_conformal(mesh_filename, verbose=verbose)
    if n_components != 1:
        raise RuntimeError(
            f"Mesh {mesh_filename} has {n_components} disconnected pieces "
            f"(expected 1). The nodes are not joined to the bath."
        )

    return mesh_filename, sender_center_x, receiver_center_x, y_center


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREATING GMSH RADIAL MESH - TRUE RADIAL REFINEMENT")
    print("="*70)
    
    # Create mesh with geometric growth (matching your requirements)
    mesh_filename, sender_x, receiver_x, y_ctr = create_conformal_radial_mesh(
        bath_width=10000.0,           # 1 cm
        bath_height=1000.0,           # 1 mm
        node_diameter=75.0,           # 75 μm diameter nodes
        distance_between_nodes=300.0, # 300 μm apart
        min_cell_size=0.75,          # 0.75 μm at node surface (for tanh)
        max_cell_size=50.0,          # 50 μm far away
        growth_rate=1.5,             #Growth Factor       
        verbose=True
    )
    
    # print("\n" + "="*70)
    # print("MESH CREATED SUCCESSFULLY!")
    # print("="*70)
    # print(f"\nMesh has {mesh.numberOfCells:,} cells")
    # print(f"This is an UNSTRUCTURED triangular mesh with TRUE radial refinement")
    # print(f"Fine cells ONLY near nodes (no horizontal/vertical bands!)")
    
    # Visualize the mesh
    # print("\nGenerating comprehensive visualization...")
    # visualize_gmsh_mesh_comprehensive(
    #     mesh, sender_x, receiver_x, y_ctr, distance_between=300.0, node_radius=37.5,
    #     save_filename='gmsh_radial_mesh_isolated_comprehensive.png'
    # )
    
    # print("\n" + "="*70)
    # print("READY TO USE IN YOUR FIPY SIMULATION!")
    # print("="*70)
    # print("\nJust use this mesh in your existing FiPy code:")
    # print("  S2 = CellVariable(name='S2', mesh=mesh, value=0.0)")
    # print("  ... all your equations work exactly the same!")
    # print("="*70 + "\n")

    # Visualize actual triangular mesh structure
    print("\nGenerating triangle mesh visualization...")
    visualize_triangle_mesh(
    mesh_filename, sender_x, receiver_x, y_ctr, node_radius=37.5,
    zoom_sender=False,  # Set False to see receiver instead
    zoom_size=250.0,   # Adjust to zoom in/out
    save_filename='gmsh_triangle_mesh_sender.png'
)